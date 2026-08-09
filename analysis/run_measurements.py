"""
run_measurements.py — 从一个 capture run 目录算出 m1-m4，输出 per-layer JSON

两种数据源：
  --source prefill   用 prefill 采样位置的真实 query（推荐）
  --source decode    用 decode step（截断模型下轨迹不可信，仅供打通链路）

"步"的统一抽象：一个样本 = (cache_len, q_I, w, kernel_topk)。
  prefill 位置 pos → cache_len = pos + 1
  decode step  s   → cache_len = n_prompt + s + 1
相邻样本用于 warm 集（τ）和 churn，故只取 cache_len 连续的样本对。

输出 JSON 可直接喂给 analysis/report.py 做一页表。

用法：
    python -m analysis.run_measurements runs/first5_32k_vllm_XXX \
        --source prefill --out runs/first5_32k_vllm_XXX/stats.json
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.loader import (load_k_I, load_step_dump, load_prefill_dump,
                           list_prefill_positions)
from replay.bounds import (compute_block_summaries_ball, ball_bound_upper,
                           compute_block_summaries_box, box_bound_upper,
                           exact_indexer_score)
from replay.exact_score import compute_exact_topk, compute_tau_from_warm_set
from replay.m1_hisa_recall import hisa_coverage
from replay.m2_w_stats import w_sign_stats, w_neg_mass
from replay.m3_block_locality import block_locality_stats
from replay.m4_bound_prune import prune_stats


def summary(values) -> dict:
    """分布摘要（约束 15：报 p5/p50/p95/p99，不报单一均值）。"""
    if len(values) == 0:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {"p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()), "n": int(a.size)}


def collect_samples(run_dir: str, layer: int, source: str, n_prompt: int):
    """返回 [(cache_len, loader_fn)]，按 cache_len 升序。"""
    if source == "prefill":
        return [(p + 1, (lambda p=p: load_prefill_dump(run_dir, layer, p)))
                for p in list_prefill_positions(run_dir, layer)]
    step = 0
    out = []
    while os.path.exists(
            os.path.join(run_dir, f"step{step:06d}_layer{layer:03d}.npz")):
        out.append((n_prompt + step + 1,
                    (lambda s=step: load_step_dump(run_dir, layer, s))))
        step += 1
    return out


def oracle_ceiling(q, w, k_I, tau, block_size):
    """块粒度剪枝的天花板：即使 bound 完全紧，能剪掉的块占比。

    = 1 - (块内真实最大分 > τ 的块占比)
    这个数低说明瓶颈在数据弥散度，不在 bound 松紧——比 F 本身更能定位问题。
    """
    n_full = k_I.shape[0] // block_size
    if n_full == 0:
        return 0.0
    sc = exact_indexer_score(q, w, k_I)
    block_max = sc[:n_full * block_size].reshape(n_full, block_size).max(1).values
    return float(1.0 - (block_max > tau).float().mean().item())


def bound_slack(q, w, k_I, block_size):
    """bound 松弛度：U / 真实块最大分 的中位数（越接近 1 越紧）。"""
    n_full = k_I.shape[0] // block_size
    if n_full == 0:
        return {}
    mu, r, _ = compute_block_summaries_ball(k_I, block_size)
    lo, hi, _ = compute_block_summaries_box(k_I, block_size)
    sc = exact_indexer_score(q, w, k_I)
    bmax = sc[:n_full * block_size].reshape(n_full, block_size).max(1).values
    denom = bmax.abs().clamp_min(1e-6)
    return {"ball": float(((ball_bound_upper(q, w, mu, r) - bmax)
                           / denom).median().item()),
            "box": float(((box_bound_upper(q, w, lo, hi) - bmax)
                          / denom).median().item())}


def measure_layer(run_dir, layer, source, n_prompt, block_size, top_k,
                  max_samples, hisa_m):
    samples = collect_samples(run_dir, layer, source, n_prompt)
    if len(samples) < 2:
        return {"error": f"need >=2 samples, got {len(samples)}"}

    k_all = load_k_I(run_dir, layer)
    # 只保留与前一个样本 cache_len 相邻的样本（warm 集/churn 要求连续）
    pairs = [(prev, cur) for prev, cur in zip(samples, samples[1:])
             if cur[0] == prev[0] + 1]
    if max_samples and len(pairs) > max_samples:
        stride = len(pairs) // max_samples
        pairs = pairs[::stride][:max_samples]

    acc = {k: [] for k in ("coverage", "touched", "churn", "jaccard",
                           "ball_oneshot", "ball_bestfirst", "box_oneshot",
                           "box_bestfirst", "tau", "oracle_ceiling",
                           "slack_ball", "slack_box")}
    w_rows = []
    for (prev_len, prev_load), (cur_len, cur_load) in pairs:
        prev, cur = prev_load(), cur_load()
        q, w = cur["q_I"], cur["w"]
        k_cur = k_all[:cur_len]
        w_rows.append(w)

        # m1
        acc["coverage"].append(
            hisa_coverage(q, w, k_cur, block_size, hisa_m, top_k)["coverage"])
        # m3（参照系是 replay exact top-k，约束 6）
        _, idx_cur = compute_exact_topk(q, w, k_cur, top_k)
        _, idx_prev = compute_exact_topk(prev["q_I"], prev["w"],
                                         k_all[:prev_len], top_k)
        loc = block_locality_stats(idx_cur, idx_prev, block_size)
        acc["touched"].append(loc["touched_blocks"])
        acc["churn"].append(loc["churn_rate"])
        acc["jaccard"].append(loc["fetched_block_jaccard"])
        # m4：warm 集 = 上一步 top-k 的 k^I，用【当前步】q 重新打分（约束 1）
        warm = k_all[idx_prev]
        pr = prune_stats(q, w, k_cur, warm, block_size, top_k)
        acc["ball_oneshot"].append(pr["ball_oneshot_F"])
        acc["ball_bestfirst"].append(pr["ball_bestfirst_F"])
        acc["box_oneshot"].append(pr["box_oneshot_F"])
        acc["box_bestfirst"].append(pr["box_bestfirst_F"])
        tau = compute_tau_from_warm_set(q, w, warm).item()
        acc["tau"].append(tau)
        acc["oracle_ceiling"].append(
            oracle_ceiling(q, w, k_cur, tau, block_size))
        sl = bound_slack(q, w, k_cur, block_size)
        acc["slack_ball"].append(sl["ball"])
        acc["slack_box"].append(sl["box"])

    W = torch.stack(w_rows)
    st, nm = w_sign_stats(W), w_neg_mass(W)
    return {
        "n_samples": len(pairs),
        "cache_len_range": [pairs[0][1][0], pairs[-1][1][0]],
        "m1_hisa": {"coverage": summary(acc["coverage"]), "m_blocks": hisa_m},
        "m2_w": {"neg_fraction": st["neg_fraction"],
                 "pos_fraction": st["pos_fraction"],
                 "neg_mass_ratio": nm["neg_mass_ratio"],
                 "w_min": st["w_min"], "w_max": st["w_max"],
                 "heads_always_neg": int(((W < 0).all(0)).sum())},
        "m3_locality": {"touched_blocks": summary(acc["touched"]),
                        "churn_rate": summary(acc["churn"]),
                        "fetched_block_jaccard": summary(acc["jaccard"])},
        "m4_prune": {"ball_oneshot_F": summary(acc["ball_oneshot"]),
                     "ball_bestfirst_F": summary(acc["ball_bestfirst"]),
                     "box_oneshot_F": summary(acc["box_oneshot"]),
                     "box_bestfirst_F": summary(acc["box_bestfirst"]),
                     "tau_0": summary(acc["tau"]),
                     "oracle_ceiling": summary(acc["oracle_ceiling"]),
                     "bound_slack_ball": summary(acc["slack_ball"]),
                     "bound_slack_box": summary(acc["slack_box"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--source", choices=("prefill", "decode"), default="prefill")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--hisa-m", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "capture_config.json")))
    layers = (cfg["capture_layers"] if isinstance(cfg["capture_layers"], list)
              else list(range(cfg["num_layers"])))
    # n_prompt 由 k^I 总长减去 decode 步数反推，无需手工传入
    n_prompt = (load_k_I(args.run_dir, layers[0]).shape[0]
                - cfg.get("num_decode_steps", 0))

    out = {"run_dir": args.run_dir, "source": args.source,
           "block_size": args.block_size, "top_k": args.top_k,
           "n_prompt": int(n_prompt), "layers": {}}
    for l in layers:
        print(f"layer {l} ...", flush=True)
        out["layers"][str(l)] = measure_layer(
            args.run_dir, l, args.source, n_prompt, args.block_size,
            args.top_k, args.max_samples, args.hisa_m)

    path = args.out or os.path.join(args.run_dir, f"stats_{args.source}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")

    print(f"\n{'layer':>5} {'cov p50':>8} {'touched':>8} {'churn':>7} "
          f"{'ball_F':>7} {'oracle_ceil':>12} {'slack_ball':>11} {'w<0':>6}")
    for l in layers:
        r = out["layers"][str(l)]
        if "error" in r:
            print(f"{l:>5} {r['error']}")
            continue
        print(f"{l:>5} {r['m1_hisa']['coverage']['p50']:>8.4f} "
              f"{r['m3_locality']['touched_blocks']['p50']:>8.0f} "
              f"{r['m3_locality']['churn_rate']['p50']:>7.4f} "
              f"{r['m4_prune']['ball_oneshot_F']['p50']:>7.4f} "
              f"{r['m4_prune']['oracle_ceiling']['p50']:>12.4f} "
              f"{r['m4_prune']['bound_slack_ball']['p50']:>11.2f} "
              f"{r['m2_w']['neg_fraction']:>6.3f}")


if __name__ == "__main__":
    main()
