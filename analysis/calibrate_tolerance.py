"""
calibrate_tolerance.py — 标定全模型金丝雀的容差 DSA_TOL

为什么需要：不同 TP 度下 all-reduce 的求和顺序不同，L0-4 的 prefill k^I
不可能逐位相同。金丝雀默认要求逐位（DSA_TOL=0），直接拿去比 TP=8 vs TP=16
必然满屏红；而容差如果靠猜，又可能松到失去鉴别力。

做法：同一个 prompt、同一个截断模型，跑两个只有 TP 不同的 run，
量 k^I / q^I / w 的相对偏差分布，取一个比"数值噪声"高一个量级、
又远低于"语义错误"的阈值。

判据参考：真正的 capture 错误（hook 接错、层错位、w 少乘 scale）
会造成 O(1) 的相对偏差，而 TP 求和顺序只造成 O(1e-3) 量级（bf16 尾数）。
两者相差几个数量级，阈值很好选。

用法：
    python -m analysis.calibrate_tolerance RUN_TP8 RUN_TP4
    python -m analysis.calibrate_tolerance RUN_A RUN_B --layers 0,1,2,3,4
    python -m analysis.calibrate_tolerance RUN_A RUN_B --out-json tol.json
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.loader import list_prefill_positions, load_prefill_dump


def n_prefill(run: str, layer: int) -> int:
    cfg = json.load(open(os.path.join(run, "capture_config.json")))
    total = np.load(os.path.join(run, f"k_I_layer{layer:03d}.npy"),
                    mmap_mode="r").shape[0]
    return total - cfg.get("num_decode_steps", 0)


def rel_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """逐元素相对偏差；分母加 floor 以免小值放大噪声。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, np.percentile(denom, 50) * 1e-3 + 1e-12)
    rel = np.abs(a - b) / denom
    return {"max": float(rel.max()), "p99": float(np.percentile(rel, 99)),
            "p50": float(np.percentile(rel, 50)),
            "exact_frac": float((a == b).mean())}


# 容差上限：再往上就和 O(1) 的语义错误挨上了，金丝雀失去鉴别力。
# 实测最坏值超过 TOL_CEILING/10 时应当怀疑不是 TP 噪声，而不是把阈值放宽。
TOL_CEILING = 0.1


def suggest_tol(overall: float) -> float:
    """由实测最坏相对偏差给出金丝雀容差：10x 余量，且封顶。

    不向上取整到 10 的整数次幂 —— 那会把 1.2e-3 抬到 1e-1（83x 余量），
    把 3e-2 抬到 1.0，而语义错误正是 O(1)，阈值到了 1.0 就等于没检查。
    """
    if overall <= 0:
        return 0.0
    return min(overall * 10.0, TOL_CEILING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--layers", default="0,1,2,3,4")
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--out-json", default=None,
                    help="把 worst/suggested_tol 落成 JSON，供脚本串联")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x]

    for run in (args.run_a, args.run_b):
        cfg_path = os.path.join(run, "capture_config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"not a capture run dir: {run}")

    # 先确认输入一致，否则比什么都没意义
    fps = []
    for run in (args.run_a, args.run_b):
        p = os.path.join(run, "prompt_token_ids.npy")
        fps.append(np.load(p) if os.path.exists(p) else None)
    if all(f is not None for f in fps):
        if fps[0].shape != fps[1].shape or not (fps[0] == fps[1]).all():
            raise SystemExit(
                "prompt token ids differ between runs — 标定必须用同一个 prompt")
        print(f"prompt identical: {fps[0].shape[0]} tokens\n")
    else:
        print("WARNING: prompt_token_ids.npy missing, 无法确认输入一致\n")

    worst = {"k_I": 0.0, "q_I": 0.0, "w": 0.0}
    print(f"{'layer':>5} {'tensor':>7} {'rel max':>11} {'rel p99':>11} "
          f"{'rel p50':>11} {'bit-exact':>10}")
    for layer in layers:
        n = min(n_prefill(args.run_a, layer), n_prefill(args.run_b, layer))
        a = np.asarray(np.load(os.path.join(
            args.run_a, f"k_I_layer{layer:03d}.npy"), mmap_mode="r")[:n])
        b = np.asarray(np.load(os.path.join(
            args.run_b, f"k_I_layer{layer:03d}.npy"), mmap_mode="r")[:n])
        st = rel_stats(a, b)
        worst["k_I"] = max(worst["k_I"], st["max"])
        print(f"{layer:>5} {'k_I':>7} {st['max']:>11.3e} {st['p99']:>11.3e} "
              f"{st['p50']:>11.3e} {st['exact_frac']:>10.4f}")

        common = sorted(set(list_prefill_positions(args.run_a, layer))
                        & set(list_prefill_positions(args.run_b, layer)))
        for tag in ("q_I", "w"):
            if not common:
                continue
            vals = []
            for pos in common[:args.max_positions]:
                x = load_prefill_dump(args.run_a, layer, pos)[tag].numpy()
                y = load_prefill_dump(args.run_b, layer, pos)[tag].numpy()
                vals.append(rel_stats(x, y))
            st = {k: max(v[k] for v in vals) for k in ("max", "p99", "p50")}
            st["exact_frac"] = min(v["exact_frac"] for v in vals)
            worst[tag] = max(worst[tag], st["max"])
            print(f"{layer:>5} {tag:>7} {st['max']:>11.3e} {st['p99']:>11.3e} "
                  f"{st['p50']:>11.3e} {st['exact_frac']:>10.4f}")

    overall = max(worst.values())
    suggested = suggest_tol(overall)
    if args.out_json:
        json.dump({"run_a": args.run_a, "run_b": args.run_b,
                   "layers": layers, "worst": worst, "overall": overall,
                   "suggested_tol": suggested},
                  open(args.out_json, "w"), indent=2)
    print(f"\nworst relative diff: "
          + ", ".join(f"{k}={v:.3e}" for k, v in worst.items()))
    if overall == 0:
        print("\n两个 run 逐位相同 —— 可以直接用 DSA_TOL=0（最严）。")
    else:
        print(f"\n建议 DSA_TOL={suggested:.1e}"
              f"   （实测最坏 {overall:.3e}，留 10x 余量；"
              f"语义错误是 O(1)，仍有 {1/suggested:.0e}x 鉴别力）")
        if overall * 10 > TOL_CEILING:
            print(f"\n⚠ 实测偏差 {overall:.3e} 比 TP 求和顺序该有的量级"
                  f"（bf16 尾数 ~1e-3）大得多，容差已封顶在 {TOL_CEILING:.0e}。"
                  f"\n  正确反应是怀疑两个 run 不只差 TP（prompt/patch/层号/scale），"
                  f"而不是继续放宽阈值。")
        print(f"\n用法: DSA_TOL={suggested:.1e} DSA_REF_RUN=... DSA_NEW_RUN=... "
              f"pytest tests/test_full_model_consistency.py")


if __name__ == "__main__":
    main()
