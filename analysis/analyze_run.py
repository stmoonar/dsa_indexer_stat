"""
analyze_run.py — 服务器侧总驱动：一次跑完全部分析，只产出小报告

设计目的：原始 dump 有几百 MB~GB，不下载；这个脚本在服务器上跑完，
产出 stats.json（几十 KB）+ report.md（可直接读），把这两个拿走即可。

覆盖：
  m1/m2/m3/m4   —— 复用 run_measurements 的逐样本测量
  m5            —— oracle 天花板、B sweep、k-means 分组对比、松弛归因
  m6            —— 想法 2 的 k^I 重建裁决（需要 c_latent，只有主 run 有）
  谱            —— 想法 4 的 M_q 特征值能量
分组口径（约束 15）：按 (sequence, layer, 前缀长度桶) 分组报分布。

前缀长度桶：capture 时按锚点抓的连续段，天然给出 oracle 可剪率、覆盖率、
块触及数【随前缀长度】的曲线——这是判断"32K 是否系统性偏差"、
能否外推 128K 的直接依据。

用法：
    python -m analysis.analyze_run runs/A_32k_tp8_XXX
    python -m analysis.analyze_run runs/C_128k_tp8_XXX --skip-kmeans
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.loader import (load_k_I, load_prefill_dump, list_prefill_positions)
from replay.bounds import compute_block_summaries_ball, exact_indexer_score
from replay.exact_score import compute_exact_topk, compute_tau_from_warm_set
from replay.m1_hisa_recall import hisa_coverage
from replay.m2_w_stats import w_sign_stats, w_neg_mass
from replay.m3_block_locality import block_locality_stats
from replay.m4_bound_prune import prune_stats
from replay.m5_oracle_ceiling import (grouping_sweep, slack_attribution,
                                      q_weighted_spectrum, contiguous_groups,
                                      oracle_ceiling)
from analysis.run_measurements import summary


def seq_dirs(root: str):
    subs = sorted(d for d in os.listdir(root)
                  if d.startswith("seq") and os.path.isdir(os.path.join(root, d)))
    return ([(s, os.path.join(root, s)) for s in subs] if subs
            else [(os.path.basename(root), root)])


def layers_of(run: str):
    cfg = json.load(open(os.path.join(run, "capture_config.json")))
    cl = cfg.get("capture_layers")
    if isinstance(cl, list):
        return cl
    return [int(f.split("_layer")[1].split(".")[0])
            for f in sorted(os.listdir(run)) if f.startswith("k_I_layer")]


def bucket_of(pos: int, anchors, run_len: int):
    """把位置归到它所属的采样段（尾窗归为 'tail'）。"""
    for a in anchors:
        if a <= pos < a + run_len:
            return f"ctx{a}"
    return "tail"


def analyze_layer(run, layer, cfg, args) -> dict:
    positions = list_prefill_positions(run, layer)
    if len(positions) < 2:
        return {"error": f"only {len(positions)} prefill positions"}

    k_all = load_k_I(run, layer)
    anchors = cfg.get("prefill_buckets") or []
    if isinstance(anchors, str):
        anchors = [int(x) for x in anchors.split(",") if x]
    run_len = int(cfg.get("prefill_run", 32))

    pairs = [(p, q) for p, q in zip(positions, positions[1:]) if q == p + 1]
    by_bucket = defaultdict(list)
    for prev, cur in pairs:
        by_bucket[bucket_of(cur, anchors, run_len)].append((prev, cur))

    out = {"n_positions": len(positions), "buckets": {}}
    w_rows, q_rows = [], []

    for bname, bpairs in sorted(by_bucket.items()):
        if args.max_per_bucket and len(bpairs) > args.max_per_bucket:
            step = max(1, len(bpairs) // args.max_per_bucket)
            bpairs = bpairs[::step][:args.max_per_bucket]
        acc = defaultdict(list)
        sweep_acc, slack_acc = defaultdict(list), defaultdict(list)
        # 同桶内各样本的 cache_len 只差几十 token，聚类复用一份即可
        group_cache = {}

        for prev, cur in bpairs:
            d_cur, d_prev = (load_prefill_dump(run, layer, cur),
                             load_prefill_dump(run, layer, prev))
            q, w = d_cur["q_I"], d_cur["w"]
            k_cur = k_all[:cur + 1]
            w_rows.append(w)
            q_rows.append(q)

            _, idx_cur = compute_exact_topk(q, w, k_cur, args.top_k)
            _, idx_prev = compute_exact_topk(d_prev["q_I"], d_prev["w"],
                                             k_all[:prev + 1], args.top_k)
            warm = k_all[idx_prev]
            tau = compute_tau_from_warm_set(q, w, warm).item()

            loc = block_locality_stats(idx_cur, idx_prev, args.block_size)
            acc["touched_blocks"].append(loc["touched_blocks"])
            acc["churn_rate"].append(loc["churn_rate"])
            acc["coverage"].append(hisa_coverage(
                q, w, k_cur, args.block_size, args.hisa_m,
                args.top_k)["coverage"])
            pr = prune_stats(q, w, k_cur, warm, args.block_size, args.top_k)
            acc["ball_oneshot_F"].append(pr["ball_oneshot_F"])
            acc["box_oneshot_F"].append(pr["box_oneshot_F"])
            acc["tau_0"].append(tau)

            # kernel 对齐（capture 正确性的抽查，不混入 recall）
            acc["kernel_overlap"].append(
                len(set(idx_cur.tolist())
                    & set(d_cur["topk_indices"].tolist())) / args.top_k)

            # m5：天花板 + 分组 sweep
            sw = grouping_sweep(q, w, k_cur, tau,
                                block_sizes=args.block_sweep,
                                kmeans_block_sizes=args.kmeans_block_sizes,
                                kmeans_iters=args.kmeans_iters,
                                group_cache=group_cache)
            for gname, st in sw.items():
                sweep_acc[f"{gname}_group_frac"].append(st["group_frac"])
                sweep_acc[f"{gname}_token_frac"].append(st["token_frac"])

            # m5：松弛归因
            mu, r, _ = compute_block_summaries_ball(k_cur, args.block_size)
            for kk, vv in slack_attribution(q, w, k_cur, mu, r,
                                            args.block_size).items():
                slack_acc[kk].append(vv)

        out["buckets"][bname] = {
            "n_samples": len(bpairs),
            "cache_len_p50": float(np.median([c + 1 for _, c in bpairs])),
            "measures": {k: summary(v) for k, v in acc.items()},
            "oracle_sweep": {k: summary(v) for k, v in sweep_acc.items()},
            "slack": {k: summary(v) for k, v in slack_acc.items()},
        }

    if w_rows:
        W = torch.stack(w_rows)
        st, nm = w_sign_stats(W), w_neg_mass(W)
        out["m2_w"] = {"neg_fraction": st["neg_fraction"],
                       "neg_mass_ratio": nm["neg_mass_ratio"],
                       "w_min": st["w_min"], "w_max": st["w_max"],
                       "heads_always_neg": int(((W < 0).all(0)).sum())}
        out["m_q_spectrum"] = q_weighted_spectrum(
            q_rows[:args.spectrum_samples], w_rows[:args.spectrum_samples])
    return out


def analyze_reconstruction(run, layer, cfg, args) -> dict:
    """想法 2：需要 c_latent（只有开了 DSA_CAPTURE_LATENT 的 run 才有）。"""
    cp = os.path.join(run, f"c_latent_layer{layer:03d}.npy")
    pp = os.path.join(run, f"k_pe_layer{layer:03d}.npy")
    if not (os.path.exists(cp) and os.path.exists(pp)):
        return {}
    from replay.m6_reconstruction import reconstruction_test

    k_I = load_k_I(run, layer)
    c = torch.from_numpy(np.load(cp))
    kpe = torch.from_numpy(np.load(pp))
    positions = list_prefill_positions(run, layer)[-args.recon_queries:]
    qs, ws = [], []
    for p in positions:
        d = load_prefill_dump(run, layer, p)
        qs.append(d["q_I"])
        ws.append(d["w"])
    return reconstruction_test(k_I, c, kpe, qs, ws, top_k=args.top_k)


def fmt(s, key="p50", digits=4):
    if not isinstance(s, dict) or key not in s:
        return "n/a"
    return f"{s[key]:.{digits}f}"


def write_report(res: dict, path: str):
    L = []
    A = L.append
    A(f"# DSA indexer 测量报告\n")
    A(f"- run: `{res['run_dir']}`")
    A(f"- 序列数: {len(res['sequences'])}   block_size={res['block_size']}   "
      f"top_k={res['top_k']}\n")
    A("> prefill 段的 q/k/w 与全模型 layer 0-4 逐位一致；decode 段未纳入。\n")

    for seq, sres in res["sequences"].items():
        A(f"\n## 序列 {seq}\n")
        for layer, lres in sorted(sres.items(), key=lambda kv: int(kv[0])):
            if "error" in lres:
                A(f"### layer {layer}: {lres['error']}\n")
                continue
            m2 = lres.get("m2_w", {})
            A(f"### layer {layer}"
              f"  (w<0 {m2.get('neg_fraction', 0):.1%}, "
              f"恒负 head {m2.get('heads_always_neg', 0)}/64)\n")
            A("| 前缀桶 | ctx | 覆盖率 | 触及块 | churn | F_ball | "
              "oracle(块) | oracle(token) | kernel对齐 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for b, br in sorted(lres["buckets"].items(),
                                key=lambda kv: kv[1]["cache_len_p50"]):
                m, o = br["measures"], br["oracle_sweep"]
                B = res["block_size"]
                A(f"| {b} | {br['cache_len_p50']:.0f} | "
                  f"{fmt(m.get('coverage'))} | "
                  f"{fmt(m.get('touched_blocks'), digits=0)} | "
                  f"{fmt(m.get('churn_rate'))} | "
                  f"{fmt(m.get('ball_oneshot_F'))} | "
                  f"{fmt(o.get(f'contiguous_B{B}_group_frac'))} | "
                  f"{fmt(o.get(f'contiguous_B{B}_token_frac'))} | "
                  f"{fmt(m.get('kernel_overlap'))} |")
            A("")
            b0 = max(lres["buckets"].values(),
                     key=lambda x: x["cache_len_p50"])
            o = b0["oracle_sweep"]
            A("**分组方式对比**（最长前缀桶，oracle token 占比）\n")
            A("| B | 连续块 | k-means |")
            A("|---|---|---|")
            for B in res["block_sweep"]:
                A(f"| {B} | {fmt(o.get(f'contiguous_B{B}_token_frac'))} | "
                  f"{fmt(o.get(f'kmeans_B{B}_token_frac'))} |")
            A("")
            s = b0["slack"]
            if s:
                rs = s.get("radius_share", {}).get("p50", 0.0)
                nb = s.get("neg_branch_share_of_slack", {}).get("p50", 0.0)
                A("**bound 松弛归因**（占总松弛 U−真实块最大分 的比例）: "
                  f"半径项 ‖q‖r {fmt(s.get('radius_share'))}, "
                  f"块心项 q·μ {fmt(s.get('center_share'))}, "
                  f"负支贡献 {fmt(s.get('neg_branch_share_of_slack'))}\n")
                if rs > 1.0:
                    A(f"> 半径项占比 >1（块心项为负）意味着松弛**全部**来自 "
                      f"Cauchy–Schwarz 的 ‖q‖r：块均值本身的得分已经低于块内"
                      f"最佳 token。要收紧只能压 r（更紧的分组）或换更细的"
                      f"逐维上界，调 τ 机制无用。\n")
                if abs(nb) < 1e-6:
                    A("> 负支贡献为 0：所有 w<0 的 head 上 "
                      "ReLU(q·μ − ‖q‖r) 都被截断到 0，即负支虽然是 soundness "
                      "所必需，当前数值上**完全没起作用**。\n")
            sp = lres.get("m_q_spectrum", {})
            if sp:
                ef = sp.get("energy_frac", {})
                A("**q 加权谱**: "
                  + ", ".join(f"top-{m} 能量 {v:.3f}" for m, v in ef.items())
                  + f"; 有效秩 {sp.get('effective_rank', 0):.1f}\n")
            rc = lres.get("reconstruction")
            if rc:
                A(f"**想法 2 (k^I←c_s 重建)**: R²={rc.get('r2_heldout', 0):.4f} "
                  f"(打乱基线 {rc.get('r2_shuffled_baseline', 0):.4f}), "
                  f"top-k overlap p50={rc.get('topk_overlap_p50', 0):.4f}, "
                  f"min={rc.get('topk_overlap_min', 0):.4f}\n")
                mis = rc.get("row_mismatch", 0)
                if mis:
                    A(f"> ⚠ latent 比 k^I 多 {mis} 行"
                      f"（{'已改用尾对齐' if rc.get('tail_aligned') else '行数不足'}）。"
                      f"若该 run 的 patch 缺 dummy-run 守卫，两者本就错位，"
                      f"此处结果不可用，需重采后再判。\n")
                if rc.get("signal_above_shuffle", 1) < 0.02:
                    A("> ⚠ R² 与打乱基线相当 —— 配对本身没有信息"
                      "（错位或采错张量），这不是「线性关系不存在」的证据。\n")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--hisa-m", type=int, default=64)
    ap.add_argument("--block-sweep", default="16,32,64,128")
    ap.add_argument("--max-per-bucket", type=int, default=6,
                    help="每个前缀桶最多算几对（m5 的 sweep 较贵）")
    ap.add_argument("--kmeans-block-sizes", default="128",
                    help="对哪些 B 做 k-means 对比（很贵；默认只做主 B）；"
                         "留空则完全跳过")
    ap.add_argument("--kmeans-iters", type=int, default=8)
    ap.add_argument("--spectrum-samples", type=int, default=64)
    ap.add_argument("--recon-queries", type=int, default=4)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.block_sweep = [int(x) for x in args.block_sweep.split(",") if x]
    args.kmeans_block_sizes = [int(x) for x in
                               args.kmeans_block_sizes.split(",") if x]

    root = args.run_root.rstrip("/\\")
    res = {"run_dir": root, "block_size": args.block_size,
           "top_k": args.top_k, "block_sweep": args.block_sweep,
           "sequences": {}}

    for seq, path in seq_dirs(root):
        cfg = json.load(open(os.path.join(path, "capture_config.json")))
        layers = ([int(x) for x in args.layers.split(",")] if args.layers
                  else layers_of(path))
        res["sequences"][seq] = {}
        for layer in layers:
            print(f"[{seq}] layer {layer} ...", flush=True)
            lres = analyze_layer(path, layer, cfg, args)
            rc = analyze_reconstruction(path, layer, cfg, args)
            if rc:
                lres["reconstruction"] = rc
            res["sequences"][seq][str(layer)] = lres

    out = args.out or os.path.join(root, "stats.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    rep = os.path.splitext(out)[0].replace("stats", "report") + ".md"
    write_report(res, rep)
    print(f"\nwrote {out} ({os.path.getsize(out)/1e3:.0f} KB)")
    print(f"wrote {rep} ({os.path.getsize(rep)/1e3:.0f} KB)")
    print("\n把这两个文件拿走即可，原始 dump 不必下载。")


if __name__ == "__main__":
    main()
