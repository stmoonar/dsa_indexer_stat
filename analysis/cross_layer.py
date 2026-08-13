"""
cross_layer.py — 层间 indexer top-k 相似性（token 级）

问题：同一个 token 位置上，第 ℓ 层 indexer 选出的 top-k 与第 ℓ' 层选出的
top-k 有多重合？重合度高 → 一次 fetch 可以被多层共享，K 读取量按层数摊薄；
重合度低 → 每层各选各的，省不掉。

口径（docs/measurements.md「跨层 top-k overlap」）：
    O(ℓ, ℓ') = |S*_{t,ℓ} ∩ S*_{t,ℓ'}| / |S*_{t,ℓ}|
同一位置各层的 |S*| 相同，故 O 对称，且与 Jaccard 单调等价。

两个必须一起报的量（否则 O 会把平凡重合读成结论）：
  - 随机基线 k / L：两个独立均匀 k-子集的期望重合度。L=8k、k=2048 时
    基线就是 0.25，看到 O=0.3 其实几乎没有跨层结构。
  - lift = (O − k/L) / (1 − k/L)：把基线归一到 0、完全重合归一到 1。
  并且 cache_len ≤ top_k 的位置直接丢弃 —— 那里 top-k 就是全集，O ≡ 1。

索引可比性（这个统计成立的前提）：
  1. dump 的 topk_indices / k_I 行号都是序列内绝对 token 位置，层间同义；
  2. 同一次 forward 里所有层用同一份采样 mask（patch 中的 prefill_local_mask），
     故每层的 prefill 采样位置集合完全一致；这里仍显式取交集再算。

参照系默认 exact（约束 6）：用 dump 的 q^I/w 对 k^I[:pos+1] 全量重打分取
top-k，而不是 kernel 写回的 topk_indices。--source kernel 只用于抽查。

分组按 (sequence, 前缀长度桶, 层对)（约束 15），报 p5/p50/p95/p99。

用法：
    python -m analysis.cross_layer runs/A_32k_tp8_XXX
    python -m analysis.cross_layer runs/C_128k_tp8_XXX --layers 0,8,16,24 --device cuda
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
from replay.exact_score import compute_exact_topk
from replay.m3_block_locality import cross_layer_overlap
from analysis.run_measurements import summary
from analysis.analyze_run import seq_dirs, layers_of, bucket_of


def common_positions(run: str, layers) -> list:
    """所有选定层都采到的 prefill 位置（升序）。

    正常 run 里各层位置本来就一致；取交集是为了让层数不全的目录
    （slim --layers、rank 合并失败）静默降级成"少几个位置"，
    而不是在某一层上抛 FileNotFoundError。
    """
    common = None
    for l in layers:
        s = set(list_prefill_positions(run, l))
        common = s if common is None else (common & s)
    return sorted(common or [])


def topk_sets(run, layer, positions, top_k, source, device):
    """某层在给定位置上的 top-k 索引集合。

    Returns: {pos: np.ndarray[int64] 升序}
    """
    out = {}
    if source == "exact":
        k_all = load_k_I(run, layer).to(device)
    for pos in positions:
        d = load_prefill_dump(run, layer, pos)
        if source == "kernel":
            idx = d["topk_indices"].numpy().astype(np.int64)
            # kernel buffer 在 cache_len < top_k 时用 -1 补位；
            # 另有越界写回（>pos）的情形，一并剔除
            idx = idx[(idx >= 0) & (idx <= pos)]
        else:
            _, ti = compute_exact_topk(d["q_I"].to(device), d["w"].to(device),
                                       k_all[:pos + 1], top_k)
            idx = ti.cpu().numpy().astype(np.int64)
        out[pos] = np.unique(idx)
    return out


def overlap_and_lift(set_a, set_b, cache_len: int):
    """O = |A∩B| / |A|，以及对随机基线 k/L 归一后的 lift。

    lift 的分母用 |A|/cache_len 作为基线：|A| 就是这个位置真实的 k
    （末尾未满 top_k 时会小于 top_k）。
    """
    o = cross_layer_overlap(torch.from_numpy(set_a), torch.from_numpy(set_b))
    base = len(set_a) / max(cache_len, 1)
    lift = (o - base) / (1 - base) if base < 1 else 0.0
    return o, lift, base


def analyze_seq(run, layers, cfg, args) -> dict:
    """单个序列目录的层间 overlap。"""
    positions = [p for p in common_positions(run, layers)
                 if p + 1 >= args.min_cache_mult * args.top_k]
    dropped = len(common_positions(run, layers)) - len(positions)
    if args.max_positions and len(positions) > args.max_positions:
        stride = max(1, len(positions) // args.max_positions)
        positions = positions[::stride][:args.max_positions]
    if len(positions) < 1:
        return {"error": f"no position with cache_len >= "
                         f"{args.min_cache_mult * args.top_k} "
                         f"(dropped {dropped})"}

    anchors = cfg.get("prefill_buckets") or []
    if isinstance(anchors, str):
        anchors = [int(x) for x in anchors.split(",") if x]
    run_len = int(cfg.get("prefill_run", 32))

    sets = {l: topk_sets(run, l, positions, args.top_k, args.source,
                         args.device) for l in layers}

    # (层对, 桶) -> [(O, lift)]；同时按层距聚合
    by_pair = defaultdict(list)
    by_pair_bucket = defaultdict(list)
    by_dist = defaultdict(list)
    baselines = []
    for pos in positions:
        bucket = bucket_of(pos, anchors, run_len)
        for i, la in enumerate(layers):
            for lb in layers[i + 1:]:
                o, lift, base = overlap_and_lift(sets[la][pos], sets[lb][pos],
                                                 pos + 1)
                by_pair[(la, lb)].append((o, lift))
                by_pair_bucket[(la, lb, bucket)].append((o, lift))
                by_dist[lb - la].append((o, lift))
        baselines.append(len(sets[layers[0]][pos]) / (pos + 1))

    def pack(rows):
        return {"overlap": summary([o for o, _ in rows]),
                "lift": summary([x for _, x in rows])}

    return {
        "n_positions": len(positions),
        "positions_dropped_short_prefix": dropped,
        "cache_len_range": [positions[0] + 1, positions[-1] + 1],
        "random_baseline": summary(baselines),
        "adjacent": {f"{a}-{b}": pack(v) for (a, b), v in sorted(by_pair.items())
                     if b - a == 1},
        "by_distance": {str(d): pack(v) for d, v in sorted(by_dist.items())},
        "by_pair": {f"{a}-{b}": pack(v) for (a, b), v in sorted(by_pair.items())},
        "by_pair_bucket": {f"{a}-{b}|{bk}": pack(v)
                           for (a, b, bk), v in sorted(by_pair_bucket.items())},
    }


def print_report(out):
    for seq, r in out["sequences"].items():
        print(f"\n=== {seq} ===")
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  positions={r['n_positions']} "
              f"cache_len={r['cache_len_range']} "
              f"random_baseline p50={r['random_baseline']['p50']:.4f}")
        print(f"  {'d_lyr':>6} {'O p5':>7} {'O p50':>7} {'O p95':>7} "
              f"{'lift p50':>9} {'n':>6}")
        for d, s in r["by_distance"].items():
            o, li = s["overlap"], s["lift"]
            print(f"  {d:>6} {o['p5']:>7.4f} {o['p50']:>7.4f} {o['p95']:>7.4f} "
                  f"{li['p50']:>9.4f} {o['n']:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--layers", default=None,
                    help="只比较这些层，如 0,8,16（缺省 = capture 的全部层）")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--source", choices=("exact", "kernel"), default="exact",
                    help="exact = replay 全量重打分（约束 6 参照系）")
    ap.add_argument("--min-cache-mult", type=float, default=2.0,
                    help="只用 cache_len >= mult*top_k 的位置（默认 2；"
                         "cache_len<=top_k 时 top-k 是全集，overlap 恒为 1）")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="每序列最多用多少个位置（0 = 全部）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {"run_dir": args.run_dir, "top_k": args.top_k, "source": args.source,
           "min_cache_mult": args.min_cache_mult, "sequences": {}}
    for name, path in seq_dirs(args.run_dir):
        cfg = json.load(open(os.path.join(path, "capture_config.json")))
        layers = ([int(x) for x in args.layers.split(",") if x]
                  if args.layers else layers_of(path))
        layers = sorted(layers)
        if len(layers) < 2:
            out["sequences"][name] = {"error": f"need >=2 layers, got {layers}"}
            continue
        out.setdefault("layers", layers)
        print(f"{name}: layers={layers[0]}..{layers[-1]} "
              f"({len(layers)}) on {args.device} ...", flush=True)
        out["sequences"][name] = analyze_seq(path, layers, cfg, args)

    path = args.out or os.path.join(args.run_dir, "cross_layer.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    print_report(out)


if __name__ == "__main__":
    main()
