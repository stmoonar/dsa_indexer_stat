"""
null_control.py — 零假设对照：用随机/打散数据跑同一套层间与步间统计

为什么需要它：
  cross_layer.py 至少算了随机基线 k/L；churn（步间命中）那一侧**一个基线都没有**，
  "相邻步命中 80%" 这句话现在无法证伪。而且 k/L 只是"索引随机"的地板，它答不了
  一个更要命的问题：高重合会不会纯粹由 k^I 的几何造成 —— 若少数 token 的 k^I
  范数远大于其余，任何 query 都会选中它们，每层都选、每步都选，重合度天然很高，
  却与"语义结构"毫无关系。

四种零假设各自**只破坏一件事**，全部走 replay/exact_score.py 的同一条打分链路，
故与真实数字逐位可比（约束 6：exact top-k 是一切统计的参照系）：

  real        真实 k^I / q^I / w                      —— 基准
  uniform     不打分，直接在 [0, pos] 上均匀采 k 个索引 —— 解析地板 k/L 的蒙特卡洛自检
  gaussian    k^I / q^I / w 全部随机（匹配真实逐维矩） —— 检验 pipeline 自身不制造相关
  random-q    k^I 用【真实】的，q^I 换成随机            —— 若 O 仍高，说明高重合来自
                                                          k^I 几何（高范数 token 通吃），
                                                          而非 query 侧对齐
  shuffle-q   k^I / q^I / w 全真实，但把 (q,w) 与位置的
              配对随机打乱                              —— 只破坏时间连续性，
                                                          是步间命中率的正确零假设

其中 random-q 是最关键的一条：它是唯一保留了真实 k^I 的对照。

用法:
    python -m analysis.null_control runs/full61_32K_tp16_XXX --all
    python -m analysis.null_control runs/... --mode random-q --seed 7
    python -m analysis.null_control runs/... --all --layers 0,8,16,24,32,40,48,56 \
        --max-positions 32          # 抽样跑，几十秒出结果
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.loader import (load_k_I, load_prefill_dump, list_prefill_positions)
from replay.exact_score import compute_exact_topk
from replay.m3_block_locality import cross_layer_overlap, topk_churn
from analysis.run_measurements import summary
from analysis.analyze_run import seq_dirs, layers_of

MODES = ("real", "uniform", "gaussian", "random-q", "shuffle-q")

# 每种模式破坏了什么 —— 打印在报告里，避免事后误读
MODE_DOC = {
    "real": "基准（真实 k/q/w）",
    "uniform": "索引均匀随机（不打分）",
    "gaussian": "k/q/w 全随机",
    "random-q": "k^I 真实，q^I 随机",
    "shuffle-q": "全真实，位置配对打乱",
}


def common_positions(run, layers):
    """所有选定层都采到的 prefill 位置（升序）。"""
    common = None
    for l in layers:
        s = set(list_prefill_positions(run, l))
        common = s if common is None else (common & s)
    return sorted(common or [])


def _match_moments(shape, ref, gen, device):
    """按 ref 的逐维均值/标准差生成同分布的随机张量。

    零假设要公平：随机数据的尺度必须与真实数据一致，否则打分的动态范围
    不同，top-k 的选择性也不同，比较就没有意义。
    """
    mu = ref.mean(dim=0, keepdim=True)
    sd = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
    z = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    return z * sd + mu


def build_topk(run, layer, positions, mode, top_k, gen, rng, device):
    """某层在给定位置上的 top-k 索引集合。

    Returns: {pos: np.ndarray[int64] 升序}
    """
    out = {}

    if mode == "uniform":
        for pos in positions:
            L = pos + 1
            k = min(top_k, L)
            out[pos] = np.sort(rng.choice(L, size=k, replace=False)).astype(np.int64)
        return out

    # 需要打分的四种模式：先备好 k^I
    real_k = load_k_I(run, layer).to(device)
    n_rows = max(positions) + 1
    if real_k.shape[0] < n_rows:
        raise ValueError(f"layer {layer}: k_I 只有 {real_k.shape[0]} 行，"
                         f"但采样位置最大到 {max(positions)}")
    if mode == "gaussian":
        k_all = _match_moments((real_k.shape[0], real_k.shape[1]),
                               real_k, gen, device)
    else:
        k_all = real_k

    # 读入该层全部真实 (q, w)
    qs, ws = {}, {}
    for pos in positions:
        d = load_prefill_dump(run, layer, pos)
        qs[pos] = d["q_I"].to(device)
        ws[pos] = d["w"].to(device)

    # 位置重映射：shuffle-q 打乱 (q,w) 与位置的配对；其余保持恒等
    if mode == "shuffle-q":
        src = list(positions)
        perm = rng.permutation(len(src))
        remap = {positions[i]: src[perm[i]] for i in range(len(src))}
    else:
        remap = {p: p for p in positions}

    # random-q 的 w：取自另一随机位置的真实 w，保住 w 的符号结构
    # （约束 2：w 有 ~58% 为负，用标准正态会把这个结构一起抹掉）
    if mode == "random-q":
        wperm = rng.permutation(len(positions))
        wmap = {positions[i]: positions[wperm[i]] for i in range(len(positions))}

    q_ref = torch.stack([qs[p] for p in positions]).reshape(-1, real_k.shape[1])

    for pos in positions:
        if mode in ("real", "shuffle-q"):
            q, w = qs[remap[pos]], ws[remap[pos]]
        elif mode == "random-q":
            q = _match_moments(tuple(qs[pos].shape), q_ref, gen, device)
            w = ws[wmap[pos]]
        elif mode == "gaussian":
            q = _match_moments(tuple(qs[pos].shape), q_ref, gen, device)
            w = torch.randn(ws[pos].shape, generator=gen, device=device,
                            dtype=torch.float32) * ws[pos].std().clamp_min(1e-6)
        else:
            raise ValueError(f"unknown mode {mode}")
        _, ti = compute_exact_topk(q, w, k_all[:pos + 1], top_k)
        out[pos] = np.unique(ti.cpu().numpy().astype(np.int64))
    return out


def analyze(run, layers, positions, mode, args, device) -> dict:
    """一种模式下的层间 + 步间统计。"""
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sets = {}
    for l in layers:
        sets[l] = build_topk(run, l, positions, mode, args.top_k, gen, rng, device)

    # ---- 层间：相邻层对 (l, l+1) ----
    o_all, lift_all, base_all = [], [], []
    per_pair = {}
    for a, b in zip(layers, layers[1:]):
        if b - a != 1:
            continue          # --layers 抽样后不相邻的对不算"相邻层"
        ov = []
        for pos in positions:
            sa, sb = sets[a][pos], sets[b][pos]
            o = cross_layer_overlap(torch.from_numpy(sa), torch.from_numpy(sb))
            base = len(sa) / max(pos + 1, 1)
            lift = (o - base) / (1 - base) if base < 1 else 0.0
            ov.append(o)
            o_all.append(o)
            lift_all.append(lift)
            base_all.append(base)
        per_pair[f"{a}-{b}"] = summary(ov)

    # ---- 步间：同层相邻位置 (p-1, p) ----
    churn_all, hit_all = [], []
    per_layer_hit = {}
    for l in layers:
        hits = []
        for prev, cur in zip(positions, positions[1:]):
            if cur - prev != 1:
                continue      # 只配对真正相邻的采样位置（prefill_run 连续段内）
            c = topk_churn(torch.from_numpy(sets[l][cur]),
                           torch.from_numpy(sets[l][prev]))["churn_rate"]
            churn_all.append(c)
            hits.append(1.0 - c / 2.0)
            hit_all.append(1.0 - c / 2.0)
        if hits:
            per_layer_hit[str(l)] = summary(hits)

    return {
        "mode": mode,
        "breaks": MODE_DOC[mode],
        "n_positions": len(positions),
        "n_adjacent_step_pairs": len(churn_all) // max(len(layers), 1),
        "cache_len_range": [positions[0] + 1, positions[-1] + 1],
        "random_baseline": summary(base_all),
        "cross_layer_adjacent": {"overlap": summary(o_all),
                                 "lift": summary(lift_all)},
        "step_adjacent": {"churn_rate": summary(churn_all),
                          "hit_rate": summary(hit_all)},
        "by_pair_overlap": per_pair,
        "by_layer_hit": per_layer_hit,
    }


def print_report(out):
    for seq, modes in out["sequences"].items():
        print(f"\n=== {seq} ===")
        if "error" in modes:          # 序列级失败：modes 是 {"error": str}
            print(f"  {modes['error']}")
            continue
        first = next(iter(modes.values()))
        print(f"  positions={first['n_positions']} "
              f"cache_len={first['cache_len_range']} "
              f"步间相邻对/层={first['n_adjacent_step_pairs']}")
        print(f"  {'mode':<11} {'破坏':<22} {'层间O':>8} {'层间lift':>9} "
              f"{'步间hit':>8} {'步间churn':>10} {'基线k/L':>8}")
        for m, r in modes.items():
            if "error" in r:
                print(f"  {m:<11} {r['error']}")
                continue
            cl, st = r["cross_layer_adjacent"], r["step_adjacent"]
            print(f"  {m:<11} {r['breaks']:<22} "
                  f"{cl['overlap']['p50'] * 100:>7.1f}% "
                  f"{cl['lift']['p50'] * 100:>8.1f}% "
                  f"{st['hit_rate']['p50'] * 100:>7.1f}% "
                  f"{st['churn_rate']['p50']:>10.3f} "
                  f"{r['random_baseline']['p50'] * 100:>7.1f}%")
        print("\n  读法：uniform/gaussian 应贴着基线 k/L —— 贴不上说明 pipeline 自造相关。")
        print("        random-q 若仍显著高于基线，说明高重合来自 k^I 几何而非 query 对齐。")
        print("        shuffle-q 的『步间hit』才是『相邻步命中』的正确零假设。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mode", choices=MODES, default="random-q")
    ap.add_argument("--all", action="store_true", help="跑全部模式并横向对比")
    ap.add_argument("--layers", default=None, help="逗号分隔；缺省 = 全部层")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--min-cache-mult", type=float, default=2.0,
                    help="丢弃 cache_len < mult*top_k 的位置（那里 top-k 即全集）")
    ap.add_argument("--max-positions", type=int, default=0,
                    help=">0 时按步长抽样，加速；0 = 全用")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    modes = list(MODES) if args.all else [args.mode]
    out = {"run_dir": args.run_dir, "top_k": args.top_k, "seed": args.seed,
           "min_cache_mult": args.min_cache_mult, "modes": modes,
           "sequences": {}}

    for name, path in seq_dirs(args.run_dir):
        layers = ([int(x) for x in args.layers.split(",") if x]
                  if args.layers else layers_of(path))
        layers = sorted(layers)
        if len(layers) < 2:
            out["sequences"][name] = {"error": f"need >=2 layers, got {layers}"}
            continue

        allpos = common_positions(path, layers)
        positions = [p for p in allpos
                     if p + 1 >= args.min_cache_mult * args.top_k]
        if args.max_positions and len(positions) > args.max_positions:
            # 保住相邻性：按连续段整段抽，而不是隔点取
            # （隔点取会让 cur-prev != 1，步间统计直接归零）
            keep, seg = [], []
            for p in positions:
                if seg and p - seg[-1] != 1:
                    keep.append(seg)
                    seg = []
                seg.append(p)
            if seg:
                keep.append(seg)
            picked = []
            for s in keep:
                if len(picked) >= args.max_positions:
                    break
                picked.extend(s[:args.max_positions - len(picked)])
            positions = picked
        if len(positions) < 2:
            out["sequences"][name] = {"error": "位置不足（提高 --max-positions 或"
                                               "降低 --min-cache-mult）"}
            continue

        out["sequences"][name] = {}
        for m in modes:
            print(f"{name} [{m}]: layers={layers[0]}..{layers[-1]} "
                  f"({len(layers)}) positions={len(positions)} "
                  f"on {args.device} ...", flush=True)
            out["sequences"][name][m] = analyze(path, layers, positions, m,
                                                args, args.device)

    dst = args.out or os.path.join(args.run_dir, "null_control.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")
    print_report(out)


if __name__ == "__main__":
    main()
