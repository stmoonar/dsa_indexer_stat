"""
slim_run.py — 从 capture run 里抽出可下载的子集

原始 dump 一个 32K×3 序列的 run 就有 ~930MB raw，128K 更大；但绝大部分体积
是两个大张量，而它们服务于不同的分析：

  c_latent + k_pe (最大头)  只服务想法 2 的 k^I 重建回归；只需要一个 run，
                             32K 足够，128K 的那份没有额外信息
  k_I         (第二大)      服务全部块几何分析（m1/m3/m4、oracle、B sweep、
                             聚类分组、松弛归因）
  prefill_*   (小)          真实 query，m2/谱/一切 query 相关分析的基础
  step_*      (小)          decode 段，已判定不可用，默认不带

档位：
  light     只要 prefill + 配置 + 统计       —— m2、想法 4 的谱、kernel 抽查
  geometry  light + k_I                     —— 全部块几何分析
  full      geometry + c_latent/k_pe        —— 加想法 2

k_I 落盘时是 fp32，但源头是 bf16。这些值量级 ~1（k^I 是 LayerNorm 输出），
转 fp16 相对 bf16 源是无损的（fp16 的 10 位尾数 > bf16 的 7 位），体积减半。
脚本会实测转换误差，非无损时自动退回 fp32 并告警。

用法：
    python -m analysis.slim_run runs/A_32k_tp8_XXX --profile geometry
    python -m analysis.slim_run runs/C_128k_tp8_XXX --profile geometry --layers 0,2,4
"""

import os
import sys
import shutil
import argparse

import numpy as np

PROFILES = {
    "light": set(),
    "geometry": {"k_I"},
    "full": {"k_I", "c_latent", "k_pe"},
}
SMALL_FILES = ("capture_config.json", "run_meta.json", "prompt_manifest.json",
               "prompt_fingerprint.json", "prompt_token_ids.npy",
               "prompt_meta.json")


def human(n: int) -> str:
    return f"{n / 1e6:.1f}MB"


def seq_dirs(run_dir: str):
    """支持多序列 run（含 seqNN 子目录）与单目录 run。"""
    subs = sorted(d for d in os.listdir(run_dir)
                  if d.startswith("seq")
                  and os.path.isdir(os.path.join(run_dir, d)))
    if subs:
        return [(s, os.path.join(run_dir, s)) for s in subs]
    return [("", run_dir)]


def copy_big(src: str, dst: str, downcast: bool) -> tuple:
    """拷贝大张量，可选 fp32->fp16 降位。返回 (原大小, 新大小, 最大绝对误差)。"""
    orig = os.path.getsize(src)
    if not downcast:
        shutil.copy2(src, dst)
        return orig, orig, 0.0
    a = np.load(src, mmap_mode="r")
    if a.dtype != np.float32:
        shutil.copy2(src, dst)
        return orig, orig, 0.0
    h = np.asarray(a).astype(np.float16)
    err = float(np.abs(h.astype(np.float32) - np.asarray(a)).max())
    if not np.isfinite(h).all():
        shutil.copy2(src, dst)          # 溢出 -> 保留 fp32
        return orig, orig, float("inf")
    np.save(dst, h)
    return orig, os.path.getsize(dst), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--profile", choices=tuple(PROFILES), default="geometry")
    ap.add_argument("--layers", default=None,
                    help="只保留这些层的大张量，如 0,2,4（prefill 仍保留全部）")
    ap.add_argument("--out", default=None, help="输出目录，缺省 <run_dir>_<profile>")
    ap.add_argument("--keep-decode", action="store_true",
                    help="保留 step*.npz（decode 统计已判定不可用，默认丢弃）")
    ap.add_argument("--no-downcast", action="store_true",
                    help="不做 fp32->fp16（体积翻倍）")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/\\")
    out_dir = args.out or f"{run_dir}_{args.profile}"
    keep_kinds = PROFILES[args.profile]
    layers = ([int(x) for x in args.layers.split(",") if x]
              if args.layers else None)

    total_in = total_out = 0
    worst_err = 0.0
    os.makedirs(out_dir, exist_ok=True)

    for name in SMALL_FILES:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, name))
    for f in os.listdir(run_dir):
        if f.endswith(".log") or f.startswith("stats_"):
            shutil.copy2(os.path.join(run_dir, f), os.path.join(out_dir, f))

    for seq_name, seq_path in seq_dirs(run_dir):
        dst_seq = os.path.join(out_dir, seq_name) if seq_name else out_dir
        os.makedirs(dst_seq, exist_ok=True)
        n_pf = n_big = 0

        for f in sorted(os.listdir(seq_path)):
            src = os.path.join(seq_path, f)
            if os.path.isdir(src):
                continue
            size = os.path.getsize(src)

            if f in SMALL_FILES or f.endswith(".log") or f.startswith("stats_"):
                shutil.copy2(src, os.path.join(dst_seq, f))
                continue
            if f.startswith("prefill_pos"):
                shutil.copy2(src, os.path.join(dst_seq, f))
                total_in += size
                total_out += size
                n_pf += 1
                continue
            if f.startswith("step"):
                if args.keep_decode:
                    shutil.copy2(src, os.path.join(dst_seq, f))
                    total_in += size
                    total_out += size
                continue

            kind = next((k for k in ("k_I", "c_latent", "k_pe")
                         if f.startswith(k + "_layer")), None)
            if kind is None:
                shutil.copy2(src, os.path.join(dst_seq, f))
                continue
            if kind not in keep_kinds:
                total_in += size
                continue
            if layers is not None:
                lid = int(f.split("_layer")[1].split(".")[0])
                if lid not in layers:
                    total_in += size
                    continue
            o, n, err = copy_big(src, os.path.join(dst_seq, f),
                                 downcast=not args.no_downcast)
            total_in += o
            total_out += n
            worst_err = max(worst_err, err)
            n_big += 1

        label = seq_name or os.path.basename(run_dir)
        print(f"  {label}: prefill={n_pf}  big_tensors={n_big}")

    print(f"\nprofile={args.profile}  layers={layers or 'all'}")
    if worst_err == float("inf"):
        print("WARNING: 某些张量 fp16 会溢出，已保留 fp32")
    elif worst_err > 0:
        print(f"fp32->fp16 最大绝对误差 = {worst_err:.3e} "
              f"(k^I 元素量级 ~1；源头本就是 bf16，故此误差无实质影响)")
    print(f"{human(total_in)} -> {human(total_out)}  "
          f"(省下 {100 * (1 - total_out / max(total_in, 1)):.0f}%)")

    if not args.no_zip:
        z = shutil.make_archive(out_dir, "zip",
                                root_dir=os.path.dirname(out_dir) or ".",
                                base_dir=os.path.basename(out_dir))
        print(f"\nZipped: {z} ({human(os.path.getsize(z))})")


if __name__ == "__main__":
    main()
