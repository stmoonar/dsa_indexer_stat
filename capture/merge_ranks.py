"""
merge_ranks.py — TP 分片的 q/w 离线拼接

根据静态代码阅读（docs/design.md 7.6 节）：
indexer 的 wq_b, wk, weights_proj 都是 ReplicatedLinear（不切分），
所有 TP rank 上完全相同，dump 只需在 rank 0 执行。

本文件保留为空壳，以备模型更新后 TP 策略变化时使用。
"""

import os
import sys
import numpy as np


def merge_ranks(dump_dirs: list[str], output_dir: str):
    """
    合并多个 rank 的 dump。

    当前不需要：V3.2 的 indexer 全部使用 ReplicatedLinear。
    若未来版本改为 TP 切分，在此实现拼接逻辑。
    """
    if len(dump_dirs) == 1:
        print(f"Single rank, no merge needed. Data at {dump_dirs[0]}")
        return

    raise NotImplementedError(
        "V3.2 indexer uses ReplicatedLinear (no TP sharding). "
        "If this changes in a future version, implement concat logic here. "
        "Check: are wq_b, wk, weights_proj still ReplicatedLinear?"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_ranks.py <output_dir> <rank0_dir> [rank1_dir ...]")
        sys.exit(1)
    merge_ranks(sys.argv[2:], sys.argv[1])
