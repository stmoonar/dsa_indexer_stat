"""
kernel_align.py — replay-exact vs 生产 kernel top-k 的 mismatch 率

约束 6：mismatch 率单独汇报，不混入 recall。
"""

import torch
from replay.exact_score import compute_exact_topk


def compute_kernel_alignment(
    replay_topk_indices: torch.Tensor,
    kernel_topk_indices: torch.Tensor,
) -> dict:
    """
    计算 replay exact top-k 与生产 kernel top-k 的对齐率。

    Args:
        replay_topk_indices: [k], replay 精确计算的 top-k
        kernel_topk_indices: [k], 生产 kernel 输出的 top-k

    Returns:
        dict with:
            - match_rate: |replay ∩ kernel| / k
            - mismatch_rate: 1 - match_rate
            - replay_only: replay 有但 kernel 无的索引数
            - kernel_only: kernel 有但 replay 无的索引数
    """
    replay_set = set(replay_topk_indices.cpu().tolist())
    kernel_set = set(kernel_topk_indices.cpu().tolist())

    k = len(replay_set)
    if k == 0:
        return {"match_rate": 1.0, "mismatch_rate": 0.0,
                "replay_only": 0, "kernel_only": 0}

    intersection = replay_set & kernel_set
    match_rate = len(intersection) / k

    return {
        "match_rate": match_rate,
        "mismatch_rate": 1.0 - match_rate,
        "replay_only": len(replay_set - kernel_set),
        "kernel_only": len(kernel_set - replay_set),
    }
