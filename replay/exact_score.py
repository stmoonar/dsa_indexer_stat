"""
exact_score.py — 全量精确打分 + top-k

这是所有测量的参照系（约束 6）。
与生产 kernel 的 mismatch 率由 kernel_align.py 单独汇报，不混入 recall。
"""

import torch
from replay.bounds import exact_indexer_score


def compute_exact_topk(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    top_k: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    全量精确打分并返回 top-k。

    Args:
        q_I: [H, d], fp32, post-RoPE, 当前步的 indexer query
        w: [H], fp32, head 权重
        k_I: [L, d], fp32, post-RoPE, 全部 indexer key
        top_k: 选取的 token 数

    Returns:
        topk_scores: [k], fp32, top-k 分数（降序）
        topk_indices: [k], int64, top-k token 索引
    """
    scores = exact_indexer_score(q_I, w, k_I)  # [L]
    actual_k = min(top_k, scores.shape[0])
    topk_scores, topk_indices = torch.topk(scores, actual_k, largest=True, sorted=True)
    return topk_scores, topk_indices


def compute_tau_from_warm_set(
    q_I: torch.Tensor,
    w: torch.Tensor,
    warm_k_I: torch.Tensor,
) -> torch.Tensor:
    """
    用当前步的 q^I 对 warm 集重新打分，取最小值作为 τ_0。

    约束 1：这是当前步 top-k 分数的合法下界。
    绝对禁止复用上一步的 τ 数值。

    合法性：warm 集是全集子集，子集第 k 大 ≤ 全集第 k 大。
    warm 集恰好有 k=2048 个元素，其第 k 大就是最小值。

    Args:
        q_I: [H, d], 当前步的 query
        w: [H], 当前步的 weight
        warm_k_I: [k, d], 上一步 top-k 的 k^I

    Returns:
        tau_0: 标量, warm 集分数的最小值
    """
    scores = exact_indexer_score(q_I, w, warm_k_I)  # [k]
    tau_0 = scores.min()
    return tau_0


def topk_set_recall(
    pred_indices: torch.Tensor,
    true_indices: torch.Tensor,
) -> float:
    """
    计算 top-k 集合的 recall。

    recall = |pred ∩ true| / |true|

    注意：只关心 membership，不关心排序。

    Args:
        pred_indices: [k1], 预测的 top-k 索引
        true_indices: [k2], 真实的 top-k 索引

    Returns:
        recall: float in [0, 1]
    """
    pred_set = set(pred_indices.cpu().tolist())
    true_set = set(true_indices.cpu().tolist())
    if len(true_set) == 0:
        return 1.0
    return len(pred_set & true_set) / len(true_set)
