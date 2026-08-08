"""
m1_hisa_recall.py — 测量 1: HISA 块覆盖率 + adaptive-m 曲线

口径（见 docs/measurements.md）：
- B = 128, m = 64（与 HISA 论文可比）
- 强制包含首块和尾块（约束 4）
- 尾部未满块不建摘要（约束 5）
- 附加 adaptive-m 曲线：每 query 达 100%/99% 覆盖的最小 m
"""

import torch
from replay.bounds import (
    compute_block_summaries_ball,
    exact_indexer_score,
)
from replay.exact_score import compute_exact_topk


def compute_block_mean_scores(
    q_I: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """HISA Stage 1: 用块均值打分。"""
    return exact_indexer_score(q_I, w, mu)


def hisa_coverage(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    block_size: int = 128,
    m: int = 64,
    top_k: int = 2048,
) -> dict:
    """
    计算单步的 HISA 块覆盖率。

    Returns:
        dict with:
            - coverage: 真 top-k 被选中块覆盖的比例
            - n_blocks_selected: 实际选中的块数（含首尾）
            - n_full_blocks: 满块总数
            - n_tail: 尾部未满块 token 数
    """
    mu, r, n_tail = compute_block_summaries_ball(k_I, block_size)
    n_full = mu.shape[0]

    if n_full == 0:
        return {"coverage": 1.0, "n_blocks_selected": 0,
                "n_full_blocks": 0, "n_tail": n_tail}

    # 只对满块内 token 计算 exact top-k
    k_I_full = k_I[:n_full * block_size]
    _, true_topk_indices = compute_exact_topk(q_I, w, k_I_full, top_k=top_k)

    # Stage 1: 块均值打分，选 top-m
    block_scores = compute_block_mean_scores(q_I, w, mu)
    actual_m = min(m, n_full)
    _, top_block_indices = torch.topk(block_scores, actual_m, largest=True)
    selected = set(top_block_indices.tolist())

    # 强制首尾块（约束 4）
    selected.add(0)
    selected.add(n_full - 1)

    # 计算覆盖率
    covered = 0
    for idx in true_topk_indices.tolist():
        if idx // block_size in selected:
            covered += 1

    actual_k = len(true_topk_indices)
    coverage = covered / actual_k if actual_k > 0 else 1.0

    return {
        "coverage": coverage,
        "n_blocks_selected": len(selected),
        "n_full_blocks": n_full,
        "n_tail": n_tail,
    }


def adaptive_m_curve(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    block_size: int = 128,
    top_k: int = 2048,
    thresholds: tuple[float, ...] = (1.0, 0.99),
) -> dict[float, int]:
    """
    计算达到各覆盖率阈值所需的最小 m。

    Returns:
        dict: {threshold: min_m}
    """
    mu, r, n_tail = compute_block_summaries_ball(k_I, block_size)
    n_full = mu.shape[0]

    if n_full == 0:
        return {t: 0 for t in thresholds}

    k_I_full = k_I[:n_full * block_size]
    _, true_topk_indices = compute_exact_topk(q_I, w, k_I_full, top_k=top_k)
    actual_k = len(true_topk_indices)

    if actual_k == 0:
        return {t: 0 for t in thresholds}

    # 真 top-k token 所在的块
    true_blocks = set()
    for idx in true_topk_indices.tolist():
        true_blocks.add(idx // block_size)

    # 按块均值分数降序排列所有块
    block_scores = compute_block_mean_scores(q_I, w, mu)
    sorted_block_indices = torch.argsort(block_scores, descending=True).tolist()

    # 首尾块强制包含
    forced = {0, n_full - 1}
    forced_covered = sum(1 for idx in true_topk_indices.tolist()
                         if idx // block_size in forced)

    result = {}
    covered = forced_covered
    covered_blocks = set(forced)

    m_count = 0
    for thresh in sorted(thresholds, reverse=True):
        target = int(thresh * actual_k) if thresh < 1.0 else actual_k

        while covered < target and m_count < len(sorted_block_indices):
            b = sorted_block_indices[m_count]
            m_count += 1

            if b in covered_blocks:
                continue
            covered_blocks.add(b)

            # 数这个块覆盖了多少真 top-k token
            for idx in true_topk_indices.tolist():
                if idx // block_size == b and b not in forced:
                    covered += 1

            if covered >= target:
                break

        result[thresh] = m_count

    return result
