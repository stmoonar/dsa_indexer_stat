"""
m4_bound_prune.py — 测量 4: Bound 剪枝率

两个版本：
- F_oneshot: τ_0 一次剪枝
- F_bestfirst: 按 U_b 降序逐块处理、动态更新 τ

约束 1：τ_0 = 用当前步 q^I 对 warm 集重打分的最小值。
约束 2：bound 处理 w < 0。
约束 4：强制包含首块和尾块。
约束 5：尾部未满块不建摘要。
"""

import torch
from replay.bounds import (
    compute_block_summaries_ball,
    compute_block_summaries_box,
    ball_bound_upper,
    box_bound_upper,
    exact_indexer_score,
)
from replay.exact_score import compute_exact_topk, compute_tau_from_warm_set


def oneshot_prune(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    warm_k_I: torch.Tensor,
    block_size: int = 128,
    bound_type: str = "ball",
    top_k: int = 2048,
) -> dict:
    """
    One-shot 剪枝：用 τ_0 一次性剪掉所有 U_b ≤ τ_0 的块。

    Args:
        q_I: [H, d], 当前步 query
        w: [H], 当前步 weight
        k_I: [L, d], 全量 indexer key
        warm_k_I: [k, d], 上一步 top-k 的 k^I
        block_size: 块大小
        bound_type: "ball" 或 "box"
        top_k: 选取数

    Returns:
        dict with F_oneshot, tau_0, n_fetched_blocks, n_total_blocks, etc.
    """
    # 计算 τ_0（约束 1）
    tau_0 = compute_tau_from_warm_set(q_I, w, warm_k_I).item()

    # 计算块摘要和上界
    if bound_type == "ball":
        mu, r, n_tail = compute_block_summaries_ball(k_I, block_size)
        U = ball_bound_upper(q_I, w, mu, r)
    else:
        lo, hi, n_tail = compute_block_summaries_box(k_I, block_size)
        U = box_bound_upper(q_I, w, lo, hi)

    n_blocks = U.shape[0]
    if n_blocks == 0:
        return {"F_oneshot": 0.0, "tau_0": tau_0,
                "n_fetched_blocks": 0, "n_total_blocks": 0, "n_tail": n_tail}

    # 强制包含首尾块（约束 4）
    forced_blocks = {0, n_blocks - 1}

    # 剪枝：U_b > τ_0 的块需要 fetch
    fetched = set()
    for b in range(n_blocks):
        if U[b].item() > tau_0 or b in forced_blocks:
            fetched.add(b)

    F_oneshot = len(fetched) / n_blocks

    return {
        "F_oneshot": F_oneshot,
        "tau_0": tau_0,
        "n_fetched_blocks": len(fetched),
        "n_total_blocks": n_blocks,
        "n_tail": n_tail,
    }


def bestfirst_prune(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    warm_k_I: torch.Tensor,
    block_size: int = 128,
    bound_type: str = "ball",
    top_k: int = 2048,
) -> dict:
    """
    Best-first 剪枝：按 U_b 降序处理，逐块精确打分并更新 τ。

    真实算法的 F：每处理一个块，将其 token 分数纳入候选集，
    更新 τ（候选集第 k 大），直到剩余块的 U 全部 ≤ τ。
    """
    # 计算 τ_0
    tau_0 = compute_tau_from_warm_set(q_I, w, warm_k_I).item()

    # 计算块摘要和上界
    if bound_type == "ball":
        mu, r, n_tail = compute_block_summaries_ball(k_I, block_size)
        U = ball_bound_upper(q_I, w, mu, r)
    else:
        lo, hi, n_tail = compute_block_summaries_box(k_I, block_size)
        U = box_bound_upper(q_I, w, lo, hi)

    n_blocks = U.shape[0]
    if n_blocks == 0:
        return {"F_bestfirst": 0.0, "tau_0": tau_0,
                "n_fetched_blocks": 0, "n_total_blocks": 0, "n_tail": n_tail}

    # 强制包含首尾块（约束 4）
    forced_blocks = {0, n_blocks - 1}

    # 按 U 降序排列
    sorted_indices = torch.argsort(U, descending=True).tolist()

    # warm 集的分数
    warm_scores = exact_indexer_score(q_I, w, warm_k_I)  # [k]
    all_candidate_scores = list(warm_scores.tolist())

    tau = tau_0
    fetched = set(forced_blocks)

    # 先处理强制块，将其分数纳入候选集
    for b in forced_blocks:
        start = b * block_size
        end = min(start + block_size, k_I.shape[0])
        if end > start:
            block_k = k_I[start:end]
            block_scores = exact_indexer_score(q_I, w, block_k)
            all_candidate_scores.extend(block_scores.tolist())

    # 更新 τ
    if len(all_candidate_scores) >= top_k:
        sorted_scores = sorted(all_candidate_scores, reverse=True)
        tau = sorted_scores[min(top_k - 1, len(sorted_scores) - 1)]

    # Best-first: 按 U 降序处理
    for b in sorted_indices:
        if b in fetched:
            continue

        if U[b].item() <= tau:
            break

        fetched.add(b)
        start = b * block_size
        end = min(start + block_size, k_I.shape[0])
        if end > start:
            block_k = k_I[start:end]
            block_scores = exact_indexer_score(q_I, w, block_k)
            all_candidate_scores.extend(block_scores.tolist())

            if len(all_candidate_scores) >= top_k:
                sorted_scores = sorted(all_candidate_scores, reverse=True)
                tau = sorted_scores[min(top_k - 1, len(sorted_scores) - 1)]

    F_bestfirst = len(fetched) / n_blocks

    return {
        "F_bestfirst": F_bestfirst,
        "tau_final": tau,
        "tau_0": tau_0,
        "n_fetched_blocks": len(fetched),
        "n_total_blocks": n_blocks,
        "n_tail": n_tail,
    }


def prune_stats(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    warm_k_I: torch.Tensor,
    block_size: int = 128,
    top_k: int = 2048,
) -> dict:
    """综合计算 one-shot 和 best-first 的剪枝率，含 ball 和 box bound。"""
    result = {}

    for bound_type in ("ball", "box"):
        os_result = oneshot_prune(
            q_I, w, k_I, warm_k_I, block_size, bound_type, top_k
        )
        bf_result = bestfirst_prune(
            q_I, w, k_I, warm_k_I, block_size, bound_type, top_k
        )

        result[f"{bound_type}_oneshot_F"] = os_result["F_oneshot"]
        result[f"{bound_type}_bestfirst_F"] = bf_result["F_bestfirst"]
        result[f"{bound_type}_tau_0"] = os_result["tau_0"]

    return result
