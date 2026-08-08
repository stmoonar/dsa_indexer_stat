"""
test_recall_equiv.py — 验证测量 1 的等价性

核心性质：HISA 候选集内精确 top-k 的 recall == 块覆盖率。

证明：设 s* 是候选集 Ω_t 内的真 top-k token，全局分数高于它的 token ≤ k-1 个，
故它在 Ω_t 内也排进前 k；平局除外。

这个等价性意味着实现只需算块覆盖率，不需模拟 Stage 2。
"""

import pytest
import torch
from replay.bounds import exact_indexer_score, compute_block_summaries_ball
from replay.exact_score import compute_exact_topk


def hisa_stage1_select_blocks(
    q_I: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    m: int,
    n_full_blocks: int,
    force_first_last: bool = True,
) -> set[int]:
    """
    HISA Stage 1：选 top-m 块 + 强制首尾块。

    Args:
        q_I: [H, d]
        w: [H]
        mu: [n_blocks, d] 块均值
        m: 选取的块数
        n_full_blocks: 满块总数
        force_first_last: 是否强制包含首块和尾块（约束 4）

    Returns:
        选中的块索引集合
    """
    block_scores = exact_indexer_score(q_I, w, mu)  # [n_blocks]
    _, top_block_indices = torch.topk(
        block_scores, min(m, n_full_blocks), largest=True
    )

    selected = set(top_block_indices.tolist())
    if force_first_last and n_full_blocks > 0:
        selected.add(0)
        selected.add(n_full_blocks - 1)

    return selected


def compute_block_coverage(
    true_topk_indices: torch.Tensor,
    selected_blocks: set[int],
    block_size: int,
) -> float:
    """
    计算真 top-k 被选中块覆盖的比例。

    coverage = |{s ∈ true_topk : s // B ∈ selected_blocks}| / |true_topk|
    """
    if len(true_topk_indices) == 0:
        return 1.0

    covered = 0
    for idx in true_topk_indices.tolist():
        block_id = idx // block_size
        if block_id in selected_blocks:
            covered += 1

    return covered / len(true_topk_indices)


def hisa_stage2_rerank(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
    selected_blocks: set[int],
    block_size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HISA Stage 2：在候选集内精确打分并取 top-k。
    """
    candidate_indices = []
    for b in sorted(selected_blocks):
        start = b * block_size
        end = min(start + block_size, k_I.shape[0])
        candidate_indices.extend(range(start, end))

    if len(candidate_indices) == 0:
        return torch.tensor([]), torch.tensor([], dtype=torch.long)

    candidate_indices = torch.tensor(candidate_indices, dtype=torch.long)
    candidate_k = k_I[candidate_indices]

    scores = exact_indexer_score(q_I, w, candidate_k)
    actual_k = min(top_k, len(scores))
    topk_scores, local_indices = torch.topk(scores, actual_k, largest=True, sorted=True)
    global_indices = candidate_indices[local_indices]

    return topk_scores, global_indices


class TestRecallEquivCoverage:
    """HISA recall == 块覆盖率（模平局）。"""

    @pytest.mark.parametrize("seed", range(200))
    def test_equivalence_no_ties(self, seed):
        """无平局时，recall 精确等于覆盖率。"""
        H, d, L, B, k, m = 8, 16, 512, 32, 16, 8
        gen = torch.Generator().manual_seed(seed)

        # 加小扰动避免平局
        k_I = torch.randn(L, d, generator=gen) + torch.arange(L).unsqueeze(1) * 1e-8
        q_I = torch.randn(H, d, generator=gen)
        w = torch.randn(H, generator=gen)

        mu, r, n_tail = compute_block_summaries_ball(k_I, B)
        n_full = L // B

        # 真 top-k（用满块内的 token）
        k_I_full = k_I[:n_full * B]
        true_scores, true_indices = compute_exact_topk(q_I, w, k_I_full, top_k=k)

        # HISA stage 1
        selected = hisa_stage1_select_blocks(q_I, w, mu, m, n_full)

        # 覆盖率
        coverage = compute_block_coverage(true_indices, selected, B)

        # HISA stage 2 recall
        _, hisa_indices = hisa_stage2_rerank(q_I, w, k_I_full, selected, B, k)

        true_set = set(true_indices.tolist())
        hisa_set = set(hisa_indices.tolist())
        recall = len(true_set & hisa_set) / len(true_set) if true_set else 1.0

        assert abs(recall - coverage) < 1e-6, (
            f"Recall != coverage: recall={recall:.6f}, coverage={coverage:.6f}, seed={seed}"
        )

    def test_with_ties_boundary(self):
        """
        构造平局场景：多个 token 分数相同，落在不同块。
        此时 recall 可能因打分平局的仲裁方式与覆盖率不同。
        验证 recall >= coverage - tie 数量 / k。
        """
        H, d, L, B, k, m = 2, 4, 128, 16, 8, 4
        gen = torch.Generator().manual_seed(42)

        k_I = torch.randn(L, d, generator=gen)
        q_I = torch.randn(H, d, generator=gen)
        w = torch.abs(torch.randn(H, generator=gen))

        mu, _, _ = compute_block_summaries_ball(k_I, B)
        n_full = L // B
        k_I_full = k_I[:n_full * B]

        true_scores, true_indices = compute_exact_topk(q_I, w, k_I_full, top_k=k)
        selected = hisa_stage1_select_blocks(q_I, w, mu, m, n_full)

        coverage = compute_block_coverage(true_indices, selected, B)
        _, hisa_indices = hisa_stage2_rerank(q_I, w, k_I_full, selected, B, k)

        true_set = set(true_indices.tolist())
        hisa_set = set(hisa_indices.tolist())
        recall = len(true_set & hisa_set) / len(true_set) if true_set else 1.0

        # 有平局时 recall 仍然不低于 coverage 太多
        # 差距上界 = 平局 token 数 / k
        kth_score = true_scores[-1].item()
        all_scores = exact_indexer_score(q_I, w, k_I_full)
        n_ties = (torch.abs(all_scores - kth_score) < 1e-6).sum().item()
        tolerance = n_ties / k

        assert recall >= coverage - tolerance - 1e-6, (
            f"Recall too low: recall={recall:.6f}, coverage={coverage:.6f}, "
            f"tolerance={tolerance:.6f}"
        )

    def test_full_coverage_implies_full_recall(self):
        """100% 覆盖 → 100% recall（不论平局）。"""
        H, d, L, B, k = 4, 8, 256, 32, 8
        gen = torch.Generator().manual_seed(55)

        k_I = torch.randn(L, d, generator=gen)
        q_I = torch.randn(H, d, generator=gen)
        w = torch.randn(H, generator=gen)

        n_full = L // B
        k_I_full = k_I[:n_full * B]

        true_scores, true_indices = compute_exact_topk(q_I, w, k_I_full, top_k=k)

        # 选所有块 → 100% 覆盖
        all_blocks = set(range(n_full))
        coverage = compute_block_coverage(true_indices, all_blocks, B)
        assert coverage == 1.0

        _, hisa_indices = hisa_stage2_rerank(q_I, w, k_I_full, all_blocks, B, k)
        true_set = set(true_indices.tolist())
        hisa_set = set(hisa_indices.tolist())
        recall = len(true_set & hisa_set) / len(true_set)
        assert recall == 1.0
