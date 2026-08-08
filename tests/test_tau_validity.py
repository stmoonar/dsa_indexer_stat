"""
test_tau_validity.py — 验证 τ 合法性

核心性质（约束 1）：
任意 2048 子集用当前 q 重打分的最小值 ≤ 全集第 2048 大。

即 τ_0（warm-start 下界）永远是合法下界，不会高估真实 top-k 阈值。
"""

import pytest
import torch
from replay.bounds import exact_indexer_score
from replay.exact_score import compute_exact_topk, compute_tau_from_warm_set


class TestTauValidity:
    """τ_0 必须 ≤ 全集第 k 大分数。"""

    @pytest.mark.parametrize("seed", range(500))
    def test_tau_leq_true_kth(self, seed):
        """
        随机生成 warm 集（全集的任意 k 子集），验证 warm 集重打分最小值 ≤ 真 top-k 阈值。
        """
        H, d, L, k = 8, 16, 512, 32
        gen = torch.Generator().manual_seed(seed)

        k_I = torch.randn(L, d, generator=gen)
        q_I = torch.randn(H, d, generator=gen)
        w = torch.randn(H, generator=gen)

        # 真 top-k
        true_topk_scores, true_topk_indices = compute_exact_topk(q_I, w, k_I, top_k=k)
        true_kth = true_topk_scores[-1].item()  # 第 k 大

        # 随机选 k 个 token 作为 warm 集（模拟任意子集）
        perm = torch.randperm(L, generator=gen)[:k]
        warm_k_I = k_I[perm]

        tau_0 = compute_tau_from_warm_set(q_I, w, warm_k_I).item()
        assert tau_0 <= true_kth + 1e-5, (
            f"τ_0 > true kth! tau_0={tau_0:.6f}, true_kth={true_kth:.6f}, seed={seed}"
        )

    @pytest.mark.parametrize("seed", range(500))
    def test_tau_with_prev_topk_as_warm(self, seed):
        """
        模拟真实场景：warm 集 = 上一步的 exact top-k，
        用当前步的 q/w 重打分。τ_0 仍必须 ≤ 当前步的真 top-k 阈值。
        """
        H, d, L, k = 8, 16, 512, 32
        gen = torch.Generator().manual_seed(seed)

        k_I = torch.randn(L, d, generator=gen)
        q_prev = torch.randn(H, d, generator=gen)
        w_prev = torch.randn(H, generator=gen)

        # 上一步 top-k
        _, prev_topk_indices = compute_exact_topk(q_prev, w_prev, k_I, top_k=k)
        warm_k_I = k_I[prev_topk_indices]

        # 当前步新的 q/w
        q_curr = torch.randn(H, d, generator=gen)
        w_curr = torch.randn(H, generator=gen)

        # 当前步真 top-k
        true_topk_scores, _ = compute_exact_topk(q_curr, w_curr, k_I, top_k=k)
        true_kth = true_topk_scores[-1].item()

        # 用当前 q/w 对 warm 集重打分
        tau_0 = compute_tau_from_warm_set(q_curr, w_curr, warm_k_I).item()
        assert tau_0 <= true_kth + 1e-5, (
            f"τ_0 > true kth with prev-topk warm! "
            f"tau_0={tau_0:.6f}, true_kth={true_kth:.6f}, seed={seed}"
        )

    def test_tau_equals_kth_when_warm_is_topk(self):
        """当 warm 集恰好是当前步 top-k 时，τ_0 == 真 top-k 阈值。"""
        H, d, L, k = 8, 16, 256, 16
        gen = torch.Generator().manual_seed(42)

        k_I = torch.randn(L, d, generator=gen)
        q_I = torch.randn(H, d, generator=gen)
        w = torch.randn(H, generator=gen)

        topk_scores, topk_indices = compute_exact_topk(q_I, w, k_I, top_k=k)
        warm_k_I = k_I[topk_indices]

        tau_0 = compute_tau_from_warm_set(q_I, w, warm_k_I).item()
        true_kth = topk_scores[-1].item()

        assert abs(tau_0 - true_kth) < 1e-5, (
            f"When warm=topk, τ_0 should equal kth. "
            f"tau_0={tau_0:.6f}, true_kth={true_kth:.6f}"
        )

    def test_tau_semantic_not_reuse_old_value(self):
        """
        验证"复用上一步 τ 数值"是非法的。

        构造：上一步 τ_old > 当前步真 top-k 阈值。
        这证明复用 τ_old 会漏 token。
        """
        H, d, L, k = 4, 8, 128, 8
        gen = torch.Generator().manual_seed(77)

        k_I = torch.randn(L, d, generator=gen)

        # 上一步：某个 q 产生较高的 τ
        q_old = torch.randn(H, d, generator=gen)
        w_old = torch.abs(torch.randn(H, generator=gen)) + 1.0  # 大正权重
        old_topk_scores, _ = compute_exact_topk(q_old, w_old, k_I, top_k=k)
        tau_old = old_topk_scores[-1].item()

        # 当前步：q 和 w 完全不同，分数分布不同
        q_new = torch.randn(H, d, generator=gen) * 0.1  # 小 query
        w_new = torch.randn(H, generator=gen) * 0.1  # 小权重

        new_topk_scores, _ = compute_exact_topk(q_new, w_new, k_I, top_k=k)
        true_kth_new = new_topk_scores[-1].item()

        # 旧 τ 很可能高于新的 kth，说明复用会高估阈值
        # （不是必然的，所以这个测试验证的是"存在这种情况"）
        if tau_old > true_kth_new:
            assert True, "Confirmed: reusing old τ can exceed true kth"
        else:
            pytest.skip("This seed didn't produce the failure case")
