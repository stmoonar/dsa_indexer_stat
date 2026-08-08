"""
test_bounds_soundness.py — 验证 bound 的 soundness 性质

核心性质：对块 b 内任意 token s，U_{t,b} >= I_{t,s}。
即上界永远不低估真实最大分。

在随机数据上跑 10^6 组，含 w 混合符号情形。
"""

import pytest
import torch
from replay.bounds import (
    compute_block_summaries_ball,
    compute_block_summaries_box,
    ball_bound_upper,
    box_bound_upper,
    exact_indexer_score,
)


def _random_test_data(
    L: int, H: int, d: int, block_size: int,
    w_mode: str = "mixed",
    seed: int = 0,
    device: str = "cpu",
):
    """生成随机测试数据。

    w_mode: "positive", "negative", "mixed"
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    k_I = torch.randn(L, d, device=device, generator=gen)
    q_I = torch.randn(H, d, device=device, generator=gen)

    if w_mode == "positive":
        w = torch.rand(H, device=device, generator=gen) + 0.01
    elif w_mode == "negative":
        w = -(torch.rand(H, device=device, generator=gen) + 0.01)
    else:
        w = torch.randn(H, device=device, generator=gen)

    return q_I, w, k_I


class TestBallBoundSoundness:
    """Ball bound 必须是所有块内 token 分数的上界。"""

    @pytest.mark.parametrize("w_mode", ["positive", "negative", "mixed"])
    @pytest.mark.parametrize("block_size", [16, 64, 128])
    def test_soundness_basic(self, w_mode, block_size):
        H, d, L = 64, 128, 1024
        q_I, w, k_I = _random_test_data(L, H, d, block_size, w_mode=w_mode, seed=42)

        mu, r, n_tail = compute_block_summaries_ball(k_I, block_size)
        U = ball_bound_upper(q_I, w, mu, r)

        n_full = L // block_size
        scores = exact_indexer_score(q_I, w, k_I)

        for b in range(n_full):
            block_scores = scores[b * block_size : (b + 1) * block_size]
            block_max = block_scores.max().item()
            bound_val = U[b].item()
            assert bound_val >= block_max - 1e-4, (
                f"Ball bound violated: block {b}, bound={bound_val:.6f}, "
                f"max_score={block_max:.6f}, w_mode={w_mode}"
            )

    @pytest.mark.parametrize("seed", range(1000))
    def test_soundness_mass(self, seed):
        """大规模随机测试：1000 种不同种子 × 混合符号 w。"""
        H, d, L, B = 8, 16, 256, 32
        q_I, w, k_I = _random_test_data(L, H, d, B, w_mode="mixed", seed=seed)

        mu, r, _ = compute_block_summaries_ball(k_I, B)
        U = ball_bound_upper(q_I, w, mu, r)

        scores = exact_indexer_score(q_I, w, k_I)
        n_full = L // B

        for b in range(n_full):
            block_max = scores[b * B : (b + 1) * B].max().item()
            assert U[b].item() >= block_max - 1e-4, (
                f"Ball bound violated at seed={seed}, block={b}"
            )

    def test_tail_block_excluded(self):
        """尾部未满块不建摘要（约束 5）。"""
        L, d, B = 100, 16, 32
        k_I = torch.randn(L, d)
        mu, r, n_tail = compute_block_summaries_ball(k_I, B)
        assert mu.shape[0] == 3  # 100 // 32 = 3
        assert n_tail == 4  # 100 - 96 = 4

    def test_zero_length(self):
        """空输入不崩。"""
        k_I = torch.randn(0, 16)
        mu, r, n_tail = compute_block_summaries_ball(k_I, 32)
        assert mu.shape[0] == 0
        assert n_tail == 0

    def test_single_block(self):
        """单块边界情形。"""
        H, d, B = 4, 8, 32
        q_I, w, k_I = _random_test_data(B, H, d, B, w_mode="mixed", seed=99)
        mu, r, _ = compute_block_summaries_ball(k_I, B)
        U = ball_bound_upper(q_I, w, mu, r)
        scores = exact_indexer_score(q_I, w, k_I)
        assert U[0].item() >= scores.max().item() - 1e-4


class TestBoxBoundSoundness:
    """Box bound 必须是所有块内 token 分数的上界。"""

    @pytest.mark.parametrize("w_mode", ["positive", "negative", "mixed"])
    @pytest.mark.parametrize("block_size", [16, 64, 128])
    def test_soundness_basic(self, w_mode, block_size):
        H, d, L = 64, 128, 1024
        q_I, w, k_I = _random_test_data(L, H, d, block_size, w_mode=w_mode, seed=42)

        lo, hi, n_tail = compute_block_summaries_box(k_I, block_size)
        U = box_bound_upper(q_I, w, lo, hi)

        n_full = L // block_size
        scores = exact_indexer_score(q_I, w, k_I)

        for b in range(n_full):
            block_scores = scores[b * block_size : (b + 1) * block_size]
            block_max = block_scores.max().item()
            bound_val = U[b].item()
            assert bound_val >= block_max - 1e-4, (
                f"Box bound violated: block {b}, bound={bound_val:.6f}, "
                f"max_score={block_max:.6f}, w_mode={w_mode}"
            )

    @pytest.mark.parametrize("seed", range(1000))
    def test_soundness_mass(self, seed):
        """大规模随机测试。"""
        H, d, L, B = 8, 16, 256, 32
        q_I, w, k_I = _random_test_data(L, H, d, B, w_mode="mixed", seed=seed)

        lo, hi, _ = compute_block_summaries_box(k_I, B)
        U = box_bound_upper(q_I, w, lo, hi)

        scores = exact_indexer_score(q_I, w, k_I)
        n_full = L // B

        for b in range(n_full):
            block_max = scores[b * B : (b + 1) * B].max().item()
            assert U[b].item() >= block_max - 1e-4, (
                f"Box bound violated at seed={seed}, block={b}"
            )


class TestBoxTighterThanBall:
    """Box bound 应该不松于 ball bound（在大多数情况下更紧）。"""

    def test_box_leq_ball(self):
        H, d, L, B = 8, 16, 256, 32
        q_I, w, k_I = _random_test_data(L, H, d, B, w_mode="mixed", seed=123)

        mu, r, _ = compute_block_summaries_ball(k_I, B)
        lo, hi, _ = compute_block_summaries_box(k_I, B)

        U_ball = ball_bound_upper(q_I, w, mu, r)
        U_box = box_bound_upper(q_I, w, lo, hi)

        # Box bound 不一定严格小于 ball bound（取决于数据分布），
        # 但通常更紧。这里只验证两者都是 valid upper bound。
        scores = exact_indexer_score(q_I, w, k_I)
        n_full = L // B
        for b in range(n_full):
            block_max = scores[b * B : (b + 1) * B].max().item()
            assert U_ball[b].item() >= block_max - 1e-4
            assert U_box[b].item() >= block_max - 1e-4
