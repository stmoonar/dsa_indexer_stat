"""
test_integration_smoke.py — 合成数据端到端集成测试

验证完整链路：生成 → 块摘要 → exact top-k → τ → 剪枝 → 覆盖率 → churn
全部不碰真实数据，用随机张量模拟。
"""

import torch
import pytest
from replay.bounds import (
    compute_block_summaries_ball,
    compute_block_summaries_box,
    ball_bound_upper,
    box_bound_upper,
    exact_indexer_score,
)
from replay.exact_score import compute_exact_topk, compute_tau_from_warm_set
from replay.m1_hisa_recall import hisa_coverage, adaptive_m_curve
from replay.m2_w_stats import w_sign_stats, w_neg_mass
from replay.m3_block_locality import block_locality_stats, cross_layer_overlap
from replay.m4_bound_prune import oneshot_prune, bestfirst_prune, prune_stats
from replay.kernel_align import compute_kernel_alignment


@pytest.fixture
def synth_data():
    """生成合成数据，模拟 V3.2 的维度但缩小规模。"""
    H, d, L, k, B = 64, 128, 4096, 64, 128
    gen = torch.Generator().manual_seed(2026)

    k_I = torch.randn(L, d, generator=gen)
    steps = []
    for _ in range(5):
        q_I = torch.randn(H, d, generator=gen)
        w = torch.randn(H, generator=gen)
        steps.append((q_I, w))

    return {"k_I": k_I, "steps": steps, "H": H, "d": d, "L": L, "k": k, "B": B}


class TestEndToEnd:

    def test_exact_topk(self, synth_data):
        q_I, w = synth_data["steps"][0]
        k_I = synth_data["k_I"]
        k = synth_data["k"]

        scores, indices = compute_exact_topk(q_I, w, k_I, top_k=k)
        assert scores.shape[0] == k
        assert indices.shape[0] == k
        assert (scores[:-1] >= scores[1:]).all(), "Scores not sorted descending"

    def test_hisa_coverage(self, synth_data):
        q_I, w = synth_data["steps"][0]
        k_I = synth_data["k_I"]
        B = synth_data["B"]
        k = synth_data["k"]

        result = hisa_coverage(q_I, w, k_I, block_size=B, m=16, top_k=k)
        assert 0.0 <= result["coverage"] <= 1.0
        assert result["n_full_blocks"] == 4096 // 128

    def test_adaptive_m(self, synth_data):
        q_I, w = synth_data["steps"][0]
        k_I = synth_data["k_I"]
        B = synth_data["B"]
        k = synth_data["k"]

        curve = adaptive_m_curve(q_I, w, k_I, block_size=B, top_k=k)
        assert 1.0 in curve
        assert 0.99 in curve
        assert curve[0.99] <= curve[1.0]

    def test_w_stats(self, synth_data):
        w_all = torch.stack([s[1] for s in synth_data["steps"]])  # [5, H]
        stats = w_sign_stats(w_all)
        assert abs(stats["neg_fraction"] + stats["zero_fraction"]
                    + stats["pos_fraction"] - 1.0) < 1e-6
        assert len(stats["per_head_neg_fraction"]) == synth_data["H"]

    def test_w_neg_mass(self, synth_data):
        w_all = torch.stack([s[1] for s in synth_data["steps"]])
        mass = w_neg_mass(w_all)
        assert 0.0 <= mass["neg_mass_ratio"] <= 1.0

    def test_block_locality(self, synth_data):
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]
        B = synth_data["B"]

        _, idx0 = compute_exact_topk(q0, w0, k_I, top_k=k)
        _, idx1 = compute_exact_topk(q1, w1, k_I, top_k=k)

        stats = block_locality_stats(idx1, idx0, B)
        assert stats["touched_blocks"] > 0
        assert 0.0 <= stats["churn_rate"] <= 2.0
        assert stats["churn_touched_blocks"] >= 0
        assert 0.0 <= stats["fetched_block_jaccard"] <= 1.0

    def test_cross_layer_overlap(self, synth_data):
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]

        _, idx0 = compute_exact_topk(q0, w0, k_I, top_k=k)
        _, idx1 = compute_exact_topk(q1, w1, k_I, top_k=k)

        overlap = cross_layer_overlap(idx0, idx1)
        assert 0.0 <= overlap <= 1.0

    def test_oneshot_prune(self, synth_data):
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]
        B = synth_data["B"]

        _, prev_idx = compute_exact_topk(q0, w0, k_I, top_k=k)
        warm_k_I = k_I[prev_idx]

        result = oneshot_prune(q1, w1, k_I, warm_k_I, block_size=B, top_k=k)
        assert 0.0 <= result["F_oneshot"] <= 1.0
        assert result["n_total_blocks"] == 4096 // 128

    def test_bestfirst_prune(self, synth_data):
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]
        B = synth_data["B"]

        _, prev_idx = compute_exact_topk(q0, w0, k_I, top_k=k)
        warm_k_I = k_I[prev_idx]

        result = bestfirst_prune(q1, w1, k_I, warm_k_I, block_size=B, top_k=k)
        assert 0.0 <= result["F_bestfirst"] <= 1.0
        assert result["F_bestfirst"] <= 1.0

    def test_bestfirst_leq_oneshot(self, synth_data):
        """Best-first 应不劣于 one-shot。"""
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]
        B = synth_data["B"]

        _, prev_idx = compute_exact_topk(q0, w0, k_I, top_k=k)
        warm_k_I = k_I[prev_idx]

        os_result = oneshot_prune(q1, w1, k_I, warm_k_I, block_size=B, top_k=k)
        bf_result = bestfirst_prune(q1, w1, k_I, warm_k_I, block_size=B, top_k=k)

        assert bf_result["F_bestfirst"] <= os_result["F_oneshot"] + 1e-6

    def test_prune_stats_sweep(self, synth_data):
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]

        _, prev_idx = compute_exact_topk(q0, w0, k_I, top_k=k)
        warm_k_I = k_I[prev_idx]

        for B in (64, 128, 256):
            result = prune_stats(q1, w1, k_I, warm_k_I, block_size=B, top_k=k)
            assert "ball_oneshot_F" in result
            assert "box_bestfirst_F" in result

    def test_kernel_align(self, synth_data):
        q_I, w = synth_data["steps"][0]
        k_I = synth_data["k_I"]
        k = synth_data["k"]

        _, replay_idx = compute_exact_topk(q_I, w, k_I, top_k=k)
        # 模拟 kernel 输出：与 replay 大部分相同，少量差异
        kernel_idx = replay_idx.clone()
        kernel_idx[0] = (replay_idx[0] + 1) % k_I.shape[0]

        result = compute_kernel_alignment(replay_idx, kernel_idx)
        assert result["match_rate"] > 0.9
        assert result["mismatch_rate"] < 0.1

    def test_bound_soundness_on_synth(self, synth_data):
        """在合成数据上再次验证 bound soundness。"""
        q_I, w = synth_data["steps"][0]
        k_I = synth_data["k_I"]
        B = synth_data["B"]

        mu, r, _ = compute_block_summaries_ball(k_I, B)
        U_ball = ball_bound_upper(q_I, w, mu, r)

        lo, hi, _ = compute_block_summaries_box(k_I, B)
        U_box = box_bound_upper(q_I, w, lo, hi)

        scores = exact_indexer_score(q_I, w, k_I)
        n_full = k_I.shape[0] // B

        for b in range(n_full):
            block_max = scores[b * B : (b + 1) * B].max().item()
            assert U_ball[b].item() >= block_max - 1e-3
            assert U_box[b].item() >= block_max - 1e-3

    def test_prune_correctness(self, synth_data):
        """剪枝后拉取的块必须覆盖全部 true top-k。"""
        q0, w0 = synth_data["steps"][0]
        q1, w1 = synth_data["steps"][1]
        k_I = synth_data["k_I"]
        k = synth_data["k"]
        B = synth_data["B"]

        _, prev_idx = compute_exact_topk(q0, w0, k_I, top_k=k)
        warm_k_I = k_I[prev_idx]

        # true top-k for current step
        n_full = k_I.shape[0] // B
        k_I_full = k_I[:n_full * B]
        _, true_idx = compute_exact_topk(q1, w1, k_I_full, top_k=k)

        # Ball bound one-shot
        mu, r, _ = compute_block_summaries_ball(k_I, B)
        U = ball_bound_upper(q1, w1, mu, r)
        tau_0 = compute_tau_from_warm_set(q1, w1, warm_k_I).item()

        fetched = set()
        fetched.add(0)
        fetched.add(n_full - 1)
        for b in range(n_full):
            if U[b].item() > tau_0:
                fetched.add(b)

        # warm 集 token + fetched 块 token 的并集
        candidate_indices = set(prev_idx.tolist())
        for b in fetched:
            start = b * B
            end = start + B
            candidate_indices.update(range(start, end))

        # 全部 true top-k 必须在候选集中
        true_set = set(true_idx.tolist())
        missed = true_set - candidate_indices
        assert len(missed) == 0, (
            f"Pruning missed {len(missed)} true top-k tokens! "
            f"tau_0={tau_0:.4f}, fetched {len(fetched)}/{n_full} blocks"
        )
