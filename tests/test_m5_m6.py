"""
test_m5_m6.py — oracle 天花板 / 分组 sweep / 松弛归因 / 想法 2 重建

这些指标会直接进结论，所以每一个都用"答案已知"的构造来验，
而不是只验类型和范围。
"""

import numpy as np
import torch
import pytest

from replay.m5_oracle_ceiling import (contiguous_groups, kmeans_groups,
                                      group_max, oracle_ceiling,
                                      grouping_sweep, slack_attribution,
                                      q_weighted_spectrum)
from replay.m6_reconstruction import (fit_linear, apply_linear, r2_score,
                                      reconstruction_test)
from replay.bounds import compute_block_summaries_ball, exact_indexer_score

H, D = 8, 16


class TestGroups:

    def test_contiguous_exact(self):
        g = contiguous_groups(10, 4)
        assert g.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 尾部并入末组

    def test_contiguous_exact_multiple(self):
        assert contiguous_groups(8, 4).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_group_max(self):
        s = torch.tensor([1.0, 5.0, 2.0, 0.0])
        g = torch.tensor([0, 0, 1, 1])
        assert group_max(s, g, 2).tolist() == [5.0, 2.0]

    def test_kmeans_assignment_valid(self):
        g = torch.Generator().manual_seed(0)
        k = torch.randn(200, D, generator=g)
        a = kmeans_groups(k, 8, iters=3)
        assert a.shape == (200,)
        assert int(a.min()) >= 0 and int(a.max()) < 8

    def test_kmeans_separates_well_separated_clusters(self):
        g = torch.Generator().manual_seed(1)
        blobs = torch.cat([torch.randn(50, D, generator=g) + 100 * i
                           for i in range(3)])
        a = kmeans_groups(blobs, 3, iters=20, seed=3)
        for i in range(3):
            assert len(set(a[i * 50:(i + 1) * 50].tolist())) == 1, \
                "well-separated blob was split"


class TestOracleCeiling:

    def setup_method(self):
        self.scores = torch.tensor([1.0, 2.0, 10.0, 0.5, 0.1, 0.2])
        self.groups = torch.tensor([0, 0, 1, 1, 2, 2])

    def test_tau_below_all(self):
        r = oracle_ceiling(self.scores, self.groups, tau=-1.0)
        assert r["group_frac"] == 0.0 and r["token_frac"] == 0.0

    def test_tau_above_all(self):
        r = oracle_ceiling(self.scores, self.groups, tau=100.0)
        assert r["group_frac"] == 1.0 and r["token_frac"] == 1.0

    def test_partial(self):
        # 组最大分 = [2, 10, 0.2]；tau=1 -> 只有第 3 组可跳过
        r = oracle_ceiling(self.scores, self.groups, tau=1.0)
        assert r["group_frac"] == pytest.approx(1 / 3)
        assert r["token_frac"] == pytest.approx(2 / 6)

    def test_token_frac_differs_when_sizes_uneven(self):
        """簇大小不等时 token 占比才是真实读放大，组占比会高估收益。"""
        scores = torch.tensor([0.1] * 9 + [10.0])
        groups = torch.tensor([0] * 9 + [1])     # 大组可跳，小组不可
        r = oracle_ceiling(scores, groups, tau=1.0)
        assert r["group_frac"] == pytest.approx(0.5)
        assert r["token_frac"] == pytest.approx(0.9)

    def test_monotone_in_tau(self):
        vals = [oracle_ceiling(self.scores, self.groups, t)["token_frac"]
                for t in (-1, 0.15, 1, 5, 100)]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))


class TestGroupingSweep:

    def test_sweep_keys_and_ranges(self):
        g = torch.Generator().manual_seed(2)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(256, D, generator=g)
        out = grouping_sweep(q, w, k, tau=0.0, block_sizes=(32, 64),
                             kmeans_block_sizes=(32, 64), kmeans_iters=3)
        assert set(out) == {"contiguous_B32", "kmeans_B32",
                            "contiguous_B64", "kmeans_B64"}
        for st in out.values():
            assert 0.0 <= st["group_frac"] <= 1.0
            assert 0.0 <= st["token_frac"] <= 1.0

    def test_kmeans_restricted_to_selected_block_sizes(self):
        """k-means 很贵，只应在指定的 B 上跑。"""
        g = torch.Generator().manual_seed(2)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(256, D, generator=g)
        out = grouping_sweep(q, w, k, tau=0.0, block_sizes=(32, 64),
                             kmeans_block_sizes=(64,), kmeans_iters=2)
        assert "kmeans_B64" in out and "kmeans_B32" not in out

    def test_group_cache_reused_across_calls(self):
        """同桶复用聚类：第二次调用不得重新聚类（缓存被命中）。"""
        g = torch.Generator().manual_seed(5)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(256, D, generator=g)
        cache = {}
        grouping_sweep(q, w, k, 0.0, block_sizes=(64,),
                       kmeans_block_sizes=(64,), kmeans_iters=2,
                       group_cache=cache)
        assert 64 in cache
        stamped = cache[64].clone()
        cache[64][:] = 0                      # 篡改缓存
        out = grouping_sweep(q, w, k, 0.0, block_sizes=(64,),
                             kmeans_block_sizes=(64,), kmeans_iters=2,
                             group_cache=cache)
        assert (cache[64] == 0).all(), "缓存未被复用（被重新计算覆盖）"
        assert out["kmeans_B64"]["n_groups"] == 1   # 全 0 分组 -> 单组
        assert not torch.equal(stamped, cache[64])

    def test_same_group_count_for_fair_comparison(self):
        g = torch.Generator().manual_seed(3)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(256, D, generator=g)
        out = grouping_sweep(q, w, k, tau=1e9, block_sizes=(64,),
                             kmeans_block_sizes=(64,), kmeans_iters=2)
        assert out["contiguous_B64"]["n_groups"] == out["kmeans_B64"]["n_groups"]


class TestSlackAttribution:

    def _run(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(256, D, generator=g)
        mu, r, _ = compute_block_summaries_ball(k, 64)
        return slack_attribution(q, w, k, mu, r, 64)

    def test_shares_sum_to_one(self):
        """radius + center 必须恰好等于总松弛（分解无遗漏项）。"""
        a = self._run()
        assert a["radius_share"] + a["center_share"] == pytest.approx(1.0, abs=0.05)

    def test_total_slack_nonnegative(self):
        """U 是上界，总松弛不得为负。"""
        for s in range(4):
            assert self._run(s)["total_slack_p50"] >= 0

    def test_neg_branch_recovered_nonnegative(self):
        """只留正支的上界不可能低于双符号上界。"""
        for s in range(4):
            assert self._run(s)["neg_branch_recovered_p50"] >= -1e-6

    def test_empty_when_no_full_block(self):
        g = torch.Generator().manual_seed(0)
        q, w = torch.randn(H, D, generator=g), torch.randn(H, generator=g)
        k = torch.randn(10, D, generator=g)
        mu, r, _ = compute_block_summaries_ball(k, 64)
        assert slack_attribution(q, w, k, mu, r, 64) == {}


class TestSpectrum:

    def test_rank_one_concentrates(self):
        v = torch.zeros(D)
        v[0] = 1.0
        q = v.unsqueeze(0).repeat(H, 1)
        w = torch.ones(H)
        sp = q_weighted_spectrum([q], [w], top_m=(1, 2))
        assert sp["energy_frac"][1] == pytest.approx(1.0, abs=1e-6)
        assert sp["effective_rank"] == pytest.approx(1.0, abs=1e-3)

    def test_isotropic_spreads(self):
        g = torch.Generator().manual_seed(0)
        qs = [torch.randn(H, D, generator=g) for _ in range(200)]
        ws = [torch.ones(H) for _ in range(200)]
        sp = q_weighted_spectrum(qs, ws, top_m=(1, D // 2))
        assert sp["energy_frac"][1] < 0.25
        assert sp["effective_rank"] > D * 0.6


class TestReconstruction:

    def test_linear_fit_is_exact(self):
        g = torch.Generator().manual_seed(0)
        X = torch.randn(500, 12, generator=g)
        Wtrue = torch.randn(12, 5, generator=g)
        Y = X @ Wtrue + 3.0
        W = fit_linear(X, Y, ridge=0.0)
        assert r2_score(Y, apply_linear(X, W)) == pytest.approx(1.0, abs=1e-6)

    def test_reconstruction_succeeds_when_truly_linear(self):
        """k^I 真是 c 的线性函数时，R² 和 top-k overlap 都应接近 1。"""
        g = torch.Generator().manual_seed(1)
        L = 800
        c = torch.randn(L, 32, generator=g)
        kpe = torch.randn(L, 8, generator=g)
        M = torch.randn(40, D, generator=g)
        k_I = torch.cat([c, kpe], 1) @ M
        qs = [torch.randn(H, D, generator=g) for _ in range(3)]
        ws = [torch.randn(H, generator=g) for _ in range(3)]
        out = reconstruction_test(k_I, c, kpe, qs, ws, top_k=64, n_fit=300)
        assert out["r2_heldout"] > 0.999
        assert out["topk_overlap_p50"] > 0.95

    def test_reconstruction_fails_on_unrelated_latent(self):
        """c 与 k^I 无关时必须给出低 R²——否则指标没有鉴别力。"""
        g = torch.Generator().manual_seed(2)
        L = 800
        c = torch.randn(L, 32, generator=g)
        kpe = torch.randn(L, 8, generator=g)
        k_I = torch.randn(L, D, generator=g)
        qs = [torch.randn(H, D, generator=g) for _ in range(3)]
        ws = [torch.randn(H, generator=g) for _ in range(3)]
        out = reconstruction_test(k_I, c, kpe, qs, ws, top_k=64, n_fit=300)
        assert out["r2_heldout"] < 0.1
        assert out["topk_overlap_p50"] < 0.5

    def test_shuffled_baseline_separates_signal_from_misalignment(self):
        """真有线性关系时，R² 必须明显高于打乱基线；否则该指标无法把
        '关系不存在' 和 '配对错位' 区分开。"""
        g = torch.Generator().manual_seed(4)
        L = 800
        c = torch.randn(L, 32, generator=g)
        kpe = torch.randn(L, 8, generator=g)
        k_I = torch.cat([c, kpe], 1) @ torch.randn(40, D, generator=g)
        out = reconstruction_test(k_I, c, kpe, [], [], n_fit=300)
        assert out["signal_above_shuffle"] > 0.9
        assert out["r2_shuffled_baseline"] < 0.1

    def test_detects_leading_offset_in_latent(self):
        """latent 多出开头几行（dummy forward 被误采）时必须报出来，
        并改用尾对齐把关系找回来。"""
        g = torch.Generator().manual_seed(5)
        L, pad = 800, 37
        c = torch.randn(L, 32, generator=g)
        kpe = torch.randn(L, 8, generator=g)
        k_I = torch.cat([c, kpe], 1) @ torch.randn(40, D, generator=g)
        c_bad = torch.cat([torch.randn(pad, 32, generator=g), c])
        kpe_bad = torch.cat([torch.randn(pad, 8, generator=g), kpe])

        out = reconstruction_test(k_I, c_bad, kpe_bad, [], [], n_fit=300)
        assert out["row_mismatch"] == pad
        assert out["tail_aligned"] is True
        assert out["r2_heldout"] > 0.99, "尾对齐后应恢复出线性关系"

    def test_misaligned_pairing_looks_like_no_signal(self):
        """行数相同但内容错位时 R² 会塌到打乱基线水平——
        这正是必须靠 signal_above_shuffle 才能识别的情形。"""
        g = torch.Generator().manual_seed(6)
        L = 800
        c = torch.randn(L, 32, generator=g)
        kpe = torch.randn(L, 8, generator=g)
        k_I = torch.cat([c, kpe], 1) @ torch.randn(40, D, generator=g)
        roll = torch.roll(torch.arange(L), 137)
        out = reconstruction_test(k_I, c[roll], kpe[roll], [], [], n_fit=300)
        assert out["row_mismatch"] == 0
        assert out["signal_above_shuffle"] < 0.05

    def test_fit_and_eval_positions_disjoint(self):
        g = torch.Generator().manual_seed(3)
        L = 400
        c, kpe = torch.randn(L, 8, generator=g), torch.randn(L, 4, generator=g)
        out = reconstruction_test(torch.randn(L, D, generator=g), c, kpe,
                                  [], [], n_fit=100)
        assert out["n_fit"] + out["n_eval"] == L
