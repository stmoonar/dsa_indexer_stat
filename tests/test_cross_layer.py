"""
test_cross_layer.py — analysis/cross_layer.py 的守护测试

用构造数据把三件事钉死：
- overlap 的口径（同层=1、构造的不相交层对=0）与随机基线归一（lift）
- 位置筛选：层间取交集、cache_len <= top_k 的平凡位置必须丢弃
- kernel 源的 -1 补位/越界写回必须被剔除
"""

import os
import json
import argparse

import numpy as np
import pytest

from analysis.cross_layer import (common_positions, topk_sets, overlap_and_lift,
                                  analyze_seq)

H, D, L, K = 1, 4, 64, 8
POSITIONS = (31, 47)


def _args(**kw):
    base = dict(top_k=K, source="exact", min_cache_mult=2.0, max_positions=0,
                device="cpu")
    base.update(kw)
    return argparse.Namespace(**base)


def _q(dim: int, sign: float = 1.0):
    q = np.zeros((H, D), np.float32)
    q[0, dim] = sign
    return q


def _write_layer(run, layer, k_col, q_dim, positions=POSITIONS, topk=None):
    """一层的 k^I + 若干位置的 prefill dump。

    k^I 只有一列非零，q 是该列的单位向量 → 分数 = 该列的值，
    top-k 集合完全可手算。
    """
    k = np.zeros((L, D), np.float32)
    k[:, k_col] = np.arange(L, dtype=np.float32) if q_dim == 0 \
        else (L - 1 - np.arange(L, dtype=np.float32))
    np.save(os.path.join(run, f"k_I_layer{layer:03d}.npy"), k)
    for p in positions:
        np.savez(os.path.join(run, f"prefill_pos{p:08d}_layer{layer:03d}.npz"),
                 q_I=_q(q_dim), w=np.ones(H, np.float32),
                 topk_indices=(topk if topk is not None
                               else np.zeros(K, np.int32)))


@pytest.fixture
def run(tmp_path):
    """三层：0 与 2 完全相同（选末尾 K 个），1 反向（选开头 K 个）。"""
    d = str(tmp_path)
    _write_layer(d, 0, k_col=0, q_dim=0)
    _write_layer(d, 1, k_col=1, q_dim=1)
    _write_layer(d, 2, k_col=0, q_dim=0)
    json.dump({"capture_layers": [0, 1, 2], "num_layers": 3,
               "prefill_buckets": [], "prefill_run": 32},
              open(os.path.join(d, "capture_config.json"), "w"))
    return d


class TestPositions:

    def test_common_is_intersection(self, run, tmp_path):
        os.remove(os.path.join(run, f"prefill_pos{POSITIONS[0]:08d}_layer001.npz"))
        assert common_positions(run, [0, 1, 2]) == [POSITIONS[1]]
        assert common_positions(run, [0, 2]) == list(POSITIONS)

    def test_short_prefix_dropped(self, run):
        """cache_len <= top_k 时 top-k 就是全集，overlap 恒为 1，必须丢弃。"""
        for l in (0, 1, 2):
            _write_layer(run, l, k_col=(0 if l != 1 else 1),
                         q_dim=(0 if l != 1 else 1), positions=(5,))
        cfg = {"prefill_buckets": [], "prefill_run": 32}
        r = analyze_seq(run, [0, 1, 2], cfg, _args())
        assert r["n_positions"] == len(POSITIONS)      # pos=5 被丢
        assert r["positions_dropped_short_prefix"] == 1
        assert r["cache_len_range"] == [POSITIONS[0] + 1, POSITIONS[1] + 1]

    def test_all_dropped_reports_error(self, run):
        r = analyze_seq(run, [0, 1, 2], {}, _args(min_cache_mult=100.0))
        assert "error" in r


class TestTopkSets:

    def test_exact_sets_are_hand_computable(self, run):
        s0 = topk_sets(run, 0, [31], K, "exact", "cpu")[31]
        s1 = topk_sets(run, 1, [31], K, "exact", "cpu")[31]
        assert s0.tolist() == list(range(24, 32))      # 分数 = 位置，取末尾 K 个
        assert s1.tolist() == list(range(0, 8))        # 分数递减，取开头 K 个

    def test_kernel_source_filters_padding_and_out_of_range(self, run):
        raw = np.array([24, 25, 26, 27, -1, -1, 999, 30], np.int32)
        _write_layer(run, 0, k_col=0, q_dim=0, positions=(31,), topk=raw)
        s = topk_sets(run, 0, [31], K, "kernel", "cpu")[31]
        assert s.tolist() == [24, 25, 26, 27, 30]


class TestOverlap:

    def test_identical_sets(self):
        a = np.arange(8)
        o, lift, base = overlap_and_lift(a, a.copy(), cache_len=32)
        assert o == 1.0 and lift == 1.0 and base == pytest.approx(0.25)

    def test_disjoint_sets_give_negative_lift(self):
        o, lift, base = overlap_and_lift(np.arange(8), np.arange(8, 16),
                                         cache_len=32)
        assert o == 0.0
        assert lift == pytest.approx((0 - 0.25) / 0.75)

    def test_lift_zero_at_random_baseline(self):
        """基线 k/L=0.25：8 个里恰好命中 2 个 → lift ≈ 0。"""
        o, lift, _ = overlap_and_lift(np.arange(8),
                                      np.array([0, 1, 8, 9, 10, 11, 12, 13]),
                                      cache_len=32)
        assert o == pytest.approx(0.25) and lift == pytest.approx(0.0)

    def test_full_set_baseline_gives_zero_lift(self):
        """cache_len == k 时基线为 1，lift 无定义 → 定义为 0，且不得除零。"""
        a = np.arange(8)
        o, lift, base = overlap_and_lift(a, a.copy(), cache_len=8)
        assert base == 1.0 and lift == 0.0


class TestAnalyzeSeq:

    def test_pair_values_match_construction(self, run):
        r = analyze_seq(run, [0, 1, 2], {"prefill_buckets": [],
                                         "prefill_run": 32}, _args())
        assert r["by_pair"]["0-2"]["overlap"]["p50"] == 1.0
        assert r["by_pair"]["0-1"]["overlap"]["p50"] == 0.0
        assert r["by_pair"]["1-2"]["overlap"]["p50"] == 0.0
        # 距离 1 有两对 × 两个位置；距离 2 有一对 × 两个位置
        assert r["by_distance"]["1"]["overlap"]["n"] == 4
        assert r["by_distance"]["2"]["overlap"]["n"] == 2
        assert set(r["adjacent"]) == {"0-1", "1-2"}

    def test_baseline_reported_per_position(self, run):
        r = analyze_seq(run, [0, 1, 2], {}, _args())
        # 位置 31/47 的基线 = K/(pos+1)
        assert r["random_baseline"]["p50"] == pytest.approx(
            (K / 32 + K / 48) / 2, rel=1e-6)

    def test_max_positions_subsamples(self, run):
        r = analyze_seq(run, [0, 1, 2], {}, _args(max_positions=1))
        assert r["n_positions"] == 1
