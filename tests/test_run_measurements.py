"""
test_run_measurements.py — analysis/run_measurements.py 的守护测试

重点是两个新指标的语义：
- oracle_ceiling: bound 完全紧时的剪枝上限（定位瓶颈在数据还是在 bound）
- bound_slack:   (U - 真实块最大分) / |真实块最大分|，必须 >= 0（soundness）
以及 prefill / decode 两种数据源的样本枚举。
"""

import os
import json

import numpy as np
import torch
import pytest

from analysis.run_measurements import (summary, oracle_ceiling, bound_slack,
                                       collect_samples)
from replay.bounds import exact_indexer_score

H, D, B = 8, 16, 32


@pytest.fixture
def synth():
    g = torch.Generator().manual_seed(7)
    return (torch.randn(H, D, generator=g),
            torch.randn(H, generator=g),
            torch.randn(B * 10, D, generator=g))


class TestSummary:

    def test_percentiles(self):
        s = summary(list(range(101)))
        assert s["p5"] == 5 and s["p50"] == 50 and s["p95"] == 95
        assert s["n"] == 101

    def test_empty(self):
        assert summary([])["n"] == 0


class TestOracleCeiling:

    def test_tau_below_everything_prunes_nothing(self, synth):
        q, w, k = synth
        assert oracle_ceiling(q, w, k, tau=-1e9, block_size=B) == 0.0

    def test_tau_above_everything_prunes_all(self, synth):
        q, w, k = synth
        assert oracle_ceiling(q, w, k, tau=1e9, block_size=B) == 1.0

    def test_matches_direct_count(self, synth):
        q, w, k = synth
        sc = exact_indexer_score(q, w, k)
        n_full = k.shape[0] // B
        bmax = sc[:n_full * B].reshape(n_full, B).max(1).values
        tau = float(bmax.median())
        expect = 1.0 - (bmax > tau).float().mean().item()
        assert abs(oracle_ceiling(q, w, k, tau, B) - expect) < 1e-9

    def test_monotone_in_tau(self, synth):
        """τ 越大，可剪块越多——单调不减。"""
        q, w, k = synth
        vals = [oracle_ceiling(q, w, k, t, B)
                for t in (-100, -10, 0, 10, 100)]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))


class TestBoundSlack:

    def test_nonnegative(self, synth):
        """bound 是上界，松弛量不得为负（否则 soundness 被破坏）。"""
        q, w, k = synth
        sl = bound_slack(q, w, k, B)
        assert sl["ball"] >= -1e-6 and sl["box"] >= -1e-6

    def test_empty_when_no_full_block(self, synth):
        q, w, k = synth
        assert bound_slack(q, w, k[:B - 1], B) == {}


class TestCollectSamples:

    def _write(self, d, name, **arrays):
        np.savez(os.path.join(d, name), **arrays)

    def _payload(self):
        return dict(q_I=np.zeros((H, D), np.float32),
                    w=np.zeros(H, np.float32),
                    topk_indices=np.zeros(4, np.int32))

    def test_prefill_source_cache_len_is_pos_plus_one(self, tmp_path):
        for pos in (10, 11, 12):
            self._write(tmp_path, f"prefill_pos{pos:08d}_layer000.npz",
                        **self._payload())
        got = collect_samples(str(tmp_path), 0, "prefill", n_prompt=0)
        assert [c for c, _ in got] == [11, 12, 13]
        assert got[0][1]()["q_I"].shape == (H, D)

    def test_decode_source_offsets_by_prompt(self, tmp_path):
        for s in (0, 1, 2):
            self._write(tmp_path, f"step{s:06d}_layer000.npz", **self._payload())
        got = collect_samples(str(tmp_path), 0, "decode", n_prompt=100)
        assert [c for c, _ in got] == [101, 102, 103]

    def test_decode_stops_at_gap(self, tmp_path):
        for s in (0, 1, 3):        # 缺 step 2
            self._write(tmp_path, f"step{s:06d}_layer000.npz", **self._payload())
        got = collect_samples(str(tmp_path), 0, "decode", n_prompt=0)
        assert [c for c, _ in got] == [1, 2]

    def test_prefill_gap_tolerated(self, tmp_path):
        """stride 采样会产生不连续位置，枚举时不得截断（配对时再筛）。"""
        for pos in (0, 4096, 8192):
            self._write(tmp_path, f"prefill_pos{pos:08d}_layer000.npz",
                        **self._payload())
        got = collect_samples(str(tmp_path), 0, "prefill", n_prompt=0)
        assert [c for c, _ in got] == [1, 4097, 8193]
