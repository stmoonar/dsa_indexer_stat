"""
test_calibrate_tolerance.py — 金丝雀容差标定的守护测试

容差的唯一用处是把"TP 求和顺序"（O(1e-3)，bf16 尾数）和"语义错误"
（O(1)：hook 接错、层号错位、w 少乘 scale）分开。所以两个方向都要钉：
阈值不能紧到把噪声判成错误，也不能松到把 O(1) 的错误放过去。
"""

import numpy as np
import pytest

from analysis.calibrate_tolerance import suggest_tol, rel_stats, TOL_CEILING


class TestSuggestTol:

    def test_bit_exact_stays_strictest(self):
        assert suggest_tol(0.0) == 0.0

    def test_ten_x_margin(self):
        assert suggest_tol(1.2e-3) == pytest.approx(1.2e-2)

    def test_capped_below_semantic_errors(self):
        """封顶后仍须与 O(1) 的语义错误留出至少一个数量级。"""
        assert suggest_tol(1.0) == TOL_CEILING
        assert TOL_CEILING <= 0.1

    def test_monotone(self):
        vals = [suggest_tol(v) for v in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_covers_measured_noise(self):
        """标定值必须大于实测最坏偏差，否则金丝雀会自我否定。"""
        for v in (1e-9, 3.7e-4, 2e-3, 4e-2):
            assert suggest_tol(v) > v


class TestRelStats:

    def test_identical_is_zero(self):
        a = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
        st = rel_stats(a, a.copy())
        assert st["max"] == 0.0 and st["exact_frac"] == 1.0

    def test_semantic_scale_error_is_order_one(self):
        """w 少乘一个 scale 这类错误必须落在 O(1)，远超任何标定容差。"""
        a = np.abs(np.random.default_rng(1).standard_normal(256)) + 1.0
        st = rel_stats(a.astype(np.float32), (a * 8).astype(np.float32))
        assert st["p50"] > 0.5 > suggest_tol(2e-3)

    def test_small_values_do_not_blow_up(self):
        """分母 floor 存在的理由：近零元素的相对偏差不该主导 max。"""
        a = np.array([1.0, 1e-12, 2.0], np.float32)
        b = np.array([1.0, 9e-12, 2.0], np.float32)
        assert rel_stats(a, b)["max"] < 1e-6
