"""
test_null_control.py — analysis/null_control.py 的守护测试

零假设对照只有在"随机时确实塌回基线、有结构时确实抬起来"两侧都钉死之后
才有说服力。这里用构造数据把四件事钉死：

- 正对照：两层 k^I/q/w 完全相同 → 层间 overlap 必须恰好 1.0
- 负对照：uniform / gaussian 的层间 overlap 必须贴着解析基线 k/L
          （贴不上就说明 pipeline 自己制造了相关性，那一切结论作废）
- 步间配对只在【真正相邻】的采样位置之间发生（cur-prev==1），
  不连续的位置不得产生步间样本
- shuffle-q 必须真的打乱了 (q,w) 与位置的配对
"""

import os
import json
import argparse

import numpy as np
import pytest

from analysis.null_control import analyze, common_positions, MODES

H, D, L, K = 4, 8, 1024, 128
START, NPOS = 512, 24
POSITIONS = list(range(START, START + NPOS))          # 连续段
LAYERS = [0, 1, 2]


def _args(**kw):
    base = dict(top_k=K, min_cache_mult=2.0, max_positions=0, seed=0,
                device="cpu")
    base.update(kw)
    return argparse.Namespace(**base)


def _write_run(root, shared: bool, positions=POSITIONS):
    """写一个最小 run 目录。

    shared=True  → 所有层用同一份 k^I 和同一份 (q, w)（正对照，overlap≡1）
    shared=False → 每层独立随机（层间无对应关系）

    w 用有正有负的随机值而非全 1：真实 w 约 58% 为负（约束 2），且全常数
    会让 gaussian 模式的 w 标准差为 0，退化成病态输入。
    """
    rng = np.random.default_rng(1234)
    k0 = rng.standard_normal((L, D)).astype(np.float32)
    q0 = {p: rng.standard_normal((H, D)).astype(np.float32) for p in positions}
    w0 = {p: rng.standard_normal(H).astype(np.float32) for p in positions}
    for layer in LAYERS:
        k = k0 if shared else rng.standard_normal((L, D)).astype(np.float32)
        np.save(os.path.join(root, f"k_I_layer{layer:03d}.npy"), k)
        for p in positions:
            q = q0[p] if shared else rng.standard_normal((H, D)).astype(np.float32)
            w = w0[p] if shared else rng.standard_normal(H).astype(np.float32)
            np.savez(os.path.join(root, f"prefill_pos{p:08d}_layer{layer:03d}.npz"),
                     q_I=q, w=w, topk_indices=np.arange(K, dtype=np.int64))
    json.dump({"num_layers": len(LAYERS), "capture_layers": LAYERS},
              open(os.path.join(root, "capture_config.json"), "w"))
    return root


def _mk(tmp_path, name, shared):
    d = tmp_path / name
    d.mkdir()
    return _write_run(str(d), shared)


@pytest.fixture
def shared_run(tmp_path):
    return _mk(tmp_path, "shared", True)


@pytest.fixture
def indep_run(tmp_path):
    return _mk(tmp_path, "indep", False)


def _baseline(positions):
    """解析基线 k/L 在这批位置上的中位数。"""
    return float(np.median([min(K, p + 1) / (p + 1) for p in positions]))


def test_positive_control_identical_layers(shared_run):
    """完全相同的两层：real 模式下层间 overlap 必须恰好 1.0。

    这一条不过，说明打分或集合运算本身有问题，其余数字都不用看。
    """
    r = analyze(shared_run, LAYERS, POSITIONS, "real", _args(), "cpu")
    assert r["cross_layer_adjacent"]["overlap"]["p50"] == pytest.approx(1.0)
    assert r["step_adjacent"]["hit_rate"]["n"] > 0


@pytest.mark.parametrize("mode", ["uniform", "gaussian"])
def test_null_modes_collapse_to_baseline(indep_run, mode):
    """随机模式的层间 overlap 必须贴着 k/L。

    容差 0.05 绝对值：K=128、每对 ~24 个位置、2 个相邻对，
    超几何抽样下均值的标准差约 0.005，0.05 是 10 sigma 的余量。
    """
    r = analyze(indep_run, LAYERS, POSITIONS, mode, _args(), "cpu")
    base = _baseline(POSITIONS)
    got = r["cross_layer_adjacent"]["overlap"]["p50"]
    assert abs(got - base) < 0.05, f"{mode}: overlap {got:.4f} vs 基线 {base:.4f}"
    assert abs(r["cross_layer_adjacent"]["lift"]["p50"]) < 0.08


@pytest.mark.parametrize("mode", ["uniform", "gaussian"])
def test_null_modes_step_hit_collapses(indep_run, mode):
    """随机模式的步间命中率也必须贴着基线。

    churn 这一侧在主分析里【没有】任何基线，这条测试是它唯一的守护。
    hit = 1 - churn/2，独立采样下期望 = k/L。
    """
    r = analyze(indep_run, LAYERS, POSITIONS, mode, _args(), "cpu")
    base = _baseline(POSITIONS)
    got = r["step_adjacent"]["hit_rate"]["p50"]
    assert abs(got - base) < 0.05, f"{mode}: hit {got:.4f} vs 基线 {base:.4f}"


def test_step_pairs_require_consecutive_positions(indep_run):
    """不连续的采样位置不得产生步间样本。

    prefill 采样是连续段（prefill_run），孤立位置在配对时必须被丢弃 ——
    否则会把"跨越几千 token 的两次查询"当成相邻步，churn 被系统性高估。
    """
    sparse = list(range(START, START + 40, 4))         # 步长 4，无相邻对
    _write_run(indep_run, False, positions=sparse)
    r = analyze(indep_run, LAYERS, sparse, "real", _args(), "cpu")
    assert r["step_adjacent"]["hit_rate"]["n"] == 0
    assert r["n_adjacent_step_pairs"] == 0
    # 层间不受影响，仍应有样本
    assert r["cross_layer_adjacent"]["overlap"]["n"] > 0


def _write_smooth_run(root, positions=POSITIONS):
    """构造【有时间结构】的 run：q 沿位置缓慢旋转。

    q_p = cos(θ_p)·u + sin(θ_p)·v，θ 在整段上转 90°。
    于是相邻位置的 q 几乎同向（top-k 高度重合），远距离位置差异很大。
    这正是 shuffle-q 应当摧毁的结构 —— 若它摧毁不了，这个零假设就是空壳。
    """
    rng = np.random.default_rng(7)
    k = rng.standard_normal((L, D)).astype(np.float32)
    u = rng.standard_normal((H, D)).astype(np.float32)
    v = rng.standard_normal((H, D)).astype(np.float32)
    w = rng.standard_normal(H).astype(np.float32)
    n = len(positions)
    for layer in LAYERS:
        np.save(os.path.join(root, f"k_I_layer{layer:03d}.npy"), k)
        for i, p in enumerate(positions):
            th = (i / max(n - 1, 1)) * (np.pi / 2)
            q = (np.cos(th) * u + np.sin(th) * v).astype(np.float32)
            np.savez(os.path.join(root, f"prefill_pos{p:08d}_layer{layer:03d}.npz"),
                     q_I=q, w=w, topk_indices=np.arange(K, dtype=np.int64))
    json.dump({"num_layers": len(LAYERS), "capture_layers": LAYERS},
              open(os.path.join(root, "capture_config.json"), "w"))
    return root


@pytest.fixture
def smooth_run(tmp_path):
    d = tmp_path / "smooth"
    d.mkdir()
    return _write_smooth_run(str(d))


def test_shuffle_q_destroys_temporal_structure(smooth_run):
    """shuffle-q 必须真的摧毁时间连续性。

    在 q 沿位置缓慢旋转的构造上，real 的步间命中应当很高；把 (q,w) 与位置的
    配对打乱之后，"相邻两步"变成两个随机远隔的查询，命中率必须显著下降。
    若两者接近，说明 remap 没生效，这个零假设就是个空壳，
    拿它去证明"真实数据有时间结构"也就没有意义。
    """
    a = analyze(smooth_run, LAYERS, POSITIONS, "real", _args(), "cpu")
    b = analyze(smooth_run, LAYERS, POSITIONS, "shuffle-q", _args(), "cpu")
    real_hit = a["step_adjacent"]["hit_rate"]["p50"]
    null_hit = b["step_adjacent"]["hit_rate"]["p50"]
    assert real_hit > 0.6, f"构造的时间结构不够强: real hit={real_hit:.3f}"
    assert real_hit - null_hit > 0.10, \
        f"shuffle-q 没起作用: real={real_hit:.3f} null={null_hit:.3f}"


def test_output_schema(indep_run):
    """每种模式的输出结构一致，便于横向比较。"""
    for m in MODES:
        r = analyze(indep_run, LAYERS, POSITIONS, m, _args(), "cpu")
        assert r["mode"] == m and r["breaks"]
        for key in ("random_baseline", "cross_layer_adjacent", "step_adjacent",
                    "by_pair_overlap", "by_layer_hit", "cache_len_range"):
            assert key in r, f"{m} 缺 {key}"
        assert set(r["by_pair_overlap"]) == {"0-1", "1-2"}


def test_common_positions(indep_run):
    assert common_positions(indep_run, LAYERS) == POSITIONS
