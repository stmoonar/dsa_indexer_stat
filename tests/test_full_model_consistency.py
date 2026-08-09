"""
test_full_model_consistency.py — 全模型 run 的金丝雀

前提（已在 docs 中论证）：截断模型的 layer 0-4 前向不受后 56 层影响，
所以【prefill 段】的 k^I / q^I / w 应与全 61 层 run 逐位一致。
decode 段不在此列（截断模型生成的是乱码 token，轨迹本就不同）。

用途：16 卡的全模型 run 很贵，capture 出错重跑的代价远大于这个检查。
拿到全模型 dump 后立刻跑一次，能在分析开始前抓出 hook 接错、
层号错位、w 少乘 scale、prompt 不一致这几类致命错误。

用法：
    DSA_REF_RUN=runs/first5_32k_vllm_XXX \
    DSA_NEW_RUN=runs/full61_32k_XXX \
    pytest tests/test_full_model_consistency.py -v

两个环境变量都没设时整文件跳过（CI/日常开发不受影响）。

关于容差：不同 TP 度下 all-reduce 的求和顺序不同，逐位相等不成立。
DSA_TOL 给出相对容差，缺省 0（要求逐位）；跨 TP 比较时应先用
tools 侧的敏感性实验标定一个值，再传进来。
"""

import os
import json

import numpy as np
import pytest

REF = os.environ.get("DSA_REF_RUN", "")
NEW = os.environ.get("DSA_NEW_RUN", "")
TOL = float(os.environ.get("DSA_TOL", "0"))
LAYERS = [int(x) for x in os.environ.get("DSA_CANARY_LAYERS", "0,1,2,3,4")
          .split(",") if x]

pytestmark = pytest.mark.skipif(
    not (REF and NEW),
    reason="set DSA_REF_RUN and DSA_NEW_RUN to run the full-model canary",
)


def n_prefill(run: str) -> int:
    """prefill token 数 = k^I 总行数 − decode 步数。"""
    cfg = json.load(open(os.path.join(run, "capture_config.json")))
    layer = LAYERS[0]
    total = np.load(os.path.join(run, f"k_I_layer{layer:03d}.npy"),
                    mmap_mode="r").shape[0]
    return total - cfg.get("num_decode_steps", 0)


def compare(a: np.ndarray, b: np.ndarray, what: str):
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    if TOL == 0:
        n_diff = int((a != b).sum())
        assert n_diff == 0, (
            f"{what}: {n_diff}/{a.size} elements differ (bit-exact required; "
            f"max|Δ|={np.abs(a.astype(np.float64) - b).max():.3e}). "
            f"若两个 run 的 TP 不同，用 DSA_TOL 传相对容差。")
        return
    denom = np.maximum(np.abs(a), 1e-6)
    rel = np.abs(a.astype(np.float64) - b) / denom
    assert rel.max() <= TOL, (
        f"{what}: max relative diff {rel.max():.3e} > tol {TOL:.3e}")


def test_prompt_identical():
    """输入必须完全一致，否则后面所有比较都无意义。"""
    paths = [os.path.join(r, "prompt_token_ids.npy") for r in (REF, NEW)]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("prompt_token_ids.npy missing (run predates fingerprinting)")
    ref, new = (np.load(p) for p in paths)
    assert ref.shape == new.shape, f"prompt length {ref.shape} vs {new.shape}"
    assert (ref == new).all(), "prompt token ids differ between runs"


@pytest.mark.parametrize("layer", LAYERS)
def test_prefill_k_I_matches(layer):
    """k^I 是全部块几何统计的基础，最重要的一项。"""
    n = min(n_prefill(REF), n_prefill(NEW))
    assert n > 0
    ref = np.load(os.path.join(REF, f"k_I_layer{layer:03d}.npy"),
                  mmap_mode="r")[:n]
    new = np.load(os.path.join(NEW, f"k_I_layer{layer:03d}.npy"),
                  mmap_mode="r")[:n]
    compare(np.asarray(ref), np.asarray(new), f"k_I layer {layer}")


@pytest.mark.parametrize("layer", LAYERS)
def test_prefill_q_w_match(layer):
    """两个 run 共有的 prefill 采样位置上，q^I 和 w 也应一致。"""
    from replay.loader import list_prefill_positions, load_prefill_dump

    common = sorted(set(list_prefill_positions(REF, layer))
                    & set(list_prefill_positions(NEW, layer)))
    if not common:
        pytest.skip(f"no shared prefill positions for layer {layer}")
    for pos in common[:8]:
        a = load_prefill_dump(REF, layer, pos)
        b = load_prefill_dump(NEW, layer, pos)
        compare(a["q_I"].numpy(), b["q_I"].numpy(), f"q_I L{layer} @{pos}")
        compare(a["w"].numpy(), b["w"].numpy(), f"w L{layer} @{pos}")


@pytest.mark.parametrize("layer", LAYERS)
def test_prefill_topk_overlap(layer):
    """kernel top-k 的重合度。FP8 边界抖动允许少量差异，但必须极高。"""
    from replay.loader import list_prefill_positions, load_prefill_dump

    common = sorted(set(list_prefill_positions(REF, layer))
                    & set(list_prefill_positions(NEW, layer)))
    if not common:
        pytest.skip(f"no shared prefill positions for layer {layer}")
    for pos in common[:8]:
        a = set(load_prefill_dump(REF, layer, pos)["topk_indices"].tolist())
        b = set(load_prefill_dump(NEW, layer, pos)["topk_indices"].tolist())
        overlap = len(a & b) / max(len(a), 1)
        assert overlap > 0.99, (
            f"topk overlap L{layer} @{pos} = {overlap:.4f}; "
            f"低于此说明 capture 语义不一致，而不是量化抖动")
