"""
test_prompt_prep.py — 多序列 / 128K 拼接的 prompt 准备，以及容差标定统计

多序列是"序列间方差"的前提，拼接是 128K 的前提（单条 agentic trace
不够长）。两者都会直接决定烧机时的数据可用性，故用假 tokenizer 做
确定性测试，不依赖真实模型。
"""

import numpy as np
import pytest

from capture.prepare_prompt import fit_to_budget, build_sequences
from analysis.calibrate_tolerance import rel_stats


class FakeTok:
    """1 字符 = 1 token，encode/decode 精确互逆，便于断言 token 数。"""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def cands(*lengths):
    """构造候选：(char_len, path, row_idx, text)，文本用不同字母区分来源。"""
    out = []
    for i, n in enumerate(lengths):
        ch = chr(ord("a") + i)
        out.append((n, f"/data/part{i}.parquet", i, ch * n))
    return out


@pytest.fixture
def tok():
    return FakeTok()


class TestFitToBudget:

    def test_no_truncation_when_short(self, tok):
        text, n = fit_to_budget(tok, "abcd", 10)
        assert text == "abcd" and n == 4

    def test_truncates_to_budget(self, tok):
        text, n = fit_to_budget(tok, "a" * 100, 30)
        assert n == 30 and len(text) == 30


class TestBuildSequences:

    def test_single_sample_long_enough(self, tok):
        seqs = build_sequences(cands(1000), tok, budget=900,
                               n_samples=1, allow_concat=False)
        assert len(seqs) == 1
        assert seqs[0]["n_tokens"] == 900        # 截断到 budget
        assert seqs[0]["truncated"] and seqs[0]["concat_parts"] == 1

    def test_multiple_sequences_are_distinct_sources(self, tok):
        seqs = build_sequences(cands(1000, 990, 980), tok, budget=900,
                               n_samples=3, allow_concat=False)
        assert len(seqs) == 3
        rows = [s["row_index"] for s in seqs]
        assert len(set(rows)) == 3, f"sequences reused a source row: {rows}"
        for s in seqs:
            assert s["n_tokens"] == 900

    def test_stops_when_candidates_exhausted(self, tok):
        seqs = build_sequences(cands(1000, 990), tok, budget=900,
                               n_samples=5, allow_concat=False)
        assert len(seqs) == 2

    def test_concat_fills_budget(self, tok):
        """128K 场景：单条只有 ~1/4 budget，必须拼接。"""
        seqs = build_sequences(cands(*([1000] * 8)), tok, budget=3600,
                               n_samples=1, allow_concat=True)
        assert len(seqs) == 1
        assert seqs[0]["n_tokens"] == 3600
        assert seqs[0]["concat_parts"] >= 4
        assert "," in seqs[0]["source_file"]      # 记录了全部来源

    def test_concat_sequences_do_not_share_parts(self, tok):
        seqs = build_sequences(cands(*([1000] * 12)), tok, budget=2900,
                               n_samples=2, allow_concat=True)
        assert len(seqs) == 2
        parts = [set(s["source_file"].split(",")) for s in seqs]
        assert not (parts[0] & parts[1]), "两条序列复用了同一段素材"

    def test_underfilled_flagged_without_concat(self, tok):
        seqs = build_sequences(cands(100), tok, budget=1000,
                               n_samples=1, allow_concat=False)
        assert seqs[0]["underfilled"] is True
        assert seqs[0]["n_tokens"] == 100

    def test_no_underfilled_flag_when_full(self, tok):
        seqs = build_sequences(cands(2000), tok, budget=1000,
                               n_samples=1, allow_concat=False)
        assert "underfilled" not in seqs[0]


class TestRelStats:

    def test_identical_is_zero(self):
        a = np.random.RandomState(0).randn(100, 8).astype(np.float32)
        st = rel_stats(a, a.copy())
        assert st["max"] == 0.0 and st["exact_frac"] == 1.0

    def test_detects_relative_scale(self):
        a = np.full((50, 4), 2.0, dtype=np.float32)
        b = a * 1.001
        st = rel_stats(a, b)
        assert 5e-4 < st["max"] < 2e-3
        assert st["exact_frac"] == 0.0

    def test_small_values_do_not_blow_up(self):
        """接近 0 的元素不得把相对偏差放大成假警报。"""
        a = np.concatenate([np.full(90, 5.0), np.full(10, 1e-9)]).astype(np.float32)
        b = a.copy()
        b[90:] = 2e-9          # 小值翻倍，但绝对差可忽略
        assert rel_stats(a, b)["max"] < 1e-3
