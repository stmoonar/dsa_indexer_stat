"""
test_capture_layers.py — 层过滤 + 自动步进 + 环境变量惰性初始化 + prompt 挑选

覆盖本分支新增的 capture 侧功能：
- parse_layer_spec 的各种表达式
- capture_layers 过滤下只有目标层被记录
- capture_decode_step 的同层重现自动步进（无须显式 advance_decode_step）
- get_indexer_state_capturer 从 DSA_CAPTURE_* 环境变量惰性初始化
- save() 输出的 capture_config.json 含层过滤快照（约束 13）
- prepare_prompt 的 row_to_text / 样本挑选逻辑（不碰真实数据）
"""

import os
import json

import numpy as np
import torch
import pytest

import capture.indexer_state_capturer as isc
from capture.indexer_state_capturer import (
    IndexerStateCapturer,
    parse_layer_spec,
    get_indexer_state_capturer,
    set_indexer_state_capturer,
)
from capture.prepare_prompt import row_to_text, detect_text_column


H, D, K = 64, 128, 16


def make_capturer(tmp_path, layers=None):
    return IndexerStateCapturer(
        output_dir=str(tmp_path),
        num_layers=61,
        top_k=K,
        capture_layers=layers,
    )


def fake_step_tensors():
    q = torch.randn(1, H, D)
    w = torch.randn(1, H)
    topk = torch.arange(K, dtype=torch.int32).unsqueeze(0)
    return q, w, topk


class TestParseLayerSpec:

    def test_range(self):
        assert parse_layer_spec("0-4") == frozenset({0, 1, 2, 3, 4})

    def test_list(self):
        assert parse_layer_spec("0,2,60") == frozenset({0, 2, 60})

    def test_mixed(self):
        assert parse_layer_spec("0-2,60") == frozenset({0, 1, 2, 60})

    def test_all(self):
        assert parse_layer_spec("") is None
        assert parse_layer_spec("all") is None

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            parse_layer_spec("4-0")


class TestLayerFilter:

    def test_only_target_layers_captured(self, tmp_path):
        cap = make_capturer(tmp_path, layers=frozenset(range(5)))
        for layer in range(61):
            cap.capture_prefill_k(layer, torch.randn(8, D))
            q, w, topk = fake_step_tensors()
            cap.capture_decode_step(layer, q, w, topk)

        assert set(cap._k_buffers.keys()) == set(range(5))
        assert {l for (_, l) in cap._step_data} == set(range(5))

    def test_should_capture_layer_no_filter(self, tmp_path):
        cap = make_capturer(tmp_path, layers=None)
        assert cap.should_capture_layer(0)
        assert cap.should_capture_layer(60)

    def test_save_writes_layer_snapshot(self, tmp_path):
        cap = make_capturer(tmp_path, layers=frozenset(range(5)))
        cap.capture_prefill_k(0, torch.randn(8, D))
        q, w, topk = fake_step_tensors()
        cap.capture_decode_step(0, q, w, topk)
        cap.save()

        with open(tmp_path / "capture_config.json") as f:
            config = json.load(f)
        assert config["capture_layers"] == [0, 1, 2, 3, 4]
        assert config["num_decode_steps"] == 1
        assert (tmp_path / "k_I_layer000.npy").exists()
        assert (tmp_path / "step000000_layer000.npz").exists()


class TestAutoStepAdvance:

    def test_layer_revisit_advances_step(self, tmp_path):
        """同一层在 decode 中再次出现 → 新 step，数据不互相覆盖。"""
        cap = make_capturer(tmp_path, layers=frozenset(range(5)))
        n_steps = 3
        for _ in range(n_steps):
            for layer in range(5):
                q, w, topk = fake_step_tensors()
                cap.capture_decode_step(layer, q, w, topk)

        assert cap.num_decode_steps == n_steps
        assert len(cap._step_data) == n_steps * 5
        for step in range(n_steps):
            for layer in range(5):
                assert (step, layer) in cap._step_data

    def test_data_not_overwritten(self, tmp_path):
        cap = make_capturer(tmp_path, layers=frozenset({0}))
        q0, w, topk = fake_step_tensors()
        cap.capture_decode_step(0, q0, w, topk)
        q1 = torch.randn(1, H, D)
        cap.capture_decode_step(0, q1, w, topk)

        np.testing.assert_allclose(
            cap._step_data[(0, 0)]["q_I"], q0.squeeze(0).numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            cap._step_data[(1, 0)]["q_I"], q1.squeeze(0).numpy(), rtol=1e-6)

    def test_explicit_advance_still_compatible(self, tmp_path):
        """显式 advance_decode_step 与自动步进混用不产生空洞或覆盖。"""
        cap = make_capturer(tmp_path, layers=frozenset({0}))
        q, w, topk = fake_step_tensors()
        cap.capture_decode_step(0, q, w, topk)   # step 0
        cap.advance_decode_step()                 # → step 1
        cap.capture_decode_step(0, q, w, topk)   # step 1 (无碰撞，不再自增)
        assert sorted(s for (s, _) in cap._step_data) == [0, 1]


class TestEnvLazyInit:

    def _reset_global(self):
        set_indexer_state_capturer(None)
        isc._env_init_attempted = False

    def test_env_init(self, tmp_path, monkeypatch):
        self._reset_global()
        monkeypatch.setenv("DSA_CAPTURE_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("DSA_CAPTURE_LAYERS", "0-4")

        cap = get_indexer_state_capturer()
        assert cap is not None
        assert cap.output_dir == str(tmp_path)
        assert cap.capture_layers == frozenset(range(5))
        assert cap.rank == 0
        self._reset_global()

    def test_no_env_no_capturer(self, monkeypatch):
        self._reset_global()
        monkeypatch.delenv("DSA_CAPTURE_OUTPUT_DIR", raising=False)
        assert get_indexer_state_capturer() is None
        self._reset_global()


class TestPreparePrompt:

    def test_row_to_text_plain_string(self):
        assert row_to_text("hello") == "hello"

    def test_row_to_text_messages(self):
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        text = row_to_text(msgs)
        assert "user: question" in text
        assert "assistant: answer" in text

    def test_row_to_text_sharegpt_style(self):
        msgs = [{"from": "human", "value": "hi"}]
        assert "human: hi" in row_to_text(msgs)

    def test_row_to_text_multimodal_content(self):
        msgs = [{"role": "user", "content": [{"text": "part1"}, {"text": "part2"}]}]
        text = row_to_text(msgs)
        assert "part1" in text and "part2" in text

    def test_detect_text_column(self):
        assert detect_text_column(["id", "messages", "meta"]) == "messages"
        assert detect_text_column(["id", "text"]) == "text"
        with pytest.raises(ValueError):
            detect_text_column(["id", "blob"])
