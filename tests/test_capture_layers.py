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


class TestSaveEvery:

    def test_periodic_save(self, tmp_path):
        """save_every=2：每进入第 2、4… 个 step 时落盘一次。"""
        cap = make_capturer(tmp_path, layers=frozenset({0}))
        cap.save_every = 2
        q, w, topk = fake_step_tensors()
        for _ in range(5):  # steps 0..4
            cap.capture_decode_step(0, q, w, topk)

        # 进入 step 2 和 step 4 时各保存过一次 → 磁盘上已有 step 0-3 的 dump
        assert (tmp_path / "step000000_layer000.npz").exists()
        assert (tmp_path / "step000002_layer000.npz").exists()
        assert not (tmp_path / "step000004_layer000.npz").exists()

        cap.save()  # 结束时全量落盘
        assert (tmp_path / "step000004_layer000.npz").exists()

    def test_env_save_every(self, tmp_path, monkeypatch):
        set_indexer_state_capturer(None)
        isc._env_init_attempted = False
        monkeypatch.setenv("DSA_CAPTURE_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("DSA_CAPTURE_SAVE_EVERY", "64")
        cap = get_indexer_state_capturer()
        assert cap.save_every == 64
        set_indexer_state_capturer(None)
        isc._env_init_attempted = False

    def test_sigterm_handler_registered(self, tmp_path):
        """rank0 init 时注册 SIGTERM 落盘 handler（主线程内）。"""
        import signal
        prev = signal.getsignal(signal.SIGTERM)
        try:
            isc.init_indexer_state_capturer(output_dir=str(tmp_path))
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler) and handler is not prev
        finally:
            signal.signal(signal.SIGTERM, prev)
            set_indexer_state_capturer(None)


class TestPrefillCapture:
    """prefill q/w 采样：滚动尾窗 + stride，chunked prefill 下必须收敛到
    全局最后 tail 个位置（这批数据的 query 是真实的，见 patch 注释）。"""

    def make(self, tmp_path, tail=0, stride=0):
        cap = make_capturer(tmp_path, layers=frozenset({0}))
        cap.prefill_tail, cap.prefill_stride = tail, stride
        return cap

    def feed_chunk(self, cap, positions, layer=0):
        """模拟一个 prefill chunk：按 mask 选行后交给 capturer。"""
        pos = torch.tensor(positions)
        mask = cap.prefill_local_mask(pos)
        sel = mask.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return
        n = len(positions)
        q = torch.randn(n, H, D)
        w = torch.randn(n, H)
        topk = torch.arange(n * K, dtype=torch.int32).reshape(n, K)
        cap.capture_prefill_step(layer, pos[sel].tolist(),
                                 q[sel], w[sel], topk[sel])

    def captured(self, cap, layer=0):
        return sorted(p for (p, l) in cap._prefill_data if l == layer)

    def test_disabled_by_default(self, tmp_path):
        cap = self.make(tmp_path)
        assert not cap.capture_prefill_qw
        self.feed_chunk(cap, list(range(100)))
        assert cap._prefill_data == {}

    def test_tail_single_chunk(self, tmp_path):
        cap = self.make(tmp_path, tail=8)
        self.feed_chunk(cap, list(range(100)))
        assert self.captured(cap) == list(range(92, 100))

    def test_tail_across_chunks(self, tmp_path):
        """chunked prefill：全局尾窗，且最后一个 chunk 比 tail 短。"""
        cap = self.make(tmp_path, tail=8)
        self.feed_chunk(cap, list(range(0, 64)))
        self.feed_chunk(cap, list(range(64, 128)))
        self.feed_chunk(cap, list(range(128, 131)))   # 只有 3 个位置
        assert self.captured(cap) == list(range(123, 131))

    def test_tail_is_contiguous(self, tmp_path):
        """尾窗必须连续——warm 集/churn 分析依赖相邻位置。"""
        cap = self.make(tmp_path, tail=16)
        for start in range(0, 200, 32):
            self.feed_chunk(cap, list(range(start, min(start + 32, 200))))
        got = self.captured(cap)
        assert got == list(range(184, 200))
        assert all(b - a == 1 for a, b in zip(got, got[1:]))

    def test_stride_kept_across_trim(self, tmp_path):
        cap = self.make(tmp_path, tail=4, stride=50)
        for start in range(0, 200, 40):
            self.feed_chunk(cap, list(range(start, start + 40)))
        got = self.captured(cap)
        assert {0, 50, 100, 150} <= set(got), got     # stride 命中不被裁掉
        assert set(range(196, 200)) <= set(got), got  # 尾窗保留

    def test_saved_files_and_config(self, tmp_path):
        cap = self.make(tmp_path, tail=4)
        self.feed_chunk(cap, list(range(20)))
        cap.save()
        for p in range(16, 20):
            assert (tmp_path / f"prefill_pos{p:08d}_layer000.npz").exists()
        cfg = json.load(open(tmp_path / "capture_config.json"))
        assert cfg["prefill_tail"] == 4
        assert cfg["prefill_positions"] == [16, 17, 18, 19]

    def test_roundtrip_through_loader(self, tmp_path):
        from replay.loader import (list_prefill_positions, load_prefill_dump,
                                   iter_prefill)
        cap = self.make(tmp_path, tail=3)
        self.feed_chunk(cap, list(range(10)))
        cap.save()
        assert list_prefill_positions(str(tmp_path), 0) == [7, 8, 9]
        d = load_prefill_dump(str(tmp_path), 0, 9)
        assert d["q_I"].shape == (H, D)
        assert d["w"].shape == (H,)
        assert d["topk_indices"].shape == (K,)
        assert len(list(iter_prefill(str(tmp_path), 0))) == 3

    def test_layer_filter_applies(self, tmp_path):
        cap = self.make(tmp_path, tail=4)     # capture_layers = {0}
        self.feed_chunk(cap, list(range(10)), layer=3)
        assert cap._prefill_data == {}

    def test_env_init(self, tmp_path, monkeypatch):
        set_indexer_state_capturer(None)
        isc._env_init_attempted = False
        monkeypatch.setenv("DSA_CAPTURE_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("DSA_CAPTURE_PREFILL_TAIL", "64")
        monkeypatch.setenv("DSA_CAPTURE_PREFILL_STRIDE", "4096")
        cap = get_indexer_state_capturer()
        assert (cap.prefill_tail, cap.prefill_stride) == (64, 4096)
        assert cap.capture_prefill_qw
        set_indexer_state_capturer(None)
        isc._env_init_attempted = False


class TestVllmPerfDefaults:

    def test_with_deep_gemm(self):
        from capture.vllm_generate import resolve_perf_defaults
        chunk, util = resolve_perf_defaults(True)
        assert chunk is None and util == 0.85
        chunk, util = resolve_perf_defaults(True, truncated=True)
        assert chunk is None and util == 0.5

    @pytest.mark.parametrize("truncated", [False, True])
    def test_without_deep_gemm_fits_memory(self, truncated):
        """兜底配置的 einsum 峰值 + torch 预算必须 < 79GB (H800)。"""
        from capture.vllm_generate import resolve_perf_defaults
        chunk, util = resolve_perf_defaults(False, truncated=truncated)
        ctx, heads = 33792, 64
        einsum_gb = heads * chunk * ctx * 4 / 1e9
        torch_budget_gb = util * 79.1
        assert einsum_gb + torch_budget_gb < 75, (
            f"fallback config OOMs: einsum={einsum_gb:.1f}GB "
            f"+ budget={torch_budget_gb:.1f}GB")

    def test_truncated_budget_covers_weights_and_kv(self):
        """截断预算 0.45×79≈35.6GB 需容纳每卡权重 ~5GB + KV/激活。"""
        from capture.vllm_generate import resolve_perf_defaults
        _, util = resolve_perf_defaults(False, truncated=True)
        assert util * 79.1 > 5 + 10  # 权重 + 充裕的 KV/激活余量


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
