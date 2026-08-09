"""
test_vllm_patch.py — vLLM capture patch 的守护测试

- patch 文件本身的结构不变量（单文件目标、标记成对）
- patch 能干净应用到 tmp/vllm 的 HEAD 基线，且应用后语法合法
  （tmp/vllm 不存在时跳过——它是 gitignored 的本地检出）
"""

import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_FILE = os.path.join(
    REPO_ROOT, "capture", "patch_vllm", "indexer_capture.diff")
VLLM_DIR = os.path.join(REPO_ROOT, "tmp", "vllm")
TARGET = "vllm/model_executor/models/deepseek_v2.py"


def read_patch() -> str:
    with open(PATCH_FILE, "r", encoding="utf-8") as f:
        return f.read()


class TestPatchStructure:

    def test_single_target_file(self):
        patch = read_patch()
        targets = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
        assert targets == [TARGET]

    def test_markers_balanced(self):
        patch = read_patch()
        added = [ln for ln in patch.splitlines() if ln.startswith("+")]
        begin = sum("=== DSA-STAT PATCH" in ln for ln in added)
        end = sum("=== END DSA-STAT PATCH" in ln for ln in added)
        assert begin == end == 4

    def test_no_crlf(self):
        with open(PATCH_FILE, "rb") as f:
            assert b"\r\n" not in f.read(), "patch must be LF-only"

    def test_contains_key_hooks(self):
        patch = read_patch()
        assert "self, positions, q, k, _dsa_raw_weights" in patch
        assert "get_indexer_state_capturer" in patch
        assert "num_hidden_layers" in patch      # 截断跳层逻辑
        assert "capture_prefill_step" in patch   # prefill 真实 query 采样
        assert "prefill_local_mask" in patch


def extract_helper_source() -> str:
    """从 diff 的新增行里抽出 capture helper 的源码本体。

    这样即使没有 tmp/vllm 检出，也能对 patch 里"真正会跑的那段代码"
    做行为测试（而不是测一份复制品）。
    """
    lines = []
    started = False
    for ln in read_patch().splitlines():
        if not ln.startswith("+"):
            continue
        body = ln[1:]
        if body.startswith("def _dsa_stat_layer_id"):
            started = True
        if body.startswith("# === END DSA-STAT PATCH") and started:
            break
        if started:
            lines.append(body)
    assert lines, "helper source not found in patch"
    return "\n".join(lines)


class _FakeMeta:
    def __init__(self, num_decodes, num_decode_tokens):
        self.num_decodes = num_decodes
        self.num_decode_tokens = num_decode_tokens


class _FakeIndexer:
    prefix = "model.layers.3.self_attn.indexer"
    n_head, head_dim, topk_tokens = 4, 8, 6

    class k_cache:
        prefix = "model.layers.3.self_attn.indexer.k_cache"


class TestPatchHelperBehavior:
    """对 patch 内的 _dsa_stat_capture_indexer 做真实形状的行为测试：
    decode/prefill 分支的切片必须取到正确的行。"""

    def _run(self, capturer, num_decode_tokens, n_tokens, positions):
        import sys
        import types
        import torch

        idx = _FakeIndexer()
        H, Dh, K = idx.n_head, idx.head_dim, idx.topk_tokens

        fake_ctx = types.ModuleType("vllm.forward_context")
        fake_ctx.get_forward_context = lambda: types.SimpleNamespace(
            attn_metadata={
                idx.k_cache.prefix: _FakeMeta(
                    1 if num_decode_tokens else 0, num_decode_tokens)
            }
        )
        vllm_pkg = sys.modules.get("vllm") or types.ModuleType("vllm")
        saved = (sys.modules.get("vllm"),
                 sys.modules.get("vllm.forward_context"))
        sys.modules["vllm"] = vllm_pkg
        sys.modules["vllm.forward_context"] = fake_ctx
        try:
            ns = {"get_indexer_state_capturer": lambda: capturer}
            exec(extract_helper_source(), ns)
            # 行 i 的 q/k/w 全部填成 i，便于验证切片取对了行
            q = (torch.arange(n_tokens, dtype=torch.float32)
                 .repeat_interleave(H).unsqueeze(1).expand(-1, Dh).contiguous())
            k = (torch.arange(n_tokens, dtype=torch.float32)
                 .unsqueeze(1).expand(-1, Dh).contiguous())
            w = (torch.arange(n_tokens, dtype=torch.float32)
                 .unsqueeze(1).expand(-1, H).contiguous())
            topk = (torch.arange(n_tokens, dtype=torch.int32)
                    .unsqueeze(1).expand(-1, K).contiguous())
            ns["_dsa_stat_capture_indexer"](
                idx, torch.tensor(positions), q, k, w, topk)
        finally:
            for name, mod in zip(("vllm", "vllm.forward_context"), saved):
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_prefill_rows_and_positions(self, tmp_path):
        from capture.indexer_state_capturer import IndexerStateCapturer
        cap = IndexerStateCapturer(output_dir=str(tmp_path), prefill_tail=3)
        self._run(cap, num_decode_tokens=0, n_tokens=10,
                  positions=list(range(100, 110)))

        got = sorted(p for (p, _) in cap._prefill_data)
        assert got == [107, 108, 109]
        # 位置 109 是本 batch 的第 9 行 -> q/w/topk 全应等于 9
        d = cap._prefill_data[(109, 3)]
        assert d["q_I"].shape == (4, 8)
        assert (d["q_I"] == 9).all() and (d["w"] == 9).all()
        assert (d["topk_indices"] == 9).all()
        # k^I 全部 prefill 行都要进 buffer
        assert cap._k_buffers[3][0].shape[0] == 10

    def test_decode_rows_not_confused_with_prefill(self, tmp_path):
        """混合 batch（bs=1，约束 7）：第 0 行 decode、其余 prefill，
        两段的行偏移不得串位。"""
        from capture.indexer_state_capturer import IndexerStateCapturer
        cap = IndexerStateCapturer(output_dir=str(tmp_path), prefill_tail=2)
        self._run(cap, num_decode_tokens=1, n_tokens=5,
                  positions=[50, 51, 52, 53, 54])

        # decode 段取第 0 行，且落到 schema 形状 [H, d]
        dec = cap._step_data[(0, 3)]
        assert dec["q_I"].shape == (4, 8) and dec["w"].shape == (4,)
        assert (dec["q_I"] == 0).all() and (dec["topk_indices"] == 0).all()
        # prefill 段尾窗 = 位置 53,54 -> 第 3,4 行
        assert sorted(p for (p, _) in cap._prefill_data) == [53, 54]
        assert (cap._prefill_data[(54, 3)]["w"] == 4).all()
        assert (cap._prefill_data[(53, 3)]["w"] == 3).all()

    def test_prefill_disabled_still_captures_k(self, tmp_path):
        from capture.indexer_state_capturer import IndexerStateCapturer
        cap = IndexerStateCapturer(output_dir=str(tmp_path))  # tail=0
        self._run(cap, num_decode_tokens=0, n_tokens=6,
                  positions=list(range(6)))
        assert cap._prefill_data == {}
        assert cap._k_buffers[3][0].shape[0] == 6

    def test_layer_filter_skips_everything(self, tmp_path):
        from capture.indexer_state_capturer import IndexerStateCapturer
        cap = IndexerStateCapturer(output_dir=str(tmp_path), prefill_tail=3,
                                   capture_layers=frozenset({0, 1}))
        self._run(cap, num_decode_tokens=0, n_tokens=6,
                  positions=list(range(6)))
        assert cap._prefill_data == {} and cap._k_buffers == {}


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(VLLM_DIR, ".git")),
    reason="tmp/vllm checkout not available (gitignored)",
)
class TestPatchAppliesToBaseline:

    def test_apply_and_compile(self, tmp_path):
        blob = subprocess.run(
            ["git", "-C", VLLM_DIR, "show", f"HEAD:{TARGET}"],
            capture_output=True, check=True,
        ).stdout

        target_path = tmp_path / TARGET
        target_path.parent.mkdir(parents=True)
        target_path.write_bytes(blob)

        subprocess.run(
            ["git", "apply", "--check", PATCH_FILE],
            cwd=tmp_path, check=True,
        )
        subprocess.run(
            ["git", "apply", PATCH_FILE],
            cwd=tmp_path, check=True,
        )

        patched = target_path.read_text(encoding="utf-8")
        assert patched.count("DSA-STAT PATCH") == 8

        compile(patched, str(target_path), "exec")  # SyntaxError → fail
