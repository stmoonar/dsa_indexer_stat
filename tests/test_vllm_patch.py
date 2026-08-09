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
        assert "_dsa_stat_capture_indexer(self, q, k, _dsa_raw_weights" in patch
        assert "get_indexer_state_capturer" in patch
        assert "num_hidden_layers" in patch  # 截断跳层逻辑


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
