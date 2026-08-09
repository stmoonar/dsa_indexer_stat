"""
test_slim_run.py — 下载瘦身工具

要守住的性质：
- 各档位保留/丢弃的文件类别正确（丢错了要重新下载几百 MB）
- fp32->fp16 对 k^I 无损（源头是 bf16，尾数更短），且 loader 读得回来
- 多序列 run 的目录结构被保留（seqNN 子目录）
"""

import os
import json
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_run(root, seqs=("",), n_tok=256, layers=2):
    """造一个迷你 run：k_I / c_latent / k_pe / prefill / step 各若干。"""
    os.makedirs(root, exist_ok=True)
    json.dump({"num_layers": layers, "capture_layers": "all",
               "num_decode_steps": 2},
              open(os.path.join(root, "capture_config.json"), "w"))
    rs = np.random.RandomState(0)
    for s in seqs:
        d = os.path.join(root, s) if s else root
        os.makedirs(d, exist_ok=True)
        for l in range(layers):
            # k^I 是 LayerNorm 输出，元素量级 ~1；先过一遍 fp16 模拟 bf16 源头
            k = rs.randn(n_tok, 128).astype(np.float16).astype(np.float32)
            np.save(os.path.join(d, f"k_I_layer{l:03d}.npy"), k)
            np.save(os.path.join(d, f"c_latent_layer{l:03d}.npy"),
                    rs.randn(n_tok, 512).astype(np.float16))
            np.save(os.path.join(d, f"k_pe_layer{l:03d}.npy"),
                    rs.randn(n_tok, 64).astype(np.float16))
            for pos in (100, 101):
                np.savez(os.path.join(d, f"prefill_pos{pos:08d}_layer{l:03d}.npz"),
                         q_I=rs.randn(64, 128).astype(np.float32),
                         w=rs.randn(64).astype(np.float32),
                         topk_indices=np.arange(8, dtype=np.int32))
            for st in (0, 1):
                np.savez(os.path.join(d, f"step{st:06d}_layer{l:03d}.npz"),
                         q_I=rs.randn(64, 128).astype(np.float32),
                         w=rs.randn(64).astype(np.float32),
                         topk_indices=np.arange(8, dtype=np.int32))
    return root


def run_slim(run_dir, out, *extra):
    cmd = [sys.executable, "-m", "analysis.slim_run", run_dir,
           "--out", out, "--no-zip", *extra]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return r.stdout.decode("utf-8", "replace")


def names(d):
    return set(os.listdir(d))


class TestProfiles:

    def test_light_drops_all_big_tensors(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "light")
        got = names(out)
        assert not any(f.startswith(("k_I", "c_latent", "k_pe")) for f in got)
        assert sum(f.startswith("prefill_pos") for f in got) == 4
        assert "capture_config.json" in got

    def test_geometry_keeps_only_k_I(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry")
        got = names(out)
        assert sum(f.startswith("k_I_layer") for f in got) == 2
        assert not any(f.startswith(("c_latent", "k_pe")) for f in got)

    def test_full_keeps_everything_big(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "full")
        got = names(out)
        for kind in ("k_I_layer", "c_latent_layer", "k_pe_layer"):
            assert sum(f.startswith(kind) for f in got) == 2, kind

    def test_decode_dropped_by_default(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry")
        assert not any(f.startswith("step") for f in names(out))

    def test_keep_decode_flag(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry", "--keep-decode")
        assert sum(f.startswith("step") for f in names(out)) == 4

    def test_layer_subset(self, tmp_path):
        run = make_run(str(tmp_path / "run"), layers=3)
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry", "--layers", "0,2")
        got = {f for f in names(out) if f.startswith("k_I_layer")}
        assert got == {"k_I_layer000.npy", "k_I_layer002.npy"}
        # prefill 不受 --layers 影响（体积小、且是真实 query）
        assert sum(f.startswith("prefill_pos") for f in names(out)) == 6


class TestDowncast:

    def test_k_I_downcast_is_lossless_for_bf16_source(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry")
        a = np.load(os.path.join(run, "k_I_layer000.npy"))
        b = np.load(os.path.join(out, "k_I_layer000.npy"))
        assert b.dtype == np.float16
        assert np.array_equal(a, b.astype(np.float32)), "downcast lost data"
        assert os.path.getsize(os.path.join(out, "k_I_layer000.npy")) < \
            os.path.getsize(os.path.join(run, "k_I_layer000.npy"))

    def test_no_downcast_flag_preserves_fp32(self, tmp_path):
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry", "--no-downcast")
        assert np.load(os.path.join(out, "k_I_layer000.npy")).dtype == np.float32

    def test_loader_reads_fp16_back_as_fp32(self, tmp_path):
        from replay.loader import load_k_I
        run = make_run(str(tmp_path / "run"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry")
        t = load_k_I(out, 0)
        assert t.dtype.__str__() == "torch.float32"
        assert t.shape == (256, 128)


class TestMultiSequence:

    def test_seq_subdirs_preserved(self, tmp_path):
        run = make_run(str(tmp_path / "run"), seqs=("seq00", "seq01", "seq02"))
        out = str(tmp_path / "out")
        run_slim(run, out, "--profile", "geometry")
        for s in ("seq00", "seq01", "seq02"):
            d = os.path.join(out, s)
            assert os.path.isdir(d), s
            assert sum(f.startswith("k_I_layer") for f in names(d)) == 2
            assert sum(f.startswith("prefill_pos") for f in names(d)) == 4
