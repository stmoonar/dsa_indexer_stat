"""
dump_schema.py — dump 格式的唯一权威定义

capture 和 replay 的唯一接口是磁盘格式。
每个字段的语义在此写死，不得在其他地方重新定义。
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os

SCHEMA_VERSION = "1.0.0"

FIELD_SPECS = {
    "k_I": "post-RoPE, dequantized fp32, shape [L, d_I=128], layer-major, 连续追加",
    "q_I": "post-RoPE, fp32, shape [H_I=64, d_I=128], 每步每层一组",
    "w": "fp32, shape [H_I=64], 每步每层一组, 保留原始符号(不做abs)",
    "topk_indices": "int32, shape [k=2048], 生产 kernel 输出的 top-k token 索引",
    "fp8_scale": "fp32, 标量或 per-block, FP8 反量化所用 scale (仅当原始为 FP8 时)",
}


@dataclass
class StepDump:
    """单步单层的 dump 数据。"""
    step: int
    layer: int
    q_I: "np.ndarray"       # [H_I, d_I], fp32, post-RoPE
    w: "np.ndarray"         # [H_I], fp32, 保留符号
    topk_indices: "np.ndarray"  # [k], int32, kernel 输出


@dataclass
class LayerKIBuffer:
    """单层的 k^I 连续 buffer，prefill 和 decode 追加写入。"""
    layer: int
    k_I: "np.ndarray"       # [current_len, d_I], fp32, post-RoPE, dequantized


@dataclass
class CaptureConfig:
    """capture 运行的配置快照。"""
    schema_version: str = SCHEMA_VERSION
    git_hash: str = ""
    model_name: str = "DeepSeek-V3.2"
    num_layers: int = 61
    num_indexer_heads: int = 64
    d_I: int = 128
    top_k: int = 2048
    context_length: int = 0
    num_decode_steps: int = 0
    block_size_fp8: str = "per-tensor"
    tp_world_size: int = 1
    rank_id: int = 0
    task_name: str = ""
    launch_args: dict = field(default_factory=dict)


def save_config(config: CaptureConfig, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "capture_config.json")
    with open(path, "w") as f:
        json.dump({
            "schema_version": config.schema_version,
            "field_specs": FIELD_SPECS,
            **{k: v for k, v in config.__dict__.items() if k != "launch_args"},
            "launch_args": config.launch_args,
        }, f, indent=2)


def load_config(output_dir: str) -> CaptureConfig:
    path = os.path.join(output_dir, "capture_config.json")
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: file={data.get('schema_version')}, "
            f"expected={SCHEMA_VERSION}"
        )
    cfg = CaptureConfig()
    for k, v in data.items():
        if k in ("schema_version", "field_specs"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
