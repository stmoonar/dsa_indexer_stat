"""
loader.py — 读取 dump 数据，校验 schema 版本
"""

import os
import json
import numpy as np
import torch
from capture.dump_schema import SCHEMA_VERSION, CaptureConfig, load_config


def load_capture_run(output_dir: str) -> CaptureConfig:
    """加载并校验 capture 配置。"""
    return load_config(output_dir)


def load_k_I(output_dir: str, layer: int) -> torch.Tensor:
    """
    加载单层的 k^I buffer。

    Returns:
        k_I: [L, d_I], fp32, post-RoPE
    """
    path = os.path.join(output_dir, f"k_I_layer{layer:03d}.npy")
    arr = np.load(path)
    return torch.from_numpy(arr).float()


def load_step_dump(output_dir: str, layer: int, step: int) -> dict:
    """
    加载单步单层的 dump 数据。

    Returns:
        dict with keys: q_I [H, d], w [H], topk_indices [k]
    """
    path = os.path.join(output_dir, f"step{step:06d}_layer{layer:03d}.npz")
    data = np.load(path)
    return {
        "q_I": torch.from_numpy(data["q_I"]).float(),
        "w": torch.from_numpy(data["w"]).float(),
        "topk_indices": torch.from_numpy(data["topk_indices"]).long(),
    }


def iter_steps(output_dir: str, layer: int, start_step: int = 0):
    """
    迭代某层的所有步骤数据。

    Yields:
        (step, dict) where dict has keys: q_I, w, topk_indices
    """
    step = start_step
    while True:
        path = os.path.join(output_dir, f"step{step:06d}_layer{layer:03d}.npz")
        if not os.path.exists(path):
            break
        yield step, load_step_dump(output_dir, layer, step)
        step += 1
