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


def list_prefill_positions(output_dir: str, layer: int) -> list:
    """列出某层已采样的 prefill 绝对位置（升序）。"""
    prefix, suffix = "prefill_pos", f"_layer{layer:03d}.npz"
    positions = []
    for name in os.listdir(output_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            positions.append(int(name[len(prefix):-len(suffix)]))
    return sorted(positions)


def load_prefill_dump(output_dir: str, layer: int, pos: int) -> dict:
    """
    加载单个 prefill 位置的 dump。

    该位置的 indexer 对 k_I[:pos+1] 打分（因果），
    与 decode 步的唯一差别是"步"的含义变成绝对位置。

    Returns:
        dict with keys: q_I [H, d], w [H], topk_indices [k]
    """
    path = os.path.join(output_dir, f"prefill_pos{pos:08d}_layer{layer:03d}.npz")
    data = np.load(path)
    return {
        "q_I": torch.from_numpy(data["q_I"]).float(),
        "w": torch.from_numpy(data["w"]).float(),
        "topk_indices": torch.from_numpy(data["topk_indices"]).long(),
    }


def iter_prefill(output_dir: str, layer: int):
    """
    迭代某层全部 prefill 采样位置。

    Yields:
        (pos, dict) where dict has keys: q_I, w, topk_indices
    """
    for pos in list_prefill_positions(output_dir, layer):
        yield pos, load_prefill_dump(output_dir, layer, pos)


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
