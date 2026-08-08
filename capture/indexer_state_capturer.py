"""
indexer_state_capturer.py — 截获 DSA indexer 的 q^I, w, k^I, topk_indices

设计原则：
- 侵入面只在 Indexer.forward_cuda 内部（约束 10）
- 不解析 paged KV pool（约束 8），k^I 在 forward 内截获
- TP: q, k, w 都是 ReplicatedLinear 不切分，只在 rank 0 dump（7.6 节结论）
- dump 格式遵循 dump_schema.py
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class IndexerStateCapturer:
    """
    逐步逐层截获 indexer 状态。

    用法：在 Indexer.forward_cuda 中，query/key/weights 计算完成后调用 capture()。
    生成结束后调用 save() 写磁盘。
    """
    output_dir: str
    num_layers: int = 61
    num_heads: int = 64
    head_dim: int = 128
    top_k: int = 2048
    enabled: bool = True
    rank: int = 0

    # 运行时状态
    _step_data: dict = field(default_factory=dict)  # {(step, layer): {q, w, topk}}
    _k_buffers: dict = field(default_factory=dict)   # {layer: [k tensors]}
    _current_decode_step: int = field(default=0)
    _is_prefill: bool = field(default=True)

    def should_capture(self) -> bool:
        return self.enabled and self.rank == 0

    def capture_prefill_k(self, layer_id: int, key: torch.Tensor):
        """
        Prefill 阶段截获 k^I。key 是 post-RoPE, post-Hadamard, bf16。
        key shape: [num_tokens, head_dim]
        """
        if not self.should_capture():
            return
        if layer_id not in self._k_buffers:
            self._k_buffers[layer_id] = []
        self._k_buffers[layer_id].append(key.detach().float().cpu())

    def capture_decode_k(self, layer_id: int, key: torch.Tensor):
        """
        Decode 阶段截获新 token 的 k^I。
        key shape: [1, head_dim] 或 [batch, head_dim]（bs=1 时为 [1, head_dim]）
        """
        if not self.should_capture():
            return
        if layer_id not in self._k_buffers:
            self._k_buffers[layer_id] = []
        self._k_buffers[layer_id].append(key.detach().float().cpu())

    def capture_decode_step(
        self,
        layer_id: int,
        query: torch.Tensor,
        weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        """
        Decode 阶段截获 q^I, w, topk_indices。

        query shape: [1, H, d_I] → 存 [H, d_I]
        weights shape: [1, H] 或 [1, H, 1] → 存 [H]（raw w * n_heads^{-0.5}，不含 q_scale/softmax_scale）
        topk_indices shape: [1, top_k] → 存 [top_k]
        """
        if not self.should_capture():
            return

        step = self._current_decode_step

        q = query.detach().float().cpu()
        if q.dim() == 3:
            q = q.squeeze(0)  # [H, d_I]

        w = weights.detach().float().cpu()
        if w.dim() == 3:
            w = w.squeeze(-1)  # [B, H]
        if w.dim() == 2:
            w = w.squeeze(0)  # [H]

        topk = topk_indices.detach().cpu().to(torch.int32)
        if topk.dim() == 2:
            topk = topk.squeeze(0)  # [top_k]

        self._step_data[(step, layer_id)] = {
            "q_I": q.numpy(),
            "w": w.numpy(),
            "topk_indices": topk.numpy(),
        }

    def advance_decode_step(self):
        """每个 decode step 结束后调用（所有层处理完后）。"""
        if self.should_capture():
            self._current_decode_step += 1

    def set_prefill_done(self):
        """标记 prefill 完成。"""
        self._is_prefill = False
        self._current_decode_step = 0

    def save(self, config_dict: Optional[dict] = None):
        """
        保存所有截获数据到 output_dir。

        文件布局：
        - k_I_layer{NNN}.npy: [L, d_I] fp32
        - step{NNNNNN}_layer{NNN}.npz: {q_I, w, topk_indices}
        - capture_config.json: 配置快照
        """
        if not self.should_capture():
            return

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Saving indexer state to {self.output_dir}")

        # 保存 k^I buffers
        for layer_id, k_list in self._k_buffers.items():
            if k_list:
                k_concat = torch.cat(k_list, dim=0).numpy()
                path = os.path.join(self.output_dir, f"k_I_layer{layer_id:03d}.npy")
                np.save(path, k_concat)
                logger.info(
                    f"  Layer {layer_id}: k_I shape={k_concat.shape}, "
                    f"saved to {path}"
                )

        # 保存 step data
        n_steps = 0
        for (step, layer_id), data in sorted(self._step_data.items()):
            path = os.path.join(
                self.output_dir, f"step{step:06d}_layer{layer_id:03d}.npz"
            )
            np.savez(path, **data)
            n_steps += 1

        logger.info(f"  Saved {n_steps} step-layer dumps")

        # 保存 config
        if config_dict:
            import json
            config_path = os.path.join(self.output_dir, "capture_config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

        logger.info("Indexer state save complete")

    @property
    def num_decode_steps(self) -> int:
        return self._current_decode_step


_global_capturer: Optional[IndexerStateCapturer] = None


def get_indexer_state_capturer() -> Optional[IndexerStateCapturer]:
    return _global_capturer


def set_indexer_state_capturer(capturer: Optional[IndexerStateCapturer]):
    global _global_capturer
    _global_capturer = capturer


def init_indexer_state_capturer(
    output_dir: str,
    num_layers: int = 61,
    rank: int = 0,
    enabled: bool = True,
) -> IndexerStateCapturer:
    capturer = IndexerStateCapturer(
        output_dir=output_dir,
        num_layers=num_layers,
        rank=rank,
        enabled=enabled,
    )
    set_indexer_state_capturer(capturer)
    return capturer
