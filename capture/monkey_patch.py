"""
monkey_patch.py — 对 SGLang Indexer.forward_cuda 的非侵入式 monkey-patch

原理：替换 Indexer.forward_cuda 为包装版本，在 query/key 计算后、
FP8 量化前截获 bf16 张量。

使用方式：在 SGLang 启动后、推理前调用 patch_indexer()。
约束 10：不改调度器、不改内存池，只 hook 模型文件层的 indexer forward。

SGLang 版本：v0.5.13.post1
"""

import os
import sys
import logging
import functools
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def patch_indexer(output_dir: str, num_layers: int = 61,
                  capture_layers: Optional[str] = None):
    """
    Monkey-patch SGLang 的 Indexer，截获 decode 阶段的 q^I, w, k^I, topk_indices。

    必须在 model 加载之后、推理之前调用。

    约束 7: 需搭配 --disable-cuda-graph 使用。
    约束 8: 不解析 paged KV pool，直接在 forward 内截获 pre-quantization 的 key。

    capture_layers: 层过滤表达式（如 "0-4" 只采前 5 层）；
                    None 时读环境变量 DSA_CAPTURE_LAYERS，均未设置则采全部层。
    """
    from capture.indexer_state_capturer import (
        IndexerStateCapturer,
        set_indexer_state_capturer,
        parse_layer_spec,
    )

    if capture_layers is None:
        capture_layers = os.environ.get("DSA_CAPTURE_LAYERS", "")

    capturer = IndexerStateCapturer(
        output_dir=output_dir,
        num_layers=num_layers,
        rank=_get_rank(),
        enabled=True,
        capture_layers=parse_layer_spec(capture_layers),
    )
    set_indexer_state_capturer(capturer)

    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

    original_forward = Indexer.forward_cuda

    @functools.wraps(original_forward)
    def patched_forward(self, x, q_lora, positions, forward_batch, layer_id,
                        return_indices=True):
        # 仅在需要 capture 且不在 CUDA graph 中时截获；
        # 层被过滤时直接走原始 forward，避免额外的 q/k/w 重复计算
        if not capturer.should_capture() or not capturer.should_capture_layer(layer_id):
            return original_forward(
                self, x, q_lora, positions, forward_batch, layer_id, return_indices
            )

        # 先计算 q, k (bf16, post-RoPE, post-Hadamard) 和 raw weights
        # 复用 Indexer 的内部方法
        x_meta = x[0] if isinstance(x, tuple) else x
        is_decode = forward_batch.forward_mode.is_decode_or_idle()

        # 截获 weights（raw, 不含 q_scale/softmax_scale）
        if isinstance(x, tuple) and len(x) == 3:
            x_for_gate = x
        elif isinstance(x, tuple):
            x_q, x_s = x[0], x[1]
            if (x_s is not None and x_q.dim() == 2 and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]):
                m, n = x_q.shape
                ng = x_s.shape[1]
                if ng > 0 and n % ng == 0:
                    group = n // ng
                    x_for_gate = (
                        x_q.to(torch.float32)
                        .view(m, ng, group)
                        .mul_(x_s.to(torch.float32).unsqueeze(-1))
                        .view(m, n)
                        .to(torch.bfloat16)
                    )
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x_q.to(torch.bfloat16)
        else:
            x_for_gate = x

        # 计算 raw weights (w * n_heads^{-0.5}), 不含 q_scale/softmax_scale
        raw_weights = self._project_and_scale_head_gates(x_for_gate)

        # 计算 q, k (bf16)
        query, key = self._get_q_k_bf16(
            q_lora, x, positions,
            enable_dual_stream=False,
            forward_batch=forward_batch,
        )

        # 截获 k^I
        if is_decode:
            capturer.capture_decode_k(layer_id, key)
        else:
            capturer.capture_prefill_k(layer_id, key)

        # 调用原始 forward 获取 topk_result
        topk_result = original_forward(
            self, x, q_lora, positions, forward_batch, layer_id, return_indices
        )

        # 截获 decode step 的 q, w, topk
        if is_decode and topk_result is not None:
            capturer.capture_decode_step(
                layer_id=layer_id,
                query=query,
                weights=raw_weights,
                topk_indices=topk_result,
            )

        return topk_result

    Indexer.forward_cuda = patched_forward
    logger.info(
        f"Indexer.forward_cuda patched for state capture. "
        f"Output: {output_dir}, rank: {capturer.rank}"
    )
    return capturer


def _get_rank() -> int:
    """获取当前 TP rank。"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0
