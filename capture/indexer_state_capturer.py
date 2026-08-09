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
from typing import Optional, FrozenSet
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_layer_spec(spec: str) -> Optional[FrozenSet[int]]:
    """
    解析层过滤表达式，返回层 id 集合；None 表示不过滤（采全部层）。

    支持格式（可混用，逗号分隔）："0-4"、"0,1,2"、"0-2,60"。
    空串 / "all" → None。
    """
    spec = spec.strip()
    if not spec or spec.lower() == "all":
        return None
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"Invalid layer range: {part}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(part))
    if not layers:
        return None
    return frozenset(layers)


@dataclass
class IndexerStateCapturer:
    """
    逐步逐层截获 indexer 状态。

    用法：在 Indexer.forward_cuda 中，query/key/weights 计算完成后调用 capture()。
    生成结束后调用 save() 写磁盘。

    capture_layers: 只截获这些层（None = 全部 61 层）。
    """
    output_dir: str
    num_layers: int = 61
    num_heads: int = 64
    head_dim: int = 128
    top_k: int = 2048
    enabled: bool = True
    rank: int = 0
    capture_layers: Optional[FrozenSet[int]] = None
    save_every: int = 0  # 每 N 个 decode step 落盘一次（0 = 只在结束时落盘）

    # 运行时状态
    _step_data: dict = field(default_factory=dict)  # {(step, layer): {q, w, topk}}
    _k_buffers: dict = field(default_factory=dict)   # {layer: [k tensors]}
    _current_decode_step: int = field(default=0)
    _is_prefill: bool = field(default=True)

    def should_capture(self) -> bool:
        return self.enabled and self.rank == 0

    def should_capture_layer(self, layer_id: int) -> bool:
        return self.capture_layers is None or layer_id in self.capture_layers

    def capture_prefill_k(self, layer_id: int, key: torch.Tensor):
        """
        Prefill 阶段截获 k^I。key 是 post-RoPE, post-Hadamard, bf16。
        key shape: [num_tokens, head_dim]
        """
        if not self.should_capture() or not self.should_capture_layer(layer_id):
            return
        if layer_id not in self._k_buffers:
            self._k_buffers[layer_id] = []
        self._k_buffers[layer_id].append(key.detach().float().cpu())

    def capture_decode_k(self, layer_id: int, key: torch.Tensor):
        """
        Decode 阶段截获新 token 的 k^I。
        key shape: [1, head_dim] 或 [batch, head_dim]（bs=1 时为 [1, head_dim]）
        """
        if not self.should_capture() or not self.should_capture_layer(layer_id):
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

        步进：同一层在 decode 中再次出现即视为进入新 step
        （集成方无须显式调用 advance_decode_step）。
        """
        if not self.should_capture() or not self.should_capture_layer(layer_id):
            return

        if (self._current_decode_step, layer_id) in self._step_data:
            self._current_decode_step += 1
            if (self.save_every > 0
                    and self._current_decode_step % self.save_every == 0):
                self.save()

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

        # 保存 config（约束 13：run 输出目录必须含配置快照）
        import json
        merged = {
            "num_layers": self.num_layers,
            "capture_layers": (
                sorted(self.capture_layers) if self.capture_layers else "all"
            ),
            "num_decode_steps": self.num_decode_steps,
            "rank_id": self.rank,
        }
        if config_dict:
            merged.update(config_dict)
        config_path = os.path.join(self.output_dir, "capture_config.json")
        with open(config_path, "w") as f:
            json.dump(merged, f, indent=2)

        logger.info("Indexer state save complete")

    @property
    def num_decode_steps(self) -> int:
        if not self._step_data:
            return 0
        return max(s for (s, _) in self._step_data) + 1


_global_capturer: Optional[IndexerStateCapturer] = None
_env_init_attempted: bool = False


def get_indexer_state_capturer() -> Optional[IndexerStateCapturer]:
    """
    获取全局 capturer。

    若尚未初始化且设置了 DSA_CAPTURE_OUTPUT_DIR，则从环境变量惰性初始化——
    这使得 patch 进 SGLang 的 dsa_indexer.py 无须任何显式 init 调用，
    每个 TP worker 进程在首次 forward 时自动建立 capturer（约束 10）。

    环境变量：
      DSA_CAPTURE_OUTPUT_DIR  dump 输出目录（未设置 → 不 capture）
      DSA_CAPTURE_LAYERS      层过滤，如 "0-4"（默认全部层）
      DSA_CAPTURE_NUM_LAYERS  模型层数（默认 61）
    """
    global _env_init_attempted
    if _global_capturer is None and not _env_init_attempted:
        _env_init_attempted = True
        output_dir = os.environ.get("DSA_CAPTURE_OUTPUT_DIR", "").strip()
        if output_dir:
            init_indexer_state_capturer(
                output_dir=output_dir,
                num_layers=int(os.environ.get("DSA_CAPTURE_NUM_LAYERS", "61")),
                rank=_detect_rank(),
                enabled=True,
                capture_layers=parse_layer_spec(
                    os.environ.get("DSA_CAPTURE_LAYERS", "")
                ),
                save_every=int(os.environ.get("DSA_CAPTURE_SAVE_EVERY", "0")),
            )
    return _global_capturer


def set_indexer_state_capturer(capturer: Optional[IndexerStateCapturer]):
    global _global_capturer
    _global_capturer = capturer


def _detect_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def init_indexer_state_capturer(
    output_dir: str,
    num_layers: int = 61,
    rank: int = 0,
    enabled: bool = True,
    capture_layers: Optional[FrozenSet[int]] = None,
    save_every: int = 0,
) -> IndexerStateCapturer:
    capturer = IndexerStateCapturer(
        output_dir=output_dir,
        num_layers=num_layers,
        rank=rank,
        enabled=enabled,
        capture_layers=capture_layers,
        save_every=save_every,
    )
    set_indexer_state_capturer(capturer)

    # worker 进程退出时自动落盘：
    # - atexit 覆盖正常退出
    # - SIGTERM handler 覆盖被 executor terminate() 的情况
    #   （vLLM/SGLang 的多进程 worker 常以 SIGTERM 结束，默认不跑 atexit）
    if capturer.should_capture():
        import atexit
        atexit.register(capturer.save)

        import signal
        try:
            prev_handler = signal.getsignal(signal.SIGTERM)

            def _save_on_term(signum, frame):
                try:
                    capturer.save()
                finally:
                    if callable(prev_handler) and prev_handler not in (
                        signal.SIG_IGN, signal.SIG_DFL
                    ):
                        prev_handler(signum, frame)
                    else:
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                        os.kill(os.getpid(), signal.SIGTERM)

            signal.signal(signal.SIGTERM, _save_on_term)
        except ValueError:
            # 非主线程无法注册 signal handler，只依赖 atexit
            logger.warning("Cannot register SIGTERM handler (non-main thread); "
                           "relying on atexit for final save")
    return capturer
