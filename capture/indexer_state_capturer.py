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


def parse_int_list(spec: str) -> tuple:
    """解析逗号分隔的整数列表，如 "2048,8192,16384"；空串 → ()。"""
    return tuple(int(x) for x in spec.replace(" ", "").split(",") if x)


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

    # prefill 期的 q/w 采样（全 0 = 不采）
    # prefill_tail:    保留最后 N 个 prefill 位置（滚动窗口）
    # prefill_buckets: 位置锚点列表，每个锚点抓 [anchor, anchor+run) 一段
    # prefill_run:     每个锚点的连续段长度
    #
    # 必须是【连续段】而不是孤立位置：warm 集 / churn / τ 都需要
    # "同层前一个采样位置"，孤立位置在配对时会被整段丢弃，
    # 前缀长度曲线会静默塌缩成单点。
    prefill_tail: int = 0
    prefill_buckets: tuple = ()
    prefill_run: int = 32

    # MLA latent 捕获（想法 2：k^I 能否由已存的 c_s 线性重建）
    # c_s = kv_c_normed [512]，k_pe [64]，两者合起来就是进 KV cache 的条目
    capture_latent: bool = False

    # 运行时状态
    _step_data: dict = field(default_factory=dict)  # {(step, layer): {q, w, topk}}
    _prefill_data: dict = field(default_factory=dict)  # {(pos, layer): {...}}
    _k_buffers: dict = field(default_factory=dict)   # {layer: [k tensors]}
    _c_buffers: dict = field(default_factory=dict)   # {layer: [kv_c tensors]}
    _kpe_buffers: dict = field(default_factory=dict)  # {layer: [k_pe tensors]}
    _current_decode_step: int = field(default=0)
    _is_prefill: bool = field(default=True)
    # 已由 vllm_generate 的 collective_rpc 显式落过盘；用于抑制 atexit 重复写
    # （61 层一次 save 是 GB 级，写两遍纯浪费）
    _explicit_save_done: bool = field(default=False)

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

    @property
    def capture_prefill_qw(self) -> bool:
        return self.prefill_tail > 0 or bool(self.prefill_buckets)

    def _in_bucket(self, pos: int) -> bool:
        return any(a <= pos < a + self.prefill_run for a in self.prefill_buckets)

    def prefill_local_mask(self, positions):
        """给定本次 forward 的 prefill 绝对位置（升序），返回需要捕获的布尔掩码。

        chunked prefill 下每个 chunk 各自贡献"本 chunk 的最后 tail 个位置"，
        跨 chunk 的全局滚动窗口由 _trim_prefill 收敛到真正的最后 tail 个；
        bucket 段按绝对位置选，天然可跨 chunk 边界。
        与 torch/numpy 都兼容。
        """
        n = int(positions.shape[0])
        mask = positions != positions  # 全 False，dtype=bool，device 跟随
        if self.prefill_tail > 0 and n > 0:
            mask[max(0, n - self.prefill_tail):] = True
        for a in self.prefill_buckets:
            mask = mask | ((positions >= a) & (positions < a + self.prefill_run))
        return mask

    def capture_prefill_step(
        self,
        layer_id: int,
        positions,
        query: torch.Tensor,
        weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        """
        Prefill 阶段截获选定位置的 q^I, w, topk_indices。

        与 decode 的语义差别只在"步"的含义：这里的键是绝对位置 pos，
        该位置的 indexer 对 k^I[:pos+1] 打分（因果）。
        前 5 层的 prefill q/k/w 不受截断影响，与全模型逐位一致。

        positions: [n] 绝对位置（int）
        query: [n, H, d_I]，weights: [n, H]，topk_indices: [n, top_k]
        """
        if not self.should_capture() or not self.should_capture_layer(layer_id):
            return
        if not self.capture_prefill_qw:
            return

        pos_list = [int(p) for p in positions]
        q = query.detach().float().cpu().numpy()
        w = weights.detach().float().cpu().numpy()
        t = topk_indices.detach().cpu().to(torch.int32).numpy()

        for i, pos in enumerate(pos_list):
            self._prefill_data[(pos, layer_id)] = {
                "q_I": q[i],
                "w": w[i],
                "topk_indices": t[i],
            }
        self._trim_prefill(layer_id)

    def _trim_prefill(self, layer_id: int):
        """只保留 bucket 段内的位置 + 最后 prefill_tail 个位置。"""
        positions = sorted(p for (p, l) in self._prefill_data if l == layer_id)
        if not positions:
            return
        keep = {p for p in positions if self._in_bucket(p)}
        if self.prefill_tail > 0:
            keep |= set(positions[-self.prefill_tail:])
        for p in positions:
            if p not in keep:
                del self._prefill_data[(p, layer_id)]

    def capture_mla_latent(self, layer_id: int, kv_c, k_pe):
        """
        截获 MLA 的 KV latent（想法 2 的裁决数据）。

        kv_c: [n_tokens, kv_lora_rank=512]  post-LayerNorm，进 KV cache 的主体
        k_pe: [n_tokens, 1, 64] 或 [n_tokens, 64]  post-RoPE
        两者拼起来就是 paged KV cache 里每 token 存的条目；问题是
        k^I（128 维）能否由它线性重建——若能，indexer 就不必单独存 k^I。

        存 fp16：c/k_pe 都是 post-norm/post-RoPE 的 O(1) 量，
        fp16 的 1e-3 相对误差远小于回归残差，可省一半体积。
        """
        if not self.should_capture() or not self.should_capture_layer(layer_id):
            return
        if not self.capture_latent:
            return
        self._c_buffers.setdefault(layer_id, []).append(
            kv_c.detach().reshape(kv_c.shape[0], -1).half().cpu())
        self._kpe_buffers.setdefault(layer_id, []).append(
            k_pe.detach().reshape(k_pe.shape[0], -1).half().cpu())

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

        # 保存 MLA latent（想法 2）
        for buffers, name in ((self._c_buffers, "c_latent"),
                              (self._kpe_buffers, "k_pe")):
            for layer_id, chunks in buffers.items():
                if not chunks:
                    continue
                arr = torch.cat(chunks, dim=0).numpy()
                np.save(os.path.join(
                    self.output_dir, f"{name}_layer{layer_id:03d}.npy"), arr)
                logger.info(f"  Layer {layer_id}: {name} shape={arr.shape}")

        # 保存 prefill 位置采样
        n_prefill = 0
        for (pos, layer_id), data in sorted(self._prefill_data.items()):
            path = os.path.join(
                self.output_dir, f"prefill_pos{pos:08d}_layer{layer_id:03d}.npz"
            )
            np.savez(path, **data)
            n_prefill += 1
        if n_prefill:
            logger.info(f"  Saved {n_prefill} prefill position dumps")

        # 保存 config（约束 13：run 输出目录必须含配置快照）
        import json
        merged = {
            "num_layers": self.num_layers,
            "capture_layers": (
                sorted(self.capture_layers) if self.capture_layers else "all"
            ),
            "num_decode_steps": self.num_decode_steps,
            "prefill_tail": self.prefill_tail,
            "prefill_buckets": list(self.prefill_buckets),
            "prefill_run": self.prefill_run,
            "prefill_positions": sorted(
                {p for (p, _) in self._prefill_data}
            ),
            "capture_latent": self.capture_latent,
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
      DSA_CAPTURE_OUTPUT_DIR    dump 输出目录（未设置 → 不 capture）
      DSA_CAPTURE_LAYERS        层过滤，如 "0-4"（默认全部层）
      DSA_CAPTURE_NUM_LAYERS    模型层数（默认 61）
      DSA_CAPTURE_SAVE_EVERY    每 N 个 decode step 落盘一次（默认 0 = 不周期落盘）
      DSA_CAPTURE_PREFILL_TAIL  保留最后 N 个 prefill 位置的 q/w（默认 0 = 不采）
      DSA_CAPTURE_PREFILL_STRIDE 额外采样 pos%N==0 的 prefill 位置（默认 0 = 关）
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
                prefill_tail=int(
                    os.environ.get("DSA_CAPTURE_PREFILL_TAIL", "0")),
                prefill_buckets=parse_int_list(
                    os.environ.get("DSA_CAPTURE_PREFILL_BUCKETS", "")),
                prefill_run=int(
                    os.environ.get("DSA_CAPTURE_PREFILL_RUN", "32")),
                capture_latent=os.environ.get(
                    "DSA_CAPTURE_LATENT", "0").lower() in ("1", "true"),
            )
    return _global_capturer


def set_indexer_state_capturer(capturer: Optional[IndexerStateCapturer]):
    global _global_capturer
    _global_capturer = capturer


def get_existing_indexer_state_capturer() -> Optional[IndexerStateCapturer]:
    """返回已存在的 capturer，绝不惰性初始化。

    显式落盘（vllm_generate 的 collective_rpc）必须用这个而不是
    get_indexer_state_capturer()：后者会在从未 capture 过的进程里凭空建一个
    capturer 并写出一份空的 capture_config.json，把"hook 根本没被调用"
    伪装成"调用了但没采到数据"——正好毁掉最关键的那条诊断信息。
    """
    return _global_capturer


def dsa_worker_save(_worker=None) -> str:
    """在 worker 进程内显式落盘，返回值是给 driver 打印的诊断字符串。

    必须定义在【真实模块】里，不能放进 vllm_generate：后者以 python -m
    运行，模块名是 __main__，collective_rpc 的 pickle 对 __main__ 里的
    函数按引用序列化（"__main__.xxx"），而 worker 侧的 __main__ 是 ray
    的入口，反序列化必然失败。

    参数是 vLLM 传进来的 Worker 实例，这里用不到。
    """
    cap = get_existing_indexer_state_capturer()
    if cap is None:
        # 该进程里 capturer 从未被创建 = patch 的 hook 一次都没被调用
        return "no-capturer"
    if not cap.should_capture():
        return f"rank={cap.rank} skip(non-rank0)"
    cap.save()
    cap._explicit_save_done = True
    return (f"rank={cap.rank} saved k_I_layers={len(cap._k_buffers)} "
            f"prefill={len(cap._prefill_data)} -> {cap.output_dir}")


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
    prefill_tail: int = 0,
    prefill_buckets: tuple = (),
    prefill_run: int = 32,
    capture_latent: bool = False,
) -> IndexerStateCapturer:
    capturer = IndexerStateCapturer(
        output_dir=output_dir,
        num_layers=num_layers,
        rank=rank,
        enabled=enabled,
        capture_layers=capture_layers,
        save_every=save_every,
        prefill_tail=prefill_tail,
        prefill_buckets=prefill_buckets,
        prefill_run=prefill_run,
        capture_latent=capture_latent,
    )
    set_indexer_state_capturer(capturer)

    # worker 进程退出时自动落盘：
    # - atexit 覆盖正常退出
    # - SIGTERM handler 覆盖被 executor terminate() 的情况
    #   （vLLM/SGLang 的多进程 worker 常以 SIGTERM 结束，默认不跑 atexit）
    if capturer.should_capture():
        import atexit

        def _final_save():
            # vllm_generate 已通过 collective_rpc 显式落过盘就不重复写。
            if not capturer._explicit_save_done:
                capturer.save()

        atexit.register(_final_save)

        import signal
        try:
            prev_handler = signal.getsignal(signal.SIGTERM)

            def _save_on_term(signum, frame):
                try:
                    _final_save()
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
