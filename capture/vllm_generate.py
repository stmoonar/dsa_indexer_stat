"""
vllm_generate.py — vLLM 离线推理入口（capture 用，bs=1）

与 SGLang server 流程不同，vLLM 用 offline LLM API：单进程驱动，
生成结束后 worker 正常退出 → capturer 的 atexit/SIGTERM handler 落盘。

约束 7 的 vLLM 对应：enforce_eager=True 同时关闭 torch.compile 与
CUDA graph（等价 -cc.mode=none -cc.cudagraph_mode=none），
max_num_seqs=1 保证 bs=1。

模型截断：hf_overrides={"num_hidden_layers": N}，配合
capture/patch_vllm/indexer_capture.diff 中的 load_weights 跳层逻辑。
--num-layers 0 = 不截断（全 61 层），走 2 节点 TP=16 + ray。

用法（DSA_CAPTURE_* 环境变量需在启动前 export，见 run_first5_32k_vllm.sh）：
    python -m capture.vllm_generate \
        --model /data1/models/DeepSeek-V3.2 \
        --prompt-file prompts/agentic_32k.txt \
        --num-layers 5 --tp 8 --max-model-len 33792 --max-new-tokens 256
    python -m capture.vllm_generate ... \
        --num-layers 0 --tp 16 --distributed-executor-backend ray
"""

import os
import json
import hashlib
import argparse
import importlib.util

import numpy as np

# collective_rpc 传 callable 需要它：vLLM 默认只接受 msgspec 可序列化的类型，
# 否则报 "Object of type <class 'function'> is not serializable"。
# 这是本地 offline 脚本，driver 与 worker 都是自己拉起的进程，不存在不可信输入。
# VLLM_ 前缀在 ray 的环境变量拷贝白名单里，会自动带到远端 worker。
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def deep_gemm_available() -> bool:
    """与 vllm.utils.deep_gemm.is_deep_gemm_supported 的软件侧条件一致：
    VLLM_USE_DEEP_GEMM 未被禁用 且 deep_gemm 包可导入。
    （硬件侧 SM90/SM100 在 H800 上恒成立，不重复判。）"""
    if os.getenv("VLLM_USE_DEEP_GEMM", "1").lower() in ("0", "false"):
        return False
    return importlib.util.find_spec("deep_gemm") is not None


def resolve_perf_defaults(has_deep_gemm: bool, truncated: bool = False):
    """返回 (max_num_batched_tokens, gpu_memory_utilization)。

    无 DeepGEMM 时 indexer 走 fp8_mqa_logits_torch 兜底，prefill 会物化
    [H=64, chunk, ctx] 的 fp32 logits：chunk=16384、ctx=32K 时 ≈ 62GB → OOM。
    兜底配置 chunk=2048 → ≈ 17.5GB，并压低 util 留出 free 显存。

    截断模型（前 5 层）权重每卡仅 ~5GB，KV 需求 ~100MB 量级，而 vLLM 会把
    util 预算内的剩余显存全部划给 KV 池——截断时预算再压一档，纯属浪费的
    KV 池让位给兜底 einsum 的 free 显存。
    """
    if has_deep_gemm:
        return None, (0.5 if truncated else 0.85)   # None = vLLM 默认 chunk
    return 2048, (0.45 if truncated else 0.65)


def missing_carry_over(environ_names, copy_set) -> list:
    """列出不会被 ray 带到远端 worker 的 DSA_CAPTURE_* 变量。

    多节点时 worker 是远端 ray actor，只继承 vLLM 白名单里的环境变量
    （vllm/ray/ray_env.py：VLLM_/NCCL_/HF_ 等前缀 + 注册过的 vllm envs）。
    DSA_CAPTURE_* 不在其中 —— 漏掉的后果不是报错而是【静默不 dump】，
    在一个要加载 685GB 权重的 run 上，这是最贵的一类失败。
    修复办法：export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DSA_
    """
    return sorted(n for n in environ_names
                  if n.startswith("DSA_CAPTURE") and n not in copy_set)


def check_ray_capture_env(backend: str, tp: int):
    """多节点 ray 下的预检：直接问 vLLM 自己的拷贝白名单，不猜版本。"""
    if not os.environ.get("DSA_CAPTURE_OUTPUT_DIR", "").strip():
        return
    try:
        import torch
        local_gpus = torch.cuda.device_count()
    except Exception:
        local_gpus = 0
    if backend != "ray" and tp <= max(local_gpus, 1):
        return
    try:
        from vllm.ray.ray_env import get_env_vars_to_copy
    except ImportError:
        print("WARNING: 无法导入 vllm.ray.ray_env，跳过环境变量传播预检；"
              "请自行确认远端 worker 能看到 DSA_CAPTURE_*")
        return
    missing = missing_carry_over(os.environ, get_env_vars_to_copy())
    if missing:
        raise RuntimeError(
            f"这些 capture 环境变量不会传到远端 ray worker: {missing}\n"
            f"  export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DSA_\n"
            f"（漏掉不会报错，只会静默不 dump——不允许在全模型 run 上赌）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=5,
                    help="截断模型到前 N 层（0 = 不截断）")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=33792)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="缺省自动：有 DeepGEMM 0.85，无 0.65")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="prefill chunk 大小；缺省自动：无 DeepGEMM 时 2048")
    ap.add_argument("--distributed-executor-backend", default=None,
                    choices=(None, "mp", "ray"),
                    help="跨节点（TP 超过单机卡数）必须是 ray")
    args = ap.parse_args()

    check_ray_capture_env(args.distributed_executor_backend, args.tp)

    has_dg = deep_gemm_available()
    auto_chunk, auto_util = resolve_perf_defaults(
        has_dg, truncated=args.num_layers > 0)
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = auto_chunk
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = auto_util
    if not has_dg:
        print("=" * 70)
        print("WARNING: deep_gemm 不可用，indexer 走 PyTorch 兜底（慢但能跑）。")
        print(f"  自动降级: max_num_batched_tokens={args.max_num_batched_tokens}, "
              f"gpu_memory_utilization={args.gpu_memory_utilization}")
        print("  推荐安装 DeepGEMM (https://github.com/deepseek-ai/DeepGEMM) "
              "后恢复默认配置。")
        print("=" * 70)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    from vllm import LLM, SamplingParams

    hf_overrides = {}
    if args.num_layers > 0:
        hf_overrides["num_hidden_layers"] = args.num_layers

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,              # 约束 7: 关 CUDA graph + torch.compile
        hf_overrides=hf_overrides or None,
        max_model_len=args.max_model_len,
        max_num_seqs=1,                  # 约束 7: bs=1
        enable_prefix_caching=False,     # 保证完整 prefill，k^I 无缺段
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = \
            args.distributed_executor_backend
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,                 # 贪心，可复现（约束 13 配套）
    )

    outputs = llm.generate([prompt], sampling)
    out = outputs[0]
    n_prompt = len(out.prompt_token_ids)
    n_gen = len(out.outputs[0].token_ids)
    print(f"\n=== Generation done: prompt_tokens={n_prompt}, "
          f"completion_tokens={n_gen} ===")
    print(f"--- first 500 chars of output ---\n{out.outputs[0].text[:500]}")

    # 落盘实际喂给模型的 token ids：全模型 run 的金丝雀测试要求输入完全一致，
    # 光靠 prompt 文本不够（tokenizer 版本/模板变化会静默改变 token 流）
    out_dir = os.environ.get("DSA_CAPTURE_OUTPUT_DIR", "").strip()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ids = np.asarray(out.prompt_token_ids, dtype=np.int32)
        np.save(os.path.join(out_dir, "prompt_token_ids.npy"), ids)
        with open(os.path.join(out_dir, "prompt_fingerprint.json"), "w") as f:
            json.dump({
                "n_prompt_tokens": int(n_prompt),
                "n_completion_tokens": int(n_gen),
                "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "prompt_file": args.prompt_file,
                "model": args.model,
            }, f, indent=2)
        print(f"Wrote prompt fingerprint to {out_dir}")

    # 主动让每个 worker 落盘。不要依赖进程退出：ray executor 关闭时对 actor
    # 用 ray.kill()（SIGKILL），atexit 与 SIGTERM handler 都不会执行，dump
    # 会【静默全丢】。mp backend 走 SIGTERM 所以单机看不出这个问题，
    # 跨节点一上 ray 就中招。
    save_capturers_now(llm)

    del llm


def save_capturers_now(llm):
    """生成结束后主动落盘，并把每个 worker 的结果打出来。

    这同时是诊断：'no-capturer' 说明 patch 的 hook 从未被调用，
    'skip(non-rank0)' 出现在全部 16 个 worker 上说明 rank0 不在本次
    executor 的 worker 列表里 —— 两种都会导致零产出，但原因完全不同。

    dsa_worker_save 从 capture.indexer_state_capturer 导入而不是定义在
    本文件里：本文件以 python -m 运行（模块名 __main__），pickle 会把
    __main__ 里的函数按引用序列化，worker 侧解不开。
    """
    try:
        from capture.indexer_state_capturer import dsa_worker_save
        results = llm.collective_rpc(dsa_worker_save)
    except Exception as e:
        print(f"WARNING: collective save 失败 ({type(e).__name__}: {e})；"
              "退回到 atexit/SIGTERM —— ray backend 下大概率丢数据。")
        return
    print("=== capturer save ===")
    for i, r in enumerate(results):
        print(f"  worker{i:02d}: {r}")
    if not any(isinstance(r, str) and "saved" in r for r in results):
        print("WARNING: 没有任何 worker 落盘，本次 run 不会有 dump。")


if __name__ == "__main__":
    main()
