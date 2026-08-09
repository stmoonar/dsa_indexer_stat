"""
vllm_generate.py — vLLM 离线推理入口（capture 用，bs=1）

与 SGLang server 流程不同，vLLM 用 offline LLM API：单进程驱动，
生成结束后 worker 正常退出 → capturer 的 atexit/SIGTERM handler 落盘。

约束 7 的 vLLM 对应：enforce_eager=True 同时关闭 torch.compile 与
CUDA graph（等价 -cc.mode=none -cc.cudagraph_mode=none），
max_num_seqs=1 保证 bs=1。

模型截断：hf_overrides={"num_hidden_layers": N}，配合
capture/patch_vllm/indexer_capture.diff 中的 load_weights 跳层逻辑。

用法（DSA_CAPTURE_* 环境变量需在启动前 export，见 run_first5_32k_vllm.sh）：
    python -m capture.vllm_generate \
        --model /data1/models/DeepSeek-V3.2 \
        --prompt-file prompts/agentic_32k.txt \
        --num-layers 5 --tp 8 --max-model-len 33792 --max-new-tokens 256
"""

import os
import argparse
import importlib.util


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
    args = ap.parse_args()

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

    # 显式销毁 → worker 优雅退出 → capturer atexit/SIGTERM 落盘
    del llm


if __name__ == "__main__":
    main()
