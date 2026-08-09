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

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=5,
                    help="截断模型到前 N 层（0 = 不截断）")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=33792)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    from vllm import LLM, SamplingParams

    hf_overrides = {}
    if args.num_layers > 0:
        hf_overrides["num_hidden_layers"] = args.num_layers

    llm = LLM(
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
