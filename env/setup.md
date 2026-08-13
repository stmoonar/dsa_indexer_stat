# 环境配置

## 硬件
- 2 × 8 H800 (16 卡)
- 节点间通信: InfiniBand

## 软件
- SGLang: (填写版本)
- 模型: DeepSeek-V3.2 (61 layers, 671B)
- Python: 3.10+
- PyTorch: 2.x with CUDA

## capture 启动命令模板

```bash
python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --tp 16 \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --max-total-tokens 131072 \
    --schedule-policy fcfs \
    2>&1 | tee logs/capture_$(date +%Y%m%d_%H%M%S).log
```

### 关键参数（约束 7）
- `--disable-cuda-graph`: 必须。CUDA graph 区内 Python dump 静默失败。
- `--tp 16`: 张量并行 = 16
- bs=1: capture 时只跑单请求
- 关闭 overlap scheduler

## 全 61 层 capture（2 节点 TP=16 + ray）

fp8 权重约 685GB，单节点 8×H800（640GB）装不下，必须两节点。

**为什么是 TP=16 而不是 TP=8 + PP=2**：capturer 只在 global rank 0 落盘
（约束 9：q/k/w 都是 Replicated，rank0 即全量）。PP 下 layer 31-60 活在
rank 8-15 上，rank != 0 → 后半段静默不 dump。TP=16 时每个 rank 都跑全部
61 层，rank0 一份就是完整数据；ray 的 worker 排序保证 rank0 落在 driver
节点，dump 写在本地盘。

```bash
# 两个节点都要：仓库与 vLLM 在同一绝对路径，且都已 apply patch
bash capture/apply_vllm_patch.sh

# head 节点
export PYTHONPATH=$(pwd)      # 必须在 ray start 之前，worker 进程继承它
ray start --head --port=6379
# worker 节点
export PYTHONPATH=/same/path/to/repo
ray start --address=<head_ip>:6379
ray status                    # 必须看到 16 GPU

# 从 head 节点跑（capture → 金丝雀 → 层间相似性 → m1-m6）
REF_RUN=runs/A_32k_tp8_XXX/seq00 DSA_TOL=<标定值> \
    VLLM_SRC=/workspace/vllm bash capture/run_full61_vllm.sh
```

**最容易踩的坑**：`DSA_CAPTURE_*` 不在 vLLM 传给远端 ray worker 的环境变量
白名单里（`vllm/ray/ray_env.py` 只带 VLLM_/NCCL_/HF_ 等前缀），漏掉不会报错，
只会静默不 dump。runner 已 export
`VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DSA_`，`vllm_generate` 里还有一道
预检直接问 vLLM 的白名单——两道都别删。

## replay 环境
- 单卡即可（H800 或 A100）
- 只需 PyTorch，不需 SGLang
- 安装: `pip install torch numpy pytest`
