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

### 步骤（**两个节点**上都要做 1-2，第 3 步只在 head 节点跑）

**1. 两节点各自准备**（缺一个都会在加载 685GB 权重之后才炸）

```bash
# 仓库与 vLLM 必须在两节点的【同一绝对路径】下（或同一共享 FS），
# 模型权重同理：同路径的本地副本，或共享盘
cd /path/to/dsa_indexer_stat
VLLM_SRC=/workspace/vllm bash capture/apply_vllm_patch.sh   # 幂等，各节点各跑一次
python -m pytest tests/ -q                                  # 约束 11
```

**2. 两节点起 ray**（head 先起）

```bash
# ---- head 节点 ----
export PYTHONPATH=/path/to/dsa_indexer_stat   # 必须在 ray start 之前：
                                              # ray worker 进程继承 ray start 的环境
export VLLM_HOST_IP=<HEAD_NODE_IP>            # 多网卡时不设会 "No available node types"
export NCCL_IB_HCA=mlx5                       # IB 设备名，按 ibstat 实际输出改
export NCCL_SOCKET_IFNAME=<IB 或内网网卡名>
ray start --head --port=6379

# ---- worker 节点 ----
export PYTHONPATH=/path/to/dsa_indexer_stat   # 同一路径
export VLLM_HOST_IP=<WORKER_NODE_IP>          # 每个节点不同，填自己的
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=<同上>
ray start --address=<HEAD_NODE_IP>:6379

# ---- 任一节点核对 ----
ray status && ray list nodes    # 必须是 2 个 node、16 GPU
```

注意 NCCL/IB 相关变量要在 `ray start` 之前 export：在 shell 里临时设只影响本节点的
后续进程，ray worker 只继承 `ray start` 时的环境（vLLM 会额外把 `NCCL_` 前缀的变量
从 driver 带过去，两处都设最稳）。

**3. head 节点一条命令跑完**

```bash
REF_RUN=runs/A_32k_tp8_XXX/seq00 \
    VLLM_SRC=/workspace/vllm bash capture/run_full61_vllm.sh
```

它按顺序做：ray 预检（GPU 数够不够）→ **5 层 × TP=16 的 smoke**（~30GB 权重，
先验证多节点通路，别拿 685GB 试错；约束 12）→ 用 smoke 与 `REF_RUN` 现场标定
`DSA_TOL` → 全 61 层 capture → 全模型金丝雀（layer 0-4）→ 层间 top-k 相似性 →
m1-m6。已经标过容差就 `SKIP_SMOKE=1 DSA_TOL=<值>` 直接烧。

`REF_RUN` 必须是**同一条 prompt**的截断 run 的 `seqNN` 目录（脚本默认
`PROMPT_PREFIX` 指向 suite A 的 manifest，manifest 存在就不重挑样本）。

### 两个坑

- **`DSA_CAPTURE_*` 不会自动传到远端 ray worker**：`vllm/ray/ray_env.py` 的白名单
  只带 `VLLM_`/`NCCL_`/`HF_` 等前缀。漏掉不报错，只会让远端 worker 的 capturer
  静默关闭。runner 已 export `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DSA_`，
  `vllm_generate` 里还有一道预检直接调 vLLM 的 `get_env_vars_to_copy()` 核对——
  两道都别删。
- **确认真的走上了 IB**：`NCCL_DEBUG=TRACE` 下日志里应出现 `via NET/IB`；
  出现 `via NET/Socket` 说明退化成 TCP，TP=16 跨节点会慢到不可用。
  smoke 阶段就该看这个，别等全模型。

## replay 环境
- 单卡即可（H800 或 A100）
- 只需 PyTorch，不需 SGLang
- 安装: `pip install torch numpy pytest`
