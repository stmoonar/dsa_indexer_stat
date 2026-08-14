# 环境配置

## 硬件（已实测）

| 角色 | bond1 IP | GPU | 说明 |
|---|---|---|---|
| head | `30.203.130.54` | 8 × H800 | driver 在这里，rank0 落这里，dump 写这里 |
| worker | `30.203.136.62` | 8 × H800 | |

- 跨节点网络：**`bond1`**，**RoCEv2 over LAG**，不是原生 InfiniBand
- RDMA 设备：`mlx5_bond_1` … `mlx5_bond_8`（LAG 后的 bond 设备，不是物理 slave）
- 两台 hostname 都是 `TENCENT64.site`，靠 `/etc/machine-id` 区分节点
- `memlock` 实测 `unlimited`
- 两台的 bond1 分属不同 /27 子网，跨子网路由 —— 所以 `NCCL_IB_GID_INDEX=3` 是硬要求

网络参数的权威来源是 H800 SGLang + TRMT-DeepEP 部署验证文档 §4 的容器 `-e` 列表
（那一组通过了 4 节点 internode DeepEP correctness），已固化进
[`net_env.sh`](net_env.sh)。换机器/换集群时先跑 [`probe_net.sh`](probe_net.sh)
重新定值，不要照抄。

## 软件

- vLLM `0.18.0`，源码在两节点容器内 `/workspace/vllm`
- 本仓库在两节点容器内 `/workspace/dsa_indexer_stat`（**同一绝对路径**，硬要求）
- conda env `vllm-td`
- 模型 DeepSeek-V3.2（61 layers, 671B），`/data1/models/DeepSeek-V3.2`（容器内外同路径）
- Python 3.10+ / PyTorch 2.x with CUDA
- SGLang 后端（`capture/run_first5_32k.sh`）仍保留，但全 61 层链路走 vLLM

## 容器

两节点必须用**完全相同的镜像 tag**，否则 ray / torch / NCCL 版本不一致会在跨节点握手时炸。

```bash
sudo docker run --privileged --gpus all \
  -dit --net=host --uts=host --ipc=host \
  --security-opt=seccomp=unconfined \
  --device=/dev/infiniband \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576 \
  --name dsa_indexer_stat \
  -v /etc/machine-id:/etc/machine-id:ro \
  -v /sys/class:/sys/class:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/devices:/sys/devices:ro \
  -v /data:/data \
  -v /data1:/data1 \
  -v /data1/work/dsa_indexer_stat:/workspace/dsa_indexer_stat \
  -v /data1/work/vllm:/workspace/vllm \
  mirrors.tencent.com/tile-overlap/vllm-triton-dist:v0.3
```

为什么是这些参数：

- `--net=host`：ray 与 NCCL 都要看见宿主的 `bond1`。**不能用 bridge**。
- `--ipc=host`：否则 `/dev/shm` 默认只有 64MB。
- `-v /etc/machine-id:...:ro`：两台 hostname 相同，唯一节点标识靠它。
- `-v .../dsa_indexer_stat`、`-v .../vllm`：仓库和 vLLM 源码放绑定挂载，
  不要留在镜像/可写层。61 层 dump 是 GB 级，写 overlay2 又慢又会撑爆
  `/var/lib/docker`；`runs/` 也应落在 `/data1` 上。
- `--ulimit memlock=-1`：RDMA 要注册大块 pinned memory。本批宿主实测已是
  `unlimited`（`--privileged` 本身不改 rlimit，是宿主 dockerd 的默认），
  加上更保险。已经起好的 privileged 容器可以事后 `ulimit -l unlimited`
  补救，但必须在 `ray start` 之前。
- `--device=/dev/infiniband` 在 `--privileged` 下冗余；去掉 privileged 时
  还需要 `--cap-add=IPC_LOCK`。

## capture 关键参数（约束 7）

- 关 CUDA graph：**必须**。CUDA graph 区内的 Python dump 会静默失败。
  vLLM 侧是 `enforce_eager=True`（同时关掉 torch.compile），SGLang 侧是
  `--disable-cuda-graph`。
- bs=1：capture 时只跑单请求（vLLM `max_num_seqs=1`）。
- 关 overlap scheduler。

## 全 61 层 capture（2 节点 TP=16 + ray）

fp8 权重约 685GB，单节点 8×H800（640GB）装不下，必须两节点。

**为什么是 TP=16 而不是 TP=8 + PP=2**：capturer 只在 global rank 0 落盘
（约束 9：q/k/w 都是 Replicated，rank0 即全量）。PP 下 layer 31-60 活在
rank 8-15 上，rank != 0 → 后半段静默不 dump。TP=16 时每个 rank 都跑全部
61 层，rank0 一份就是完整数据；ray 的 worker 排序保证 rank0 落在 driver
节点，dump 写在本地盘。

### 步骤

**0. 容器起好后先确认三件事**（两节点都做，**在容器内**）

```bash
ip -o -4 addr show dev bond1     # 必须有 IPv4 地址
ls /sys/class/infiniband/        # 必须是 mlx5_bond_1 .. mlx5_bond_8
ulimit -l                        # 必须 unlimited
```

任何一条不符，先跑 `bash env/probe_net.sh` 重新定值，不要硬上。

**1. 两节点各自准备**（缺一个都会在加载 685GB 权重之后才炸）

```bash
cd /workspace/dsa_indexer_stat
conda activate vllm-td
VLLM_SRC=/workspace/vllm bash capture/apply_vllm_patch.sh   # 幂等，各节点各跑一次
python -m pytest tests/ -q                                  # 约束 11
```

**2. 两节点起 ray**

先 `source`，再 `ray start`，顺序不能反 —— ray worker 进程只继承 `ray start`
时的环境。`net_env.sh` 必须**从仓库里** source（PYTHONPATH 是按脚本位置推的，
拷到别处会推错）。

```bash
# ---- 两个节点都先做 ----
cd /workspace/dsa_indexer_stat
source env/net_env.sh
```

54 上应打印 `net_env: bond1=30.203.130.54 …`，62 上是 `bond1=30.203.136.62`。
两边都要有 `memlock=unlimited`、`GID_INDEX=3 (RoCEv2)`。

```bash
# ---- head 节点 (30.203.130.54) ----
ray stop
ray start --head --node-ip-address="$VLLM_HOST_IP" --port=6379

# ---- worker 节点 (30.203.136.62) ----
ray stop
ray start --address="$HEAD_IP:6379" --node-ip-address="$VLLM_HOST_IP"
```

`--node-ip-address` 不能省。不给它，ray 会按默认路由自己挑网卡（本集群上会挑到
管理网 `30.205.x`），与 `VLLM_HOST_IP` 对不上，placement group 永远 pending。

**核对（任一节点，这一步不过就别往下走）**：

```bash
ray status      # 2 个 node、16 GPU

python -c "import ray;ray.init(address='auto');\
print(sorted(k for k in ray.cluster_resources() if k.startswith('node:')))"
# 必须精确打印: ['node:30.203.130.54', 'node:30.203.136.62']
```

`node:` 列表里没有本机的 `$VLLM_HOST_IP`，就是 `--node-ip-address` 没生效或
ray 没重启干净 —— 回去 `ray stop` 重来，改环境变量无效。

**3. head 节点一条命令跑完**

```bash
cd /workspace/dsa_indexer_stat
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
REF_RUN=runs/A_32k_tp8_XXX/seq00 \
VLLM_SRC=/workspace/vllm \
    bash capture/run_full61_vllm.sh
```

它按顺序做：ray 预检 → **5 层 × TP=16 的 smoke**（~30GB 权重，先验证多节点通路，
别拿 685GB 试错；约束 12）→ 用 smoke 与 `REF_RUN` 现场标定 `DSA_TOL` →
全 61 层 capture → 全模型金丝雀（layer 0-4）→ 层间 top-k 相似性 → m1-m6。
已经标过容差就 `SKIP_SMOKE=1 DSA_TOL=<值>` 直接烧（那时可以去掉 `NCCL_DEBUG`）。

`REF_RUN` 必须是**同一条 prompt**的截断 run 的 `seqNN` 目录（脚本默认
`PROMPT_PREFIX` 指向 suite A 的 manifest，manifest 存在就不重挑样本）。

### 三条通路互相独立，别混为一谈

| 变量 | 填什么 | 管什么 | 跨节点流量 |
|---|---|---|---|
| `VLLM_HOST_IP` / `ray start --node-ip-address` | bond1 的**本机 IP** | ray 节点注册（`node:<ip>` 资源）+ torch.distributed rendezvous | 极小 |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | **网卡名** `bond1` | NCCL/gloo 的带外 bootstrap；RDMA 用不上时的数据回退 | 正常情况下很小 |
| `NCCL_IB_HCA` | `ibstat` 的**设备名** `mlx5_bond_1..8` | 真正的 all-reduce 数据平面 | **全部** |

前两个必须落在同一张网卡上。第三个走 RDMA，**本集群是 RoCEv2**，所以它反过来
又依赖 `NCCL_IB_GID_INDEX=3` 和 IP 层可达（原生 IB 才与 IP 无关）。

把 `VLLM_HOST_IP` 设成"最快那张网卡"没有收益 —— 它只承载握手，带宽全在
`NCCL_IB_HCA` 上。

### 坑

- **`VLLM_HOST_IP` 必须与本节点 `ray start` 注册的 IP 逐字相同**。vLLM 用
  `get_ip()`（= `VLLM_HOST_IP` 的字面值）给 placement group 的 bundle 0 打
  `node:<ip>` 亲和标记，而 ray 只在每个节点创建 `node:<该节点注册 IP>` 这一个资源。
  对不上就是 `No available node types can fulfill resource request` 无限重试，
  **GPU 数预检照样通过**，卡死在后面。设错比不设更糟：不设时 vLLM 和 ray 各自
  探测，通常都选默认路由那张，反而一致。

- **`NCCL_IB_GID_INDEX=3` 不能漏**。本集群是 RoCEv2；缺它 NCCL 会用 GID 0
  （RoCEv1 / link-local），两节点又跨子网，直接不通。`NCCL_IB_SL=3` /
  `NCCL_IB_TC=160` 是与交换机 PFC/ECN 无损队列对齐的值，同样不能省。
  `NCCL_IB_HCA` 也**不能写成 `mlx5`** —— 那是前缀匹配，会连 bond 底下的物理
  slave 一起匹配上，选错口。

- **`GLOO_SOCKET_IFNAME` 不在 vLLM 的 ray 拷贝白名单里**。白名单见启动日志的
  `Env var prefixes to copy`：`DSA_`/`HF_`/`HUGGING_FACE_`/`LMCACHE_`/`NCCL_`/
  `UCX_`/`VLLM_`。vLLM 的 `GroupCoordinator` 会另建一个 CPU 侧 gloo group，
  它不看 `NCCL_SOCKET_IFNAME`；`--net=host` 下容器能看到宿主的 `docker0` 和
  一堆 `veth*`，自动选网卡有很大概率挑中 `docker0` 然后跨节点 hang。
  `net_env.sh` 里已 `export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_`，
  并且它在每个节点的 `ray start` 之前 source，两道保险。

- **`DSA_CAPTURE_*` 也不会自动传到远端 ray worker**。漏掉不报错，只会让远端
  worker 的 capturer 静默关闭。runner 已 export `…PREFIXES_TO_COPY` 追加
  `DSA_`，`vllm_generate` 里还有一道预检直接调 vLLM 的 `get_env_vars_to_copy()`
  核对 —— 两道都别删。

- **`VLLM_HOST_IP` 会被 vLLM 从 driver 拷到远端 worker**（它在 `VLLM_` 白名单里）。
  所以 worker 节点上这个变量对 vLLM 进程几乎不起作用，它真正的用处只有一个：
  作为该节点 `ray start --node-ip-address` 的取值。

- **确认真的走上了 RDMA**：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` 下
  日志里应出现 `NET/IB` 和 `mlx5_bond_*`；出现 `NET/Socket` 说明退化成 TCP，
  TP=16 跨节点会慢到不可用。**smoke 阶段就看**，别等全模型。
  退化最常见的两个原因：`ulimit -l` 不是 unlimited、`NCCL_IB_GID_INDEX` 没设。

## replay 环境

- 单卡即可（H800 或 A100）
- 只需 PyTorch，不需 SGLang / vLLM
- 安装：`pip install torch numpy pytest`
