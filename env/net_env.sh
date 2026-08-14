#!/bin/bash
# env/net_env.sh — 两节点通用的网络 / NCCL 环境。每个节点在 `ray start` 之前 source。
#
# 值的来源：H800 SGLang+TRMT-DeepEP 部署文档 §4 里已验证的容器 -e 列表
# （4 节点 internode DeepEP correctness 通过的那一组），加 vLLM/ray 侧的补充。
#
# 这套集群是 RoCEv2 over bonded Ethernet，不是原生 InfiniBand：
#   - bond1 承载 rendezvous（TCP）与 NCCL bootstrap
#   - RDMA 设备是 mlx5_bond_1..8（LAG 后的 bond 设备，不是物理 slave）
#   - 因此必须 NCCL_IB_GID_INDEX=3（RoCEv2）；缺它 NCCL 会选 GID 0
#     （RoCEv1 / link-local），两节点又不在同一 /27，跨子网直接不通。
#   - SL=3 / TC=160 是与交换机 PFC/ECN 无损队列对齐的值，不能省。
#
# 用法（两个节点都做，必须在 ray start 之前）：
#   cd /workspace/dsa_indexer_stat && source env/net_env.sh
#   # head:
#   ray start --head --node-ip-address="$VLLM_HOST_IP" --port=6379
#   # worker:
#   ray start --address="$HEAD_IP:6379" --node-ip-address="$VLLM_HOST_IP"
#
# 为什么必须在 ray start 之前：ray worker 进程只继承 ray start 时的环境。
# GLOO_ 前缀不在 vLLM 的 ray 拷贝白名单里（白名单见启动日志的
# "Env var prefixes to copy"：DSA_/HF_/HUGGING_FACE_/LMCACHE_/NCCL_/UCX_/VLLM_），
# 漏了不报错，只会让 CPU 侧 gloo group 自己挑网卡（--net=host 下可能挑中 docker0）。

NET_IF="${NET_IF:-bond1}"
HEAD_IP="${HEAD_IP:-30.203.130.54}"   # node0 的 bond1 IP；换 head 时改这里

SELF_IP="$(ip -o -4 addr show dev "${NET_IF}" 2>/dev/null \
           | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [[ -z "${SELF_IP}" ]]; then
    echo "ERROR: 网卡 ${NET_IF} 上没有 IPv4 地址。" >&2
    echo "       ip -o -4 addr show   看实际网卡名，再 NET_IF=<名字> source env/net_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# ---- 控制面 / rendezvous：这三者必须落在同一张网卡上 ----
# VLLM_HOST_IP 决定 vLLM 给 placement group bundle 0 打的 node:<ip> 亲和标记，
# 必须与本节点 `ray start --node-ip-address` 逐字相同，否则 PG 永远 pending。
export VLLM_HOST_IP="${SELF_IP}"
export HEAD_IP
export NCCL_SOCKET_IFNAME="${NET_IF}"
export GLOO_SOCKET_IFNAME="${NET_IF}"
export UCX_NET_DEVICES="${NET_IF}"

# ---- RDMA 数据面（RoCEv2 over LAG）----
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_IB_TC=160
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_CUDA_SUPPORT=1

# ---- 其余照抄已验证配置 ----
export NCCL_P2P_DISABLE=0
export NCCL_PXN_DISABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # smoke 阶段改 INFO，日志里确认 NET/IB

# ---- 让 GLOO_ 也跟着 ray actor 走（DSA_ 由 run_first5_32k_vllm.sh 追加）----
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="GLOO_"

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"

# ---- RDMA 需要的 rlimit。privileged 容器有 CAP_SYS_RESOURCE，可自己抬 ----
ulimit -l unlimited 2>/dev/null

echo "net_env: ${NET_IF}=${SELF_IP}  head=${HEAD_IP}  memlock=$(ulimit -l)"
echo "         HCA=mlx5_bond_1..8  GID_INDEX=3 (RoCEv2)  SL=3 TC=160"

# 注意：vLLM 会把 VLLM_HOST_IP 从 driver 拷到远端 ray worker（见启动日志的
# "Copying the following environment variables"），所以 worker 上这个变量的
# 真实用处只有一个 —— 作为本节点 ray start --node-ip-address 的取值。
