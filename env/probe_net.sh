#!/bin/bash
# probe_net.sh — 只读探测：一次性拿全选定 VLLM_HOST_IP / NCCL_SOCKET_IFNAME /
#                GLOO_SOCKET_IFNAME / NCCL_IB_HCA 所需的全部事实。
#
# 不改任何状态，不起 ray，不动网卡。两个节点各跑一次，输出一起看才能定值
# （VLLM_HOST_IP 每节点不同，另外两个通常两节点相同）。
#
# 用法:
#   bash env/probe_net.sh
#   PEER=30.205.35.67 bash env/probe_net.sh    # 已知对端 IP，多做一组连通性判定

set +e
PEER="${PEER:-}"
sec() { printf '\n\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

sec "0. 身份"
echo "hostname : $(hostname)"
echo "date     : $(date -Is)"
[[ -f /.dockerenv ]] && echo "容器     : 是 (/.dockerenv 存在)" || echo "容器     : 否"
echo "netns    : $(readlink /proc/self/ns/net)   # --net=host 时应与宿主一致"

sec "1. ray 集群（决定 VLLM_HOST_IP 的合法取值集合）"
if have ray; then
    ray status 2>&1 | head -40
    python - <<'PY' 2>&1 | grep -v "^20[0-9][0-9]-" 
import sys
try:
    import ray
except ImportError:
    print("ray python 包缺失"); sys.exit(0)
try:
    ray.init(address="auto", log_to_driver=False, ignore_reinit_error=True)
except Exception as e:
    print("!! 连不上 ray 集群:", type(e).__name__, e); sys.exit(0)
me = ray.get_runtime_context().get_node_id()
print("\n本机  NodeID       NodeManagerAddress   Alive  GPU")
for n in ray.nodes():
    mark = ">>>>" if n["NodeID"] == me else "    "
    print(f"{mark}  {n['NodeID'][:10]:<11}  {n['NodeManagerAddress']:<19}  "
          f"{str(n['Alive']):<5}  {n['Resources'].get('GPU', 0)}")
print("\n集群里实际存在的 node: 资源  —— VLLM_HOST_IP 只能从这里面选:")
for k in sorted(ray.cluster_resources()):
    if k.startswith("node:"):
        print("     ", k)
r, a = ray.cluster_resources(), ray.available_resources()
print(f"\nGPU 总数={r.get('GPU',0)}  当前可用={a.get('GPU',0)}  (可用 < 16 时 PG 也会 pending)")
PY
else
    echo "!! ray 不在 PATH 上"
fi

sec "2. 本机网卡与 IP（VLLM_HOST_IP / *_SOCKET_IFNAME 从这里选）"
ip -o -4 addr show 2>/dev/null | awk '{printf "  %-14s %s\n", $2, $4}'
echo ""
echo "-- 默认路由 --"
ip route show default 2>/dev/null
echo ""
echo "-- 需要排除的接口（NCCL 自动选中它们会跨节点 hang）--"
ip -o link show 2>/dev/null | awk -F': ' '{print $2}' \
    | grep -E '^(lo|docker|veth|br-|virbr|cni|flannel|tun)' | sed 's/^/  /'

sec "3. 到对端的可达性"
if [[ -z "${PEER}" ]]; then
    echo "未设 PEER。在另一节点跑完本脚本后，取它 [2] 里的 IP 回填："
    echo "    PEER=<对端IP> bash env/probe_net.sh"
else
    echo "-- ip route get ${PEER}  (dev = NCCL_SOCKET_IFNAME, src = VLLM_HOST_IP) --"
    ip route get "${PEER}" 2>&1 | sed 's/^/  /'
    echo ""
    echo "-- ping --"
    ping -c 2 -W 2 "${PEER}" 2>&1 | tail -3 | sed 's/^/  /'
    echo ""
    echo "-- ray GCS 端口 6379 --"
    if have nc; then nc -zv -w 3 "${PEER}" 6379 2>&1 | sed 's/^/  /'
    else timeout 3 bash -c "echo > /dev/tcp/${PEER}/6379" 2>&1 \
         && echo "  6379 open" || echo "  6379 closed/unreachable"; fi
fi

sec "4. InfiniBand / RoCE 设备（决定 NCCL_IB_HCA）"
if have ibstat; then
    echo "-- 设备列表 --"; ibstat -l 2>&1 | sed 's/^/  /'
    echo ""
    echo "-- 逐端口状态：只有 State: Active + Physical state: LinkUp 的能写进 NCCL_IB_HCA --"
    ibstat 2>&1 | grep -E "^CA '|^[[:space:]]+Port [0-9]+:|State:|Physical state:|Rate:|Link layer:"
else
    echo "!! ibstat 不存在 —— 容器里没装 MLNX OFED 用户态工具，或没挂 /dev/infiniband"
fi
echo ""
if have ibdev2netdev; then
    echo "-- HCA <-> 网卡映射（Link layer 是 Ethernet 即 RoCE，那时才与 IP 有关）--"
    ibdev2netdev -v 2>&1 | sed 's/^/  /'
else
    echo "-- ibdev2netdev 不存在，用 sysfs 代替 --"
    for d in /sys/class/infiniband/*; do
        [[ -e "$d" ]] || continue
        n=$(basename "$d")
        for p in "$d"/ports/*; do
            [[ -e "$p" ]] || continue
            echo "  $n port $(basename "$p"): state=$(cat "$p"/state 2>/dev/null) " \
                 "phys=$(cat "$p"/phys_state 2>/dev/null) " \
                 "link=$(cat "$p"/link_layer 2>/dev/null) " \
                 "rate=$(cat "$p"/rate 2>/dev/null)"
        done
    done
fi

sec "5. GPU 与 HCA 的亲和（NCCL_IB_HCA 选哪几张看这张表的 mlx5_* 列）"
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
    echo ""
    nvidia-smi topo -m 2>&1 | head -30
else
    echo "!! nvidia-smi 不存在"
fi

sec "6. rlimit（memlock 不是 unlimited 会让 IB 静默退化成 Socket）"
echo "  memlock (ulimit -l): $(ulimit -l)      <- 必须 unlimited"
echo "  stack   (ulimit -s): $(ulimit -s)"
echo "  nofile  (ulimit -n): $(ulimit -n)"
grep -E "Max locked memory|Max stack size|Max open files" /proc/self/limits 2>/dev/null | sed 's/^/  /'

sec "7. 当前环境变量现状"
for v in VLLM_HOST_IP NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA; do
    printf '  %-22s = %s\n' "$v" "${!v:-<未设置>}"
done
echo ""
echo "-- 其余相关变量 --"
env | grep -E "^(NCCL|VLLM|GLOO|RAY|UCX|DSA|CUDA_VISIBLE)" | sort | sed 's/^/  /'

sec "8. 版本（两节点必须完全一致）"
python - <<'PY' 2>&1 | sed 's/^/  /'
for mod, attr in (("torch","__version__"), ("ray","__version__"),
                  ("vllm","__version__"), ("triton","__version__")):
    try:
        m = __import__(mod); print(f"{mod:8} {getattr(m, attr, '?')}")
    except Exception as e:
        print(f"{mod:8} !! {type(e).__name__}: {e}")
try:
    import torch
    print(f"{'nccl':8} {'.'.join(map(str, torch.cuda.nccl.version()))}")
    print(f"{'cuda':8} {torch.version.cuda}")
except Exception as e:
    print(f"{'nccl':8} !! {e}")
PY

printf '\n\n========== 探测结束 ==========\n'
echo "把 [1][2][3][4][6] 四段贴出来即可定值。另一节点也要跑一遍。"
