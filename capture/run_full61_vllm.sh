#!/bin/bash
# run_full61_vllm.sh — 全 61 层 × 2 节点 TP=16：capture → 金丝雀 → 层间 top-k 相似性
#
# 为什么是 TP=16 而不是 TP=8 + PP=2（这条不能改）：
#   capturer 只在 global rank 0 落盘（约束 9：q/k/w 都是 Replicated，rank0 即全量）。
#   PP 下 layer 31-60 活在 rank 8-15 上，rank!=0 → 后半段【静默不 dump】，
#   而层间相似性恰恰要的就是深层。TP=16 每个 rank 都跑全部 61 层，rank0 一份即完整。
#   ray 的 worker 排序保证 rank0 落在 driver 节点（本节点），dump 就写在本地盘。
#
# 前置（两个节点都要，缺一个都会在加载 685GB 权重之后才炸）：
#   1) 本仓库与 vLLM 在两节点【同一绝对路径】下，且都已 apply patch
#      bash capture/apply_vllm_patch.sh   # 各节点各跑一次
#   2) ray 集群（先 export PYTHONPATH=<repo> 再 start，worker 进程继承它）：
#      head:   export PYTHONPATH=$(pwd); ray start --head --port=6379
#      worker: export PYTHONPATH=/same/path/to/repo; ray start --address=<head_ip>:6379
#      检查:   ray status   # 必须看到 16 GPU
#   3) tests/ 全绿（约束 11）
#
# 金丝雀（约束 12 的精神）：全模型 run 的 layer 0-4 prefill 段必须与已有的
#   截断 run 一致。REF_RUN 必须与本 run 用【同一条 prompt】，所以默认
#   PROMPT_PREFIX 指向 suite A 的 manifest 且 MAX_SEQS=1（不重挑样本）。
#   TP 不同（8 vs 16）→ all-reduce 求和顺序不同 → 逐位相等不成立，
#   容差用 analysis/calibrate_tolerance.py 标出来的 DSA_TOL，不要猜。
#
# 用法：
#   REF_RUN=runs/A_32k_tp8_XXX/seq00 DSA_TOL=2e-3 \
#       VLLM_SRC=/workspace/vllm bash capture/run_full61_vllm.sh
#   CONTEXT_LENGTH=131072 CONCAT=1 bash capture/run_full61_vllm.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

TP="${TP:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
CTX_TAG=$(( CONTEXT_LENGTH / 1024 ))K
REF_RUN="${REF_RUN:-}"
DSA_TOL="${DSA_TOL:-0}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-full61_${CTX_TAG}_tp${TP}}"
OUTPUT_ROOT="runs/${RUN_TAG}_${STAMP}"

echo "=== [0/4] ray 集群预检 ==="
if ! command -v ray >/dev/null 2>&1; then
    echo "ERROR: ray 不在 PATH 上。跨节点 TP=${TP} 必须走 ray。"; exit 1
fi
RAY_GPUS=$(ray status 2>/dev/null \
    | grep -oE '[0-9]+\.?[0-9]*/[0-9]+\.?[0-9]* GPU' \
    | head -1 | sed -E 's|.*/([0-9]+).*|\1|')
if [[ -z "${RAY_GPUS}" ]]; then
    echo "ERROR: ray status 读不到 GPU 数——集群没起或本节点没连上。"; exit 1
fi
echo "ray 集群 GPU 数: ${RAY_GPUS}（需要 >= ${TP}）"
if [[ "${RAY_GPUS}" -lt "${TP}" ]]; then
    echo "ERROR: ray 集群只有 ${RAY_GPUS} 张卡，不够 TP=${TP}。"
    echo "       worker 节点是否执行了 ray start --address=<head_ip>:6379 ?"
    exit 1
fi

# ---- 1) capture：不截断，全 61 层 ----
# CAPTURE_LATENT=0：想法 2 的 c_latent 只需一个 run（截断 run 已有），
#   61 层再存一份要多 2.3GB/序列，且对层间相似性毫无用处。
# MAX_SEQS=1：每条序列一个独立进程 = 重新加载一次 685GB 权重。
#   先跑通一条；要序列间方差再单独加跑。
# MAX_NEW_TOKENS=32：全模型的 decode 轨迹是真实的（截断 run 的才是乱码），
#   这是本项目第一份可用的 decode 段数据，多留几步几乎不要钱（~2.5MB/步）。
echo ""
echo "=== [1/4] capture (61 层, TP=${TP}, ray) ==="
NUM_LAYERS=0 MODEL_LAYERS=61 TP="${TP}" DIST_BACKEND=ray \
    CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
    CAPTURE_LAYERS=all CAPTURE_LATENT=0 \
    MAX_SEQS="${MAX_SEQS:-1}" N_SAMPLES="${N_SAMPLES:-1}" \
    CONCAT="${CONCAT:-0}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}" SLIM_KEEP_DECODE=1 \
    PROMPT_PREFIX="${PROMPT_PREFIX:-prompts/agentic_${CTX_TAG}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" RUN_TAG="${RUN_TAG}" \
    SLIM_PROFILE="${SLIM_PROFILE:-geometry}" \
    bash capture/run_first5_32k_vllm.sh
CAP_RC=$?
if [[ "${CAP_RC}" -ne 0 ]]; then
    echo "ERROR: capture 失败 (rc=${CAP_RC})，后续步骤跳过。"; exit "${CAP_RC}"
fi

NEW_SEQ=$(ls -d "${OUTPUT_ROOT}"/seq* 2>/dev/null | head -1)
NEW_SEQ="${NEW_SEQ:-${OUTPUT_ROOT}}"

# ---- 2) 金丝雀：layer 0-4 必须与截断 run 一致 ----
echo ""
echo "=== [2/4] 全模型金丝雀 (layer 0-4 vs ${REF_RUN:-<未指定>}) ==="
if [[ -z "${REF_RUN}" ]]; then
    echo "SKIP: 未设 REF_RUN。强烈建议指向同一条 prompt 的截断 run 的 seqNN 目录，"
    echo "      否则 hook 接错/层号错位/w 少乘 scale 这几类致命错误要到分析阶段才暴露。"
else
    DSA_REF_RUN="${REF_RUN}" DSA_NEW_RUN="${NEW_SEQ}" DSA_TOL="${DSA_TOL}" \
        python -m pytest tests/test_full_model_consistency.py -q
    if [[ $? -ne 0 ]]; then
        echo "ERROR: 金丝雀不过 —— 这个 run 的数据不可用，先查 capture 再分析。"
        exit 1
    fi
fi

# ---- 3) 层间 top-k 相似性（token 级）----
echo ""
echo "=== [3/4] 层间 top-k 相似性 ==="
python -m analysis.cross_layer "${OUTPUT_ROOT}" \
    --top-k "${TOP_K:-2048}" --out "${OUTPUT_ROOT}/cross_layer.json"

# ---- 4) 常规四项测量（层数多，默认限样本量）----
echo ""
echo "=== [4/4] m1-m6 ==="
python -m analysis.analyze_run "${OUTPUT_ROOT}" \
    --max-per-bucket "${MAX_PER_BUCKET:-2}" --kmeans-block-sizes ""

echo ""
echo "完成。拿走这三个文件即可，原始 dump 不必下载："
echo "  ${OUTPUT_ROOT}/cross_layer.json"
echo "  ${OUTPUT_ROOT}/stats.json"
echo "  ${OUTPUT_ROOT}/report.md"
