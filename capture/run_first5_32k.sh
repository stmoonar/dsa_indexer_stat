#!/bin/bash
# run_first5_32k.sh — 前 5 层 (layer 0-4) × 32K 上下文 单样本测试运行
#
# 数据: lmcache-agentic-traces (parquet), 取一条 ~32K token 的 trace
# 约束 7: --disable-cuda-graph, 关 overlap scheduler, bs=1
# 约束 12: 这是 smoke 之后、烧全量之前的中间验证档位
#
# 前置：
#   1. tests/ 全绿（约束 11）
#   2. capture/patch_sglang/indexer_dump.diff 已应用到 SGLang 源码
#      （capturer 通过 DSA_CAPTURE_* 环境变量在 worker 进程内惰性初始化，
#       无须其他集成代码）
#   3. pip install pyarrow transformers（prepare_prompt 依赖）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/data1/models/DeepSeek-V3.2}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/lmcache-agentic-traces/data}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TP="${TP:-16}"
PORT="${PORT:-30000}"
CAPTURE_LAYERS="${CAPTURE_LAYERS:-0-4}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-runs/first5_32k_${TIMESTAMP}}"
PROMPT_FILE="${PROMPT_FILE:-prompts/agentic_32k.txt}"

mkdir -p "${OUTPUT_DIR}" logs prompts
LOG_FILE="logs/capture_first5_32k_${TIMESTAMP}.log"

echo "=== First-5-layers 32K capture test ==="
echo "Model:   ${MODEL_PATH}"
echo "Dataset: ${DATASET_DIR}"
echo "Layers:  ${CAPTURE_LAYERS}"
echo "Output:  ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
echo ""

# Step 0: 准备 prompt（已存在则跳过；FORCE_PREPARE=1 强制重选）
if [[ ! -f "${PROMPT_FILE}" || "${FORCE_PREPARE:-0}" == "1" ]]; then
    echo "[1/4] Preparing 32K prompt from parquet dataset..."
    python -m capture.prepare_prompt \
        --dataset-dir "${DATASET_DIR}" \
        --tokenizer "${MODEL_PATH}" \
        --context-length "${CONTEXT_LENGTH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --output "${PROMPT_FILE}"
else
    echo "[1/4] Prompt exists: ${PROMPT_FILE} (FORCE_PREPARE=1 to regenerate)"
fi

# Step 1: 启动 SGLang server
#   DSA_CAPTURE_* 由 worker 进程继承 → capturer 惰性初始化（rank0 dump）
echo "[2/4] Launching SGLang server..."
export DSA_CAPTURE_OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
export DSA_CAPTURE_LAYERS="${CAPTURE_LAYERS}"
export DSA_CAPTURE_NUM_LAYERS=61
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MAX_TOTAL_TOKENS=$(( CONTEXT_LENGTH + MAX_NEW_TOKENS + 512 ))

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP}" \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --max-running-requests 1 \
    --schedule-policy fcfs \
    --port "${PORT}" \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() { kill -TERM "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server (model load can take several minutes)..."
READY=0
for i in $(seq 1 1800); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server died, see ${LOG_FILE}"; exit 1
    fi
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1; echo "Server ready after ${i}s"; break
    fi
    sleep 1
done
[[ "${READY}" == "1" ]] || { echo "ERROR: server not ready in 30min"; exit 1; }

# Step 2: 发送单条请求 (bs=1)
echo "[3/4] Sending capture request..."
python -m capture.send_request \
    --prompt-file "${PROMPT_FILE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --port "${PORT}"

# Step 3: 优雅关闭 server → 触发 rank0 worker 的 atexit save
echo "[4/4] Shutting down server (triggers dump save)..."
kill -TERM "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
trap - EXIT
sleep 5

# Step 4: 校验输出 + 写 run 元数据（约束 13）
echo ""
echo "=== Verifying dump output ==="
N_K=$(ls "${OUTPUT_DIR}"/k_I_layer*.npy 2>/dev/null | wc -l)
N_STEP=$(ls "${OUTPUT_DIR}"/step*_layer*.npz 2>/dev/null | wc -l)
echo "k_I buffers:      ${N_K} (expect 5: layers 000-004)"
echo "step-layer dumps: ${N_STEP} (expect ~$(( MAX_NEW_TOKENS * 5 )))"

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
cat > "${OUTPUT_DIR}/run_meta.json" <<EOF
{
  "git_hash": "${GIT_HASH}",
  "model_path": "${MODEL_PATH}",
  "dataset_dir": "${DATASET_DIR}",
  "prompt_file": "${PROMPT_FILE}",
  "context_length": ${CONTEXT_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "capture_layers": "${CAPTURE_LAYERS}",
  "tp": ${TP},
  "max_total_tokens": ${MAX_TOTAL_TOKENS},
  "timestamp": "${TIMESTAMP}"
}
EOF
cp "${PROMPT_FILE}.meta.json" "${OUTPUT_DIR}/prompt_meta.json" 2>/dev/null || true

if [[ "${N_K}" -eq 5 && "${N_STEP}" -gt 0 ]]; then
    echo "OK: capture output looks complete → ${OUTPUT_DIR}"
else
    echo "WARNING: output incomplete, check ${LOG_FILE}"
    exit 1
fi
