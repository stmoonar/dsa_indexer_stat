#!/bin/bash
# run_first5_32k.sh — 截断模型只跑前 5 层 (layer 0-4) × 32K 上下文 单样本测试
#
# 模型截断: --json-model-override-args '{"num_hidden_layers": 5}'
#   只构建、只加载前 5 层（3 dense + 2 MoE ≈ 30GB fp8），layer 5-60 及
#   MTP 层的权重在 deepseek_weight_loader 中被跳过，完全不 forward。
#   单机 TP=8 即可，不需要跨节点。
#   ⚠ 截断模型的 decode 输出是无意义 token：
#     - prefill 段 layer 0-4 的 k^I 与全模型完全一致（前 5 层不受后层影响）
#     - decode 段的 token 轨迹与全模型不同 → 本 run 只用于打通链路，
#       统计数字不进正式汇报
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
#
# 备选（若在线覆盖遇到问题）：离线截断出 5 层模型再正常启动
#   python -m sglang.srt.debug_utils.model_truncator \
#       --input /data1/models/DeepSeek-V3.2 \
#       --output /data1/models/DeepSeek-V3.2-5layer --keep-num-layers 5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# 可选 conda 环境（默认不激活；CONDA_ENV=<name> 启用）
CONDA_ENV="${CONDA_ENV-}"
if [[ -n "${CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
        echo "Activated conda env: ${CONDA_ENV} ($(which python))"
    else
        echo "WARNING: conda not on PATH; using current python ($(which python))."
    fi
fi

MODEL_PATH="${MODEL_PATH:-/data1/models/DeepSeek-V3.2}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/lmcache-agentic-traces/data}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_LAYERS="${NUM_LAYERS:-5}"          # 模型截断到前 N 层，后面的层不构建、不 forward
TP="${TP:-8}"                          # 5 层 ≈ 30GB fp8，单机 8 卡足够
PORT="${PORT:-30000}"
CAPTURE_LAYERS="${CAPTURE_LAYERS:-all}"  # 模型已截断，采全部现存层即 0-4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-runs/first5_32k_${TIMESTAMP}}"
PROMPT_FILE="${PROMPT_FILE:-prompts/agentic_32k.txt}"

mkdir -p "${OUTPUT_DIR}" logs prompts
LOG_FILE="logs/capture_first5_32k_${TIMESTAMP}.log"

echo "=== First-${NUM_LAYERS}-layers (truncated model) 32K capture test ==="
echo "Model:   ${MODEL_PATH} (truncated to ${NUM_LAYERS} layers)"
echo "Dataset: ${DATASET_DIR}"
echo "TP:      ${TP}"
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
export DSA_CAPTURE_NUM_LAYERS="${NUM_LAYERS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MAX_TOTAL_TOKENS=$(( CONTEXT_LENGTH + MAX_NEW_TOKENS + 512 ))

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --json-model-override-args "{\"num_hidden_layers\": ${NUM_LAYERS}}" \
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
echo "k_I buffers:      ${N_K} (expect ${NUM_LAYERS})"
echo "step-layer dumps: ${N_STEP} (expect ~$(( MAX_NEW_TOKENS * NUM_LAYERS )))"

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
cat > "${OUTPUT_DIR}/run_meta.json" <<EOF
{
  "git_hash": "${GIT_HASH}",
  "model_path": "${MODEL_PATH}",
  "num_hidden_layers_override": ${NUM_LAYERS},
  "model_truncated": true,
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
cp "${LOG_FILE}" "${OUTPUT_DIR}/" 2>/dev/null || true

# 打包结果（不完整也打包，便于拿下来排查）
echo ""
echo "Packaging results..."
python - "${OUTPUT_DIR}" <<'PY'
import os, sys, shutil
out = sys.argv[1].rstrip("/")
zip_path = shutil.make_archive(
    out, "zip",
    root_dir=os.path.dirname(out) or ".",
    base_dir=os.path.basename(out),
)
print(f"Zipped: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
PY

if [[ "${N_K}" -eq "${NUM_LAYERS}" && "${N_STEP}" -gt 0 ]]; then
    echo "OK: capture output complete → ${OUTPUT_DIR}"
    echo "    下载: ${OUTPUT_DIR}.zip"
else
    echo "WARNING: output incomplete (zip 仍已生成，便于排查), check ${LOG_FILE}"
    exit 1
fi
