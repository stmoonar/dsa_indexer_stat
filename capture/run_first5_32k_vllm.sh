#!/bin/bash
# run_first5_32k_vllm.sh — vLLM 后端：截断模型只跑前 5 层 × 32K 上下文 单样本测试
#
# 与 run_first5_32k.sh（SGLang 版）等价的 vLLM 流程：
#   模型截断: hf_overrides {"num_hidden_layers": 5}
#     只构建、只加载前 5 层（3 dense + 2 MoE ≈ 30GB fp8），后面的层不 forward。
#     跳层加载依赖 patch 中的 load_weights 修改（vLLM 原生会 KeyError）。
#   约束 7: enforce_eager（= 关 CUDA graph + torch.compile）, max_num_seqs=1
#   ⚠ 截断模型 decode 输出是无意义 token：prefill 段 k^I 与全模型一致，
#     decode 轨迹不同 → 本 run 只用于打通链路，统计不进正式汇报
#
# 前置：
#   1. tests/ 全绿（约束 11）
#   2. bash capture/apply_vllm_patch.sh   # 幂等：先恢复源码再打 patch
#   3. pip install pyarrow transformers（prepare_prompt 依赖）
#
# 用法：
#   VLLM_SRC=/workspace/vllm bash capture/run_first5_32k_vllm.sh
#   默认激活 conda env "vllm-td"（CONDA_ENV=xxx 换环境，CONDA_ENV="" 跳过）
#   结束后自动把输出目录打包成 ${OUTPUT_DIR}.zip 便于下载分析

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# conda 环境（默认 vllm-td；CONDA_ENV="" 可跳过激活）
CONDA_ENV="${CONDA_ENV-vllm-td}"
if [[ -n "${CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
        echo "Activated conda env: ${CONDA_ENV} ($(which python))"
    else
        echo "WARNING: conda not on PATH; using current python ($(which python))."
        echo "         Expected conda env: ${CONDA_ENV}"
    fi
fi

MODEL_PATH="${MODEL_PATH:-/data1/models/DeepSeek-V3.2}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/lmcache-agentic-traces/data}"
VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_LAYERS="${NUM_LAYERS:-5}"          # 截断到前 N 层
TP="${TP:-8}"                          # 5 层 ≈ 30GB fp8，单机 8 卡足够
CAPTURE_LAYERS="${CAPTURE_LAYERS:-all}"  # 模型已截断，现存层全采

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-runs/first5_32k_vllm_${TIMESTAMP}}"
PROMPT_FILE="${PROMPT_FILE:-prompts/agentic_32k.txt}"

mkdir -p "${OUTPUT_DIR}" logs prompts
LOG_FILE="logs/capture_first5_32k_vllm_${TIMESTAMP}.log"

echo "=== [vLLM] First-${NUM_LAYERS}-layers (truncated model) 32K capture test ==="
echo "Model:   ${MODEL_PATH} (truncated to ${NUM_LAYERS} layers)"
echo "vLLM:    ${VLLM_SRC}"
echo "Dataset: ${DATASET_DIR}"
echo "TP:      ${TP}"
echo "Output:  ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
echo ""

# Step 0: 打 patch（幂等：内部先恢复再应用）
echo "[1/4] Applying vLLM capture patch..."
VLLM_SRC="${VLLM_SRC}" bash "${REPO_ROOT}/capture/apply_vllm_patch.sh"

# Step 1: 准备 prompt（与 SGLang 版共用；已存在则跳过）
if [[ ! -f "${PROMPT_FILE}" || "${FORCE_PREPARE:-0}" == "1" ]]; then
    echo "[2/4] Preparing 32K prompt from parquet dataset..."
    python -m capture.prepare_prompt \
        --dataset-dir "${DATASET_DIR}" \
        --tokenizer "${MODEL_PATH}" \
        --context-length "${CONTEXT_LENGTH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --output "${PROMPT_FILE}"
else
    echo "[2/4] Prompt exists: ${PROMPT_FILE} (FORCE_PREPARE=1 to regenerate)"
fi

# Step 2: 离线推理 + capture
#   DSA_CAPTURE_* 被 vLLM worker 进程继承 → capturer 惰性初始化（rank0 dump）
echo "[3/4] Running vLLM offline generation..."
export DSA_CAPTURE_OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
export DSA_CAPTURE_LAYERS="${CAPTURE_LAYERS}"
export DSA_CAPTURE_NUM_LAYERS="${NUM_LAYERS}"
export DSA_CAPTURE_SAVE_EVERY="${DSA_CAPTURE_SAVE_EVERY:-64}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MAX_MODEL_LEN=$(( CONTEXT_LENGTH + MAX_NEW_TOKENS + 512 ))

# 无 DeepGEMM 时 vllm_generate 会自动降级 chunk/显存配置；
# 需要手动覆盖时设 MAX_BATCHED_TOKENS / GPU_MEM_UTIL
EXTRA_ARGS=()
[[ -n "${MAX_BATCHED_TOKENS:-}" ]] && EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS}")
[[ -n "${GPU_MEM_UTIL:-}" ]] && EXTRA_ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")

python -m capture.vllm_generate \
    --model "${MODEL_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-layers "${NUM_LAYERS}" \
    --tp "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "${LOG_FILE}"

sleep 5   # 等 worker 退出并完成最终落盘

# Step 3: 校验输出 + 写 run 元数据（约束 13）
echo ""
echo "[4/4] Verifying dump output..."
N_K=$(ls "${OUTPUT_DIR}"/k_I_layer*.npy 2>/dev/null | wc -l)
N_STEP=$(ls "${OUTPUT_DIR}"/step*_layer*.npz 2>/dev/null | wc -l)
echo "k_I buffers:      ${N_K} (expect ${NUM_LAYERS})"
echo "step-layer dumps: ${N_STEP} (expect ~$(( MAX_NEW_TOKENS * NUM_LAYERS )))"

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
VLLM_HASH=$(git -C "${VLLM_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)
cat > "${OUTPUT_DIR}/run_meta.json" <<EOF
{
  "backend": "vllm",
  "git_hash": "${GIT_HASH}",
  "vllm_git_hash": "${VLLM_HASH}",
  "model_path": "${MODEL_PATH}",
  "num_hidden_layers_override": ${NUM_LAYERS},
  "model_truncated": true,
  "dataset_dir": "${DATASET_DIR}",
  "prompt_file": "${PROMPT_FILE}",
  "context_length": ${CONTEXT_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "capture_layers": "${CAPTURE_LAYERS}",
  "tp": ${TP},
  "max_model_len": ${MAX_MODEL_LEN},
  "timestamp": "${TIMESTAMP}"
}
EOF
cp "${PROMPT_FILE}.meta.json" "${OUTPUT_DIR}/prompt_meta.json" 2>/dev/null || true
cp "${LOG_FILE}" "${OUTPUT_DIR}/" 2>/dev/null || true

# Step 4: 打包结果（不完整也打包，便于拿下来排查）
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
