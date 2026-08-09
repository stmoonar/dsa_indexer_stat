#!/bin/bash
# run_first5_32k_vllm.sh — vLLM 后端：截断模型只跑前 N 层，多序列 capture
#
# 模型截断: hf_overrides {"num_hidden_layers": 5}
#   只构建、只加载前 5 层（3 dense + 2 MoE ≈ 30GB fp8），后面的层不 forward。
#   跳层加载依赖 patch 中的 load_weights 修改（vLLM 原生会 KeyError）。
#   约束 7: enforce_eager（= 关 CUDA graph + torch.compile）, max_num_seqs=1
#
# 数据有效性（重要）：
#   prefill 段的 k^I / q^I / w 与全 61 层模型逐位一致 —— 可信。
#   decode 段是乱码 token 轨迹上的统计 —— 不可信，仅供 kernel 对齐抽查。
#   故 MAX_NEW_TOKENS 默认只留 8 步。
#
# 多序列：每条序列**单独起一个进程**（各自的 OUTPUT_DIR）。
#   capturer 的 k^I buffer 是进程内全局的，同进程连发两条请求会把
#   两条序列的 k^I 首尾相接，静默污染全部几何统计。5 层模型加载只要 ~40s，
#   用重启换取零污染风险是划算的。
#
# 前置：
#   1. tests/ 全绿（约束 11）
#   2. bash capture/apply_vllm_patch.sh   # 幂等：先恢复源码再打 patch
#   3. pip install pyarrow transformers（prepare_prompt 依赖）
#
# 用法：
#   VLLM_SRC=/workspace/vllm bash capture/run_first5_32k_vllm.sh
#   CONTEXT_LENGTH=131072 CONCAT=1 N_SAMPLES=1 bash capture/run_first5_32k_vllm.sh
#   TP=4 N_SAMPLES=1 RUN_TAG=tp4calib bash capture/run_first5_32k_vllm.sh
#   默认激活 conda env "vllm-td"（CONDA_ENV=xxx 换环境，CONDA_ENV="" 跳过）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"   # decode 统计不可用，只留少量步做 kernel 抽查
NUM_LAYERS="${NUM_LAYERS:-5}"
TP="${TP:-8}"
N_SAMPLES="${N_SAMPLES:-3}"             # 序列间方差需要 >1
CONCAT="${CONCAT:-0}"                   # 128K 时置 1（单条 trace 不够长）
CAPTURE_LAYERS="${CAPTURE_LAYERS:-all}"

CTX_TAG=$(( CONTEXT_LENGTH / 1024 ))K
RUN_TAG="${RUN_TAG:-first${NUM_LAYERS}_${CTX_TAG}_tp${TP}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/${RUN_TAG}_${TIMESTAMP}}"
PROMPT_PREFIX="${PROMPT_PREFIX:-prompts/agentic_${CTX_TAG}}"

mkdir -p "${OUTPUT_ROOT}" logs prompts
LOG_FILE="logs/capture_${RUN_TAG}_${TIMESTAMP}.log"

echo "=== [vLLM] first-${NUM_LAYERS}-layers capture ==="
echo "Model:    ${MODEL_PATH} (truncated to ${NUM_LAYERS} layers)"
echo "vLLM:     ${VLLM_SRC}"
echo "Context:  ${CONTEXT_LENGTH}   TP: ${TP}   sequences: ${N_SAMPLES}"
echo "Output:   ${OUTPUT_ROOT}"
echo "Log:      ${LOG_FILE}"
echo ""

# Step 0: 打 patch（幂等：内部先恢复再应用）
echo "[1/4] Applying vLLM capture patch..."
VLLM_SRC="${VLLM_SRC}" bash "${REPO_ROOT}/capture/apply_vllm_patch.sh" 2>&1 \
    | tee -a "${LOG_FILE}"

# Step 1: 准备 prompt（manifest 已存在则跳过；FORCE_PREPARE=1 强制重选）
MANIFEST="${PROMPT_PREFIX}_manifest.json"
if [[ ! -f "${MANIFEST}" || "${FORCE_PREPARE:-0}" == "1" ]]; then
    echo "[2/4] Preparing ${N_SAMPLES} prompt(s) from parquet dataset..."
    CONCAT_FLAG=()
    [[ "${CONCAT}" == "1" ]] && CONCAT_FLAG+=(--concat)
    python -m capture.prepare_prompt \
        --dataset-dir "${DATASET_DIR}" \
        --tokenizer "${MODEL_PATH}" \
        --context-length "${CONTEXT_LENGTH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --n-samples "${N_SAMPLES}" \
        --out-prefix "${PROMPT_PREFIX}" \
        ${CONCAT_FLAG[@]+"${CONCAT_FLAG[@]}"} 2>&1 | tee -a "${LOG_FILE}"
else
    echo "[2/4] Manifest exists: ${MANIFEST} (FORCE_PREPARE=1 to regenerate)"
fi

# 通用 capture 环境变量（每条序列只改 OUTPUT_DIR）
export DSA_CAPTURE_LAYERS="${CAPTURE_LAYERS}"
export DSA_CAPTURE_NUM_LAYERS="${NUM_LAYERS}"
export DSA_CAPTURE_SAVE_EVERY="${DSA_CAPTURE_SAVE_EVERY:-64}"
# prefill 真实 query 采样：尾窗 + 分桶【连续段】
# 必须是连续段：warm 集 / τ / churn 都要求"同层前一个采样位置"，
# 孤立位置在配对时会被丢弃，前缀长度曲线会静默塌缩成单点。
export DSA_CAPTURE_PREFILL_TAIL="${PREFILL_TAIL:-32}"
export DSA_CAPTURE_PREFILL_RUN="${PREFILL_RUN:-32}"
if [[ -z "${PREFILL_BUCKETS:-}" ]]; then
    # 按上下文长度自动铺锚点：约 1/16, 1/4, 1/2, 3/4, 结尾前
    PREFILL_BUCKETS="$(( CONTEXT_LENGTH / 16 )),$(( CONTEXT_LENGTH / 4 )),"
    PREFILL_BUCKETS+="$(( CONTEXT_LENGTH / 2 )),$(( CONTEXT_LENGTH * 3 / 4 )),"
    PREFILL_BUCKETS+="$(( CONTEXT_LENGTH - 1024 ))"
fi
export DSA_CAPTURE_PREFILL_BUCKETS="${PREFILL_BUCKETS}"
# MLA latent（想法 2：k^I 能否由已存的 c_s 线性重建）
export DSA_CAPTURE_LATENT="${CAPTURE_LATENT:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MAX_MODEL_LEN=$(( CONTEXT_LENGTH + MAX_NEW_TOKENS + 512 ))
EXTRA_ARGS=()
[[ -n "${MAX_BATCHED_TOKENS:-}" ]] && EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS}")
[[ -n "${GPU_MEM_UTIL:-}" ]] && EXTRA_ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")

# Step 2: 逐条序列跑（每条一个独立进程）
echo "[3/4] Running vLLM offline generation, one process per sequence..."
# MAX_SEQS 限制实际要跑的序列条数（manifest 里可能有更多）。
# 容差标定必须复用主 run 的同一条 prompt，故用 MAX_SEQS=1 指向同一个 manifest，
# 而不是另起一个 PROMPT_PREFIX——后者会重新挑样本，两个 run 的输入就不同了。
mapfile -t SEQS < <(python -c "
import json, os
m = json.load(open('${MANIFEST}'))
limit = int(os.environ.get('MAX_SEQS') or 0)
seqs = m['sequences'][:limit] if limit > 0 else m['sequences']
for s in seqs:
    print(s['name'], s['path'], s['n_tokens'], sep='\t')
")
echo "Sequences to run: ${#SEQS[@]}"
if [[ "${#SEQS[@]}" -eq 0 ]]; then
    echo "ERROR: no sequences in ${MANIFEST}"; exit 1
fi

for row in "${SEQS[@]}"; do
    NAME=$(cut -f1 <<< "${row}")
    PFILE=$(cut -f2 <<< "${row}")
    NTOK=$(cut -f3 <<< "${row}")
    SEQ_DIR="${OUTPUT_ROOT}/${NAME}"
    mkdir -p "${SEQ_DIR}"
    echo ""
    echo "--- ${NAME}: ${PFILE} (${NTOK} tokens) -> ${SEQ_DIR} ---"
    export DSA_CAPTURE_OUTPUT_DIR="${REPO_ROOT}/${SEQ_DIR}"
    python -m capture.vllm_generate \
        --model "${MODEL_PATH}" \
        --prompt-file "${PFILE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-layers "${NUM_LAYERS}" \
        --tp "${TP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        2>&1 | tee -a "${LOG_FILE}"
    sleep 5   # 等 worker 退出并完成最终落盘
done

# Step 3: 校验 + 元数据 + 打包
echo ""
echo "[4/4] Verifying output..."
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
VLLM_HASH=$(git -C "${VLLM_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)
cat > "${OUTPUT_ROOT}/run_meta.json" <<EOF
{
  "backend": "vllm",
  "git_hash": "${GIT_HASH}",
  "vllm_git_hash": "${VLLM_HASH}",
  "model_path": "${MODEL_PATH}",
  "num_hidden_layers_override": ${NUM_LAYERS},
  "model_truncated": true,
  "dataset_dir": "${DATASET_DIR}",
  "prompt_prefix": "${PROMPT_PREFIX}",
  "n_sequences": ${#SEQS[@]},
  "context_length": ${CONTEXT_LENGTH},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "capture_layers": "${CAPTURE_LAYERS}",
  "prefill_tail": ${DSA_CAPTURE_PREFILL_TAIL},
  "prefill_buckets": "${DSA_CAPTURE_PREFILL_BUCKETS}",
  "prefill_run": ${DSA_CAPTURE_PREFILL_RUN},
  "capture_latent": ${DSA_CAPTURE_LATENT},
  "tp": ${TP},
  "max_model_len": ${MAX_MODEL_LEN},
  "timestamp": "${TIMESTAMP}"
}
EOF
cp "${MANIFEST}" "${OUTPUT_ROOT}/prompt_manifest.json" 2>/dev/null || true
cp "${LOG_FILE}" "${OUTPUT_ROOT}/" 2>/dev/null || true

OK=1
for row in "${SEQS[@]}"; do
    NAME=$(cut -f1 <<< "${row}")
    D="${OUTPUT_ROOT}/${NAME}"
    N_K=$(ls "${D}"/k_I_layer*.npy 2>/dev/null | wc -l)
    N_PF=$(ls "${D}"/prefill_pos*_layer*.npz 2>/dev/null | wc -l)
    N_LAT=$(ls "${D}"/c_latent_layer*.npy 2>/dev/null | wc -l)
    N_STEP=$(ls "${D}"/step*_layer*.npz 2>/dev/null | wc -l)
    echo "  ${NAME}: k_I=${N_K}/${NUM_LAYERS}  prefill=${N_PF}  latent=${N_LAT}  decode=${N_STEP}"
    if [[ "${N_K}" -ne "${NUM_LAYERS}" || "${N_PF}" -eq 0 ]]; then
        OK=0
    fi
done

echo ""
echo "Packaging results..."
# 默认只打可下载的瘦身包：丢掉 decode dump 与 c_latent，k_I 降到 fp16
# （源头是 bf16，fp16 无损；实测测量数字逐位不变）。
# 原始目录本来就在服务器上，重分析在服务器跑即可；FULL_ZIP=1 才打全量包。
python -m analysis.slim_run "${OUTPUT_ROOT}" \
    --profile "${SLIM_PROFILE:-geometry}" 2>&1 | tail -6
if [[ "${FULL_ZIP:-0}" == "1" ]]; then
    python - "${OUTPUT_ROOT}" <<'PY'
import os, sys, shutil
out = sys.argv[1].rstrip("/")
z = shutil.make_archive(out, "zip", root_dir=os.path.dirname(out) or ".",
                        base_dir=os.path.basename(out))
print(f"Full zip: {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
PY
fi

if [[ "${OK}" == "1" ]]; then
    echo "OK: capture complete → ${OUTPUT_ROOT}"
    echo "    下载: ${OUTPUT_ROOT}_${SLIM_PROFILE:-geometry}.zip"
else
    echo "WARNING: output incomplete (包仍已生成，便于排查), check ${LOG_FILE}"
    exit 1
fi
