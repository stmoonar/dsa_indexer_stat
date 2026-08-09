#!/bin/bash
# run_suite.sh — 前 5 层环境的"一次跑完"套件
#
# 三个 run，都用截断到 5 层的模型（单机 8 卡足够）：
#   A) 32K × TP=8 × 3 条序列   —— 主数据。多序列才有序列间方差，
#                                 也才第一次跑通 (task, layer) 分组的代码路径
#   B) 32K × TP=4 × 1 条序列   —— 只为标定金丝雀容差 DSA_TOL。
#                                 全模型 run 会用 TP=16，求和顺序不同，
#                                 逐位一致必然不成立；容差必须实测而非猜
#   C) 128K × TP=8 × 1 条序列  —— 主配置是 128K。5 层模型的 KV 极小，
#                                 128K 直接跑得动，不必从 32K 外推
#
# B 复用 A 的第 0 条 prompt（必须同一个 prompt 才能标定），
# 所以 B 不重新准备 prompt。
#
# 用法：
#   VLLM_SRC=/workspace/vllm bash capture/run_suite.sh
#   SKIP_128K=1 bash capture/run_suite.sh     # 只跑 A + B
#
# 单个 run 失败不会中断后面的（各 run 独立），最后统一汇报。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${REPO_ROOT}/capture/run_first5_32k_vllm.sh"
STAMP=$(date +%Y%m%d_%H%M%S)
SUITE_LOG="logs/suite_${STAMP}.log"
mkdir -p logs runs

N_SAMPLES_MAIN="${N_SAMPLES_MAIN:-3}"
RESULTS=()

run_one() {
    local tag="$1"; shift
    echo ""
    echo "############################################################"
    echo "# ${tag}"
    echo "############################################################"
    if env "$@" RUN_TAG="${tag}" bash "${RUNNER}" 2>&1 | tee -a "${SUITE_LOG}"; then
        RESULTS+=("OK   ${tag}")
    else
        RESULTS+=("FAIL ${tag}")
    fi
}

# ---- A) 32K × TP=8 × N 条序列（主数据）----
run_one "A_32k_tp8" \
    CONTEXT_LENGTH=32768 TP=8 N_SAMPLES="${N_SAMPLES_MAIN}" \
    PROMPT_PREFIX=prompts/agentic_32K

# ---- B) 32K × TP=4 × 1 条（容差标定，复用 A 的 seq00）----
# 用同一个 PROMPT_PREFIX + 已存在的 manifest -> 不会重新挑样本；
# N_SAMPLES=1 时只跑 manifest 里的第一条，正好是 A 的 seq00。
run_one "B_32k_tp4_calib" \
    CONTEXT_LENGTH=32768 TP=4 N_SAMPLES=1 \
    PROMPT_PREFIX=prompts/agentic_32K_calib \
    PREFILL_TAIL=8 PREFILL_BUCKETS=16384 PREFILL_RUN=8 \
    CAPTURE_LATENT=0

# ---- C) 128K × TP=8 × 1 条（主配置的真实几何）----
if [[ "${SKIP_128K:-0}" != "1" ]]; then
    run_one "C_128k_tp8" \
        CONTEXT_LENGTH=131072 TP=8 N_SAMPLES=1 CONCAT=1 \
        PROMPT_PREFIX=prompts/agentic_128K
fi

echo ""
echo "############################################################"
echo "# Suite summary"
echo "############################################################"
printf '%s\n' "${RESULTS[@]}"
echo ""
echo "产出的 zip："
ls -1sh runs/*_${STAMP%%_*}*.zip 2>/dev/null || ls -1sh runs/*.zip 2>/dev/null | tail -5
echo ""
echo "下一步："
echo "  1) 标定容差（A 与 B 的 seq00 目录）:"
echo "     python -m analysis.calibrate_tolerance \\"
echo "         runs/A_32k_tp8_*/seq00 runs/B_32k_tp4_calib_*/seq00"
echo "  2) 出数（每条序列各跑一次，--source prefill）:"
echo "     for d in runs/A_32k_tp8_*/seq*; do \\"
echo "         python -m analysis.run_measurements \"\$d\" --source prefill; done"
