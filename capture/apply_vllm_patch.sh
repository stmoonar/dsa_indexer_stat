#!/bin/bash
# apply_vllm_patch.sh — 把 indexer capture patch 打到 vLLM 源码
#
# 可重复执行：每次先把被 patch 的文件恢复到原始状态，再重新打补丁，
# 因此修改 patch 后重跑不会因"已应用"而失败。
#
# 恢复策略：
#   - VLLM_SRC 是 git 仓库 → git checkout HEAD -- <file>
#   - 不是 git 仓库 → 首次运行时备份 <file>.dsa_orig，之后从备份恢复
#
# 用法：
#   VLLM_SRC=/workspace/vllm bash capture/apply_vllm_patch.sh
#   （VLLM_SRC 缺省即 /workspace/vllm）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
PATCH_FILE="${PATCH_FILE:-${REPO_ROOT}/capture/patch_vllm/indexer_capture.diff}"

[[ -d "${VLLM_SRC}" ]] || { echo "ERROR: VLLM_SRC not found: ${VLLM_SRC}"; exit 1; }
[[ -f "${PATCH_FILE}" ]] || { echo "ERROR: patch not found: ${PATCH_FILE}"; exit 1; }

# vLLM 可能以 pip -e 安装（源码即 ${VLLM_SRC}/vllm）——patch 路径是
# a/vllm/... 形式，直接在 VLLM_SRC 根目录应用
cd "${VLLM_SRC}"

IS_GIT_REPO=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    IS_GIT_REPO=1
fi

# 从 patch 中解析目标文件列表
FILES=$(grep -E '^\+\+\+ b/' "${PATCH_FILE}" | sed 's|^+++ b/||')

echo "=== Restoring patched files ==="
for f in ${FILES}; do
    if [[ "${IS_GIT_REPO}" == "1" ]]; then
        git checkout HEAD -- "${f}"
        rm -f "${f}.dsa_orig"
        echo "  restored (git): ${f}"
    else
        if [[ -f "${f}.dsa_orig" ]]; then
            cp -f "${f}.dsa_orig" "${f}"
            echo "  restored (backup): ${f}"
        else
            cp -f "${f}" "${f}.dsa_orig"
            echo "  backed up: ${f} -> ${f}.dsa_orig"
        fi
    fi
done

echo "=== Applying patch ==="
if git apply --check "${PATCH_FILE}" 2>/dev/null; then
    git apply "${PATCH_FILE}"
elif command -v patch >/dev/null 2>&1; then
    patch -p1 --dry-run < "${PATCH_FILE}" >/dev/null
    patch -p1 < "${PATCH_FILE}"
else
    echo "ERROR: patch does not apply cleanly. vLLM 版本与 patch 不匹配？"
    echo "       对照 tmp/vllm（patch 的生成基线）检查 ${VLLM_SRC} 的版本。"
    exit 1
fi

for f in ${FILES}; do
    N=$(grep -c "DSA-STAT PATCH" "${f}" || true)
    echo "  ${f}: ${N} DSA-STAT markers"
done
echo "Done. 注意：若 vLLM 以非 editable 方式安装（site-packages 里是拷贝），"
echo "则需对安装目录重复此操作，或改用 pip install -e ${VLLM_SRC}"
