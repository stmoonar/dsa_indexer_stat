#!/bin/bash
# run_capture.sh — 启动 capture 运行
#
# 约束 7: --disable-cuda-graph, bs=1, 关 overlap scheduler
# 约束 12: 先跑 smoke (2K×10step) 再烧全量

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to DeepSeek-V3.2 weights}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for dump output}"
TASK_FILE="${TASK_FILE:?Set TASK_FILE (prompt file path)}"

# 默认 smoke 配置
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
TP="${TP:-16}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/capture_${TIMESTAMP}.log"

echo "=== DSA Indexer State Capture ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Context: ${CONTEXT_LENGTH}, New tokens: ${MAX_NEW_TOKENS}"
echo "TP: ${TP}"
echo "Log: ${LOG_FILE}"
echo ""

# Step 1: 启动 SGLang server (约束 7: --disable-cuda-graph)
echo "Starting SGLang server..."
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP}" \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --max-total-tokens "${CONTEXT_LENGTH}" \
    --schedule-policy fcfs \
    --port 30000 \
    2>&1 | tee "${LOG_FILE}" &

SERVER_PID=$!

# 等待 server 就绪
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Step 2: 发送请求并 capture
echo "Sending capture request..."
python -c "
import sys
sys.path.insert(0, '.')
from capture.indexer_state_capturer import init_indexer_state_capturer

# Note: 在实际使用中，monkey_patch 需要在 model 加载后调用
# 这里提供的是命令行入口的参考模板
print('Capture script template - see capture/monkey_patch.py for integration')
print('Integration requires modifying SGLang server startup to call patch_indexer()')
"

echo ""
echo "=== IMPORTANT ==="
echo "This script is a template. To actually capture:"
echo "1. Modify SGLang server startup to import and call capture.monkey_patch.patch_indexer()"
echo "2. Or apply capture/patch_sglang/indexer_dump.diff to SGLang source"
echo "3. Then run inference with bs=1, --disable-cuda-graph"
echo ""
echo "For smoke test: CONTEXT_LENGTH=2048 MAX_NEW_TOKENS=10"
echo "For full run:   CONTEXT_LENGTH=131072 MAX_NEW_TOKENS=1000"

# Cleanup
kill "${SERVER_PID}" 2>/dev/null || true
