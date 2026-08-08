# 环境配置

## 硬件
- 2 × 8 H800 (16 卡)
- 节点间通信: InfiniBand

## 软件
- SGLang: (填写版本)
- 模型: DeepSeek-V3.2 (61 layers, 671B)
- Python: 3.10+
- PyTorch: 2.x with CUDA

## capture 启动命令模板

```bash
python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --tp 16 \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --max-total-tokens 131072 \
    --schedule-policy fcfs \
    2>&1 | tee logs/capture_$(date +%Y%m%d_%H%M%S).log
```

### 关键参数（约束 7）
- `--disable-cuda-graph`: 必须。CUDA graph 区内 Python dump 静默失败。
- `--tp 16`: 张量并行 = 16
- bs=1: capture 时只跑单请求
- 关闭 overlap scheduler

## replay 环境
- 单卡即可（H800 或 A100）
- 只需 PyTorch，不需 SGLang
- 安装: `pip install torch numpy pytest`
