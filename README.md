# DSA Indexer K 稀疏化测量

在 DeepSeek-V3.2 上测量 4 个统计量，判定"块摘要 + 证书剪枝"能把 indexer K 读取量压到多少。

## 四个测量

| 编号 | 名称 | 核心问题 |
|------|------|----------|
| M1 | HISA 块覆盖率 | 块均值 top-m 能覆盖多少真 top-k？ |
| M2 | w 符号分布 | 负 w 占比多大？影响 bound 紧度 |
| M3 | 块局部性与 churn | top-k 多集中？相邻步变化多大？ |
| M4 | Bound 剪枝率 | μ,r 证书能剪掉多少块？ |

## 快速开始

```bash
# 安装依赖
pip install torch numpy pytest pyyaml

# 运行性质测试（必须全绿再跑真数据）
python -m pytest tests/ -v

# 单卡 replay（需要 capture dump 数据）
python -m replay.exact_score  # 参照系
```

## 流程

1. **性质测试全绿** → 2. **Smoke capture (2K×10step)** → 3. **Smoke replay + kernel 对齐** → 4. **全量 capture (128K)** → 5. **全量 replay** → 6. **报告**

## 仓库结构

- `docs/` — 算法设计（design.md）和测量口径（measurements.md）
- `capture/` — SGLang 侵入补丁和 dump schema
- `replay/` — 离线统计模块（m1-m4, bounds, exact_score）
- `tests/` — 数学性质测试和集成测试
- `tasks/` — 任务族定义
- `analysis/` — 报告生成
- `results/` — 输出目录
