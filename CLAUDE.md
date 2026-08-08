# 项目：DSA Indexer K 稀疏化测量

## 目标
在 DeepSeek-V3.2 上测量 4 个统计量，判定"块摘要+证书剪枝"能把
indexer K 读取量压到多少。docs/design.md 是算法权威定义，
docs/measurements.md 是测量口径权威定义。冲突时以这两个文件为准。

## 不可违反的正确性约束
1. τ 的语义：用【当前步】的 q^I 对 warm 集（上一步 top-2048 的 k^I）
   重新打分，取最小值。禁止复用上一步的 τ 数值——那不是本步的合法下界。
   是第 2048 大（warm 集内最小），不是第 2048 小。
2. bound 必须处理 w<0：w_j≥0 用 ReLU(q·μ + ||q||r)，
   w_j<0 用 w_j·ReLU(q·μ − ||q||r)。不得假设 w 全正，
   除非 m2 的静态代码阅读证明了参数化保证非负。
3. 所有 k^I、q^I 统一为 post-RoPE、反量化 fp32。FP8 dump 必须带 scale。
4. HISA 复现必须强制包含首块和尾块（sink + local）。
5. 尾部未满块不建摘要、不进块统计，归入 warm/local 集合。
6. replay 的 exact top-k 是全部统计的参照系；与生产 kernel 的
   mismatch 率单独汇报（kernel_align.py），不得混入 recall。

## 不可违反的系统约束
7. capture 时：--disable-cuda-graph，关 overlap scheduler，bs=1。
   CUDA graph 区内的 Python dump 会静默失败。
8. 不解析 SGLang paged KV pool。k^I 在 indexer forward 内
   截获并追加写入自有连续 buffer。
9. 检查 indexer q head 是否被 TP 切分；若切分，dump 带 rank id，
   merge_ranks.py 离线拼接。k^I 单头共享，确认后可只在 rank0 dump。
10. 侵入面只限模型文件层的 indexer forward（diff 形式保存在
    capture/patch_sglang/），不改调度器、不改内存池。

## 流程纪律
11. 任何真数据运行前，tests/ 必须全绿。
12. 先 smoke（2K ctx × 10 step）打通 capture→replay→kernel对齐
    全链路，再烧 32K/128K。
13. 每个 run 的输出目录必须含 config 快照（git hash、启动参数、schema 版本）。
14. w 的参数化、indexer RoPE 语义：以 DeepSeek 官方 reference
    实现的静态阅读为准，写进 docs/design.md 后再写代码。
    不要用实验回答读代码能回答的问题。

## 汇报口径
15. 一切统计按 (task, layer) 分组，报分布（p5/p50/p95/p99），
    不报单一均值。主配置 128K，32K 为副配置。