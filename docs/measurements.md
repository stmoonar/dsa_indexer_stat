# 测量口径定义

本文档是四个测量的精确数学定义和汇报口径。所有实现以此为准。

## 通用口径

- 所有 q^I、k^I 为 post-RoPE、反量化 fp32（约束 3）
- 参照系：replay 的 exact top-k（全量精确打分），不是生产 kernel 的 top-k（约束 6）
- 与生产 kernel 的 mismatch 率单独由 kernel_align.py 汇报
- 汇报粒度：按 (task, layer) 分组，报分布 p5/p50/p95/p99（约束 15）
- 主配置 128K，32K 为副配置
- 尾部未满块不建摘要、不进块统计，归入 warm/local（约束 5）

## 测量 1：HISA 块覆盖率（m1_hisa_recall.py）

### 定义

对第 t 步，设 S*_t 为 exact top-2048 token 集合。
将序列按块大小 B 分块，对每块计算块均值分数：

    J_{t,b} = Σ_j w_{t,j} · ReLU(q^I_{t,j} · μ_b)

取 J 最高的 m 个块（加上强制包含的首块和尾块），记为候选块集 Ω_t。

**覆盖率** = |{s ∈ S*_t : s 落在 Ω_t 的某个块内}| / |S*_t|

### 等价性

Stage 2 在候选集内是精确打分，所以 HISA 最终 top-2048 对真 top-2048
的 recall ≡ 真 top-k 被候选块覆盖的比例（模平局）。
实现只需算覆盖率，不需模拟 Stage 2。

### 配置

- B = 128（与 HISA 论文可比）
- m = 64（HISA 默认）
- 强制包含首块（索引 0）和尾块（最后一个满块）

### 附加统计

**Adaptive-m 曲线**：每个 query 达到 100%/99% 覆盖所需的最小 m 的分布。

## 测量 2：w 符号分布（m2_w_stats.py）

### 定义

统计 w_{t,j} 的符号分布：
- 负 w 的 (t, j) 占比：|{(t,j) : w_{t,j} < 0}| / (T × H^I)
- 负部质量占比：Σ_{w<0} |w_{t,j}| · E[ReLU(q_j·k)] / Σ_all |w_{t,j}| · E[ReLU(q_j·k)]

### 前置步骤

**优先静态阅读官方 reference 代码**（约束 14）：
确认 w 的参数化（weights_proj 之后有无 softmax/sigmoid）。
若代码结构上保证非负，记录证据并简化 bound 公式。

## 测量 3：块局部性与 churn（m3_block_locality.py）

### 定义

设 B 为块大小，S*_t 为第 t 步 exact top-2048。

(a) **全 top-k 触块数**：|{b : S*_t ∩ block_b ≠ ∅}|

(b) **相邻步 churn 率**：|S*_t △ S*_{t-1}| / |S*_t|
    其中 △ 为对称差

(c) **仅 churn token 触块数**：
    churn_t = S*_t \ S*_{t-1}（新进入 top-k 的 token）
    F_oracle = |{b : churn_t ∩ block_b ≠ ∅}|
    这是带 warm 集算法的真实 fetch 下界。

### 附加统计

- 相邻步 fetched-block Jaccard：
  J_fetch = |blocks(S*_t) ∩ blocks(S*_{t-1})| / |blocks(S*_t) ∪ blocks(S*_{t-1})|

- 跨层 top-k overlap（token 级，cross_layer.py）：
  对同一位置 t 的两层 ℓ、ℓ'，

      O(ℓ, ℓ') = |S*_{t,ℓ} ∩ S*_{t,ℓ'}| / |S*_{t,ℓ}|

  同一位置各层 |S*| 相同，故 O 对称，且与 Jaccard 单调等价。

  **必须与 O 同时汇报的两项**（否则长度带来的平凡重合会被读成跨层结构）：

  1. 随机基线 = |S*| / L（L = cache_len）：两个独立均匀 k-子集的期望重合度。
     L=8192、k=2048 时基线已经是 0.25。
  2. lift = (O − |S*|/L) / (1 − |S*|/L)：基线归一到 0、完全重合归一到 1。
     结论按 lift 写，O 只作为原始量列出。

  **位置筛选**：只用 L ≥ 2k 的位置。L ≤ k 时 top-k 就是全集，O ≡ 1，
  把这些位置混进分布会把重合度整体抬高。32K 配置下这会丢掉 ctx2048 桶。

  分组同约束 15：按 (sequence, 前缀长度桶, 层对)，并另报 O 随层距 |ℓ−ℓ'| 的曲线。
  索引层间可比的前提：dump 的行号是序列内绝对 token 位置，且同一次 forward
  内所有层用同一份采样 mask，故各层采样位置集合一致（实现仍显式取交集）。

## 测量 4：Bound 剪枝率（m4_bound_prune.py）

### 定义

**Fetch 率 F**：需要从 Pool 拉取做精确打分的块数 / 总块数。
剪枝率 = 1 - F。

两个版本：
- **F_oneshot**：用 τ_0（warm 集重打分最小值）一次剪枝，
  F = |{b : U_{t,b} > τ_0}| / N_blocks
- **F_bestfirst**：按 U_{t,b} 降序逐块处理、动态更新 τ，
  F = 实际评估的块数 / N_blocks

### τ_0 的计算（约束 1，关键！）

τ_0 = min_{s ∈ warm_set} I_{t,s}

其中 warm_set = 上一步的 exact top-2048 的 k^I，
I_{t,s} 使用**当前步**的 q^I_t 和 w_t 重新计算。

**禁止复用上一步的 τ 数值。**

### 配置

- B ∈ {64, 128, 256} sweep
- Bound 类型：ball bound（默认），box bound（对比）
- 报 F 及 F - F_oracle（oracle 下界来自测量 3c）

### 附加统计

- 相邻步 fetched-block Jaccard（预取可行性指标）
