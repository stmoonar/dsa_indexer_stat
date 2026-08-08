# 算法设计文档：块摘要 + 证书剪枝

本文档是算法的权威定义。所有实现以此为准。

## 1. 问题定义

DSA indexer 的打分函数为：

    I_{t,s} = Σ_j  w_{t,j} · ReLU(q^I_{t,j} · k^I_s)

其中：
- q^I_{t,j} ∈ R^{d_I}：第 t 步、第 j 个 head 的 indexer query（post-RoPE, fp32）
- k^I_s ∈ R^{d_I}：第 s 个 token 的 indexer key（post-RoPE, fp32, MQA 共享）
- w_{t,j} ∈ R：第 j 个 head 的权重（**符号不确定，需由 m2 静态代码阅读确认**）
- H^I = 64 heads, d_I = 128, top-k = 2048

目标：每步选出 T_t = argmax_{|S|=k} min_{s∈S} I_{t,s}，即全局分数最高的 k 个 token。

## 2. 块摘要定义

将序列按块大小 B 分块。第 b 块包含 token 索引 [bB, (b+1)B)。

**尾部未满块不建摘要，归入 warm/local 集合（约束 5）。**

每块存储两个摘要：
- 块均值 μ_b = (1/B) Σ_{s∈block_b} k^I_s，形状 [d_I]
- 块半径 r_b = max_{s∈block_b} ||k^I_s - μ_b||_2，标量

可选扩展（box bound）：
- 逐维最小值 lo_b[d] = min_{s∈block_b} k^I_s[d]
- 逐维最大值 hi_b[d] = max_{s∈block_b} k^I_s[d]

## 3. 上界公式（Ball Bound，双符号版）

对块 b 内所有 token s，indexer 分数的上界为：

    U_{t,b} = Σ_{j: w_{t,j}≥0} w_{t,j} · ReLU(q^I_{t,j} · μ_b + ||q^I_{t,j}|| · r_b)
            + Σ_{j: w_{t,j}<0} w_{t,j} · ReLU(q^I_{t,j} · μ_b - ||q^I_{t,j}|| · r_b)

**证明依据**：
- w_j ≥ 0 时：由 Cauchy-Schwarz, q·k ≤ q·μ + ||q||·||k-μ|| ≤ q·μ + ||q||·r，
  ReLU 单调，故 w_j·ReLU(q·k) ≤ w_j·ReLU(q·μ + ||q||r)。
- w_j < 0 时：需要 ReLU(q·k) 的**下界**来得到 w_j·ReLU(q·k) 的上界。
  由 Cauchy-Schwarz, q·k ≥ q·μ - ||q||·r，
  故 ReLU(q·k) ≥ ReLU(q·μ - ||q||r)，
  乘以 w_j < 0 翻转方向，得 w_j·ReLU(q·k) ≤ w_j·ReLU(q·μ - ||q||r)。
- 对所有 head 求和得 I_{t,s} ≤ U_{t,b}。

**约束 2 强制要求**：不得假设 w 全正。

## 4. 上界公式（Box Bound，更紧）

对每个维度 d，q^I_{t,j}[d] · k^I_s[d] 的上界为：
- 若 q[d] ≥ 0：q[d] · hi_b[d]
- 若 q[d] < 0：q[d] · lo_b[d]

即 max_k(q·k) = Σ_d max(q[d]·lo[d], q[d]·hi[d])。

对每个 head j，box bound 给出 q_j·k_s 的精确上界（在 box 约束下），
代入 ReLU 和 w 加权后得到逐 head bound，再按 w 符号求和。

存储开销：2·d_I 标量/块 = 256 floats vs ball bound 的 d_I+1 = 129 floats。

## 5. Warm-Start τ 算法

### τ 的语义（约束 1，关键！）

τ_t 是全集 top-k 分数的合法下界。

**正确做法**：用**当前步**的 q^I_t 对 warm 集（上一步 top-2048 的 k^I）
重新打分，取 warm 集内的第 2048 大分数（即 warm 集分数的最小值）作为 τ_0。

**禁止**：复用上一步的 τ 数值。上一步的 τ 是用上一步的 q 算的，
与当前步的 q 无关，不是当前步 top-k 分数的合法下界。

**合法性证明**：warm 集是全集的子集，子集的第 k 大 ≤ 全集的第 k 大。

### 算法流程

1. 对 warm 集（2048 个 k^I）用当前 q^I_t 精确打分，取最小值得 τ_0。
2. 对所有块计算 U_{t,b}。
3. **One-shot 剪枝**：丢弃 U_{t,b} ≤ τ_0 的块，拉取其余块做精确打分。
4. **Best-first 剪枝**（更紧）：按 U_{t,b} 降序处理块，
   逐块精确打分并更新 τ（在已评估 token 中取第 k 大），
   直到剩余块的 U 全部 ≤ τ。

HISA 复现额外要求（约束 4）：强制包含首块和尾块（sink + local）。

## 6. 存储与流量三本账

| 账本 | 定义 | 度量 |
|------|------|------|
| Pool 容量 | 远端存储的 K^I 总量 | bytes/token/layer |
| HBM 常驻 | GPU 常驻的摘要 + warm 集 | bytes/layer |
| 每步 Pool 流量 | 每 decode step 从远端拉取的字节数 | bytes/step/layer |

不同方法在三本账上的贡献不同，不得直接连乘。

## 7. 静态代码阅读结论（SGLang v0.5.13.post1, dsa_indexer.py）

### 7.1 w 的参数化：**无非负保证，w 可以为负**

- `weights_proj` 是裸 `ReplicatedLinear(hidden_size, n_heads, bias=False)`（line 361）
- 输出经过 `w * n_heads^{-0.5}` 缩放（line 421），**无 softmax/sigmoid/ReLU**
- 最终传入 kernel 的 weights = `w * n_heads^{-0.5} * q_scale * softmax_scale`
- **结论：bound 公式必须使用双符号版本（约束 2）**

### 7.2 计算流程（tilelang_kernel.py line 244-247）

kernel 内部执行：
1. `fp8_q @ fp8_k -> fp32 logits`（逐 head 内积）
2. `relu(fp32_logits) * q_s(weights) -> fp32 logits`（ReLU 后乘 weights）
3. `fp32_logits -> fp32 logits_sum`（跨 head 求和）
4. `fp32_logits_sum * k_s -> fp32 index_score`（乘 k 的 FP8 scale）

等效公式：`I_{t,s} = k_scale_s * Σ_j (w_j * n_h^{-0.5} * q_scale_j * d^{-0.5}) * ReLU(q_fp8_j · k_fp8_s)`
由于 q_scale, k_scale > 0，排序等价于 `Σ_j w_j * ReLU(q_j · k_s)`。

### 7.3 q^I 和 k^I 的处理流程

1. **k^I 投影**: `key = wk(x)`，然后 `key = k_norm(key)`（LayerNorm, line 368/480）
2. **q^I 投影**: `query = wq_b(q_lora)`，reshape 为 `[L, H, d_I]`
3. **RoPE**: 只对前 `rope_head_dim` 维施加（V3.2: rope_head_dim = qk_rope_head_dim = 64）
   - query 和 key 都 split 为 `[rope_head_dim, head_dim - rope_head_dim]`
   - 只对 rope 部分施加 `rotary_emb(positions, q_rope, k_rope)`
   - 结果写回前 `rope_head_dim` 维
4. **Hadamard 旋转**: `rotate_activation(query)` 和 `rotate_activation(key)`
   - 对完整 `head_dim=128` 维施加 Hadamard 变换
   - 带缩放 `hidden_size^{-0.5} = 128^{-0.5}`

### 7.4 FP8 量化

- 量化函数: `act_quant(x, block_size=128, scale_fmt)`（triton_kernel.py line 86）
- 粒度: **per-group**，group_size = block_size = 128（即每 128 个元素共享一个 scale）
- 对 d_I=128，恰好每行一个 scale（per-token per-layer）
- 格式: float8_e4m3fn
- Scale 计算: `scale = max(abs(x_group)) / 448.0`，可选 power-of-2 rounding
- q^I 和 k^I 都用同一个 `act_quant` 量化

### 7.5 k^I cache 存储

- 存储格式：FP8 key (128 bytes) + FP32 scale (4 bytes) = 132 bytes/token/layer
- 通过 `_store_index_k_cache` 方法存入 paged KV pool（line 1228）
- 优先使用 fused kernel `fused_store_index_k_cache`

### 7.6 TP 切分

- `wq_b` 和 `weights_proj` 都使用 `ReplicatedLinear`（**不切分，全 rank 复制**）
- `wk` 也使用 `ReplicatedLinear`（不切分）
- **结论：indexer 的 q、k、w 在所有 TP rank 上完全相同，
  dump 只需在 rank 0 执行，不需要 merge_ranks.py**
