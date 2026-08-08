"""
bounds.py — Ball bound 和 Box bound 的独立实现

所有 bound 必须满足 soundness 性质：
对块 b 内任意 token s，U_{t,b} >= I_{t,s}。

约束 2：bound 必须处理 w < 0 的情形。
"""

import torch
import torch.nn.functional as F


def compute_block_summaries_ball(
    k_I: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    计算 ball bound 所需的块摘要：均值和半径。

    Args:
        k_I: [L, d_I], fp32, post-RoPE 的 indexer key
        block_size: 块大小 B

    Returns:
        mu: [n_full_blocks, d_I] 块均值
        r: [n_full_blocks] 块半径 (max ||k - mu||)
        n_tail: 尾部未满块的 token 数（不建摘要，约束 5）
    """
    L, d_I = k_I.shape
    n_full_blocks = L // block_size
    n_tail = L - n_full_blocks * block_size

    if n_full_blocks == 0:
        return (
            torch.empty(0, d_I, device=k_I.device),
            torch.empty(0, device=k_I.device),
            n_tail,
        )

    k_blocks = k_I[:n_full_blocks * block_size].reshape(n_full_blocks, block_size, d_I)
    mu = k_blocks.mean(dim=1)  # [n_blocks, d_I]
    diff = k_blocks - mu.unsqueeze(1)  # [n_blocks, B, d_I]
    norms = diff.norm(dim=2)  # [n_blocks, B]
    r = norms.max(dim=1).values  # [n_blocks]

    return mu, r, n_tail


def compute_block_summaries_box(
    k_I: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    计算 box bound 所需的块摘要：逐维 min/max。

    Args:
        k_I: [L, d_I], fp32, post-RoPE 的 indexer key
        block_size: 块大小 B

    Returns:
        lo: [n_full_blocks, d_I] 逐维最小值
        hi: [n_full_blocks, d_I] 逐维最大值
        n_tail: 尾部未满块的 token 数
    """
    L, d_I = k_I.shape
    n_full_blocks = L // block_size
    n_tail = L - n_full_blocks * block_size

    if n_full_blocks == 0:
        return (
            torch.empty(0, d_I, device=k_I.device),
            torch.empty(0, d_I, device=k_I.device),
            n_tail,
        )

    k_blocks = k_I[:n_full_blocks * block_size].reshape(n_full_blocks, block_size, d_I)
    lo = k_blocks.min(dim=1).values  # [n_blocks, d_I]
    hi = k_blocks.max(dim=1).values  # [n_blocks, d_I]

    return lo, hi, n_tail


def ball_bound_upper(
    q_I: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """
    Ball bound 上界（双符号版，约束 2）。

    对块 b 内所有 token s: I_{t,s} <= U_{t,b}

    U_{t,b} = Σ_{j: w_j>=0} w_j · ReLU(q_j · μ_b + ||q_j|| · r_b)
            + Σ_{j: w_j<0}  w_j · ReLU(q_j · μ_b - ||q_j|| · r_b)

    Args:
        q_I: [H, d] indexer query, 当前步
        w: [H] head 权重, 保留符号
        mu: [n_blocks, d] 块均值
        r: [n_blocks] 块半径

    Returns:
        U: [n_blocks] 每块的分数上界
    """
    H, d = q_I.shape
    n_blocks = mu.shape[0]

    q_norms = q_I.norm(dim=1)  # [H]
    q_dot_mu = torch.mm(q_I, mu.t())  # [H, n_blocks]

    q_norm_r = q_norms.unsqueeze(1) * r.unsqueeze(0)  # [H, n_blocks]

    w_pos_mask = (w >= 0)  # [H]
    w_neg_mask = ~w_pos_mask

    relu_pos = F.relu(q_dot_mu + q_norm_r)  # [H, n_blocks], w>=0 时用
    relu_neg = F.relu(q_dot_mu - q_norm_r)  # [H, n_blocks], w<0 时用

    contrib = torch.zeros(H, n_blocks, device=q_I.device)
    if w_pos_mask.any():
        contrib[w_pos_mask] = w[w_pos_mask].unsqueeze(1) * relu_pos[w_pos_mask]
    if w_neg_mask.any():
        contrib[w_neg_mask] = w[w_neg_mask].unsqueeze(1) * relu_neg[w_neg_mask]

    U = contrib.sum(dim=0)  # [n_blocks]
    return U


def box_bound_upper(
    q_I: torch.Tensor,
    w: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """
    Box bound 上界（双符号版）。

    对每个 head j，q_j · k_s 的上界为：
      max_val_j = Σ_d max(q_j[d]*lo_b[d], q_j[d]*hi_b[d])

    然后按 w 符号处理 ReLU：
    - w_j >= 0: w_j · ReLU(max_val_j)
    - w_j <  0: w_j · ReLU(min_val_j)，
      其中 min_val_j = Σ_d min(q_j[d]*lo_b[d], q_j[d]*hi_b[d])

    Args:
        q_I: [H, d]
        w: [H]
        lo: [n_blocks, d] 逐维最小值
        hi: [n_blocks, d] 逐维最大值

    Returns:
        U: [n_blocks]
    """
    H, d = q_I.shape
    n_blocks = lo.shape[0]

    # q_I: [H, d], lo/hi: [n_blocks, d]
    # q_lo[h, b, d] = q_I[h,d] * lo[b,d]
    q_lo = torch.einsum('hd,bd->hbd', q_I, lo)  # [H, n_blocks, d]
    q_hi = torch.einsum('hd,bd->hbd', q_I, hi)  # [H, n_blocks, d]

    max_per_dim = torch.maximum(q_lo, q_hi)  # [H, n_blocks, d]
    min_per_dim = torch.minimum(q_lo, q_hi)  # [H, n_blocks, d]

    max_dot = max_per_dim.sum(dim=2)  # [H, n_blocks]
    min_dot = min_per_dim.sum(dim=2)  # [H, n_blocks]

    w_pos_mask = (w >= 0)
    w_neg_mask = ~w_pos_mask

    contrib = torch.zeros(H, n_blocks, device=q_I.device)
    if w_pos_mask.any():
        contrib[w_pos_mask] = w[w_pos_mask].unsqueeze(1) * F.relu(max_dot[w_pos_mask])
    if w_neg_mask.any():
        contrib[w_neg_mask] = w[w_neg_mask].unsqueeze(1) * F.relu(min_dot[w_neg_mask])

    U = contrib.sum(dim=0)  # [n_blocks]
    return U


def exact_indexer_score(
    q_I: torch.Tensor,
    w: torch.Tensor,
    k_I: torch.Tensor,
) -> torch.Tensor:
    """
    精确计算 indexer 分数。

    I_{t,s} = Σ_j w_j · ReLU(q_j · k_s)

    Args:
        q_I: [H, d]
        w: [H]
        k_I: [L, d]

    Returns:
        scores: [L]
    """
    q_dot_k = torch.mm(q_I, k_I.t())  # [H, L]
    relu_scores = F.relu(q_dot_k)  # [H, L]
    weighted = w.unsqueeze(1) * relu_scores  # [H, L]
    scores = weighted.sum(dim=0)  # [L]
    return scores
