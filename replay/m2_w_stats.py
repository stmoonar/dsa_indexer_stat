"""
m2_w_stats.py — 测量 2: w 符号分布

约束 14：w 的参数化优先以官方 reference 代码静态阅读为准。
本模块统计 dump 数据中 w 的实际符号分布。
"""

import torch


def w_sign_stats(w_all: torch.Tensor) -> dict:
    """
    统计 w 的符号分布。

    Args:
        w_all: [T, H] 或 [N, H]，所有步所有层的 w 值

    Returns:
        dict with:
            - neg_fraction: w < 0 的 (t,j) 占比
            - zero_fraction: w == 0 的 (t,j) 占比
            - pos_fraction: w > 0 的占比
            - w_min: w 的最小值
            - w_max: w 的最大值
            - w_mean: w 的均值
            - w_std: w 的标准差
            - per_head_neg_fraction: [H] 每个 head 的负 w 占比
    """
    total = w_all.numel()
    neg_count = (w_all < 0).sum().item()
    zero_count = (w_all == 0).sum().item()
    pos_count = (w_all > 0).sum().item()

    per_head_neg = (w_all < 0).float().mean(dim=0)  # [H]

    return {
        "neg_fraction": neg_count / total,
        "zero_fraction": zero_count / total,
        "pos_fraction": pos_count / total,
        "w_min": w_all.min().item(),
        "w_max": w_all.max().item(),
        "w_mean": w_all.mean().item(),
        "w_std": w_all.std().item(),
        "per_head_neg_fraction": per_head_neg.tolist(),
    }


def w_neg_mass(
    w_all: torch.Tensor,
    relu_expectation: torch.Tensor | None = None,
) -> dict:
    """
    计算负 w 的质量占比。

    负部质量 = Σ_{w<0} |w| · E[ReLU(q·k)] / Σ_all |w| · E[ReLU(q·k)]

    若 relu_expectation 未提供，简化为 Σ_{w<0}|w| / Σ|w|。

    Args:
        w_all: [T, H]
        relu_expectation: [T, H] 可选，每个 (t,j) 的 E[ReLU(q_j·k)]

    Returns:
        dict with neg_mass_ratio
    """
    abs_w = w_all.abs()
    neg_mask = w_all < 0

    if relu_expectation is not None:
        weighted = abs_w * relu_expectation
    else:
        weighted = abs_w

    neg_mass = weighted[neg_mask].sum().item()
    total_mass = weighted.sum().item()

    return {
        "neg_mass_ratio": neg_mass / total_mass if total_mass > 0 else 0.0,
        "neg_mass_abs": neg_mass,
        "total_mass_abs": total_mass,
    }
