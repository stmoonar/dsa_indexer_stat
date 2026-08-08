"""
m3_block_locality.py — 测量 3: 块局部性与 churn

三个统计：
(a) 全 top-k 触块数
(b) 相邻步 churn 率
(c) 仅 churn token 触块数（oracle 下界）

附加：fetched-block Jaccard、跨层 top-k overlap
"""

import torch


def topk_touched_blocks(
    topk_indices: torch.Tensor,
    block_size: int,
) -> int:
    """(a) top-k token 触及的块数。"""
    blocks = set()
    for idx in topk_indices.tolist():
        blocks.add(idx // block_size)
    return len(blocks)


def topk_churn(
    topk_curr: torch.Tensor,
    topk_prev: torch.Tensor,
) -> dict:
    """
    (b) 相邻步 churn 率及 (c) churn token 详情。

    churn_rate = |S_t △ S_{t-1}| / |S_t|
    churn_in = S_t \\ S_{t-1}  (新进入 top-k 的)
    churn_out = S_{t-1} \\ S_t  (离开 top-k 的)
    """
    curr_set = set(topk_curr.tolist())
    prev_set = set(topk_prev.tolist())

    sym_diff = curr_set ^ prev_set
    churn_in = curr_set - prev_set
    churn_out = prev_set - curr_set

    k = len(curr_set)
    churn_rate = len(sym_diff) / k if k > 0 else 0.0

    return {
        "churn_rate": churn_rate,
        "churn_in": churn_in,
        "churn_out": churn_out,
        "n_churn_in": len(churn_in),
        "n_churn_out": len(churn_out),
    }


def churn_touched_blocks(
    churn_in_indices: set[int],
    block_size: int,
) -> int:
    """
    (c) 仅 churn-in token 触及的块数 = oracle fetch 下界。

    warm 集已有的 token 不需要 fetch，只有新进入 top-k 的 token 需要。
    """
    blocks = set()
    for idx in churn_in_indices:
        blocks.add(idx // block_size)
    return len(blocks)


def block_locality_stats(
    topk_curr: torch.Tensor,
    topk_prev: torch.Tensor | None,
    block_size: int,
) -> dict:
    """
    综合计算一步的块局部性统计。

    Args:
        topk_curr: [k], 当前步 exact top-k 索引
        topk_prev: [k] 或 None, 上一步 exact top-k 索引
        block_size: 块大小

    Returns:
        dict with all locality stats
    """
    result = {
        "touched_blocks": topk_touched_blocks(topk_curr, block_size),
    }

    if topk_prev is not None:
        churn = topk_churn(topk_curr, topk_prev)
        result["churn_rate"] = churn["churn_rate"]
        result["n_churn_in"] = churn["n_churn_in"]
        result["n_churn_out"] = churn["n_churn_out"]
        result["churn_touched_blocks"] = churn_touched_blocks(
            churn["churn_in"], block_size
        )

        # Fetched-block Jaccard
        curr_blocks = set(idx // block_size for idx in topk_curr.tolist())
        prev_blocks = set(idx // block_size for idx in topk_prev.tolist())
        union = curr_blocks | prev_blocks
        inter = curr_blocks & prev_blocks
        result["fetched_block_jaccard"] = len(inter) / len(union) if union else 1.0

    return result


def cross_layer_overlap(
    topk_layer_a: torch.Tensor,
    topk_layer_b: torch.Tensor,
) -> float:
    """
    跨层 top-k overlap = |S_{ℓ} ∩ S_{ℓ+1}| / |S_{ℓ}|
    """
    set_a = set(topk_layer_a.tolist())
    set_b = set(topk_layer_b.tolist())
    if len(set_a) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a)
