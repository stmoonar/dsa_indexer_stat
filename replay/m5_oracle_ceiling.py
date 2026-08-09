"""
m5_oracle_ceiling.py — 剪枝天花板、分组方式 sweep、bound 松弛归因

三个问题，一个模块：

1) oracle 天花板：即使 bound 完全紧，能跳过多少？
   = 组内真实最大分 <= τ 的组占比。这个数低说明瓶颈在数据弥散度，
   收紧 bound 不会有帮助——比 F 本身更能定位问题。

2) 分组方式：低天花板是"弥散本身"还是"连续位置分块"这个假设造成的？
   同样的组数下对比连续块 vs 位置无关的 k-means 聚类。
   若聚类把天花板从 8% 拉到 40%，死穴在分块方式而非 τ 机制。

3) 松弛归因：U 比真实块最大分高出的那部分，分别来自哪一项？
   要收紧先动哪一项，这个分解直接回答。

口径注意：
- 天花板同时报"组占比"和"token 占比"。连续等长块两者相同；聚类的簇大小
  不等，此时 token 占比才是真正的读放大指标，组占比会高估收益。
- 聚类天花板忽略了分组自身的代价（簇分配表、gather 变随机访问），
  是【上界】，必须与通信侧的 BW_eff 曲线成对解读。
"""

import torch
import torch.nn.functional as F

from replay.bounds import exact_indexer_score, ball_bound_upper


def contiguous_groups(n_tokens: int, block_size: int) -> torch.Tensor:
    """连续位置分块。返回 [n_tokens] 的组 id；尾部不满块归入最后一组。"""
    idx = torch.arange(n_tokens) // block_size
    n_full = n_tokens // block_size
    if n_full == 0:
        return torch.zeros(n_tokens, dtype=torch.long)
    return idx.clamp_max(n_full - 1).long()


def _kmeanspp_init(k_I: torch.Tensor, n_groups: int, gen):
    """k-means++ 初始化（D² 采样）。

    随机初始化会把同一个簇劈成两半、同时把两个真簇并掉，导致聚类质量
    虚低。而"聚类分组能否救回天花板"是本模块的决定性对比，坏初始化会让
    结论以错误的理由成立，所以这里必须用 ++。
    """
    L = k_I.shape[0]
    centers = torch.empty(n_groups, k_I.shape[1], dtype=k_I.dtype)
    first = int(torch.randint(L, (1,), generator=gen).item())
    centers[0] = k_I[first]
    d2 = ((k_I - centers[0]) ** 2).sum(1)
    for i in range(1, n_groups):
        total = float(d2.sum().item())
        if total <= 0:
            centers[i:] = k_I[torch.randperm(L, generator=gen)[:n_groups - i]]
            break
        pick = int(torch.multinomial(d2 / total, 1, generator=gen).item())
        centers[i] = k_I[pick]
        d2 = torch.minimum(d2, ((k_I - centers[i]) ** 2).sum(1))
    return centers


def _assign_all(k_I, centers, chunk):
    out = torch.zeros(k_I.shape[0], dtype=torch.long)
    for s in range(0, k_I.shape[0], chunk):
        e = min(s + chunk, k_I.shape[0])
        out[s:e] = torch.cdist(k_I[s:e], centers).argmin(dim=1)
    return out


def kmeans_groups(k_I: torch.Tensor, n_groups: int, iters: int = 8,
                  seed: int = 0, chunk: int = 8192,
                  fit_sample: int = 16384) -> torch.Tensor:
    """位置无关的 k-means 分组（k-means++ 初始化 + 固定轮数 Lloyd）。

    代价：一次全量指派是 O(L · n_groups · d)，128K × 8192 簇就有 ~137 GFLOP，
    所以中心点只在 fit_sample 个采样点上拟合，最后再对全量指派一次。

    迭代数有限、且在子样本上拟合 -> 聚类略偏离最优 -> 天花板被【低估】，
    方向保守："聚类也救不了"的结论是稳的；若聚类天花板已经很高，
    真实最优只会更高。
    """
    L, d = k_I.shape
    n_groups = max(1, min(n_groups, L))
    gen = torch.Generator().manual_seed(seed)

    fit_n = min(L, max(fit_sample, n_groups * 4))
    fit_X = (k_I if fit_n >= L
             else k_I[torch.randperm(L, generator=gen)[:fit_n]])
    centers = _kmeanspp_init(fit_X, n_groups, gen).clone()

    for _ in range(iters):
        a = _assign_all(fit_X, centers, chunk)
        new = torch.zeros_like(centers)
        cnt = torch.zeros(n_groups, dtype=torch.long)
        new.index_add_(0, a, fit_X)
        cnt.index_add_(0, a, torch.ones(fit_X.shape[0], dtype=torch.long))
        alive = cnt > 0
        centers[alive] = new[alive] / cnt[alive].unsqueeze(1).to(k_I.dtype)
        # 空簇：搬到离自身中心最远的点上（逐点距离，不要构造 L×L 矩阵）
        n_dead = int((~alive).sum().item())
        if n_dead:
            d2 = ((fit_X - centers[a]) ** 2).sum(1)
            centers[~alive] = fit_X[d2.topk(min(n_dead, fit_X.shape[0])).indices]

    return _assign_all(k_I, centers, chunk)


def group_max(scores: torch.Tensor, groups: torch.Tensor,
              n_groups: int) -> torch.Tensor:
    """每组内的最大分数（空组为 -inf）。"""
    out = torch.full((n_groups,), float("-inf"))
    return out.index_reduce_(0, groups, scores, "amax", include_self=True)


def oracle_ceiling(scores: torch.Tensor, groups: torch.Tensor,
                   tau: float) -> dict:
    """bound 完全紧时可跳过的比例。"""
    n_groups = int(groups.max().item()) + 1
    gmax = group_max(scores, groups, n_groups)
    alive = torch.isfinite(gmax)
    skip = alive & (gmax <= tau)

    sizes = torch.zeros(n_groups, dtype=torch.long)
    sizes.index_add_(0, groups, torch.ones_like(groups))
    n_tok = int(sizes.sum().item())
    return {
        "group_frac": float(skip.sum().item() / max(int(alive.sum()), 1)),
        "token_frac": float(sizes[skip].sum().item() / max(n_tok, 1)),
        "n_groups": int(alive.sum().item()),
        "median_group_size": float(sizes[alive].float().median().item()),
    }


def grouping_sweep(q_I, w, k_I, tau, block_sizes=(16, 32, 64, 128),
                   kmeans_block_sizes=(128,), kmeans_iters=8,
                   group_cache=None) -> dict:
    """同一步上扫 B，并在同样组数下对比 k-means 分组。

    group_cache: 可选 dict，跨样本复用分组。k-means 很贵，而同一前缀桶内
    各样本的 cache_len 只差几十个 token（3 万分之一），复用同一份聚类
    不影响结论，却把调用次数从"每样本每 B"降到"每桶每 B"。
    调用方需保证同一 cache 只用于 k_I 前缀相同的样本。
    """
    scores = exact_indexer_score(q_I, w, k_I)
    L = k_I.shape[0]
    out = {}
    for B in block_sizes:
        g = contiguous_groups(L, B)
        n_groups = int(g.max().item()) + 1
        out[f"contiguous_B{B}"] = oracle_ceiling(scores, g, tau)
        if B not in kmeans_block_sizes:
            continue
        gk = None if group_cache is None else group_cache.get(B)
        if gk is None or gk.shape[0] < L:
            gk = kmeans_groups(k_I, n_groups, iters=kmeans_iters)
            if group_cache is not None:
                group_cache[B] = gk
        out[f"kmeans_B{B}"] = oracle_ceiling(scores, gk[:L], tau)
    return out


def slack_attribution(q_I, w, k_I, mu, r, block_size: int) -> dict:
    """把 U_ball 与真实块最大分之间的差分解到各项。

    U_full     = Σ_{w>=0} w ReLU(q·mu + ||q||r) + Σ_{w<0} w ReLU(q·mu - ||q||r)
    U_noradius = 令 r=0（两支合一）：Σ_j w_j ReLU(q_j·mu)
    U_posonly  = 只算 w>=0 的支（丢掉 <=0 的负支项，故 >= U_full）

    分解：
      radius_term   = U_full - U_noradius     ||q||r 这一项贡献的松弛
      center_term   = U_noradius - true_max   块均值代替最佳 token 的松弛
      neg_branch    = U_posonly - U_full      负 head 支已经帮忙压下去的量
                                              （小 -> 负支没起作用）
    """
    n_full = mu.shape[0]
    if n_full == 0:
        return {}
    scores = exact_indexer_score(q_I, w, k_I)
    true_max = scores[:n_full * block_size].reshape(n_full, block_size).max(1).values

    U_full = ball_bound_upper(q_I, w, mu, r)
    U_noradius = ball_bound_upper(q_I, w, mu, torch.zeros_like(r))

    w_pos = w.clamp_min(0.0)
    U_posonly = ball_bound_upper(q_I, w_pos, mu, r)

    total = (U_full - true_max).clamp_min(1e-9)
    radius = (U_full - U_noradius)
    center = (U_noradius - true_max)

    def med(x):
        return float(x.median().item())

    return {
        "U_full_p50": med(U_full),
        "true_block_max_p50": med(true_max),
        "total_slack_p50": med(total),
        "radius_term_p50": med(radius),
        "center_term_p50": med(center),
        "radius_share": med(radius / total),
        "center_share": med(center / total),
        "neg_branch_recovered_p50": med(U_posonly - U_full),
        "neg_branch_share_of_slack": med((U_posonly - U_full) / total),
    }


def q_weighted_spectrum(q_list, w_list, top_m=(16, 32, 64)) -> dict:
    """想法 4：M_q = E[Σ_j w_j^2 q_j q_j^T] 的谱。

    若 top-m 特征值已占绝大部分能量，indexer 的 q/k 维度可以降到 m，
    是与一切其他优化正交的常数倍收益。
    """
    d = q_list[0].shape[-1]
    M = torch.zeros(d, d, dtype=torch.float64)
    for q, w in zip(q_list, w_list):
        wq = (w.double() ** 2).unsqueeze(1) * q.double()   # [H, d]
        M += q.double().t() @ wq
    M /= max(len(q_list), 1)
    evals = torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
    total = float(evals.sum().item()) or 1.0
    return {
        "d": d,
        "energy_frac": {int(m): float(evals[:m].sum().item() / total)
                        for m in top_m if m <= d},
        "effective_rank": float(
            torch.exp(-(evals / total * (evals / total).clamp_min(1e-12).log())
                      .sum()).item()),
        "top_eigenvalues": [float(x) for x in evals[:8]],
    }
