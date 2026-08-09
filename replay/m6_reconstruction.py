"""
m6_reconstruction.py — 想法 2 的裁决：k^I 能否由已存的 MLA latent 线性重建

若 k^I ≈ W·[c_s; k_pe_s] 成立得足够好，indexer 就不必单独存一份 k^I，
省掉的是 indexer KV 的全部容量与带宽。

裁决口径（两个都要，缺一不可）：
  R²            重建的数值保真度
  top-k overlap 用重建的 k̂ 重跑 top-k，与真 k 的 top-k 的重合度
                —— 这才是下游真正在意的量。R² 高但 overlap 塌掉是可能的：
                indexer 分数在排名边界极其拥挤（实测第 2048 名与 2098 名
                只差 0.1~0.4，而分数量级是 7~38）。

拟合与评估必须分开位置，否则 R² 是自欺欺人。
"""

import torch

from replay.bounds import exact_indexer_score


def fit_linear(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-3):
    """最小二乘（含偏置项）拟合 Y ≈ [X,1] @ W。返回 W。"""
    X = X.double()
    Y = Y.double()
    ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
    Xb = torch.cat([X, ones], dim=1)
    A = Xb.t() @ Xb
    A += ridge * torch.eye(A.shape[0], dtype=torch.float64) * A.diagonal().mean()
    return torch.linalg.solve(A, Xb.t() @ Y)


def apply_linear(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
    return (torch.cat([X.double(), ones], dim=1) @ W).float()


def r2_score(Y: torch.Tensor, Yhat: torch.Tensor) -> float:
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - (ss_res / ss_tot.clamp_min(1e-12)).item())


def reconstruction_test(k_I, c_latent, k_pe, q_samples, w_samples,
                        top_k=2048, n_fit=8192, seed=0) -> dict:
    """
    k_I:       [L, 128]  真 indexer key
    c_latent:  [L, 512]  MLA 的 kv_c_normed
    k_pe:      [L, 64]   post-RoPE 的 k_pe（与 c 一起构成 KV cache 条目）
    q_samples / w_samples: 若干真实 query，用来测 top-k overlap

    拟合位置与评估位置不相交。
    """
    L = min(k_I.shape[0], c_latent.shape[0], k_pe.shape[0])
    k_I, c_latent, k_pe = k_I[:L], c_latent[:L].float(), k_pe[:L].float()
    X = torch.cat([c_latent, k_pe], dim=1)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(L, generator=g)
    fit_idx = perm[:min(n_fit, L // 2)]
    W = fit_linear(X[fit_idx], k_I[fit_idx])

    held = perm[len(fit_idx):]
    k_hat_held = apply_linear(X[held], W)
    out = {
        "n_fit": int(len(fit_idx)),
        "n_eval": int(len(held)),
        "r2_heldout": r2_score(k_I[held], k_hat_held),
        "input_dim": int(X.shape[1]),
        "output_dim": int(k_I.shape[1]),
    }

    # 用重建的 k̂ 重跑 top-k（全序列），与真 k 的 top-k 比
    k_hat_all = apply_linear(X, W)
    overlaps, tau_shift = [], []
    for q, w in zip(q_samples, w_samples):
        s_true = exact_indexer_score(q, w, k_I)
        s_hat = exact_indexer_score(q, w, k_hat_all)
        kk = min(top_k, L)
        t_true = torch.topk(s_true, kk).indices
        t_hat = torch.topk(s_hat, kk).indices
        overlaps.append(len(set(t_true.tolist()) & set(t_hat.tolist())) / kk)
        tau_shift.append(float((s_hat[t_hat[-1]] - s_true[t_true[-1]]).item()))
    if overlaps:
        ov = torch.tensor(overlaps)
        out["topk_overlap_p50"] = float(ov.median().item())
        out["topk_overlap_min"] = float(ov.min().item())
        out["tau_shift_p50"] = float(torch.tensor(tau_shift).median().item())
    return out
