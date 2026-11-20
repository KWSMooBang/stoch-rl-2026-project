"""T035 base-anchored GRPO: advantage and loss mask (pure torch, no RLinf import -> CPU unit tests).

Group layout (set by ResidualVLAEnv with vla_residual.anchor_n_base): group_size consecutive batch columns share one
episode id; the first n_base columns of every group are frozen-base members (executed residual 0), the rest are
residual members.
    b        = mean episode score of the n_base base members of the group
    sigma    = unbiased std of the episode scores of the residual members of the group (same estimator as RLinf GRPO)
    A_i      = (R_i - b) / (sigma + eps)                       residual members, broadcast to all their valid steps
    sigma==0 -> sigma := sigma_floor (default n_res ** -0.5 = smallest non-zero unbiased std of n_res binary scores,
                0.408 for 6). The literal "+eps only" would give |A| = 1e6 when all residual members agree with each
                other but not with the base; the floor keeps that case at |A| <= 1 / 0.408 = 2.45.
    base members: loss_mask = 0 (no policy gradient), advantage 0.
n_base == 0 is not handled here: the caller falls back to RLinf's own GRPO function.
"""
from __future__ import annotations

import json
import os
import time

import torch

EPS = 1e-6


def base_columns(batch_size: int, group_size: int, n_base: int) -> torch.Tensor:
    """Bool [batch_size]: True for base-member columns (first n_base of every group)."""
    assert batch_size % group_size == 0, f"batch {batch_size} not divisible by group_size {group_size}"
    assert 0 <= n_base < group_size, f"expected 0 <= n_base < group_size, found {n_base} / {group_size}"
    return (torch.arange(batch_size) % group_size) < n_base


def mask_base(loss_mask: torch.Tensor, group_size: int, n_base: int) -> torch.Tensor:
    """loss_mask [n_steps, batch, ...] (bool) with the base-member columns set to False (new tensor)."""
    bc = base_columns(loss_mask.shape[1], group_size, n_base).to(loss_mask.device)
    shape = [1, -1] + [1] * (loss_mask.dim() - 2)
    return loss_mask & ~bc.view(shape)


def anchor_advantages(scores: torch.Tensor, group_size: int, n_base: int, sigma_floor: float | None = None,
                      eps: float = EPS):
    """scores [n_groups, group_size] -> (advantages [n_groups, group_size] with base columns 0, stats dict)."""
    scores = scores.view(-1, group_size).float()
    assert 0 < n_base < group_size - 1, f"need >= 2 residual members, found n_base={n_base}, group_size={group_size}"
    n_res = group_size - n_base
    if sigma_floor is None:
        sigma_floor = n_res ** -0.5
    base, res = scores[:, :n_base], scores[:, n_base:]
    b = base.mean(dim=1, keepdim=True)
    sigma = res.std(dim=1, keepdim=True)
    zero = sigma < 1e-8
    sigma_used = torch.where(zero, torch.full_like(sigma, float(sigma_floor)), sigma)
    a_res = (res - b) / (sigma_used + eps)
    adv = torch.cat([torch.zeros_like(base), a_res], dim=1)
    stats = dict(b=b[:, 0], sigma=sigma[:, 0], sigma_zero=zero[:, 0], a_res=a_res, r_res=res, r_base=base)
    return adv, stats


def summarize(stats: dict, loss_mask_steps: torch.Tensor | None, group_size: int, n_base: int) -> dict:
    """JSON-able per-update summary. loss_mask_steps: valid step count per batch column (after base masking)."""
    a = stats["a_res"]
    row = dict(
        n_groups=int(a.shape[0]), n_res_members=int(a.numel()),
        adv_pos=int((a > 0).sum()), adv_neg=int((a < 0).sum()), adv_zero=int((a == 0).sum()),
        adv_mean=float(a.mean()), adv_abs_mean=float(a.abs().mean()), adv_min=float(a.min()), adv_max=float(a.max()),
        sigma_zero_groups=int(stats["sigma_zero"].sum()),
        sigma_zero_nonzero_adv_groups=int((stats["sigma_zero"] & (a.abs().sum(1) > 0)).sum()),
        base_score_mean=float(stats["r_base"].mean()), res_score_mean=float(stats["r_res"].mean()),
        base_pair_disagree_groups=int((stats["r_base"].max(1).values != stats["r_base"].min(1).values).sum()),
        b=[round(float(x), 4) for x in stats["b"]], sigma=[round(float(x), 4) for x in stats["sigma"]],
        r_res=[[round(float(x), 4) for x in r] for r in stats["r_res"]],
    )
    if loss_mask_steps is not None:
        bc = base_columns(loss_mask_steps.numel(), group_size, n_base)
        # step-weighted sign counts = what the policy loss sees (every valid step carries its episode's A)
        w = loss_mask_steps[~bc].view(-1, group_size - n_base).float()
        row.update(res_samples=int(w.sum()), base_samples_in_loss=int(loss_mask_steps[bc].sum()),
                   adv_pos_steps=int(w[a > 0].sum()), adv_neg_steps=int(w[a < 0].sum()),
                   adv_zero_steps=int(w[a == 0].sum()))
    return row


def log_row(row: dict, name: str = "anchor_adv.jsonl") -> None:
    d = os.environ.get("VR_LOG_DIR")
    if not d:
        return
    row = dict(row, wall=round(time.time(), 1))
    with open(os.path.join(d, name), "a") as f:
        f.write(json.dumps(row) + "\n")
