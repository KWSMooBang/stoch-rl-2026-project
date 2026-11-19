"""T024: frozen T023 success predictors as PBRS potentials Phi(s) inside ResidualVLAEnv (torch port of
repcritic.t023_value_audit.load_potential, Platt calibration; checked against it in validate_t024 --mode offline).

Inputs per env slot at a decision boundary (the state the policy sees):
    feat_raw [B, 2048]  VLM layer-17 image-token mean, NOT standardized / clipped (T018 cache convention)
    qpos     [B, 8]     wrapped-env "states" (T017 buffer `state` convention)
    dec      [B]        decision index 0..16 (time lookup has 16 entries; d=16 uses entry 15)
Names: vla | raw_q | time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

PRED_DIR = Path("/outputs/StochRL/t023/predictors")


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def _poly2(X):
    d = X.shape[1]
    iu = torch.triu_indices(d, d, device=X.device)
    return torch.cat([X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]], 1)


class Potential:
    def __init__(self, name, device="cpu", root=PRED_DIR):
        self.name = name
        cal = json.load(open(Path(root) / "calibration.json"))[name]
        self.a, self.b = float(cal["platt_a"]), float(cal["platt_b"])
        dd = dict(dtype=torch.float64, device=device)
        if name == "time":
            self.rate = torch.as_tensor(np.load(Path(root) / "lookup.npz")["time"], **dd)
            return
        P = dict(np.load(Path(root) / f"{name}.npz", allow_pickle=True))
        assert "Q" not in P, f"{name}: random-projection predictors are not supported"
        self.kind = str(P["kind"])
        self.feature = str(P["feature"])
        exp = {"vla": "vlm17_img_mean", "raw_q": "raw_proprio"}[name]
        assert self.feature == exp, f"{name}: expected feature {exp}, found {self.feature}"
        self.W = torch.as_tensor(P["W"], **dd)
        self.b0 = float(P["b"])
        self.mu = torch.as_tensor(P["mu"], **dd)
        self.sd = torch.as_tensor(P["sd"], **dd)
        if self.kind == "poly2":
            self.mu2 = torch.as_tensor(P["mu2"], **dd)
            self.sd2 = torch.as_tensor(P["sd2"], **dd)

    def uncalibrated(self, feat_raw=None, qpos=None, dec=None):
        if self.name == "time":
            d = dec.to(self.rate.device).long().clamp(0, len(self.rate) - 1)
            return self.rate[d]
        X = feat_raw if self.name == "vla" else qpos
        X = X.to(self.W.device, torch.float64)
        Z = (X - self.mu) / self.sd
        if self.kind == "poly2":
            Z = (_poly2(Z) - self.mu2) / self.sd2
        return torch.sigmoid(Z @ self.W + self.b0)

    def __call__(self, feat_raw=None, qpos=None, dec=None):
        """Platt-calibrated V_hat in (0, 1), float32 [B]."""
        v = self.uncalibrated(feat_raw, qpos, dec)
        return torch.sigmoid(self.a * _logit(v) + self.b).float()


def shaped_reward(r, phi, phi_next, term, beta, gamma, phi_term=1.0, active=None):
    """r' = r + beta (gamma Phi(s') - Phi(s)); Phi(s') = phi_term on success termination; truncation keeps V_hat(s').
    `active` masks steps after the episode already ended (auto_reset=False tails; loss-masked anyway)."""
    pn = torch.where(term, torch.full_like(phi_next, phi_term), phi_next)
    shp = beta * (gamma * pn - phi)
    if active is not None:
        shp = torch.where(active, shp, torch.zeros_like(shp))
    return r + shp, shp
