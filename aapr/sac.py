"""Minimal SAC (Haarnoja et al. 2018) for CPU, identical network/hyper-parameters for all methods."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, od, ad):
        super().__init__()
        self.net = mlp(od, 2 * ad)

    def forward(self, o, deterministic=False):
        mu, log_std = self.net(o).chunk(2, -1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        z = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(-1)
        return a, logp


class Replay:
    def __init__(self, od, ad, cap):
        self.o = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros(cap, np.float32); self.o2 = np.zeros((cap, od), np.float32)
        self.d = np.zeros(cap, np.float32); self.n = 0; self.cap = cap; self.i = 0

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, rng, b):
        idx = rng.integers(0, self.n, b)
        return (torch.from_numpy(self.o[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.o2[idx]), torch.from_numpy(self.d[idx]))


class SAC:
    def __init__(self, od, ad, gamma=0.99, tau=0.005, lr=3e-4, alpha0=0.05, target_entropy=None, seed=0):
        torch.manual_seed(seed)
        self.actor = Actor(od, ad)
        self.q1, self.q2 = mlp(od + ad, 1), mlp(od + ad, 1)
        self.q1t, self.q2t = mlp(od + ad, 1), mlp(od + ad, 1)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(np.log(alpha0), dtype=torch.float32, requires_grad=True)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -ad if target_entropy is None else target_entropy

    @torch.no_grad()
    def act(self, o, deterministic=False):
        a, _ = self.actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), deterministic)
        return a[0].numpy().astype(np.float64)

    def update(self, batch):
        o, a, r, o2, d = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            oa2 = torch.cat([o2, a2], -1)
            qt = torch.min(self.q1t(oa2), self.q2t(oa2)).squeeze(-1) - alpha * logp2
            y = r + self.gamma * (1 - d) * qt
        oa = torch.cat([o, a], -1)
        q1, q2 = self.q1(oa).squeeze(-1), self.q2(oa).squeeze(-1)
        lq = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); lq.backward(); self.q_opt.step()

        an, logp = self.actor(o)
        oan = torch.cat([o, an], -1)
        qn = torch.min(self.q1(oan), self.q2(oan)).squeeze(-1)
        lpi = (alpha * logp - qn).mean()
        self.pi_opt.zero_grad(); lpi.backward(); self.pi_opt.step()

        la = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); la.backward(); self.a_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(lq), float(lpi), float(alpha), float(q1.mean().detach())
