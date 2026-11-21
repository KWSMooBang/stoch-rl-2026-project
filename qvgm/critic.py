"""T040 chunk critic (Q-VGM-style): 10-head ensemble Q(z, A), widths [512,512,256], A re-injected at every hidden layer,
IQL V(z) (expectile 0.8), EMA target, plus an action-only Q(A) trained on the same targets (prereg 1.2-1.4).

  CUDA_VISIBLE_DEVICES=1 python -m qvgm.critic --folds 0,1,2,3,4,all --seeds 0,1   # cross-fit + closed-loop critic
Outputs /workspace/stochrl_t040/critic/{fold}_s{seed}.pt and oof predictions oof_{fold}_s{seed}.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path("/workspace/stochrl_t040")
ADIM, ZDIM, NH = 35, 2048, 10
WIDTHS = (512, 512, 256)


class EnsLinear(nn.Module):
    def __init__(self, n, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, i, o))
        self.b = nn.Parameter(torch.zeros(n, 1, o))
        for k in range(n):
            nn.init.kaiming_uniform_(self.w.data[k].T, a=5 ** 0.5)
            bound = 1 / i ** 0.5
            nn.init.uniform_(self.b.data[k], -bound, bound)

    def forward(self, x):  # x [n, B, i]
        return torch.baddbmm(self.b, x, self.w)


class EnsLN(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n, 1, d))
        self.b = nn.Parameter(torch.zeros(n, 1, d))

    def forward(self, x):
        return Fn.layer_norm(x, x.shape[-1:]) * self.g + self.b


class QEns(nn.Module):
    """Q_h(z, A) for h = 1..n; zdim = 0 -> action-only."""

    def __init__(self, zdim=ZDIM, adim=ADIM, n=NH, widths=WIDTHS):
        super().__init__()
        self.zdim, self.n = zdim, n
        dims, layers, lns = zdim + adim, [], []
        for w in widths:
            layers.append(EnsLinear(n, dims, w)); lns.append(EnsLN(n, w))
            dims = w + adim  # action re-injection
        self.layers, self.lns = nn.ModuleList(layers), nn.ModuleList(lns)
        self.out = EnsLinear(n, dims, 1)

    def forward(self, z, a):  # -> [n, B]
        B = a.shape[0]
        a_ = a[None].expand(self.n, B, a.shape[-1])
        h = torch.cat([z[None].expand(self.n, B, z.shape[-1]), a_], -1) if self.zdim else a_
        for L, N in zip(self.layers, self.lns):
            h = torch.cat([Fn.relu(N(L(h))), a_], -1)
        return self.out(h)[..., 0]


class VNet(nn.Module):
    def __init__(self, zdim=ZDIM, widths=WIDTHS):
        super().__init__()
        mods, d = [], zdim
        for w in widths:
            mods += [nn.Linear(d, w), nn.LayerNorm(w), nn.ReLU()]
            d = w
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, z):
        return self.net(z)[:, 0]


def expectile(diff, tau=0.8):
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff ** 2).mean()


def load_data(path=ROOT / "data.npz"):
    return dict(np.load(path, allow_pickle=False))


def train(D, fold, seed, steps=15000, dev="cuda", gamma=0.99, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    bm = (D["buf_split"] != "test")
    rm = np.ones(len(D["br_G"]), bool)
    if fold != "all":
        f = int(fold)
        bm &= D["buf_fold"] != f
        rm &= D["br_fold"] != f
    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
    bz, bzn, bA = T(D["buf_z"][bm]), T(D["buf_zn"][bm]), T(D["buf_A"][bm])
    br_, bt = T(D["buf_r"][bm]), T(D["buf_term"][bm].astype(np.float32))
    rz, rA, rG = T(D["br_z"][rm]), T(D["br_A"][rm]), T(D["br_G"][rm])
    q, q_t, qa, v = QEns().to(dev), QEns().to(dev), QEns(zdim=0).to(dev), VNet().to(dev)
    q_t.load_state_dict(q.state_dict())
    for p in q_t.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([*q.parameters(), *qa.parameters(), *v.parameters()], lr=3e-4, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb, nr = len(bA), len(rA)
    t0 = time.time()
    hist = []
    for it in range(steps):
        i = torch.randint(0, nb, (256,), generator=g).to(dev)
        j = torch.randint(0, nr, (64,), generator=g).to(dev)
        with torch.no_grad():
            tq = q_t(bz[i], bA[i]).mean(0)
            y = br_[i] + gamma * (1 - bt[i]) * v(bzn[i])
        lv = expectile(tq - v(bz[i]))
        lq = ((q(bz[i], bA[i]) - y) ** 2).mean(1).sum() + ((q(rz[j], rA[j]) - rG[j]) ** 2).mean(1).sum()
        la = ((qa(None, bA[i]) - y) ** 2).mean(1).sum() + ((qa(None, rA[j]) - rG[j]) ** 2).mean(1).sum()
        loss = lv + lq + la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), q_t.parameters()):
                pt.lerp_(p, 0.005)
        if it % 2500 == 0 or it == steps - 1:
            hist.append(dict(it=it, lv=float(lv), lq=float(lq) / NH, la=float(la) / NH, s=round(time.time() - t0, 1)))
            if log:
                print(f"[{fold} s{seed}] {hist[-1]}", flush=True)
    return dict(q=q.eval(), qa=qa.eval(), v=v.eval()), dict(n_buf=int(nb), n_br=int(nr), hist=hist,
                                                            wall_s=time.time() - t0)


def qbar(m, z, a):
    return m(z, a).mean(0)


def predict(models, D, dev="cuda"):
    """Scores for every branch row + gradient at each row's action; V / Q on T017 test buffer."""
    q, qa, v = models["q"], models["qa"], models["v"]
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    z, A = T(D["br_z"]), T(D["br_A"]).requires_grad_(True)
    Qf = qbar(q, z, A)
    gf = torch.autograd.grad(Qf.sum(), A)[0]
    A2 = T(D["br_A"]).requires_grad_(True)
    Qa = qbar(qa, None, A2)
    ga = torch.autograd.grad(Qa.sum(), A2)[0]
    with torch.no_grad():
        out = dict(Qf=Qf.cpu().numpy(), Qa=Qa.cpu().numpy(), V=v(z).cpu().numpy(), Qf_std=q(z, A).std(0).cpu().numpy(),
                   gf=gf.cpu().numpy(), ga=ga.cpu().numpy())
        tm = D["buf_split"] == "test"
        out["test_V"] = v(T(D["buf_z"][tm])).cpu().numpy()
        out["test_Q"] = qbar(q, T(D["buf_z"][tm]), T(D["buf_A"][tm])).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4,all")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "T040: physical GPU 1 only"
    D = load_data()
    od = ROOT / f"critic{a.tag}"; od.mkdir(parents=True, exist_ok=True)
    for fold in a.folds.split(","):
        for seed in map(int, a.seeds.split(",")):
            if fold == "all" and seed != 0:
                continue  # closed-loop critic: seed 0 only (prereg 2)
            fn = od / f"{fold}_s{seed}.pt"
            if fn.exists():
                print("exists", fn); continue
            models, info = train(D, fold, seed, steps=a.steps, log=True)
            pr = predict(models, D)
            np.savez_compressed(od / f"oof_{fold}_s{seed}.npz", **pr)
            torch.save({k: m.state_dict() for k, m in models.items()} | dict(info=info, fold=fold, seed=seed,
                       act_scale=D["act_scale"], act_bias=D["act_bias"]), fn)
            print(json.dumps(dict(fold=fold, seed=seed, n_buf=info["n_buf"], n_br=info["n_br"],
                                  wall_s=round(info["wall_s"], 1))), flush=True)
    print("peak MB", torch.cuda.max_memory_allocated() / 2 ** 20)


if __name__ == "__main__":
    main()
