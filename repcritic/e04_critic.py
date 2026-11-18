"""E04 (T019) representation-aware critics f(z(o,l), E_a(a_{t:t+H}) [, g]) on frozen pi0.5 features.

Objectives (same data, same trunk/action-encoder budget):
  bce : sigmoid(f) ~ success_once after executing the chunk (buffer: logged episode outcome on pre-success decisions;
        branch: per-candidate matched-state outcome, soft for the 4-continuation oracle candidates)
  q   : semi-MDP Q^pi (gamma=0.98/step, Gamma=0.98^5, r = gamma^j at first success).  buffer: TD with EMA target and the
        logged next chunk (a sample of the frozen VLA itself -> SARSA = Q^pi);  branch: MC target.
  nce : InfoNCE f(z,a,g)=<phi(z,a), psi(g)>, positive g = frozen feature (vlm17_img_mean) of an achieved future decision
        (k ~ Geom, gamma_dec=0.7; or k=1), negatives = batch futures, same-episode negatives masked (false negatives).
        Score for G0 = logmeanexp_g f over a bank of TRAIN success-state goals of the same instruction (<=64).
        In the buf+br regime the branch rows add BCE on (score + b) (hybrid; stated as such).
Inputs z (candidate-independent, z_crossfit for concept probes; manifest T018): see INPUTS.
Train/val/test = T017 episode split. Early stopping = val objective loss. No hyper-parameter is chosen on test.

  python -m repcritic.e04_critic --shard i --nshard n [--plan main|frac|all]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repcritic.e03_probe import episode_subsample

T18 = "/workspace/stochrl_t018"
E4IN = f"{T18}/results/e04_inputs"
OUT = "/workspace/stochrl_t019"
RUNS = os.environ.get("E04_RUNS", f"{OUT}/runs")
CKPT = os.environ.get("E04_CKPT", f"{OUT}/ckpt")
GAMMA, K = 0.98, 5
GK = GAMMA ** K
HID, ZD, GD = 256, 256, 64
STEPS, BS, LR, WD, DROP = int(os.environ.get("E04_STEPS", 3000)), 256, 3e-4, 1e-4, 0.1
EVAL_EVERY = 150
BANK = 64

INPUTS = {  # name -> builder key
    "Raw": "raw_cnn",            # C0: small CNN on 64x48 RGB + proprio + t
    "Proprio": "proprio_time",   # extra cheap raw: qpos + t
    "Full": "C-Full",            # vlm17_img_mean 2048 standardized
    "Goal": "C-Goal", "Geo": "C-Geo", "Phase": "C-Phase", "Fusion": "C-Fusion",
    "RandomProj": "C-RandomProj", "PCA": "C-PCA", "Shuffle": "C-Shuffle",
    "Fusion+Full": "fusion_full",
    "AECand": "aecand13",        # candidate-DEPENDENT ablation (not usable for E07 guidance)
}


# ------------------------------------------------------------------ data
def std_tr(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def load_z(name, D, frac=None, seed=None):
    """-> (Zbuf [8000,d], Zbr [6352,d] or None (=Zbuf[b_row])), kind"""
    tr = D["split"] == "train"
    key = INPUTS.get(name, name)
    if name == "FusionLC":
        Z = np.load(f"{OUT}/lc/C-FusionLC_f{frac}_s{seed}.npy") if frac < 1 else np.load(f"{E4IN}/C-Fusion.npz")["z_crossfit"]
        return std_tr(Z, tr), None, "vec"
    if key == "raw_cnn":
        rgb = np.load(f"{T18}/cache/feat_raw_rgb64.npy").astype(np.float32) - 0.5
        pt = std_tr(np.load(f"{T18}/cache/feat_raw_proprio_time.npy").astype(np.float32), tr)
        return np.concatenate([rgb, pt], 1), None, "raw"
    if key == "proprio_time":
        return std_tr(np.load(f"{T18}/cache/feat_raw_proprio_time.npy").astype(np.float32), tr), None, "vec"
    if key == "C-Full":
        return std_tr(np.load(f"{T18}/cache/feat_vlm17_img_mean.npy").astype(np.float32), tr), None, "vec"
    if key == "fusion_full":
        a, _, _ = load_z("Fusion", D)
        b, _, _ = load_z("Full", D)
        return np.concatenate([a, b], 1), None, "vec"
    if key == "aecand13":
        Xb = np.load(f"{T18}/cache/feat_aecand13_mean.npy").astype(np.float32)
        Xc = np.load(f"{T18}/cache/cand_feat_aecand13_mean.npy").astype(np.float32)
        mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-6
        return ((Xb - mu) / sd).astype(np.float32), ((Xc - mu) / sd).astype(np.float32), "vec"
    f = np.load(f"{E4IN}/{key}.npz")
    if "z_crossfit" in f.files:
        Z = f["z_crossfit"]
    else:  # RandomProj / PCA: unsupervised projections (no label leakage)
        Z = f["z"]
    return std_tr(Z.astype(np.float32), tr), None, "vec"


# ------------------------------------------------------------------ model
class RawEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 12 + 9, ZD), nn.GELU())

    def forward(self, x):
        img = x[:, :9216].view(-1, 3, 48, 64)
        return self.fc(torch.cat([self.cnn(img), x[:, 9216:]], 1))


class Critic(nn.Module):
    def __init__(self, dz, kind, obj, adim=56, gdim=2048):
        super().__init__()
        self.obj = obj
        self.zenc = RawEnc() if kind == "raw" else nn.Sequential(nn.Linear(dz, ZD), nn.GELU())
        self.aenc = nn.Sequential(nn.Linear(adim, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(ZD + HID, HID), nn.GELU(), nn.Dropout(DROP), nn.Linear(HID, HID), nn.GELU(),
                                   nn.Dropout(DROP))
        if obj == "nce":
            self.head = nn.Linear(HID, GD)
            self.genc = nn.Sequential(nn.Linear(gdim, HID), nn.GELU(), nn.Linear(HID, GD))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.head = nn.Linear(HID, 1)

    def phi(self, z, a):
        return self.head(self.trunk(torch.cat([self.zenc(z), self.aenc(a.flatten(1))], 1)))

    def score(self, z, a, gbank=None):
        """G0 ranking score (differentiable in a). gbank: [n, BANK, gdim] goal features or psi [n,BANK,GD] if pre-encoded."""
        h = self.phi(z, a)
        if self.obj != "nce":
            return h[:, 0]
        psi = self.genc(gbank) if gbank.shape[-1] != GD else gbank
        return torch.logsumexp(torch.einsum("nd,nkd->nk", h, psi), 1) - math.log(psi.shape[1])


# ------------------------------------------------------------------ helpers
def conc_parts(s, y, groups):
    """per-group (num, den) of pairwise concordance on pairs with y_i > y_j (ties in s = 0.5)."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        ss, t = s[m], y[m]
        ds = ss[:, None] - ss[None, :]
        pm = (t[:, None] - t[None, :]) > 1e-9
        out[g] = ((ds[pm] > 0).sum() + 0.5 * (ds[pm] == 0).sum(), pm.sum())
    return out


def conc(parts):
    n = sum(v[0] for v in parts.values())
    d = sum(v[1] for v in parts.values())
    return n / d if d else float("nan")


def auroc(s, y):
    y = np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)  # average ranks for ties
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ece(p, y, nb=10):
    bins = np.clip((p * nb).astype(int), 0, nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


class Ctx:
    """GPU tensors shared by all runs."""

    def __init__(self, dev):
        D = dict(np.load(f"{OUT}/data.npz"))
        self.D, self.dev = D, dev
        self.split = D["split"]
        tr = self.split == "train"
        G = std_tr(np.load(f"{T18}/cache/feat_vlm17_img_mean.npy").astype(np.float32), tr)
        self.G = torch.from_numpy(G).to(dev)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
        self.A, self.R, self.done, self.nxt = t(D["A"]), t(D["R"]), t(D["done"]), t(D["nxt"], torch.long)
        self.bA, self.b_row = t(D["b_A"]), t(D["b_row"], torch.long)
        self.b_y, self.b_q = t(D["b_y"]), t(D["b_q"])
        self.y_buf = t(D["success_once"])
        self.val_buf = t(D["value"])
        self.dec = D["decision"]
        self.obj = D["obj_id"]
        # goal bank: TRAIN rows already in the success state, per instruction (obj id = language goal)
        rs = np.random.RandomState(0)
        bank = np.zeros((16, BANK), np.int64)
        for o in range(16):
            cand = np.where(tr & D["success_before"] & (D["obj_id"] == o))[0]
            bank[o] = rs.choice(cand, BANK, replace=len(cand) < BANK)
        self.bank = t(bank, torch.long)
        self.obj_t = t(D["obj_id"], torch.long)
        self.ep_code = t(np.unique(D["episode"], return_inverse=True)[1], torch.long)
        self.dec_t = t(D["decision"], torch.long)

    def rows(self, sp, frac=None, seed=None):
        D = self.D
        bm = (self.split == sp) & D["pre"]
        brm = D["b_split"] == sp
        if sp == "train" and frac is not None and frac < 1:
            sub = episode_subsample({"episode": D["episode"], "split": self.split}, frac, seed)
            bm &= sub
            brm &= sub[D["b_row"]]
        return np.where(bm)[0], np.where(brm)[0]

    def gbank(self, rows_obj):
        return self.G[self.bank[rows_obj]]  # [n, BANK, 2048]


def future_rows(ctx, rows, gen, mode):
    d = ctx.dec_t[rows]
    if mode == "k1":
        k = torch.ones_like(d)
    else:  # geometric, gamma_dec = 0.7
        u = torch.rand(len(rows), generator=gen, device=rows.device).clamp_min(1e-6)
        k = 1 + torch.floor(torch.log(u) / math.log(0.7)).long()
    k = torch.minimum(k, 15 - d)
    return rows + k


# ------------------------------------------------------------------ training
def batch_loss(m, ctx, Zb, Zbr, br_rows, bf_rows, obj, regime, nce_mode, gen, targ=None, pair_rows=None):
    out = {}
    z = Zb[bf_rows]
    a = ctx.A[bf_rows]
    if obj in ("bce", "pair"):
        out["buf"] = F.binary_cross_entropy_with_logits(m.score(z, a), ctx.y_buf[bf_rows])
    elif obj == "q":
        with torch.no_grad():
            nx = ctx.nxt[bf_rows].clamp_min(0)
            qn = targ.score(Zb[nx], ctx.A[nx]) if targ is not None else ctx.val_buf[nx]
            y = ctx.R[bf_rows] + (1 - ctx.done[bf_rows]) * GK * qn
        out["buf"] = F.mse_loss(m.score(z, a), y)
    else:
        ok = ctx.dec_t[bf_rows] < 15
        r = bf_rows[ok]
        fut = future_rows(ctx, r, gen, nce_mode)
        h = m.phi(Zb[r], ctx.A[r])
        psi = m.genc(ctx.G[fut])
        lg = h @ psi.T
        same = ctx.ep_code[r][:, None] == ctx.ep_code[r][None, :]
        eye = torch.eye(len(r), dtype=torch.bool, device=lg.device)
        lg = lg.masked_fill(same & ~eye, -1e9)
        out["buf"] = F.cross_entropy(lg, torch.arange(len(r), device=lg.device))
    if regime == "buf+br" and br_rows is not None and len(br_rows):
        zr = Zbr[br_rows] if Zbr is not None else Zb[ctx.b_row[br_rows]]
        ar = ctx.bA[br_rows]
        if obj in ("bce", "pair"):
            out["br"] = F.binary_cross_entropy_with_logits(m.score(zr, ar), ctx.b_y[br_rows])
        elif obj == "q":
            out["br"] = F.mse_loss(m.score(zr, ar), ctx.b_q[br_rows])
        else:
            s = m.score(zr, ar, ctx.gbank(ctx.obj_t[ctx.b_row[br_rows]])) + m.bias
            out["br"] = F.binary_cross_entropy_with_logits(s, ctx.b_y[br_rows])
    if obj == "pair" and pair_rows is not None:
        S, C = pair_rows.shape
        fr = pair_rows.flatten()
        zr = Zbr[fr] if Zbr is not None else Zb[ctx.b_row[fr]]
        sc = m.score(zr, ctx.bA[fr]).view(S, C)
        yy = ctx.b_y[fr].view(S, C)
        pm = (yy[:, :, None] - yy[:, None, :]) > 1e-6
        if pm.any():
            out["pair"] = F.softplus(-(sc[:, :, None] - sc[:, None, :])[pm]).mean()
    return out


@torch.no_grad()
def val_loss(m, ctx, Zb, Zbr, obj, regime, nce_mode, vb, vbr):
    m.eval()
    gen = torch.Generator(device=ctx.dev).manual_seed(123)
    if obj == "q":  # MC targets on val (no bootstrapping)
        l1 = F.mse_loss(m.score(Zb[vb], ctx.A[vb]), ctx.val_buf[vb]).item()
        zr = Zbr[vbr] if Zbr is not None else Zb[ctx.b_row[vbr]]
        l2 = F.mse_loss(m.score(zr, ctx.bA[vbr]), ctx.b_q[vbr]).item() if regime == "buf+br" else 0.0
        m.train()
        return l1 + l2
    tot = 0.0
    for i in range(0, len(vb), 256):  # NCE batches of 256 like training
        o = batch_loss(m, ctx, Zb, Zbr, vbr if i == 0 else None, vb[i:i + 256], obj, regime, nce_mode, gen,
                       pair_rows=state_groups(ctx, vbr) if (i == 0 and obj == "pair") else None)
        tot += sum(v.item() for v in o.values())
    m.train()
    return tot / math.ceil(len(vb) / 256)


def state_groups(ctx, br_rows):
    """branch rows -> [n_states, 16] row matrix (every branch state has 16 candidates, rows contiguous by state)."""
    r = br_rows.sort().values
    return r.view(-1, 16)


def train_one(ctx, cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = ctx.dev
    Zb_np, Zbr_np, kind = load_z(cfg["input"], ctx.D, cfg.get("frac"), cfg["seed"])
    Zb = torch.from_numpy(Zb_np).to(dev)
    Zbr = torch.from_numpy(Zbr_np).to(dev) if Zbr_np is not None else None
    m = Critic(Zb.shape[1], kind, cfg["obj"]).to(dev)
    targ = None
    if cfg["obj"] == "q":
        targ = Critic(Zb.shape[1], kind, "q").to(dev)
        targ.load_state_dict(m.state_dict())
        targ.eval()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    tb, tbr = ctx.rows("train", cfg.get("frac"), cfg["seed"])
    vb, vbr = ctx.rows("val")
    tb, tbr, vb, vbr = (torch.as_tensor(x, device=dev) for x in (tb, tbr, vb, vbr))
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    tgroups = state_groups(ctx, tbr) if len(tbr) else None
    best, best_state, best_step, curve = float("inf"), None, 0, []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(1, STEPS + 1):
        bf = tb[torch.randint(len(tb), (min(BS, len(tb)),), generator=gen, device=dev)]
        br = tbr[torch.randint(len(tbr), (min(BS, len(tbr)),), generator=gen, device=dev)] if len(tbr) else None
        pr = None
        if cfg["obj"] == "pair":
            sg = tgroups[torch.randint(len(tgroups), (32,), generator=gen, device=dev)]
            pr = torch.gather(sg, 1, torch.argsort(torch.rand(sg.shape, generator=gen, device=dev), 1)[:, :8])
        o = batch_loss(m, ctx, Zb, Zbr, br, bf, cfg["obj"], cfg["regime"], cfg["nce"], gen,
                       targ=targ if step > 200 else None, pair_rows=pr)  # first 200 steps: bootstrap from MC value (warm start)
        loss = sum(o.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if targ is not None:
            with torch.no_grad():
                for p, pt in zip(m.parameters(), targ.parameters()):
                    pt.lerp_(p, 0.005)
        if step % EVAL_EVERY == 0:
            vl = val_loss(m, ctx, Zb, Zbr, cfg["obj"], cfg["regime"], cfg["nce"], vb, vbr)
            curve.append((step, float(loss.item()), vl))
            if vl < best:
                best, best_step = vl, step
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    info = dict(train_s=time.time() - t0, peak_mb=torch.cuda.max_memory_allocated(dev) / 2**20, best_step=best_step,
                best_val_loss=best, n_train_buf=len(tb), n_train_br=len(tbr))
    return m, Zb, Zbr, info, curve


# ------------------------------------------------------------------ evaluation
def score_rows(m, ctx, Zb, Zbr, rows_buf=None, rows_br=None, A=None, grad=False):
    """Scores for buffer rows (executed action) or branch rows (candidate action); A overrides the actions."""
    if rows_br is not None:
        z = Zbr[rows_br] if Zbr is not None else Zb[ctx.b_row[rows_br]]
        a = ctx.bA[rows_br] if A is None else A
        objr = ctx.obj_t[ctx.b_row[rows_br]]
    else:
        z = Zb[rows_buf]
        a = ctx.A[rows_buf] if A is None else A
        objr = ctx.obj_t[rows_buf]
    psi = None
    if m.obj == "nce":
        with torch.no_grad():
            psi_all = m.genc(ctx.G[ctx.bank])  # [16, BANK, GD]
        psi = psi_all[objr]
    if grad:
        a = a.detach().clone().requires_grad_(True)
        s = m.score(z, a, psi)
        g, = torch.autograd.grad(s.sum(), a)
        return s.detach(), g.detach(), a
    with torch.no_grad():
        out = []
        for i in range(0, len(z), 2048):
            out.append(m.score(z[i:i + 2048], a[i:i + 2048], None if psi is None else psi[i:i + 2048]))
        return torch.cat(out)


def evaluate(m, ctx, Zb, Zbr, sp, cfg, deep=False):
    D = ctx.D
    res = {}
    brr = np.where(D["b_split"] == sp)[0]
    br_t = torch.as_tensor(brr, device=ctx.dev)
    s = score_rows(m, ctx, Zb, Zbr, rows_br=br_t).cpu().numpy()
    st = D["b_state"][brr]
    y = D["b_y"][brr]
    nc = D["b_ncont"][brr]
    res["conc_1c_parts"] = conc_parts(s, y, st)
    res["conc_1c"] = conc(res["conc_1c_parts"])
    om = nc >= 4
    res["conc_or_parts"] = conc_parts(s[om], y[om], st[om])
    res["conc_or"] = conc(res["conc_or_parts"])
    # selection gain (argmax vs mean) on single-continuation and oracle p-hat
    g1, go = [], []
    for g in np.unique(st):
        mm = st == g
        g1.append(y[mm][np.argmax(s[mm])] - y[mm].mean())
        mo = mm & om
        if mo.any():
            go.append(y[mo][np.argmax(s[mo])] - y[mo].mean())
    res["pick_gain_1c"], res["pick_gain_or"] = float(np.mean(g1)), float(np.mean(go)) if go else float("nan")
    one = nc == 1
    res["auroc_br"] = auroc(s[one], y[one])
    # action reliance: share of score variance that is within-state
    wv = np.mean([s[st == g].var() for g in np.unique(st)])
    res["within_var_frac"] = float(wv / (s.var() + 1e-12))
    # buffer pre-success decisions (executed action)
    bb = np.where((D["split"] == sp) & D["pre"])[0]
    sb = score_rows(m, ctx, Zb, Zbr, rows_buf=torch.as_tensor(bb, device=ctx.dev)).cpu().numpy()
    yb = D["success_once"][bb].astype(float)
    res["auroc_buf"] = auroc(sb, yb)
    aw = [auroc(sb[D["decision"][bb] == d], yb[D["decision"][bb] == d]) for d in range(16)]
    wts = [((D["decision"][bb] == d)).sum() for d in range(16)]
    aw = np.array(aw)
    ok = ~np.isnan(aw)
    res["auroc_buf_within_t"] = float(np.average(aw[ok], weights=np.array(wts)[ok]))
    if m.obj in ("bce", "pair"):
        p = 1 / (1 + np.exp(-s[one]))
        res["ece_br"], res["brier_br"] = ece(p, y[one]), float(np.mean((p - y[one]) ** 2))
        pb = 1 / (1 + np.exp(-sb))
        res["ece_buf"], res["brier_buf"] = ece(pb, yb), float(np.mean((pb - yb) ** 2))
    res["_scores_br"] = s
    if deep:
        res.update(ood_and_grad(m, ctx, Zb, Zbr, brr, s, cfg))
    return res


def ood_and_grad(m, ctx, Zb, Zbr, brr, s_in, cfg):
    D = ctx.D
    dev = ctx.dev
    out = {}
    br_t = torch.as_tensor(brr, device=dev)
    A0 = ctx.bA[br_t]
    st = D["b_state"][brr]
    asd = ctx.A.reshape(-1, 7).std(0)  # per-dim action std (buffer)
    gen = torch.Generator(device=dev).manual_seed(7)
    groups = {g: np.where(st == g)[0] for g in np.unique(st)}
    smax = np.zeros(len(brr))
    smu = np.zeros(len(brr))
    ssd = np.zeros(len(brr))
    for g, ii in groups.items():
        smax[ii], smu[ii], ssd[ii] = s_in[ii].max(), s_in[ii].mean(), s_in[ii].std() + 1e-6

    def ood_stats(tag, A):
        so = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=A).cpu().numpy()
        out[f"ood_{tag}_win"] = float((so > smax).mean())       # beats every in-distribution candidate of the state
        out[f"ood_{tag}_z"] = float(np.mean((so - smu) / ssd))   # shift in units of within-state candidate spread

    for sig in (0.5, 1.0, 2.0):
        ood_stats(f"gauss{sig}", A0 + sig * asd * torch.randn(A0.shape, generator=gen, device=dev))
    ood_stats("uniform", torch.rand(A0.shape, generator=gen, device=dev) * 2 - 1)
    perm = torch.randperm(len(A0), generator=gen, device=dev)
    ood_stats("otherstate", A0[perm])
    # ---- gradients wrt action (normalized model space), for E07
    s, g, _ = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, grad=True)
    gn = g.flatten(1).norm(dim=1)
    out["grad_finite_frac"] = float(torch.isfinite(g).all(dim=(1, 2)).float().mean())
    out["grad_norm_med"], out["grad_norm_p90"] = float(gn.median()), float(gn.quantile(0.9))
    for nm, sl in (("trans", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7))):
        out[f"grad_share_{nm}"] = float((g[..., sl] ** 2).sum((1, 2)).div(gn ** 2 + 1e-12).mean())
    # finite-difference check along random unit directions
    u = torch.randn(A0.shape, generator=gen, device=dev)
    u = u / u.flatten(1).norm(dim=1)[:, None, None]
    eps = 1e-2
    fp = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=A0 + eps * u)
    fm = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=A0 - eps * u)
    fd = (fp - fm) / (2 * eps)
    ad = (g * u).sum((1, 2))
    out["fd_relerr_med"] = float(((fd - ad).abs() / (ad.abs() + fd.abs() + 1e-8)).median())
    # smoothness: gradient direction stability under small action noise (0.1 std)
    _, g2, _ = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=A0 + 0.1 * asd * torch.randn(A0.shape, generator=gen, device=dev),
                          grad=True)
    out["grad_cos_local"] = float(F.cosine_similarity(g.flatten(1), g2.flatten(1)).median())
    # gradient alignment with better candidates of the same state: cos(grad f(a_j), a_i - a_j) for y_i > y_j
    y = D["b_y"][brr]
    nc = D["b_ncont"][brr]
    al1, alo = [], []
    gf, Af = g.flatten(1).cpu().numpy(), A0.flatten(1).cpu().numpy()
    for gg, ii in groups.items():
        for ia in ii:
            for jb in ii:
                if y[ia] > y[jb] + 1e-9:
                    d = Af[ia] - Af[jb]
                    c = float(gf[jb] @ d / (np.linalg.norm(gf[jb]) * np.linalg.norm(d) + 1e-12))
                    (alo if (nc[ia] >= 4 and nc[jb] >= 4) else al1).append(c > 0)
    out["grad_align_1c"] = float(np.mean(al1)) if al1 else float("nan")
    out["grad_align_or"] = float(np.mean(alo)) if alo else float("nan")
    # gradient-ascent hacking probe: 20 normalized ascent steps of size 0.05 (L2 over the chunk)
    a = A0.clone()
    for _ in range(20):
        _, ga, _ = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=a, grad=True)
        a = a + 0.05 * ga / (ga.flatten(1).norm(dim=1)[:, None, None] + 1e-12)
    sa = score_rows(m, ctx, Zb, Zbr, rows_br=br_t, A=a).cpu().numpy()
    out["ascent_dz"] = float(np.mean((sa - smu) / ssd))
    out["ascent_win"] = float((sa > smax).mean())
    out["ascent_clip_frac"] = float((a.abs() > 1.0).float().mean())
    return out


# ------------------------------------------------------------------ plans
def plan(which):
    cfgs = []
    if which in ("main", "all"):
        for inp in INPUTS:
            for obj in ("bce", "q", "nce"):
                for regime in ("buf+br", "buf"):
                    for seed in range(3):
                        cfgs.append(dict(input=inp, obj=obj, regime=regime, nce="geom", seed=seed, frac=1.0))
        for inp in ("Full", "Fusion"):
            for regime in ("buf+br", "buf"):
                for seed in range(3):
                    cfgs.append(dict(input=inp, obj="nce", regime=regime, nce="k1", seed=seed, frac=1.0))
        for inp in ("Raw", "Proprio", "Full", "Fusion", "RandomProj", "AECand"):  # diagnostic: + within-state pair loss
            for seed in range(3):
                cfgs.append(dict(input=inp, obj="pair", regime="buf+br", nce="geom", seed=seed, frac=1.0))
    if which in ("frac", "all"):
        for inp in ("Raw", "Proprio", "Full", "Fusion", "RandomProj", "FusionLC"):
            for obj in ("bce", "q", "nce"):
                for frac in (0.1, 0.25, 0.5):
                    for seed in range(3):
                        cfgs.append(dict(input=inp, obj=obj, regime="buf+br", nce="geom", seed=seed, frac=frac))
    return cfgs


def cfg_name(c):
    return f"{c['input']}_{c['obj']}{'' if c['nce'] == 'geom' or c['obj'] != 'nce' else '-k1'}_{c['regime']}_f{c['frac']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"), "use GPU 0 or 1 only"
    dev = torch.device("cuda")
    ctx = Ctx(dev)
    cfgs = plan(args.plan)
    if args.only:
        cfgs = [c for c in cfgs if args.only in cfg_name(c)]
    cfgs = cfgs[args.shard::args.nshard]
    if args.limit:
        cfgs = cfgs[:args.limit]
    os.makedirs(f"{RUNS}", exist_ok=True)
    fout = open(f"{RUNS}/results_{args.plan}_{args.shard}.jsonl", "a")
    done = set()
    for f in os.listdir(f"{RUNS}"):
        if f.endswith(".jsonl"):
            for line in open(f"{RUNS}/{f}"):
                r = json.loads(line)
                done.add((r["name"], r["seed"]))
    t_start = time.time()
    for c in cfgs:
        name = cfg_name(c)
        if (name, c["seed"]) in done:
            continue
        m, Zb, Zbr, info, curve = train_one(ctx, c)
        row = dict(name=name, **c, **info, curve=curve)
        deep = c["frac"] == 1.0
        for sp in ("val", "test"):
            r = evaluate(m, ctx, Zb, Zbr, sp, c, deep=deep)
            for k, v in r.items():
                if k.endswith("_parts"):
                    row[f"{sp}_{k}"] = {g: [float(a), float(b)] for g, (a, b) in v.items()}
                elif k == "_scores_br":
                    row[f"{sp}_scores_br"] = [round(float(x), 5) for x in v]
                else:
                    row[f"{sp}_{k}"] = v
        if c["frac"] == 1.0:
            os.makedirs(f"{CKPT}/{name}", exist_ok=True)
            torch.save(dict(state_dict=m.state_dict(), cfg=c, dz=Zb.shape[1],
                            kind="raw" if c["input"] == "Raw" else "vec"), f"{CKPT}/{name}/seed{c['seed']}.pt")
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"[{time.time() - t_start:7.0f}s] {name} s{c['seed']} best@{info['best_step']} "
              f"val conc {row['val_conc_1c']:.3f} or {row['val_conc_or']:.3f} | train {info['train_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
