"""E06 (T020) candidate scorers for closed-loop reranking. All fixed BEFORE any closed-loop run (no test tuning).

- GeoProbe : online C-Geo z(o,l) = ((h - mu)/sd) @ W + b on vlm9_img_grid4 (T018 probe, train-fit), then the T019 critic
             standardiser (mean/sd of z_crossfit on TRAIN rows).  Same as the val/test rows the critic was evaluated on.
- PhaseGate: C-Phase probe on vlm9_txt_mean; "approach" iff argmax over the 4 phase heads is ph_approach (policy-observable,
             no privileged label).
- CriticScorer: T019 `Geo_q_buf+br` (3 seeds). Ensemble = mean over seeds of the within-candidate-set z-scored Q (T019's
             "seed ensemble" rule).  Single-seed scores also returned for logging.
- ActionOnly: T019 within-state demeaned ridge on chunk_model[..., :7] flattened (56), lambda chosen on VAL branch
             concordance -- recomputed here exactly as in repcritic/e04_actiononly.py (deterministic).

Offline check:  python -m repcritic.e06_scorers   (reproduces T019 test concordance of both scorers on branch rows)
"""
import json

import numpy as np
import torch

from repcritic.e04_critic import Critic, conc, conc_parts

T18 = "/workspace/stochrl_t018"
E4IN = f"{T18}/results/e04_inputs"
T19 = "/workspace/stochrl_t019"
CKPT = "/outputs/StochRL/t019/critic_ckpt/Geo_q_buf+br_f1.0"


class Probe:
    def __init__(self, name, dev):
        f = np.load(f"{E4IN}/{name}.npz")
        self.key = str(f["feature_key"])
        self.names = list(f["target_names"])
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)  # noqa: E731
        self.mu, self.sd, self.W, self.b = t(f["mu"]), t(f["sd"]), t(f["W"]), t(f["b"])
        self.z_ref, self.z_cf = f["z"], f["z_crossfit"]

    def __call__(self, h):
        """h: [B, ...] fp16 feature (same pooling as T017 cache) -> [B, d] float64"""
        h = torch.as_tensor(h).to(self.mu.device).reshape(h.shape[0], -1).double()
        return ((h - self.mu) / self.sd) @ self.W + self.b


class GeoZ:
    def __init__(self, dev):
        self.p = Probe("C-Geo", dev)
        tr = np.load(f"{T19}/data.npz")["split"] == "train"
        Z = self.p.z_cf.astype(np.float32)
        self.zmu = torch.as_tensor(Z[tr].mean(0), device=dev)
        self.zsd = torch.as_tensor(Z[tr].std(0) + 1e-6, device=dev)

    def __call__(self, h):
        return ((self.p(h).float() - self.zmu) / self.zsd)


class PhaseGate:
    def __init__(self, dev):
        self.p = Probe("C-Phase", dev)
        assert self.p.names[:4] == ["ph_approach", "ph_grasp_transport", "ph_post_grasp_lost", "ph_placed"]

    def __call__(self, h):
        return (self.p(h)[:, :4].argmax(1) == 0).cpu().numpy()


class CriticScorer:
    def __init__(self, dev):
        self.ms = []
        for s in range(3):
            c = torch.load(f"{CKPT}/seed{s}.pt", map_location="cpu", weights_only=False)
            assert c["cfg"]["input"] == "Geo" and c["cfg"]["obj"] == "q" and c["cfg"]["regime"] == "buf+br"
            m = Critic(c["dz"], c["kind"], "q")
            m.load_state_dict(c["state_dict"])
            self.ms.append(m.to(dev).eval())
        self.dev = dev

    @torch.no_grad()
    def __call__(self, z, A):
        """z [n, 12] standardized, A [n, N, 8, 7] model-space chunks -> ens [n,N], per-seed raw [3,n,N]"""
        n, N = A.shape[:2]
        zz = z[:, None].expand(n, N, z.shape[-1]).reshape(n * N, -1)
        aa = A.reshape(n * N, 8, 7).float()
        raw = torch.stack([m.score(zz, aa).view(n, N) for m in self.ms])
        zs = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True) + 1e-8)
        return zs.mean(0), raw


class ActionOnly:
    """Exact recomputation of T019 `ridge_action_within`."""

    def __init__(self, dev):
        D = dict(np.load(f"{T19}/data.npz"))
        bA, y, st, sp = D["b_A"].reshape(len(D["b_A"]), -1).astype(np.float64), D["b_y"], D["b_state"], D["b_split"]

        def demean(X, g):
            X = X.copy()
            for s in np.unique(g):
                m = g == s
                X[m] -= X[m].mean(0)
            return X
        tr, va = sp == "train", sp == "val"
        Xtr, ytr = demean(bA[tr], st[tr]), demean(y[tr][:, None], st[tr])[:, 0]
        best = None
        for lam in 10.0 ** np.arange(-2, 5):
            w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
            v = conc(conc_parts(bA[va] @ w, y[va], st[va]))
            if best is None or v > best[0]:
                best = (v, lam, w)
        self.val, self.lam, w = best
        self.w = torch.as_tensor(w, dtype=torch.float32, device=dev)

    @torch.no_grad()
    def __call__(self, A):
        n, N = A.shape[:2]
        return (A.reshape(n, N, -1).float() @ self.w)


def offline_check(dev="cpu"):
    """Reproduce T019 numbers (test branch concordance) + probe reproduction from cached features."""
    D = dict(np.load(f"{T19}/data.npz"))
    out = {}
    g = GeoZ(dev)
    H = np.load(f"{T18}/cache/feat_vlm9_img_grid4.npy", mmap_mode="r")
    sp = D["split"]
    rows = np.where(sp != "train")[0][:400]
    zr = g.p(np.asarray(H[rows])).cpu().numpy()
    out["geo_probe_max_abs_err_vs_T018_z(val/test rows)"] = float(np.abs(zr - g.p.z_ref[rows]).max())
    Zstd = ((torch.as_tensor(g.p.z_cf.astype(np.float32)) - g.zmu.cpu()) / g.zsd.cpu())
    cs, ao = CriticScorer(dev), ActionOnly(dev)
    te = D["b_split"] == "test"
    st = D["b_state"][te]
    y = D["b_y"][te]
    A = torch.as_tensor(D["b_A"][te])
    z = Zstd[D["b_row"][te]]
    raw = torch.stack([m.score(z, A).detach() for m in cs.ms]).numpy()
    out["critic_test_conc_per_seed"] = [conc(conc_parts(r, y, st)) for r in raw]
    # ensemble with within-state z-score
    ens = np.zeros(len(y))
    for s in np.unique(st):
        m = st == s
        r = raw[:, m]
        ens[m] = ((r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-8)).mean(0)
    out["critic_test_conc_ensemble"] = conc(conc_parts(ens, y, st))
    out["actiononly_lam"] = ao.lam
    out["actiononly_val"] = ao.val
    out["actiononly_test_conc"] = conc(conc_parts((A.reshape(len(y), -1) @ ao.w.cpu()).numpy(), y, st))
    # agreement of argmax between critic ensemble and action-only on test states (top-1 on same candidate set)
    agree, tot = 0, 0
    sa = (A.reshape(len(y), -1) @ ao.w.cpu()).numpy()
    for s in np.unique(st):
        m = np.where(st == s)[0]
        agree += int(np.argmax(ens[m]) == np.argmax(sa[m])); tot += 1
    out["argmax_agreement_critic_vs_actiononly_test_states(16 cand)"] = agree / tot
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    offline_check()
