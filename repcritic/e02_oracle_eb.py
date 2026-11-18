"""Model-based E[max_i p_i] - E[p_i] for the 8x4 oracle: per state p_i ~ Beta(mean=pbar_s, ICC=rho),
rho = pooled between-candidate ICC (variance components). Monte Carlo over 8 draws. Also per-state
Gaussian 1.4236*sqrt(max(s_b^2,0)). usage: python -m repcritic.e02_oracle_eb analysis_main.json"""
import json, sys
import numpy as np
r = json.load(open(sys.argv[1])); rng = np.random.RandomState(0); out = {}
for m in ("success_once", "success_at_end"):
    o = r[f"oracle_{m}"]; ph = np.array(o["per_state"]["p_hat"]); vb = np.array(o["per_state"]["var_between"])
    def G(rho, idx):
        g = []
        for s in idx:
            p = ph[s].mean()
            if p <= 0 or p >= 1 or rho <= 0:
                g.append(0.0); continue
            k = 1 / rho - 1  # alpha+beta
            x = rng.beta(p * k, (1 - p) * k, size=(4000, 8))
            g.append(float((x.max(1) - x.mean(1)).mean()))
        return float(np.mean(g))
    S = len(ph); rho = o["icc_pooled"]
    est = G(rho, range(S))
    bs = []
    for _ in range(200):
        idx = rng.randint(0, S, S)
        pb = ph[idx].mean(1); tv = pb * (1 - pb); mx = tv > 0
        rb = vb[idx][mx].sum() / tv[mx].sum() if mx.any() else 0
        bs.append(G(rb, idx))
    out[m] = dict(icc=rho, G_betabinom=est, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  G_gauss_perstate=float((1.4236 * np.sqrt(np.clip(vb, 0, None))).mean()))
print(json.dumps(out, indent=1))
