"""E06 (T020) report: per-condition success (Wilson), paired episode bootstrap differences, McNemar, latency, diagnostics.

  python -m repcritic.e06_report --out /outputs/StochRL/t020
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SRC = Path("/workspace/stochrl_t020")
CONDS = ["B0", "Random8", "ActOnly8", "Critic8", "CriticApp8"]
PAIRS = [("Critic8", "ActOnly8"), ("Critic8", "B0"), ("Critic8", "Random8"), ("ActOnly8", "B0"), ("Random8", "B0"),
         ("CriticApp8", "B0"), ("CriticApp8", "ActOnly8"), ("ActOnly8", "Random8")]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t020")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for f in sorted(glob.glob(str(SRC / "episodes_p*.jsonl"))) for l in open(f)]
    df = pd.DataFrame(rows)
    # keep only episodes completed under every condition (paired)
    cnt = df.groupby("eid").cond.nunique()
    full = cnt[cnt == len(CONDS)].index
    df = df[df.eid.isin(full)]
    n = len(full)
    res = dict(n_episodes=int(n), conds={}, pairs={})
    recs = []
    for c in CONDS:
        d = df[df.cond == c]
        r = dict(cond=c, n=len(d))
        for m in ("success_once", "success_at_end"):
            k = int(d[m].sum())
            lo, hi = wilson(k, len(d))
            r[m] = k / len(d); r[f"{m}_lo"] = lo; r[f"{m}_hi"] = hi
        r["mean_steps"] = float(d.steps.mean())
        # amortised latency per env-decision (batch of n_batch envs)
        r["ms_per_decision_batch"] = float(d.groupby("batch").tot_ms.first().mean())
        r["ms_per_env_decision"] = float((d.tot_ms / d.n_batch).mean())
        r["score_ms_per_batch_decision"] = float(d.groupby("batch").score_ms.first().mean())
        r["frac_sel_c0"] = float(d.frac_sel_c0.mean())
        r["frac_gate"] = float(d.frac_gate.mean())
        recs.append(r)
        res["conds"][c] = r
    piv = {m: df.pivot(index="eid", columns="cond", values=m).astype(float) for m in ("success_once", "success_at_end")}
    rs = np.random.RandomState(0)
    B = 10000
    idx = rs.randint(0, n, size=(B, n))
    for a, b in PAIRS:
        pr = {}
        for m, P in piv.items():
            diff = (P[a] - P[b]).values
            bs = diff[idx].mean(1)
            n10, n01 = int(((P[a] == 1) & (P[b] == 0)).sum()), int(((P[a] == 0) & (P[b] == 1)).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            pr[m] = dict(diff=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                         a_only=n10, b_only=n01, mcnemar_p=float(p))
        res["pairs"][f"{a}-{b}"] = pr
        for m, v in pr.items():
            recs.append(dict(cond=f"diff:{a}-{b}", metric=m, **v))
    # decision diagnostics
    diag = {}
    for c in CONDS:
        fs = sorted(glob.glob(str(SRC / "decisions" / f"p*_{c}.npz")))
        D = [np.load(f) for f in fs]
        ph = np.concatenate([d["phase"] for d in D])
        sel = np.concatenate([d["sel"] for d in D])
        dv = np.concatenate([d["div"] for d in D])
        g = np.concatenate([d["gate"] for d in D])
        cl = np.concatenate([d["clip_frac"] for d in D])
        dd = dict(sel_hist=np.bincount(sel.ravel(), minlength=8).tolist(), mean_div=float(dv.mean()),
                  clip_frac=float(cl.mean()),
                  phase_frac={p: float((ph == p).mean()) for p in np.unique(ph)})
        if c == "CriticApp8":
            dd["gate_rate"] = float(g.mean())
            dd["gate_given_priv_approach"] = float(g[ph == "approach"].mean())
            dd["gate_given_priv_not_approach"] = float(g[ph != "approach"].mean())
        diag[c] = dd
    # d=0: identical candidate sets in all N=8 conds -> agreement of argmax between Critic8 and ActOnly8
    agree = []
    for fc in sorted(glob.glob(str(SRC / "decisions" / "p*_Critic8.npz"))):
        fa = fc.replace("_Critic8", "_ActOnly8")
        c, a = np.load(fc), np.load(fa)
        agree.append(c["sel"][:, 0] == a["sel"][:, 0])
    diag["d0_argmax_agree_Critic8_vs_ActOnly8"] = float(np.concatenate(agree).mean())
    diag["d0_argmax_agree_chance"] = 1 / 8
    res["diag"] = diag
    bench = SRC / "bench_latency.json"
    if bench.exists():
        res["bench_batch1"] = json.loads(bench.read_text())
    pd.DataFrame(recs).to_csv(out / "rerank_metrics.csv", index=False)
    df.to_csv(out / "rerank_episodes.csv", index=False)
    (out / "rerank_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
