"""T040 closed-loop analysis (prereg 2): paired vs B0 (episode bootstrap 5000), adoption / modification size.

  python -m qvgm.analyze_closed_loop --raw /workspace/stochrl_t040/closed_loop --out /outputs/StochRL/t040
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NB = 5000


def ep_table(df):
    last = df.sort_values("dec").groupby("eid").last()
    return pd.DataFrame(dict(once=last.ever_success.astype(bool), end=last.success_now.astype(bool)))


def metrics(x, b):
    out = {}
    for k in ("once", "end"):
        xs, bs = x[k].values, b[k].values
        out[k] = xs.mean()
        out[f"{k}_preservation"] = xs[bs].mean() if bs.any() else np.nan
        out[f"{k}_recovery"] = xs[~bs].mean() if (~bs).any() else np.nan
        out[f"{k}_regression"] = 1 - out[f"{k}_preservation"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/stochrl_t040/closed_loop")
    ap.add_argument("--out", default="/outputs/StochRL/t040")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    conds = [c for c in ("b0", "qa0.05", "qa0.1", "rg0.1", "rn0.1", "rg0.05", "rn0.05", "b0rep")
             if (raw / f"{c}.parquet").exists()]
    dfs = {c: pd.read_parquet(raw / f"{c}.parquet") for c in conds}
    E = {c: ep_table(d) for c, d in dfs.items()}
    eids = E["b0"].index
    for c in conds:
        assert (E[c].index == eids).all()
    rng = np.random.default_rng(0)
    n = len(eids)
    bix = rng.integers(0, n, (NB, n))
    res = {"n_episodes": int(n), "conds": {}}
    for c in conds:
        x, b = E[c], E["b0"]
        m = metrics(x, b)
        d = dfs[c]
        r = dict(point={k: float(v) for k, v in m.items()},
                 adopt_rate=float((d.n_acc > 0).mean()), mean_acc_steps=float(d.n_acc.mean()),
                 mean_acc_steps_if_adopted=float(d.n_acc[d.n_acc > 0].mean()) if (d.n_acc > 0).any() else 0.0,
                 d_rms_mean=float(d.d_rms.mean()), d_rms_if_adopted=float(d.d_rms[d.n_acc > 0].mean())
                 if (d.n_acc > 0).any() else 0.0,
                 d_xyz_mm_mean=float(d.d_xyz_mm.mean()), d_grip_mean=float(d.d_grip.mean()),
                 dq_mean=float((d.q1 - d.q0).mean()), clip_frac=float(d["clip"].mean()), q0_mean=float(d.q0.mean()))
        if c != "b0":
            diffs = {}
            for k in ("once", "end"):
                xv, bv = x[k].values.astype(float), b[k].values.astype(float)
                bs = (xv[bix] - bv[bix]).mean(1)
                diffs[k] = dict(est=float(xv.mean() - bv.mean()), ci=[float(np.percentile(bs, 2.5)),
                                                                       float(np.percentile(bs, 97.5))],
                                n_rescue=int(((xv == 1) & (bv == 0)).sum()), n_harm=int(((xv == 0) & (bv == 1)).sum()))
            r["vs_b0"] = diffs
            r["episode_agree_once"] = float((x.once == b.once).mean())
        res["conds"][c] = r
    # QA vs random at the same alpha (paired, episode bootstrap)
    res["qa_vs_random"] = {}
    for al in ("0.05", "0.1"):
        qa = f"qa{al}"
        if qa not in E:
            continue
        for rc in (f"rg{al}", f"rn{al}"):
            if rc not in E:
                continue
            o = {}
            for k in ("once", "end"):
                xv, yv = E[qa][k].values.astype(float), E[rc][k].values.astype(float)
                bs = (xv[bix] - yv[bix]).mean(1)
                o[k] = dict(est=float(xv.mean() - yv.mean()),
                            ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            res["qa_vs_random"][f"{qa}-{rc}"] = o
    # prereg condition 2
    qas = [q for q in ("qa0.05", "qa0.1") if q in E]
    best = max(qas, key=lambda q: E[q].once.mean() + E[q].end.mean()) if qas else None
    c2 = False
    if best:
        al = best[2:]
        v = res["conds"][best]["vs_b0"]
        for k in ("once", "end"):
            if v[k]["ci"][0] > 0:
                rnds = [r for r in (f"rg{al}", f"rn{al}") if r in E]
                if rnds and all(E[best][k].mean() > E[r][k].mean() for r in rnds):
                    c2 = True
    res["condition2"] = dict(best_qa=best, satisfied=bool(c2))
    (out / "closed_loop.json").write_text(json.dumps(res, indent=1))
    tab = []
    for c in conds:
        r = res["conds"][c]
        p = r["point"]
        row = dict(cond=c, once=p["once"], end=p["end"], pres_once=p["once_preservation"],
                   rec_once=p["once_recovery"], reg_once=p["once_regression"], pres_end=p["end_preservation"],
                   rec_end=p["end_recovery"], adopt=r["adopt_rate"], acc_steps=r["mean_acc_steps"],
                   d_rms=r["d_rms_mean"], d_xyz_mm=r["d_xyz_mm_mean"], dQ=r["dq_mean"])
        if "vs_b0" in r:
            for k in ("once", "end"):
                row[f"d{k}"] = r["vs_b0"][k]["est"]; row[f"d{k}_lo"] = r["vs_b0"][k]["ci"][0]
                row[f"d{k}_hi"] = r["vs_b0"][k]["ci"][1]
        tab.append(row)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(tab).round(3).to_string())
    print(json.dumps(res["qa_vs_random"], indent=1), json.dumps(res["condition2"]))


if __name__ == "__main__":
    main()
