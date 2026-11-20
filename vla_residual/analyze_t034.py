"""T034 analysis (CPU): residual size / direction before vs after success, E0 vs E2 vs B0.

Input: per-run decision logs written by vla_residual/eval_t034.py. Output: decision_logs.parquet (all runs),
analysis.json, tables.md, residual_by_phase.png.

Phase of a decision = state in which the decision is taken:
  pre        not yet succeeded (includes the decision that reaches the first success)
  post       any decision after the first success            (prev_ever_success)
  post1      the first decision after the first success
  in_succ    success flag currently true                     (prev_success_now)
    hold     ... and still true after the decision
    lose     ... and false after the decision  ("loss decision")
  lost       succeeded before, flag currently false
Means are decision-weighted ratios (sum over decisions / number of decisions); CIs are two-stage bootstraps that
resample episode ids (shared by all runs) and seeds (within condition), percentile 95 %.

  python vla_residual/analyze_t034.py --raw /outputs/StochRL/t034/raw --out /outputs/StochRL/t034
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COMP = ["x", "y", "z", "rx", "ry", "rz", "grip"]
ALPHA = np.array([0.02] * 3 + [0.1] * 3 + [2.0])
PHASES = ["pre", "post", "post1", "in_succ", "hold", "lose", "lost"]
METRICS = {  # name -> (column, scale, unit)
    "tanh_norm": ("res_tanh_norm", 1.0, "‖tanh u‖ (0–2.65)"),
    "pos_norm_mm": ("res_pos_norm", 1e3, "mm/step"),
    "rot_norm": ("res_rot_norm", 1.0, "rad/step"),
    "abs_grip": ("res_abs_grip", 1.0, "grip units"),
    "grip": ("res_grip", 1.0, "signed (+ open)"),
    "z_mm": ("res_z", 1e3, "signed mm/step (+ up)"),
    "xy_norm_mm": ("res_xy_norm", 1e3, "mm/step"),
    "base_pos_norm_mm": ("base_pos_norm", 1e3, "mm/step"),
    "base_grip": ("base_grip", 1.0, "signed"),
    "base_z_mm": ("base_z", 1e3, "mm/step"),
    "exec_grip": ("exec_grip", 1.0, "signed"),
    "exec_z_mm": ("exec_z", 1e3, "mm/step"),
    "ee_above_plate_mm": ("pre_ee_dz_plate", 1e3, "mm, privileged, before the decision"),
    "ee_carrot_mm": ("pre_ee_carrot", 1e3, "mm, privileged, before the decision"),
    "grip_flip_close": ("grip_flip_close", 1.0, "frac"),
    "grip_flip_open": ("grip_flip_open", 1.0, "frac"),
}
COL = {"E2": "#2a78d6", "E0": "#eb6834", "B0": "#1baf7a"}


def load(raw):
    dfs = [pd.read_parquet(p) for p in sorted(Path(raw).glob("*.parquet"))]
    d = pd.concat(dfs, ignore_index=True)
    d["res_abs_grip"] = d.res_grip.abs()
    d["res_xy_norm"] = np.hypot(d.res_x, d.res_y)
    d["base_pos_norm"] = np.sqrt(d.base_x ** 2 + d.base_y ** 2 + d.base_z ** 2)
    bc = np.stack(d.base_chunk.to_numpy()).reshape(len(d), 5, 7)[:, :, 6]
    ec = np.stack(d.exec_chunk.to_numpy()).reshape(len(d), 5, 7)[:, :, 6]
    d["grip_flip_close"] = ((bc > 0) & (ec < 0)).any(1)  # base says open, executed command closes
    d["grip_flip_open"] = ((bc < 0) & (ec > 0)).any(1)
    first = d[d.ever_success].groupby(["name", "eid"]).dec.min().rename("d_first")
    d = d.join(first, on=["name", "eid"])
    d["rel"] = d.dec - d.d_first  # 0 = decision reaching first success; NaN = never succeeded
    d["ph_pre"] = ~d.prev_ever_success
    d["ph_post"] = d.prev_ever_success
    d["ph_post1"] = d.rel == 1
    d["ph_in_succ"] = d.prev_success_now
    d["ph_hold"] = d.prev_success_now & d.success_now
    d["ph_lose"] = d.prev_success_now & ~d.success_now
    d["ph_lost"] = d.prev_ever_success & ~d.prev_success_now
    d["d_cp_dz"] = d.post_cp_dz - d.pre_cp_dz
    d["d_cp_xy"] = d.post_cp_xy - d.pre_cp_xy
    return d


class Boot:
    """Two-stage bootstrap over (seed within condition, episode id)."""

    def __init__(self, d, n=5000, seed=0):
        self.d = d
        self.eids = np.sort(d.eid.unique())
        self.pos = {e: i for i, e in enumerate(self.eids)}
        self.names = {c: sorted(d[d.cond == c].name.unique()) for c in sorted(d.cond.unique())}
        g = np.random.default_rng(seed)
        E = len(self.eids)
        self.eidx = g.integers(0, E, (n, E))
        self.sidx = {c: g.integers(0, len(v), (n, len(v))) for c, v in self.names.items()}
        self.n = n

    def sums(self, val, mask):
        """-> {cond: (S[n_seed,E], N[n_seed,E])}"""
        d = self.d
        out = {}
        ei = d.eid.map(self.pos).to_numpy()
        v = np.where(mask, val, 0.0).astype(float)
        m = mask.astype(float)
        for c, names in self.names.items():
            S = np.zeros((len(names), len(self.eids))); N = np.zeros_like(S)
            for k, nm in enumerate(names):
                sel = (d.name == nm).to_numpy()
                np.add.at(S[k], ei[sel], v[sel]); np.add.at(N[k], ei[sel], m[sel])
            out[c] = (S, N)
        return out

    @staticmethod
    def _stat(S, N):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(S.sum(-1) / N.sum(-1), axis=-1)  # mean over seeds of per-run ratio

    def summary(self, val, mask):
        sn = self.sums(np.asarray(val, float), np.asarray(mask, bool))
        res = {}
        bs = {}
        for c, (S, N) in sn.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                per = S.sum(1) / N.sum(1)
            b = np.empty(self.n)
            for i in range(self.n):
                s = self.sidx[c][i]; e = self.eidx[i]
                b[i] = self._stat(S[s][:, e], N[s][:, e])
            bs[c] = b
            res[c] = dict(mean=float(np.nanmean(per)), per_seed=[float(x) for x in per], n_dec=[int(x) for x in N.sum(1)],
                          n_ep=[int((x > 0).sum()) for x in N], ci=[float(x) for x in np.nanpercentile(b, [2.5, 97.5])])
        if "E0" in bs and "E2" in bs:
            df = bs["E0"] - bs["E2"]
            res["E0-E2"] = dict(mean=res["E0"]["mean"] - res["E2"]["mean"],
                                ci=[float(x) for x in np.nanpercentile(df, [2.5, 97.5])])
        return res


def mlp(sd, x):
    h = x
    for i in (0, 2, 4):
        h = torch.tanh(h @ sd[f"backbone.{i}.weight"].T + sd[f"backbone.{i}.bias"])
    return h @ sd["actor_mean.weight"].T + sd["actor_mean.bias"]


def cross_policy(d, raw, runs):
    """All six actors evaluated offline on the states visited by B0 (identical state distribution for every actor)."""
    from vla_residual.eval_t034 import RUNS, ckpt_file

    b = d[d.name == "b0"].reset_index(drop=True)
    if not len(b):
        return None
    feat = np.load(Path(raw) / "b0_feat.npz")["feat"].astype(np.float32)
    o44 = np.stack(b.obs44.to_numpy()).astype(np.float32)
    assert len(feat) == len(b)
    out = {}
    for nm in runs:
        cond, seed, rid = RUNS[nm]
        sd = torch.load(ckpt_file(rid), map_location="cpu")
        f = np.zeros_like(feat) if cond == "E0" else feat
        u = mlp(sd, torch.from_numpy(np.concatenate([f, o44], 1))).numpy()
        t = np.tanh(u)
        out[nm] = dict(cond=cond, tanh=t)
    rows = []
    for ph in ("pre", "post", "in_succ"):
        m = b[f"ph_{ph}"].to_numpy()
        for nm, o in out.items():
            t = o["tanh"][m]
            rows.append(dict(phase=ph, name=nm, cond=o["cond"], n=int(m.sum()), tanh_norm=float(np.linalg.norm(t, axis=1).mean()),
                             pos_norm_mm=float(np.linalg.norm(t[:, :3] * 0.02, axis=1).mean() * 1e3),
                             grip=float((t[:, 6] * 2).mean()), z_mm=float((t[:, 2] * 20).mean()),
                             abs_grip=float(np.abs(t[:, 6] * 2).mean())))
    return pd.DataFrame(rows)


def fmt(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def phase_table(res, metric, phases):
    L = ["| 구간 | E2 평균 (seed별) | E0 평균 (seed별) | E0 − E2 [95 % CI] | B0 |", "|---|---|---|---|---|"]
    for ph in phases:
        r = res[metric][ph]
        cell = lambda c: (f"{fmt(r[c]['mean'])} ({', '.join(fmt(v) for v in r[c]['per_seed'])})" if c in r else "–")
        dd = r.get("E0-E2")
        L.append(f"| {ph} | {cell('E2')} | {cell('E0')} | "
                 + (f"{dd['mean']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]" if dd else "–")
                 + f" | {fmt(r['B0']['mean']) if 'B0' in r else '–'} |")
    return "\n".join(L)


def loss_mechanism(d):
    """Physical change of the object during loss decisions + what the residual did there."""
    L = d[d.ph_lose].copy()
    L["mech"] = np.where(L.d_cp_dz > 0.01, "lift(>1cm up)", np.where(L.d_cp_xy.abs() > 0.01, "xy move(>1cm)", "small move"))
    rows = []
    for (cond, nm), g in L.groupby(["cond", "name"]):
        r = dict(cond=cond, name=nm, n_lose=len(g))
        for k in ("lift(>1cm up)", "xy move(>1cm)", "small move"):
            r[k] = float((g.mech == k).mean())
        r.update(pre_grasp=float(g.pre_grasp.mean()), post_grasp=float(g.post_grasp.mean()),
                 pre_fingers_mm=float(g.pre_fingers.mean() * 1e3),
                 pre_ee_above_plate_mm=float(g.pre_ee_dz_plate.mean() * 1e3), pre_ee_carrot_mm=float(g.pre_ee_carrot.mean() * 1e3),
                 d_xy_mm=float(g.d_cp_xy.mean() * 1e3), d_dz_mm=float(g.d_cp_dz.mean() * 1e3),
                 res_grip=float(g.res_grip.mean()), base_grip=float(g.base_grip.mean()), exec_grip=float(g.exec_grip.mean()),
                 res_z_mm=float(g.res_z.mean() * 1e3), base_z_mm=float(g.base_z.mean() * 1e3),
                 exec_z_mm=float(g.exec_z.mean() * 1e3), res_xy_mm=float(g.res_xy_norm.mean() * 1e3),
                 flip_close=float(g.grip_flip_close.mean()), flip_open=float(g.grip_flip_open.mean()),
                 exec_close=float((g.exec_grip < 0).mean()), base_close=float((g.base_grip < 0).mean()))
        rows.append(r)
    return pd.DataFrame(rows), L


def figure(d, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a",
                         "ytick.color": "#3d3d3a", "axes.grid": True, "grid.color": "#e8e7e1", "grid.linewidth": 0.8,
                         "axes.axisbelow": True})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
    # ---- row 1: residual size by phase (bars = condition mean, dots = seeds, whiskers = bootstrap CI)
    phs = ["pre", "post1", "hold", "lose", "lost"]
    lab = ["pre-success", "1st decision\nafter success", "in success:\nheld", "in success:\nlost (loss dec.)", "after loss"]
    for a, (met, yl) in zip(ax[0], [("tanh_norm", "residual size  ‖tanh(u)‖"), ("pos_norm_mm", "position residual  (mm / sim step)"),
                                    ("z_mm", "vertical residual  (mm / sim step, + = up)")]):
        for j, c in enumerate(("E2", "E0")):
            xs = np.arange(len(phs)) + (j - 0.5) * 0.36
            ms = [res[met][p][c]["mean"] for p in phs]
            a.bar(xs, ms, 0.32, color=COL[c], label=f"{c} ({'VLM feature' if c == 'E2' else 'feature = 0'})", zorder=2)
            for x, p in zip(xs, phs):
                r = res[met][p][c]
                a.plot([x, x], r["ci"], color="#3d3d3a", lw=1.2, zorder=3)
                a.scatter([x] * len(r["per_seed"]), r["per_seed"], s=14, color="white", edgecolor="#3d3d3a", lw=0.8, zorder=4)
        a.set_xticks(np.arange(len(phs))); a.set_xticklabels(lab, fontsize=8.5); a.set_ylabel(yl); a.grid(axis="x", visible=False)
        if met != "z_mm":
            a.set_ylim(0, None)
        else:
            a.axhline(0, color="#8a8a85", lw=0.8)
    ax[0, 2].legend(*ax[0, 0].get_legend_handles_labels(), frameon=False, loc="lower left")
    ax[0, 0].set_title("Residual size by phase (3-seed mean, seed dots, 95 % CI)", loc="left", fontsize=10.5)
    # ---- row 2: aligned to the first success
    al = d[d.rel.notna() & (d.rel >= -4) & (d.rel <= 8)]

    def line(a, col, scale=1.0, conds=("B0", "E2", "E0"), direct=True):
        for c in conds:
            g = al[al.cond == c]
            if not len(g):
                continue
            per = g.groupby(["name", "rel"])[col].mean().unstack(0) * scale
            for nm in per.columns:
                a.plot(per.index, per[nm], color=COL[c], lw=0.8, alpha=0.35)
            m = per.mean(1)
            a.plot(m.index, m, color=COL[c], lw=2, marker="o", ms=4, label=c)
            if direct:
                a.annotate(c, (m.index[-1], m.iloc[-1]), xytext=(5, 0), textcoords="offset points", color="#3d3d3a",
                               fontsize=9, va="center")
        a.axvline(0.5, color="#8a8a85", lw=1, ls="--")
        a.set_xlabel("decision relative to first success (0 = decision that reaches it)")
        a.set_xlim(-4.3, 9)

    line(ax[1, 0], "res_tanh_norm", conds=("E2", "E0")); ax[1, 0].set_ylabel("residual size  ‖tanh(u)‖"); ax[1, 0].set_ylim(0, None)
    ax[1, 0].set_title("Aligned to first success (thin = seeds)", loc="left", fontsize=10.5)
    line(ax[1, 1], "exec_z", 1e3, direct=False); ax[1, 1].set_ylabel("executed vertical action (mm / step, + = up)")
    ax[1, 1].set_ylim(-9, 4); ax[1, 1].set_title("base + residual; pre-success peaks clipped", loc="left", fontsize=9.5)
    ax[1, 1].axhline(0, color="#8a8a85", lw=0.8)
    line(ax[1, 2], "success_now"); ax[1, 2].set_ylabel("P(success flag true after the decision)")
    ax[1, 2].set_ylim(0, 1.02)
    ax[1, 1].legend(frameon=False, loc="lower right")
    fig.suptitle("T034  Residual before / after success: E0 (VLM feature zeroed) vs E2 vs B0 (no residual) — held-out 128, GPU 0",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/outputs/StochRL/t034/raw")
    ap.add_argument("--out", default="/outputs/StochRL/t034")
    ap.add_argument("--n_boot", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.out)
    d = load(a.raw)
    d.to_parquet(out / "decision_logs.parquet")
    B = Boot(d, a.n_boot)
    res = {m: {ph: B.summary(d[col].to_numpy() * sc, d[f"ph_{ph}"].to_numpy()) for ph in PHASES}
           for m, (col, sc, _) in METRICS.items()}
    # hazards / episode-level
    res["hazard"] = dict(lose_given_in_succ=B.summary(d.ph_lose.to_numpy(), d.ph_in_succ.to_numpy()),
                         regain_given_lost=B.summary(d.success_now.to_numpy(), d.ph_lost.to_numpy()))
    lastd = d.dec == d.dec.max()
    res["episode"] = dict(once=B.summary(d.ever_success.to_numpy(), lastd.to_numpy()),
                          end=B.summary(d.success_now.to_numpy(), lastd.to_numpy()),
                          end_given_once=B.summary(d.success_now.to_numpy(), (lastd & d.ever_success).to_numpy()))
    # time-matched: post vs pre residual size at the same decision index (decisions 4..15)
    tm = []
    for (cond, nm), g in d[d.cond != "B0"].groupby(["cond", "name"]):
        p = g.groupby(["dec", "ph_post"]).res_tanh_norm.mean().unstack()
        n = g.groupby(["dec", "ph_post"]).size().unstack().fillna(0)
        ok = (n.get(True, 0) >= 5) & (n.get(False, 0) >= 5)
        w = np.minimum(n[True], n[False])[ok]
        tm.append(dict(cond=cond, name=nm, n_dec_idx=int(ok.sum()), pre=float((p[False][ok] * w).sum() / w.sum()),
                       post=float((p[True][ok] * w).sum() / w.sum())))
    tm = pd.DataFrame(tm); tm["post_minus_pre"] = tm.post - tm.pre
    mech, L = loss_mechanism(d)
    actors = [n for n in sorted(d.name.unique()) if n != "b0"]
    cp = cross_policy(d, a.raw, actors)
    rep = [json.loads(p.read_text()) for p in sorted(Path(a.raw).glob("*_repro.json"))]
    (out / "analysis.json").write_text(json.dumps(dict(res=res, time_matched=tm.to_dict("records"),
                                                       loss_mechanism=mech.to_dict("records"),
                                                       cross_policy=None if cp is None else cp.to_dict("records"),
                                                       repro=rep), indent=1))
    T = ["# T034 tables (analyze_t034.py)\n", "## 재현 확인\n",
         "| run | once | end | 저장된 once | 저장된 end | episode 일치 once | 일치 end | wall s |", "|---|---|---|---|---|---|---|---|"]
    for r in rep:
        T.append(f"| {r['name']} | {r['once']:.3f} | {r['end']:.3f} | {r['ref_once']:.3f} | {r['ref_end']:.3f} | "
                 f"{r['agree_once']:.3f} | {r['agree_end']:.3f} | {r['wall_s']:.0f} |")
    for m, (col, sc, unit) in METRICS.items():
        T += [f"\n## {m} [{unit}]\n", phase_table(res, m, PHASES)]
    T += ["\n## 구간별 결정 수 (seed별)\n", "| 구간 | E2 | E0 | B0 |", "|---|---|---|---|"]
    for ph in PHASES:
        r = res["tanh_norm"][ph]
        T.append(f"| {ph} | {r['E2']['n_dec']} | {r['E0']['n_dec']} | {r['B0']['n_dec'] if 'B0' in r else '–'} |")
    T += ["\n## hazard / episode\n", phase_table(res, "hazard", ["lose_given_in_succ", "regain_given_lost"]),
          "", phase_table(res, "episode", ["once", "end", "end_given_once"])]
    T += ["\n## 같은 결정 번호에서 post − pre (‖tanh u‖)\n", tm.round(3).to_markdown(index=False)]
    T += ["\n## 성공 상실 결정\n", mech.round(3).T.to_markdown()]
    if cp is not None:
        T += ["\n## B0가 방문한 상태에서 여섯 actor의 residual (offline)\n", cp.round(3).to_markdown(index=False)]
    (out / "tables.md").write_text("\n".join(T) + "\n")
    figure(d, res, out / "residual_by_phase.png")
    print("\n".join(T))


if __name__ == "__main__":
    main()
