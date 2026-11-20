"""T031 analysis — implements the rules of repcritic/reports/t031_prereg.md (no new decisions here).

usage: .venv/bin/python -m repcritic.t031_analyze --out /outputs/StochRL/t031
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repcritic.t031_branch import CANDS, NREP, OUT

GROUPS = {"NOISE": ["u0_ind"], "E1": ["E1"], "E3": ["E3"], "T1": ["T1_+u", "T1_-u", "T1_+v", "T1_-v"],
          "T05": ["T05_+u", "T05_-u"], "YAW": ["yaw_+", "yaw_-"], "GRIP": ["grip_flip"]}
NONPRIV = ["T1", "T05", "YAW", "GRIP"]
NONPRIV_CANDS = [c for g in NONPRIV for c in GROUPS[g]]
STRATA = ["S0_init", "S1_pre", "S2_post", "S3_lowV"]
NB = 4000


def load(path):
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    df["gx"] = [a["pos_gripper"][0] for a in df.afterK]
    df["gy"] = [a["pos_gripper"][1] for a in df.afterK]
    df["gz"] = [a["pos_gripper"][2] for a in df.afterK]
    df["grasp_afterK"] = [bool(a["grasping_now"]) for a in df.afterK]
    return df


def cube(df, metric):
    """Y[state, cand, rep] in {0,1}; states in first-appearance order."""
    sids = list(dict.fromkeys(df.state_id))
    si = {s: i for i, s in enumerate(sids)}
    Y = np.full((len(sids), len(CANDS), NREP), np.nan)
    Y[df.state_id.map(si), df.cand, df.rep] = df[metric].astype(float)
    assert not np.isnan(Y).any(), "incomplete state"
    return sids, Y


def effects(Y, cands):
    """Per-state rescue/harm/net/disc averaged over reps and over the candidate list. -> dict of [S] arrays."""
    ci = [CANDS.index(c) for c in cands]
    R0 = Y[:, :1, :]
    Rc = Y[:, ci, :]
    resc = ((R0 == 0) & (Rc == 1)).mean((1, 2))
    harm = ((R0 == 1) & (Rc == 0)).mean((1, 2))
    return dict(rescue=resc, harm=harm, net=resc - harm, disc=resc + harm, p_cand=Rc.mean((1, 2)), p_base=R0.mean((1, 2)))


def boot(vals, rng, lo=2.5, hi=97.5, idx=None):
    """vals: [S] per-state values; cluster (state) bootstrap of the mean."""
    S = len(vals)
    if idx is None:
        idx = rng.integers(0, S, (NB, S))
    bs = vals[idx].mean(1)
    return [float(vals.mean()), float(np.percentile(bs, lo)), float(np.percentile(bs, hi))]


def split_half(Y, cset):
    """T026 split-half selection gain vs base. Pick argmax p over reps A (ties -> base = cand 0, listed first),
    evaluate on reps B minus base on B; average both directions. Returns [S]."""
    ci = [0] + [CANDS.index(c) for c in cset]
    out = np.zeros(Y.shape[0])
    for A, B in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
        pA = Y[:, ci][:, :, A].mean(2)
        pick = np.array(ci)[pA.argmax(1)]  # argmax returns first max -> base on ties
        pB = Y[np.arange(Y.shape[0]), pick][:, B].mean(1)
        out += (pB - Y[:, 0, B].mean(1)) / 2
    return out


def table(Y, rng, idx, groups, lohi=(2.5, 97.5)):
    out = {}
    noise = effects(Y, GROUPS["NOISE"])
    for name, cands in groups.items():
        e = effects(Y, cands)
        out[name] = {k: boot(e[k], rng, *lohi, idx=idx) for k in ("rescue", "harm", "net", "disc", "p_cand")}
        out[name]["d_disc_vs_noise"] = boot(e["disc"] - noise["disc"], rng, *lohi, idx=idx)
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1]), facecolor=SURF)
    # left: per candidate rescue (up) / harm (down) bars, net dot + 95% CI
    ax = axes[0]
    names = CANDS[1:]
    C = res["cands"]
    x = np.arange(len(names))
    ax.bar(x, [C[n]["rescue"][0] for n in names], 0.62, color=BLUE, label="rescue (base 0 → cand 1)")
    ax.bar(x, [-C[n]["harm"][0] for n in names], 0.62, color=ORANGE, label="harm (base 1 → cand 0)")
    net = np.array([C[n]["net"] for n in names])
    ax.errorbar(x, net[:, 0], yerr=[net[:, 0] - net[:, 1], net[:, 2] - net[:, 0]], fmt="o", color=INK, ms=6,
                capsize=3, lw=1.5, label="net = rescue − harm (95% CI)")
    nz = C["u0_ind"]
    ax.axhspan(-nz["harm"][0], nz["rescue"][0], color=MUTED, alpha=0.12, lw=0, label="noise baseline (u0_ind) band")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("rate per (state, rep) pair", color=INK)
    ax.set_title(f"One-shot candidates vs base (success_once, {res['n_states']} states × 4 CRN reps)", fontsize=11,
                 color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    # right: group net by stratum
    ax = axes[1]
    G = ["NOISE", "E1", "E3", "T1", "T05", "YAW", "GRIP"]
    st = list(res["strata"])
    w = 0.8 / max(1, len(st))
    shades = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # one-hue sequential steps = stratum order
    for j, s in enumerate(st):
        v = np.array([res["strata"][s]["groups"][g]["net"] for g in G])
        xx = np.arange(len(G)) + (j - (len(st) - 1) / 2) * w
        ax.errorbar(xx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=5, capsize=2, lw=1.2,
                    color=shades[j % 4], label=f"{s} (n={res['strata'][s]['n']})")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(G)), G, fontsize=9)
    ax.set_ylabel("net (95% CI, state bootstrap)", color=INK)
    ax.set_title("Net by stratum and candidate group", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for a in axes:
        a.set_facecolor(SURF)
        a.grid(axis="y", color="#e6e5e1", lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t031")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    df = load(OUT / "branch_results_main.jsonl")
    sids, Y = cube(df, "success_once")
    _, Ye = cube(df, "success_at_end")
    S = len(sids)
    idx = rng.integers(0, S, (NB, S))
    meta = df.drop_duplicates("state_id").set_index("state_id").loc[sids]
    res = dict(n_states=S, n_branches=int(len(df)), strata_n=meta.stratum.value_counts().to_dict(),
               p_base=float(Y[:, 0].mean()))
    # ---- main group table (success_once) + success_at_end
    res["groups"] = table(Y, rng, idx, GROUPS)
    res["groups_at_end"] = table(Ye, rng, idx, GROUPS)
    # ---- individual candidates, unadjusted and Bonferroni (12 comparisons)
    bon = 2.5 / 12
    res["cands"] = table(Y, rng, idx, {c: [c] for c in CANDS[1:]})
    res["cands_bonf"] = table(Y, rng, idx, {c: [c] for c in CANDS[1:]}, lohi=(bon, 100 - bon))
    # ---- strata
    res["strata"] = {}
    for st in STRATA:
        m = (meta.stratum == st).values
        if m.sum() < 2:
            continue
        Ys = Y[m]
        ii = rng.integers(0, m.sum(), (NB, m.sum()))
        res["strata"][st] = dict(n=int(m.sum()), p_base=float(Ys[:, 0].mean()), groups=table(Ys, rng, ii, GROUPS))
    # ---- base-outcome subgroup (episode logged success)
    res["by_episode_success"] = {}
    for v in (True, False):
        m = (meta.episode_success_once == v).values
        ii = rng.integers(0, m.sum(), (NB, m.sum()))
        res["by_episode_success"][str(v)] = dict(n=int(m.sum()), p_base=float(Y[m][:, 0].mean()),
                                                 groups=table(Y[m], rng, ii, GROUPS))
    # ---- paired contrasts
    def contrast(a, b, key="net"):
        return boot(effects(Y, a)[key] - effects(Y, b)[key], rng, idx=idx)
    res["contrasts"] = {
        "E3_minus_E1_net": contrast(["E3"], ["E1"]),
        "T1u_minus_T05u_net": contrast(["T1_+u", "T1_-u"], ["T05_+u", "T05_-u"]),
        "T1u_minus_T05u_disc": contrast(["T1_+u", "T1_-u"], ["T05_+u", "T05_-u"], "disc"),
    }
    # ---- expert first-chunk diagnostics (post hoc, descriptive): what does the E1/E3 first chunk do?
    pb = Y[:, 0].mean(1)
    er = []
    for i, s in enumerate(sids):
        z = np.load(OUT / "candidates" / f"{s}.npz")
        d, bch = z["expert_delta"], z["base"]
        er.append(dict(state_id=s, stratum=meta.stratum.iloc[i], expert_phase0=int(z["expert_phase0"]),
                       dz_cm=float(d[2] * 5 * 100), dxy_cm=float(np.linalg.norm(d[:2]) * 5 * 100),
                       dyaw_rad=float(d[5] * 5), base_grip=float(bch[:, 6].mean()), expert_grip_delta=float(d[6]),
                       opens_closing_gripper=bool(bch[:, 6].mean() < 0 and d[6] > 1.0),
                       net_E1=float(Y[i, CANDS.index("E1")].mean() - pb[i]),
                       net_E3=float(Y[i, CANDS.index("E3")].mean() - pb[i])))
    er = pd.DataFrame(er)
    er.to_csv(out / "expert_first_chunk_diag.csv", index=False, float_format="%.4f")
    res["expert_diag"] = {str(k): dict(n=int(len(g)), net_E1=float(g.net_E1.mean()), net_E3=float(g.net_E3.mean()))
                          for k, g in er.groupby("opens_closing_gripper")}
    # ---- split-half selection
    res["split_half"] = {"nonpriv": boot(split_half(Y, NONPRIV_CANDS), rng, idx=idx),
                         "nonpriv+expert": boot(split_half(Y, NONPRIV_CANDS + ["E1", "E3"]), rng, idx=idx),
                         "noise_only(u0_ind)": boot(split_half(Y, ["u0_ind"]), rng, idx=idx)}
    # ---- after-K recovery diagnostics (vs base, same state & rep)
    b = df[df.cand == 0].set_index(["state_id", "rep"])
    diag = {}
    for c in CANDS[1:]:
        x = df[df.cand_name == c].set_index(["state_id", "rep"])
        bb = b.loc[x.index]
        disp = np.sqrt((x.gx.values - bb.gx.values) ** 2 + (x.gy.values - bb.gy.values) ** 2 + (x.gz.values - bb.gz.values) ** 2)
        diag[c] = dict(gripper_disp_median_cm=float(np.median(disp) * 100),
                       grasp_state_differs=float((x.grasp_afterK.values != bb.grasp_afterK.values).mean()),
                       agree_with_base=float((x.success_once.values == bb.success_once.values).mean()),
                       bound_clip=float(x.bound_clip.mean()))
    res["afterK"] = diag
    e = df[df.cand_name == "E1"].drop_duplicates("state_id")
    res["expert"] = dict(alpha_clip_mean=float(e.expert_alpha_clip.mean()),
                         phase0_counts=e.expert_phase0.value_counts().to_dict())
    # ---- replay floor (restore test: u=0, same CRN noise, 13 copies per rep)
    rt = OUT / "branch_results_restore_test.jsonl"
    if rt.exists():
        r = pd.DataFrame([json.loads(l) for l in open(rt)])
        agree, disc = [], []
        for (s, rep), g in r.groupby(["state_id", "rep"]):
            y = g.sort_values("cand").success_once.values.astype(int)
            agree.append(max(y.mean(), 1 - y.mean()))
            disc.append(float((y[1:] != y[0]).mean()))
        res["replay_floor"] = dict(n_groups=len(agree), mean_majority_agreement=float(np.mean(agree)),
                                   discordance_vs_slot0=float(np.mean(disc)), per_group_disc=disc)
    # ---- verdict (pre-registered)
    G = res["groups"]
    go = [g for g in NONPRIV if G[g]["net"][1] > 0 and G[g]["d_disc_vs_noise"][1] > 0]
    e1_sig = G["E1"]["net"][1] > 0
    e3_sig = G["E3"]["net"][1] > 0
    explo = [c for c in NONPRIV_CANDS if res["cands"][c]["net"][1] > 0]
    if go:
        verdict = "Go"
    elif not e1_sig:
        verdict = "setting No-Go" + (" (E3 significant: repeated intervention / mu=residual needed)" if e3_sig else "")
    else:
        verdict = "Partial (E1 significant, non-privileged candidates not)"
    res["verdict"] = dict(verdict=verdict, go_groups=go, e1_significant=e1_sig, e3_significant=e3_sig,
                          exploratory_single_cands=explo)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    # ---- tables / raw
    df.drop(columns=["afterK"]).assign(afterK=df.afterK.map(json.dumps), e3_later=df.get("e3_later", pd.Series(dtype=object)).map(
        lambda v: json.dumps(v) if isinstance(v, list) else None)).to_parquet(out / "branch_rollouts.parquet")
    rows = []
    for level, d in (("group", res["groups"]), ("cand", res["cands"])):
        for name, v in d.items():
            row = dict(level=level, name=name)
            for k in ("rescue", "harm", "net", "disc", "d_disc_vs_noise", "p_cand"):
                row[k], row[k + "_lo"], row[k + "_hi"] = v[k]
            if level == "cand":
                row["net_bonf_lo"], row["net_bonf_hi"] = res["cands_bonf"][name]["net"][1:]
                row.update({f"afterK_{k}": x for k, x in res["afterK"][name].items()})
            rows.append(row)
    for st, d in res["strata"].items():
        for name, v in d["groups"].items():
            row = dict(level=f"stratum:{st}", name=name)
            for k in ("rescue", "harm", "net", "disc", "d_disc_vs_noise", "p_cand"):
                row[k], row[k + "_lo"], row[k + "_hi"] = v[k]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "effects_by_candidate.csv", index=False, float_format="%.4f")
    plot(res, out / "rescue_harm.png")
    print(json.dumps({k: res[k] for k in ("n_states", "p_base", "verdict", "split_half", "contrasts")}, indent=1))
    for g, v in res["groups"].items():
        print(f"{g:6s} resc {v['rescue'][0]:.3f} harm {v['harm'][0]:.3f} net {v['net'][0]:+.3f} "
              f"[{v['net'][1]:+.3f},{v['net'][2]:+.3f}] disc {v['disc'][0]:.3f} d_disc {v['d_disc_vs_noise'][0]:+.3f} "
              f"[{v['d_disc_vs_noise'][1]:+.3f},{v['d_disc_vs_noise'][2]:+.3f}]")


if __name__ == "__main__":
    main()
