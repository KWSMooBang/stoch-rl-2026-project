"""E09 (T027) final summary: figures + summary_results.csv from EXISTING T016–T020 outputs. CPU only, no new runs.

Sources (read-only):
  /outputs/StochRL/t016/oracle_diversity_analysis.json   (E01/Oracle-N)
  /outputs/StochRL/t017/oracle_eb_main.json              (E02 multi-continuation oracle)
  /outputs/StochRL/t018/small_data_curves.csv            (E03)
  /outputs/StochRL/t019/critic_eval.csv, selection.json, action_only_baselines.json  (E04)
  /outputs/StochRL/t020/rerank_metrics.csv               (E06)
  /outputs/StochRL/t027/hacking_case.json                (repcritic.e09_hacking_case, CPU rerun of the T019 probe)
Newly computed here (and only here): Holm-adjusted McNemar p for the 8 T020 comparisons.

  python -m repcritic.e09_report [--out /outputs/StochRL/t027]
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "/outputs/StochRL"
# reference categorical palette (light), fixed order; text in neutral ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold",
})
for fam in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
    from matplotlib import font_manager
    if any(fam == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
KO = plt.rcParams["font.family"] != ["sans-serif"] and plt.rcParams["font.family"] != "sans-serif"


def L(ko, en):
    return ko if KO else en


def rd_csv(p):
    return list(csv.DictReader(open(p)))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ figures
def fig_sample_eff(out, rows_csv):
    ce = rd_csv(f"{R}/t019/critic_eval.csv")
    sel = json.load(open(f"{R}/t019/selection.json"))
    ceil = sel["noise_ceiling_test"]["conc_1c"]
    series = [("Full", "C-Full (2048)"), ("Fusion", "C-Fusion (21)"), ("FusionLC", "FusionLC"),
              ("RandomProj", "RandomProj (21)"), ("Raw", "Raw CNN")]
    fr = [0.1, 0.25, 0.5, 1.0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (inp, lab) in enumerate(series):
        au, ausd, co = [], [], []
        for f in fr:
            r = [x for x in ce if x["input"] == inp and x["obj"] == "q" and x["regime"] == "buf+br" and float(x["frac"]) == f]
            if inp == "FusionLC" and f == 1.0:  # FusionLC at 100% == Fusion (probe already fit on all train)
                r = [x for x in ce if x["input"] == "Fusion" and x["obj"] == "q" and x["regime"] == "buf+br" and float(x["frac"]) == f]
            au.append(float(r[0]["test_auroc_buf_within_t"])); ausd.append(float(r[0]["test_auroc_buf_within_t_sd"]))
            co.append(float(r[0]["test_conc_1c"]))
            rows_csv.append(dict(exp="E04", quantity=f"q_{inp}_frac{f}_auroc_within_t", value=au[-1], lo="", hi="",
                                 n="3 seeds", source="t019/critic_eval.csv", status="exploratory(curve)"))
            rows_csv.append(dict(exp="E04", quantity=f"q_{inp}_frac{f}_conc_1c", value=co[-1], lo="", hi="",
                                 n="3 seeds", source="t019/critic_eval.csv", status="exploratory(curve)"))
        ls = "--" if inp == "FusionLC" else "-"
        ax[0].errorbar(fr, au, yerr=ausd, color=C[k], ls=ls, marker="o", ms=5, capsize=0, elinewidth=1, label=lab)
        ax[1].plot(fr, co, color=C[k], ls=ls, marker="o", ms=5, label=lab)
    ax[0].set_title(L("(a) 상태 가치: 같은 t AUROC", "(a) state value: within-t AUROC"))
    ax[0].set_ylabel("test AUROC (within decision index)")
    ax[1].axhline(ceil, color=INK2, ls=":", lw=1.2); ax[1].text(0.11, ceil - 0.025, L(f"noise 천장 {ceil:.2f}", f"noise ceiling {ceil:.2f}"), color=INK2, fontsize=9)
    ax[1].axhline(0.5, color=INK2, ls="-", lw=0.8); ax[1].text(0.11, 0.505, "chance 0.5", color=INK2, fontsize=9)
    ax[1].axhline(0.543, color=C[7], ls="--", lw=1.2); ax[1].text(0.11, 0.6, L("action-only 0.543", "action-only 0.543"), color=INK2, fontsize=9)
    ax[1].set_ylim(0.45, 0.85)
    ax[1].set_title(L("(b) 같은 상태 후보 ranking 일치도", "(b) same-state candidate concordance"))
    ax[1].set_ylabel("test conc_1c")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(fr); a.set_xticklabels(["10%", "25%", "50%", "100%"])
        a.set_xlabel(L("train episode 비율 (350 ep = 100%)", "train episode fraction"))
    ax[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(L("Fig A. critic sample efficiency (Q objective, buf+br, 3 seed, T019)", "Fig A. critic sample efficiency (T019)"), fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/figs/figA_sample_efficiency.png", dpi=160); plt.close(fig)


def fig_ranking(out, rows_csv):
    ce = {x["name"]: x for x in rd_csv(f"{R}/t019/critic_eval.csv")}
    ao = json.load(open(f"{R}/t019/action_only_baselines.json"))["ridge_action_within"]["test"]
    sel = json.load(open(f"{R}/t019/selection.json"))
    names = ["Geo_q_buf+br", "Fusion_q_buf+br", "Full_q_buf+br", "Geo_nce_buf+br", "Fusion_nce_buf+br",
             "Full_bce_buf+br", "RandomProj_q_buf+br", "Raw_q_buf+br", "Raw_bce_buf+br", "AECand_pair_buf+br"]
    items = []
    for n in names:
        x = ce[n + "_f1.0"]
        items.append((n, float(x["test_conc_1c"]), float(x["test_conc_1c_lo"]), float(x["test_conc_1c_hi"]), "critic"))
    items.append(("action-only ridge (no state)", ao[0], ao[1], ao[2], "ao"))
    nc = sel["noise_ceiling_test"]
    items.append(("noise ceiling (true p_i)", nc["conc_1c"], nc["ci"][0], nc["ci"][1], "ceil"))
    for n, v, lo, hi, k in items:
        rows_csv.append(dict(exp="E04", quantity=f"test_conc_1c:{n}", value=v, lo=lo, hi=hi, n="120 states",
                             source="t019/critic_eval.csv|action_only_baselines.json|selection.json",
                             status="prereg(gate)" if n == "Geo_q_buf+br" else "exploratory"))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ys = np.arange(len(items))[::-1]
    for y, (n, v, lo, hi, k) in zip(ys, items):
        col = {"critic": C[0], "ao": C[1], "ceil": INK2}[k]
        if n == "Geo_q_buf+br":
            col = C[6]
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(v, y, "o", color=col, ms=8, mec=SURF, mew=2)
        ax.text(hi + 0.006, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([n + ("  ← 선택(val)" if (n == "Geo_q_buf+br" and KO) else "") for n, *_ in items], fontsize=8.5)
    ax.set_xlim(0.45, 0.9); ax.grid(axis="y", visible=False)
    ax.set_xlabel(L("test 같은 상태 일치도 conc_1c [95% CI, 상태 bootstrap]", "test same-state concordance [95% CI]"))
    ax.set_title(L("Fig B. 같은 상태 ranking: critic ≈ action-only ≪ noise 천장 (T019)", "Fig B. same-state ranking (T019)"))
    fig.tight_layout(); fig.savefig(f"{out}/figs/figB_same_state_ranking.png", dpi=160); plt.close(fig)


def fig_oracle(out, rows_csv):
    o = json.load(open(f"{R}/t016/oracle_diversity_analysis.json"))["oracle"]["pre_success"]
    eb = json.load(open(f"{R}/t017/oracle_eb_main.json"))["success_once"]
    N = [1, 4, 8, 16]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1, 1.25]})
    for k, (key, lab) in enumerate([("cand_once", L("cand: 후보 i 실행 후 continuation", "cand arm")),
                                    ("ctrl_once", L("ctrl: 같은 후보 0, continuation만 다름", "ctrl arm"))]):
        m = [o[key][str(n)]["mean"] for n in N]
        lo = [o[key][str(n)]["ci"][0] for n in N]; hi = [o[key][str(n)]["ci"][1] for n in N]
        ax[0].fill_between(N, lo, hi, color=C[k], alpha=0.12, lw=0)
        ax[0].plot(N, m, color=C[k], marker="o", ms=6, label=lab)
        ax[0].text(16.6, m[-1], f"{m[-1]:.3f}", va="center", fontsize=8.5, color=INK2)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(N); ax[0].set_xticklabels(N); ax[0].set_xlim(0.9, 22)
    ax[0].set_xlabel("N (OraclePass@N, unbiased from 16 branches)"); ax[0].set_ylabel("success_once")
    ax[0].set_title(L("(a) T016 Oracle-N, 성공 전 결정 59개", "(a) T016 Oracle-N (59 pre-success states)"))
    ax[0].legend(loc="lower right", fontsize=8.5)
    # (b) decomposition of the chunk-selection contribution
    items = [
        ("T016 OraclePass@16 − Pass@1", 0.207, 0.15, 0.27, INK2),
        ("T016 cand16 − ctrl16 (" + L("첫 chunk 몫", "chunk share") + ")", 0.034, 0.00, 0.08, C[0]),
        ("T017 E[max p_i]−E[p_i] (" + L("완벽한 critic", "perfect critic") + ")", eb["G_betabinom"], eb["ci95"][0], eb["ci95"][1], C[6]),
        ("T017 excess over noise null", 0.035, 0.013, 0.062, C[0]),
        ("T017 split-half selector", 0.039, -0.015, 0.099, C[0]),
        ("T019 pick gain: noise ceiling", 0.20154, np.nan, np.nan, INK2),
        ("T019 pick gain: Geo_q critic", 0.019, np.nan, np.nan, C[1]),
        ("T020 Critic8 − B0 (closed loop)", 0.073, -0.013, 0.167, C[1]),
        ("T020 Critic8 − ActOnly8 (" + L("critic 몫", "critic share") + ")", 0.007, -0.087, 0.100, C[1]),
    ]
    ys = np.arange(len(items))[::-1]
    for y, (n, v, lo, hi, col) in zip(ys, items):
        if not np.isnan(lo):
            ax[1].plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax[1].plot(v, y, "o", color=col, ms=8, mec=SURF, mew=2)
        ax[1].text((hi if not np.isnan(hi) else v) + 0.008, y, f"{v:+.3f}", va="center", fontsize=8.5, color=INK2)
        rows_csv.append(dict(exp="E01/E02/E04/E06", quantity=f"chunk_contrib:{n}", value=v, lo="" if np.isnan(lo) else lo,
                             hi="" if np.isnan(hi) else hi, n="", source="t016/t017/t019/t020 summaries (verbatim)",
                             status="prereg(primary)" if n.startswith("T020 Critic8") else "diagnostic"))
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_yticks(ys); ax[1].set_yticklabels([i[0] for i in items], fontsize=8.5); ax[1].grid(axis="y", visible=False)
    ax[1].set_xlim(-0.12, 0.33); ax[1].set_xlabel(L("성공률 차이 (결정당 또는 episode당)", "success-rate difference"))
    ax[1].set_title(L("(b) chunk 선택 기여 분해", "(b) chunk-selection contribution"))
    fig.suptitle(L("Fig C. Oracle-N과 chunk 선택 기여: 이득 대부분은 continuation 운", "Fig C. Oracle-N decomposition"), fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/figs/figC_oracle_decomposition.png", dpi=160); plt.close(fig)


def fig_rerank(out, rows_csv):
    rm = rd_csv(f"{R}/t020/rerank_metrics.csv")
    conds = [r for r in rm if r["cond"] and not r["cond"].startswith("diff")]
    diffs = [r for r in rm if r["cond"].startswith("diff") and r["metric"] == "success_once"]
    adj = holm([float(r["mcnemar_p"]) for r in diffs])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.3]})
    for i, r in enumerate(conds):
        v, lo, hi = float(r["success_once"]), float(r["success_once_lo"]), float(r["success_once_hi"])
        col = C[1] if r["cond"] in ("Critic8", "CriticApp8") else (C[0] if r["cond"] == "B0" else INK2)
        ax[0].plot([i, i], [lo, hi], color=col, lw=2, solid_capstyle="round")
        ax[0].plot(i, v, "o", color=col, ms=8, mec=SURF, mew=2)
        ax[0].text(i + 0.12, v, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
        rows_csv.append(dict(exp="E06", quantity=f"success_once:{r['cond']}", value=v, lo=lo, hi=hi, n=150,
                             source="t020/rerank_metrics.csv", status="prereg"))
    ax[0].set_xticks(range(len(conds))); ax[0].set_xticklabels([r["cond"] for r in conds], fontsize=8.5)
    ax[0].set_ylim(0.4, 0.8); ax[0].set_ylabel("success_once [Wilson 95%]"); ax[0].grid(axis="x", visible=False)
    ax[0].set_title(L("(a) 조건별 성공률 (n=150 paired)", "(a) success rate (n=150 paired)"))
    ys = np.arange(len(diffs))[::-1]
    for y, r, pa in zip(ys, diffs, adj):
        n = r["cond"].replace("diff:", "")
        v, lo, hi = float(r["diff"]), float(r["lo"]), float(r["hi"])
        prim = n in ("Critic8-ActOnly8", "Critic8-B0")
        col = C[6] if prim else INK2
        ax[1].plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax[1].plot(v, y, "o", color=col, ms=8, mec=SURF, mew=2)
        ax[1].text(hi + 0.006, y, f"{v:+.3f}  p={float(r['mcnemar_p']):.2f}, Holm {pa:.2f}", va="center", fontsize=8, color=INK2)
        rows_csv.append(dict(exp="E06", quantity=f"paired_diff_success_once:{n}", value=v, lo=lo, hi=hi, n=150,
                             source="t020/rerank_metrics.csv (Holm: e09_report.py)",
                             status=("prereg(primary)" if prim else "prereg(secondary)") + f";mcnemar_p={float(r['mcnemar_p']):.4f};holm_p={pa:.3f}"))
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_yticks(ys); ax[1].set_yticklabels([r["cond"].replace("diff:", "") + (" ★" if r["cond"].replace("diff:", "") in ("Critic8-ActOnly8", "Critic8-B0") else "") for r in diffs], fontsize=8.5)
    ax[1].grid(axis="y", visible=False); ax[1].set_xlim(-0.12, 0.42)
    ax[1].set_xlabel(L("paired 차이 [episode bootstrap 95%]  ★=사전 등록 주 비교", "paired diff [95% CI]  ★=primary"))
    ax[1].set_title(L("(b) paired 차이, McNemar p와 Holm 보정 p", "(b) paired differences"))
    fig.suptitle(L("Fig D. E06 closed-loop reranking N=8 (T020)", "Fig D. E06 reranking (T020)"), fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/figs/figD_rerank_success.png", dpi=160); plt.close(fig)
    return diffs, adj


def fig_hacking(out, rows_csv):
    h = json.load(open(f"{out}/hacking_case.json"))
    ce = {x["name"]: x for x in rd_csv(f"{R}/t019/critic_eval.csv")}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    steps = np.arange(21)
    for k, (s, v) in enumerate(h["seeds"].items()):
        ax[0].plot(steps, v["ascent_mean_z"], color=C[k], label=f"seed {s}")
        ax[0].axhline(v["in_dist_max_z_mean"], color=C[k], ls=":", lw=1)
    ax[0].text(0.5, 3.5, L("in-dist 후보 최대 (점선, ≈1.8σ)", "in-dist candidate max (dotted)"), fontsize=8.5, color=INK2)
    ax[0].set_xlabel(L("gradient 상승 step (L2 0.05/step)", "ascent step")); ax[0].set_ylabel(L("점수 (상태 내 후보 z)", "score (within-state z)"))
    ax[0].set_title(L("(a) 상승 궤적 평균, 120 test 상태", "(a) ascent, mean over 120 states")); ax[0].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.9))
    # (b) single example state
    sid, ex = next(iter(h["examples"].items()))
    zt = np.array(ex["z_traj_seed0"])  # [21, 16]
    y = np.array(ex["y"])
    for j in range(zt.shape[1]):
        col = C[2] if y[j] > 0.5 else C[7]
        ax[1].plot(steps, zt[:, j], color=col, lw=1.2, alpha=0.85)
    ax[1].plot([], [], color=C[2], label=L("branch 성공 후보", "branch success")); ax[1].plot([], [], color=C[7], label=L("branch 실패 후보", "branch failure"))
    ax[1].legend(fontsize=8.5, loc="upper left")
    ax[1].set_xlabel(L("gradient 상승 step", "ascent step")); ax[1].set_ylabel(L("점수 (상태 내 후보 z)", "score (z)"))
    ax[1].set_title(L(f"(b) 사례: test 상태 {sid}, 후보 16개", f"(b) case: state {sid}"), fontsize=10)
    sig = [float(k) for k in h["seeds"]["0"]["gauss_win"]]
    for k, (s, v) in enumerate(h["seeds"].items()):
        ax[2].plot(sig, [v["gauss_win"][str(x)] for x in sig], color=C[k], marker="o", ms=5, label=f"seed {s}")
    ax[2].axhline(1 / 17, color=INK2, ls=":", lw=1.2); ax[2].text(1.9, 1 / 17 + 0.012, L("in-dist 기대치 1/17", "in-dist expectation 1/17"), fontsize=8.5, color=INK2)
    ax[2].set_xlabel(L("Gaussian 행동 섭동 σ (action std 단위)", "Gaussian action perturbation σ")); ax[2].set_ylabel(L("상태 내 후보 최대를 넘는 비율", "frac beating in-dist max"))
    ax[2].set_title(L("(c) 정책에서 먼 행동일수록 점수 상승", "(c) OOD actions score higher")); ax[2].legend(fontsize=8.5, loc="upper left")
    fig.suptitle(L("Fig E. critic hacking: Geo_q_buf+br (T019 프로토콜 CPU 재실행, e09_hacking_case.py)", "Fig E. critic hacking"), fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/figs/figE_critic_hacking.png", dpi=160); plt.close(fig)
    zf = [v["ascent_mean_z"][-1] for v in h["seeds"].values()]
    wf = [v["ascent_win"][-1] for v in h["seeds"].values()]
    rows_csv.append(dict(exp="E04", quantity="Geo_q ascent20 dz (T019 csv)", value=float(ce["Geo_q_buf+br_f1.0"]["test_ascent_dz"]),
                         lo="", hi="", n="1920 rows", source="t019/critic_eval.csv", status="diagnostic"))
    rows_csv.append(dict(exp="E04", quantity="Geo_q ascent20 dz (rerun, mean of 3 seeds)", value=float(np.mean(zf)), lo=min(zf), hi=max(zf),
                         n="1920 rows", source="e09_hacking_case.py", status="diagnostic(recomputed)"))
    rows_csv.append(dict(exp="E04", quantity="Geo_q ascent20 win (rerun)", value=float(np.mean(wf)), lo=min(wf), hi=max(wf),
                         n="1920 rows", source="e09_hacking_case.py", status="diagnostic(recomputed)"))


def static_rows(rows):
    S = lambda exp, q, v, lo="", hi="", n="", src="", st="": rows.append(dict(exp=exp, quantity=q, value=v, lo=lo, hi=hi, n=n, source=src, status=st))
    S("E01", "success_once frozen pi0.5 (3 seeds x 30 ep)", 0.533, 0.431, 0.633, 90, "t016/summary_ko.md", "baseline")
    S("E01", "success_at_end", 0.511, 0.410, 0.612, 90, "t016/summary_ko.md", "baseline")
    S("E01", "latency batch1 ms/decision", 141, n="", src="t016/summary_ko.md", st="baseline")
    S("E01", "OraclePass@16 pre-success", 0.610, 0.45, 0.78, 59, "t016/summary_ko.md", "diagnostic")
    S("E01", "cand16-ctrl16 pre-success", 0.034, 0.00, 0.08, 59, "t016/summary_ko.md", "diagnostic")
    S("E02", "dataset success_once all 500 ep", 0.586, n=500, src="t017/summary_ko.md", st="data")
    S("E02", "ICC (candidate share of outcome variance)", 0.21, n="50x8x4", src="t017/summary_ko.md", st="diagnostic")
    S("E02", "E[max p_i]-E[p_i] beta-binomial", 0.118, 0.074, 0.154, "50 states", "t017/summary_ko.md", "diagnostic")
    S("E02", "excess over noise null", 0.035, 0.013, 0.062, "50 states", "t017/summary_ko.md", "diagnostic;perm p=0.0005")
    S("E02", "split-half selector gain", 0.039, -0.015, 0.099, "50 states", "t017/summary_ko.md", "diagnostic")
    S("E02", "Pass@1/Pass@4/Pass@16 (397 states)", "0.553/0.689/0.748", n=397, src="t017/summary_ko.md", st="diagnostic")
    for c, f, v, lo, hi, raw, g, glo, ghi in [
        ("succ_pre AUC", "vlm17_img_mean", 0.894, 0.86, 0.93, 0.817, 0.077, 0.044, 0.110),
        ("succ_pre AUC within-t", "vlm17_img_mean", 0.800, 0.74, 0.86, 0.646, 0.154, 0.087, 0.222),
        ("succk15_pre AUC", "vlm13_img_mean", 0.971, 0.96, 0.98, 0.861, 0.111, 0.069, 0.158),
        ("value R2", "vlm13_img_mean", 0.731, 0.67, 0.78, 0.142, 0.589, 0.50, 0.68),
        ("value_pre R2", "vlm17_img_mean", 0.478, 0.39, 0.56, 0.225, 0.253, 0.18, 0.32),
        ("geo_rel R2", "vlm9_img_grid4", 0.943, 0.93, 0.96, 0.340, 0.603, 0.57, 0.64),
        ("phase4 macro-F1", "vlm9_txt_mean", 0.695, 0.64, 0.74, 0.413, 0.282, 0.22, 0.35),
        ("grasp AUC", "vlm9_img_mean", 0.989, 0.98, 0.99, 0.874, 0.114, 0.07, 0.17)]:
        S("E03", f"probe {c} best ({f})", v, lo, hi, "75 test ep", "t018/summary_ko.md", "exploratory(val-selected)")
        S("E03", f"probe {c} gain vs proprio+t ({raw})", g, glo, ghi, "75 test ep", "t018/summary_ko.md", "exploratory(val-selected)")
    S("E04", "gate: Geo_q_buf+br test conc_1c", 0.544, 0.501, 0.586, "120 states", "t019/summary_ko.md", "prereg(gate, passed at boundary)")
    S("E04", "noise ceiling conc_1c", 0.81, 0.79, 0.84, "120 states", "t019/summary_ko.md", "diagnostic")
    S("E06", "latency batch1 ms: B0/Random8/ActOnly8/Critic8", "266/782/788/827", src="t020/bench_latency.json", st="cost")
    S("cost", "GPU-h T016/T017/T018/T019/T020/T027", "2.5/3.2/0/0.95/1.5/0", src="summaries", st="cost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/t027")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/figs", exist_ok=True)
    rows = []
    static_rows(rows)
    fig_sample_eff(a.out, rows)
    fig_ranking(a.out, rows)
    fig_oracle(a.out, rows)
    diffs, adj = fig_rerank(a.out, rows)
    fig_hacking(a.out, rows)
    with open(f"{a.out}/summary_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "quantity", "value", "lo", "hi", "n", "source", "status"])
        w.writeheader(); w.writerows(rows)
    json.dump({r["cond"]: {"mcnemar_p": float(r["mcnemar_p"]), "holm_p": float(p)} for r, p in zip(diffs, adj)},
              open(f"{a.out}/holm_t020.json", "w"), indent=1)
    print("KO font:", KO, "rows:", len(rows))
    for r, p in zip(diffs, adj):
        print(r["cond"], r["mcnemar_p"], round(float(p), 3))


if __name__ == "__main__":
    main()
