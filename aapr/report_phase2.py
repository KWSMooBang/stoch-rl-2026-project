"""Markdown tables from phase2.json.  usage: python -m aapr.report_phase2 phase2.json"""
import json
import sys


def main(p):
    R = json.load(open(p))
    L = []
    L.append("### (a) 고정 상태별 보상 기대값 (단위: 최대 1-step 이동 0.1, 2000 noise draw)\n")
    L.append("| 바람 | 상태 | action | E[prog] ± sd | b(s)=E[prog\\|a0] | E[AAPR1] ± sd | E[MH] ± sd | AAPR(absent) |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        for r in R["canon"][c]:
            L.append(f"| {c} | {r['state']} | {r['action']} | {r['prog_mean']:+.3f} ± {r['prog_std']:.3f} | "
                     f"{r['b']:+.3f} | {r['aapr_mean']:+.3f} ± {r['aapr_std']:.3f} | {r['mh_mean']:+.3f} ± {r['mh_std']:.3f} | "
                     f"{r['aapr_absent_mean']:+.3f} |")
    L.append("\n### (b) 방문 상태의 b(s) 분포 (scripted / random 정책, 상태 600개 × draw 64)\n")
    L.append("| 바람 | 정책 | b 평균 | b sd | b q05 / q50 / q95 | \\|b\\|>0.1 | \\|b\\|>0.25 | b>0 / b<0 | b 접촉 / 비접촉 | E[prog] | E[AAPR] | within-var prog / AAPR |")
    L.append("|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---|")
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        for pn in ["scripted", "random"]:
            b = R["bdist"][c][pn]
            L.append(f"| {c} | {pn} | {b['b_mean']:+.3f} | {b['b_std']:.3f} | {b['b_q05']:+.2f} / {b['b_q50']:+.2f} / {b['b_q95']:+.2f} | "
                     f"{b['frac_absb_gt_0p1']:.2f} | {b['frac_absb_gt_0p25']:.2f} | {b['frac_b_pos']:.2f} / {b['frac_b_neg']:.2f} | "
                     f"{b['b_mean_touching']:+.3f} / {b['b_mean_free']:+.3f} | {b['Eprog_mean']:+.3f} | {b['Eaapr_mean']:+.3f} | "
                     f"{b['within_state_var_prog']:.4f} / {b['within_state_var_aapr']:.4f} |")
    L.append("\n### (c) paired(CRN) vs independent 추정 분산 (scripted 방문 상태, 상태별 draw 64의 분산 평균)\n")
    L.append("| 바람 | K=1 paired / indep | K=4 paired / indep | K=8 paired / indep |")
    L.append("|---|---|---|---|")
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        b = R["bdist"][c]["scripted"]
        vp, vi = b["var_paired_K1_4_8"], b["var_indep_K1_4_8"]
        L.append(f"| {c} | " + " | ".join(f"{vp[i]:.4f} / {vi[i]:.4f}" for i in range(3)) + " |")
    L.append("\n### (d) 보상 단위 테스트: 에피소드 합 (λ=1, 100 episode, 비할인 / γ=0.99 할인)\n")
    ms = ["sparse", "dist", "prog", "pbrs", "aapr1", "mh", "mh_ind"]
    L.append("| 바람 | 정책 | SR | " + " | ".join(ms) + " |")
    L.append("|---|---|---:|" + "---:|" * len(ms))
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        u = R["unit"][c]
        for pn in ["scripted", "random", "zero"]:
            L.append(f"| {c} | {pn} | {u[pn]['sr']:.2f} | " + " | ".join(
                f"{u[pn]['ret'][m]:.1f} / {u[pn]['disc'][m]:.1f}" for m in ms) + " |")
        L.append(f"| {c} | **pass** (scripted > random, zero) | | " + " | ".join(
            f"{'O' if u['pass_undisc'][m] else 'X'} / {'O' if u['pass_disc'][m] else 'X'}" for m in ms) + " |")
    L.append("\n### (e) per-step 보상 통계 (실현값, 단위 0.1)\n")
    L.append("| 바람 | 정책 | prog 평균 / 분산 | AAPR1 평균 / 분산 |")
    L.append("|---|---|---|---|")
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        for pn in ["scripted", "random", "zero"]:
            u = R["unit"][c][pn]
            L.append(f"| {c} | {pn} | {u['step_prog_mean']:+.3f} / {u['step_prog_var']:.4f} | {u['step_aapr_mean']:+.3f} / {u['step_aapr_var']:.4f} |")
    L.append("\n### (f) 바람이 도와주는 초기 상태(정지 반례)에서 SR\n")
    L.append("| 바람 | zero 정책 | scripted |")
    L.append("|---|---:|---:|")
    for c in ["ar1", "bias", "iid", "ar1_hi"]:
        h = R["helps"][c]
        L.append(f"| {c} | {h['zero']:.2f} | {h['scripted']:.2f} |" if h else f"| {c} | n/a | n/a |")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
