"""T033 analysis (CPU): where does the E0 - E2 success_at_end gap come from?

Uses only saved logs: eval_episodes.jsonl of the 3 E2 and 3 E0 GRPO runs (all 3 evaluations, 128 held-out ids each)
and /outputs/StochRL/t022/val/b0.json. No rollout, no GPU.

Per run and evaluation:
    once      = P(success_once)
    end       = P(success_at_end)
    drop      = P(success_once and not success_at_end)       (unconditional; "once success -> end failure")
    drop_cond = P(not success_at_end | success_once)
    late      = P(success_at_end and not success_once)        (must be 0: end success implies once success)
Because late == 0, end = once - drop exactly, so for the condition means
    (E0 - E2) end = (E0 - E2) once - (E0 - E2) drop
and share_drop = -(E0 - E2) drop / (E0 - E2) end is the fraction of the end gap explained by once->end failures.
Also split drop_cond by the base (B0) outcome of the same episode id, and compare episodes that BOTH conditions
(same seed index) solved once.
CI: the two-stage bootstrap of analyze_t032.py (seeds within condition x episode ids shared by all runs), percentile 95 %.

  python vla_residual/analyze_t033.py --out /outputs/StochRL/t032/analysis
"""
import argparse
import json
import os

import numpy as np

from analyze_t032 import B0_JSON, E0, E2, RUNS

STATS = ["once", "end", "drop", "drop_cond", "drop_cond_base_succ", "drop_cond_base_fail"]


def load_evals(run_id):
    """{eval index: {eid: (once, end)}}"""
    out = {}
    for line in open(os.path.join(RUNS, run_id, "eval_episodes.jsonl")):
        e = json.loads(line)
        d = out.setdefault(e["eval"], {})
        assert e["eid"] not in d, (run_id, e)
        d[e["eid"]] = (bool(e["success_once"]), bool(e["success_at_end"]))
    return out


def div(a, b):
    return float(a) / float(b) if b else np.nan


def stats(o, e, bo):
    """o, e: residual once/end bool arrays; bo: base once."""
    d = o & ~e
    return dict(once=o.mean(), end=e.mean(), drop=d.mean(), drop_cond=div(d.sum(), o.sum()),
                drop_cond_base_succ=div((d & bo).sum(), (o & bo).sum()),
                drop_cond_base_fail=div((d & ~bo).sum(), (o & ~bo).sum()), late=(e & ~o).mean(),
                n_once=int(o.sum()), n_drop=int(d.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t032/analysis")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b0 = {e["eid"]: e for e in json.load(open(B0_JSON))["episodes"]}
    ev = {c: [load_evals(r) for r in ids] for c, ids in (("E2", E2), ("E0", E0))}
    n_eval = min(len(x) for v in ev.values() for x in v)
    last = n_eval - 1
    eids = sorted(set(b0).intersection(*[set(x[k]) for v in ev.values() for x in v for k in range(n_eval)]))
    n = len(eids)
    bo = np.array([b0[i]["success_once"] for i in eids], bool)
    be = np.array([b0[i]["success_at_end"] for i in eids], bool)
    # consistency with metrics.json final_eval_episodes (what analyze_t032.py used)
    for c, ids in (("E2", E2), ("E0", E0)):
        for r, x in zip(ids, ev[c]):
            fin = {e["eid"]: (e["success_once"], e["success_at_end"])
                   for e in json.load(open(os.path.join(RUNS, r, "metrics.json")))["final_eval_episodes"]}
            assert fin == x[last], f"{r}: final_eval_episodes != eval {last} of eval_episodes.jsonl"

    def arr(c, k, j):  # [n_seed, n] bool
        return np.array([[x[k][i][j] for i in eids] for x in ev[c]], bool)

    out = dict(n_episodes=n, n_evals=n_eval, runs=dict(E2=E2, E0=E0), n_boot=a.n_boot,
               b0=stats(bo, be, bo), per_run={}, cond={}, diff_E0_minus_E2={}, by_eval={}, both_once={})
    L = [f"# T033 once→end 실패 분해 (held-out {n} episode, base = b0.json)", "",
         f"- B0(잔차 0): once {bo.mean():.3f}, end {be.mean():.3f}, once→end 실패 {out['b0']['drop']:.3f} "
         f"({out['b0']['n_drop']}/{out['b0']['n_once']} = 조건부 {out['b0']['drop_cond']:.3f})",
         f"- end 성공인데 once 실패인 episode: 모든 run·eval에서 0 (아래 검증) → end = once − drop 이 정확히 성립", ""]
    # ---- per evaluation table
    L += ["## eval별 (조건 평균, seed별 값)", "",
          "| eval | 조건 | once | end | drop = P(once ∧ ¬end) | drop_cond = P(¬end \\| once) |", "|---|---|---|---|---|---|"]
    for k in range(n_eval):
        for c in ("E2", "E0"):
            O, E = arr(c, k, 0), arr(c, k, 1)
            ps = [stats(o, e, bo) for o, e in zip(O, E)]
            assert all(p["late"] == 0 for p in ps)
            out["by_eval"].setdefault(str(k), {})[c] = {s: [float(p[s]) for p in ps] for s in STATS}
            f = lambda s: f"{np.mean([p[s] for p in ps]):.3f} ({' / '.join(f'{p[s]:.3f}' for p in ps)})"
            L.append(f"| {k} | {c} | {f('once')} | {f('end')} | {f('drop')} | {f('drop_cond')} |")
    L.append("")
    # ---- last evaluation: per run + bootstrap
    M = {c: (arr(c, last, 0), arr(c, last, 1)) for c in ("E2", "E0")}
    L += [f"## 마지막 eval (eval {last}) run별", "",
          "| 조건 | run | once | end | once 성공 수 | once→end 실패 수 | drop | drop_cond | drop_cond (base 성공) | drop_cond (base 실패) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    point = {}
    for c, ids in (("E2", E2), ("E0", E0)):
        ps = [stats(o, e, bo) for o, e in zip(*M[c])]
        for r, p in zip(ids, ps):
            out["per_run"][r] = {k: (float(v) if not isinstance(v, int) else v) for k, v in p.items()}
            L.append(f"| {c} | {r} | {p['once']:.3f} | {p['end']:.3f} | {p['n_once']} | {p['n_drop']} | {p['drop']:.3f} | "
                     f"{p['drop_cond']:.3f} | {p['drop_cond_base_succ']:.3f} | {p['drop_cond_base_fail']:.3f} |")
        point[c] = {s: dict(mean=float(np.mean([p[s] for p in ps])), seed_sd=float(np.std([p[s] for p in ps], ddof=1)))
                    for s in STATS}
    L.append("")
    rng = np.random.default_rng(a.seed)
    bs = {c: {s: [] for s in STATS} for c in M}
    bd = {s: [] for s in STATS + ["share_drop"]}
    for _ in range(a.n_boot):
        ei = rng.integers(0, n, n)
        cur = {}
        for c, (O, E) in M.items():
            si = rng.integers(0, len(O), len(O))
            ps = [stats(o, e, bo[ei]) for o, e in zip(O[si][:, ei], E[si][:, ei])]
            cur[c] = {s: np.mean([p[s] for p in ps]) for s in STATS}
            for s in STATS:
                bs[c][s].append(cur[c][s])
        for s in STATS:
            bd[s].append(cur["E0"][s] - cur["E2"][s])
        de = cur["E0"]["end"] - cur["E2"]["end"]
        bd["share_drop"].append(-(cur["E0"]["drop"] - cur["E2"]["drop"]) / de if de != 0 else np.nan)

    def ci(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    L += ["## 마지막 eval 조건 비교 (3 seed 평균, seed SD, 2단계 bootstrap 95 % CI)", "",
          "| 지표 | E2 평균 (SD) [CI] | E0 평균 (SD) [CI] | E0 − E2 [CI] |", "|---|---|---|---|"]
    for s in STATS:
        for c in M:
            point[c][s]["ci95"] = ci(bs[c][s])
        d = dict(diff=point["E0"][s]["mean"] - point["E2"][s]["mean"], ci95=ci(bd[s]))
        out["diff_E0_minus_E2"][s] = d
        g = lambda c: f"{point[c][s]['mean']:.3f} ({point[c][s]['seed_sd']:.3f}) [{point[c][s]['ci95'][0]:.3f}, {point[c][s]['ci95'][1]:.3f}]"
        L.append(f"| {s} | {g('E2')} | {g('E0')} | {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    out["cond"] = point
    D = out["diff_E0_minus_E2"]
    share = -D["drop"]["diff"] / D["end"]["diff"]
    out["share_drop"] = dict(value=share, ci95=ci(bd["share_drop"]),
                             share_once=D["once"]["diff"] / D["end"]["diff"])
    L += ["", "## end 차이 분해", "",
          f"- (E0 − E2) end {D['end']['diff']:+.3f} = (E0 − E2) once {D['once']['diff']:+.3f} − (E0 − E2) drop {D['drop']['diff']:+.3f}",
          f"- once→end 실패가 설명하는 비율: **{share:.3f}** [CI {out['share_drop']['ci95'][0]:.3f}, {out['share_drop']['ci95'][1]:.3f}], "
          f"once 차이가 설명하는 비율: {out['share_drop']['share_once']:.3f}", ""]
    # ---- episodes solved once by both conditions (same seed index)
    L += ["## 두 조건이 모두 once 성공한 episode에서의 end 유지율 (같은 seed 짝)", "",
          "| seed | 둘 다 once 성공 수 | E2 end 유지 | E0 end 유지 | E2만 유지 | E0만 유지 |", "|---|---|---|---|---|---|"]
    for s in range(len(E2)):
        both = M["E2"][0][s] & M["E0"][0][s]
        h2, h0 = M["E2"][1][s] & both, M["E0"][1][s] & both
        r = dict(n_both=int(both.sum()), e2_hold=div(h2.sum(), both.sum()), e0_hold=div(h0.sum(), both.sum()),
                 only_e2=int((h2 & ~h0).sum()), only_e0=int((h0 & ~h2).sum()))
        out["both_once"][str(s)] = r
        L.append(f"| {s} | {r['n_both']} | {r['e2_hold']:.3f} | {r['e0_hold']:.3f} | {r['only_e2']} | {r['only_e0']} |")
    L += ["", f"- CI: seed(조건 안) × episode id(모든 run 공유) 2단계 bootstrap, {a.n_boot}회, percentile 95 %.",
          "- drop_cond의 base 성공/실패 구분은 b0.json의 success_once 기준이다(b0.json은 GPU 1 측정, T029 주의)."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "hold_metrics.json"), "w"), indent=1)
    open(os.path.join(a.out, "hold_metrics.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
