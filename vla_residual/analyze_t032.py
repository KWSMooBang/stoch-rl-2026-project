"""T032 analysis (CPU): design doc 9.1 paired metrics for E2 (GRPO, VLM feature) vs E0 (VLM feature = 0).

Per run, from metrics.json `final_eval_episodes` (last evaluation, 128 held-out ids) paired by episode id with the
B0 episodes in /outputs/StochRL/t022/val/b0.json, for success_once and success_at_end separately:
    success      = residual success rate
    preservation = P(residual success | base success)
    recovery     = P(residual success | base failure)
    regression   = P(residual failure | base success) = 1 - preservation
Condition value = mean over seeds; seed SD = sample SD (ddof 1) over seeds.
CI: two-stage bootstrap. Each replicate resamples seeds (runs) with replacement inside each condition, and resamples
episode ids with replacement once, shared by all runs and B0 (keeps the episode pairing). Percentile 95 %.
Verdict (task spec): "E2 clearly better" iff the CI lower bound of E2 - E0 success (once or end) is > 0.
Caveat: b0.json was measured on GPU 1; GPU 0 B0 may be lower (T029), which shifts preservation/recovery of BOTH
conditions the same way. E0 vs E2 are all GPU 0 runs.

  python vla_residual/analyze_t032.py --out /outputs/StochRL/t032/analysis
"""
import argparse
import json
import os

import numpy as np

RUNS = "/outputs/StochRL/runs"
B0_JSON = "/outputs/StochRL/t022/val/b0.json"
E2 = ["t022-grpo-s0", "t030-grpo-s1", "t030-grpo-s2"]
E0 = ["t032-grpo-e0-s0", "t032-grpo-e0-s1", "t032-grpo-e0-s2"]
KEYS = {"once": "success_once", "end": "success_at_end"}
METRICS = ["success", "preservation", "recovery", "regression"]


def load_run(run_id):
    """Returns (run_id actually used, {eid: episode}) or None. Accepts a -rN resubmission of the same run."""
    cands = [run_id] + [f"{run_id}-r{n}" for n in range(1, 21)]
    for c in cands:
        p = os.path.join(RUNS, c, "metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        if m.get("return_code") == 0 and len(m.get("final_eval_episodes", [])) > 0:
            return c, {e["eid"]: e for e in m["final_eval_episodes"]}
    return None


def rates(res, base):
    """res, base: bool arrays over episodes."""
    nb, nf = base.sum(), (~base).sum()
    pres = (res & base).sum() / nb if nb else np.nan
    return dict(success=res.mean(), preservation=pres, recovery=(res & ~base).sum() / nf if nf else np.nan,
                regression=1 - pres if nb else np.nan)


def cond_stats(R, base):
    """R: [n_seed, n_ep] bool -> per-seed dicts."""
    return [rates(r, base) for r in R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t032/analysis")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b0 = {e["eid"]: e for e in json.load(open(B0_JSON))["episodes"]}
    conds = {}
    for name, ids in (("E2", E2), ("E0", E0)):
        got = [load_run(r) for r in ids]
        conds[name] = [g for g in got if g is not None]
        missing = [r for r, g in zip(ids, got) if g is None]
        if missing:
            print(f"[warn] {name}: missing/unfinished runs {missing}")
    eids = sorted(set(b0).intersection(*[set(ep) for c in conds.values() for _, ep in c]))
    n = len(eids)
    out = dict(n_episodes=n, b0_json=B0_JSON, runs={k: [r for r, _ in v] for k, v in conds.items()},
               n_boot=a.n_boot, b0={}, per_run={}, cond={}, diff_E0_minus_E2={})
    rng = np.random.default_rng(a.seed)
    lines = [f"# T032 paired metrics (held-out {n} episodes, last evaluation; base = b0.json)", ""]
    for tag, key in KEYS.items():
        base = np.array([b0[e][key] for e in eids], bool)
        out["b0"][tag] = float(base.mean())
        M = {c: np.array([[ep[e][key] for e in eids] for _, ep in v], bool).reshape(len(v), n) for c, v in conds.items()}
        point = {}
        for c, R in M.items():
            ps = cond_stats(R, base)
            for (rid, _), p in zip(conds[c], ps):
                out["per_run"].setdefault(rid, {})[tag] = {k: float(v) for k, v in p.items()}
            point[c] = {k: dict(mean=float(np.mean([p[k] for p in ps])) if ps else None,
                                seed_sd=float(np.std([p[k] for p in ps], ddof=1)) if len(ps) > 1 else None,
                                per_seed=[float(p[k]) for p in ps]) for k in METRICS}
        # two-stage bootstrap
        bs = {c: {k: [] for k in METRICS} for c in M}
        bd = {k: [] for k in METRICS}
        for _ in range(a.n_boot):
            ei = rng.integers(0, n, n)
            bb = base[ei]
            cur = {}
            for c, R in M.items():
                if len(R) == 0:
                    continue
                si = rng.integers(0, len(R), len(R))
                ps = cond_stats(R[si][:, ei], bb)
                cur[c] = {k: np.mean([p[k] for p in ps]) for k in METRICS}
                for k in METRICS:
                    bs[c][k].append(cur[c][k])
            if len(cur) == 2:
                for k in METRICS:
                    bd[k].append(cur["E0"][k] - cur["E2"][k])

        def ci(v):
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None

        for c in M:
            for k in METRICS:
                point[c][k]["ci95"] = ci(bs[c][k])
        out["cond"][tag] = point
        both = all(len(M[c]) > 0 for c in ("E0", "E2"))
        if both:
            out["diff_E0_minus_E2"][tag] = {k: dict(diff=point["E0"][k]["mean"] - point["E2"][k]["mean"], ci95=ci(bd[k]))
                                            for k in METRICS}
        lines += [f"## {tag} (B0 {base.mean():.3f})", "",
                  "| metric | E2 mean (seed SD) [CI] | E2 per seed | E0 mean (seed SD) [CI] | E0 per seed | E0 − E2 [CI] |",
                  "|---|---|---|---|---|---|"]

        def cell(c, k):
            p = point[c][k]
            if p["mean"] is None:
                return "–", "–"
            sd = f"{p['seed_sd']:.3f}" if p["seed_sd"] is not None else "–"
            return (f"{p['mean']:.3f} ({sd}) [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]",
                    " / ".join(f"{x:.3f}" for x in p["per_seed"]))

        for k in METRICS:
            d = out["diff_E0_minus_E2"].get(tag, {}).get(k)
            ds = f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "–"
            lines.append(f"| {k} | " + " | ".join(cell("E2", k)) + " | " + " | ".join(cell("E0", k)) + f" | {ds} |")
        lines.append("")
    d = out["diff_E0_minus_E2"]
    if d and len(conds["E0"]) == len(E0) and len(conds["E2"]) == len(E2):
        # E2 - E0 CI lower bound > 0  <=>  E0 - E2 CI upper bound < 0
        better = [t for t in KEYS if d[t]["success"]["ci95"][1] < 0]
        out["verdict"] = ("E2 clearly better (" + ", ".join(better) + "): recommend A2 (E1/E3/E4)") if better else \
            "E0 ~= E2 or E0 > E2: GRPO residual gain is not due to the VLA feature; A2 / CCFR not recommended"
    else:
        out["verdict"] = "incomplete: need all 3 E0 and 3 E2 runs"
    lines += [f"**verdict**: {out['verdict']}", "",
              "- regression = 1 − preservation. CI = two-stage bootstrap (seeds within condition × shared episode ids), "
              f"{a.n_boot} replicates, percentile 95 %.",
              "- b0.json was measured on GPU 1; GPU 0 B0 may be lower (T029). This affects both conditions equally."]
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "paired_metrics.json"), "w"), indent=1)
    open(os.path.join(a.out, "paired_metrics.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
