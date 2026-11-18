"""Aggregate oracle/diversity jsonl -> tables (markdown + json).

pass@N uses the unbiased estimator 1 - C(n-c, N)/C(n, N) over the n=16 branches of a decision
(c = successful branches), i.e. the expected OraclePass of a random size-N candidate subset.
'cand' = 16 different first chunks; 'ctrl' = the same first chunk (candidate 0) x 16 continuations.
CIs: bootstrap over episodes (2000 resamples).
"""
from __future__ import annotations

import glob
import json
import sys
from math import comb

import numpy as np

NS = (1, 4, 8, 16)


def pass_at(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def boot(ep_vals, f=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(ep_vals)
    stats = []
    for _ in range(B):
        ks = rng.choice(len(keys), len(keys))
        v = np.concatenate([ep_vals[keys[i]] for i in ks])
        stats.append(f(v))
    return np.percentile(stats, [2.5, 97.5]).tolist()


def main(pattern):
    recs = [json.loads(line) for fn in sorted(glob.glob(pattern)) for line in open(fn)]
    eps = sorted({r["episode_id"] for r in recs})
    out = {"n_episodes": len(eps), "n_decision_states": len(recs)}
    # ---------- diversity ----------
    keys = ["l2_env_K_perstep", "l2_env_H_perstep", "l2_norm_K_perstep", "l2_norm_H_perstep",
            "ee_disp_var_K", "ee_disp_var_H", "ee_disp_mean_norm_K", "ee_disp_mean_norm_H",
            "rot_disp_var_K", "grip_unique_patterns_K", "grip_unique_patterns_H", "grip_pattern_entropy_H",
            "grip_majority_frac_H", "grip_value_std_mean"]
    phases = ["all", "approach", "transport", "post_grasp_lost", "placed"]
    div = {}
    for ph in phases:
        rs = [r for r in recs if ph == "all" or r["phase"] == ph]
        if not rs:
            continue
        d = {"n": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            d[k] = dict(mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                        p90=float(np.percentile(v, 90)))
        # relative EE spread: std of predicted displacement / mean displacement norm
        rel = np.array([np.sqrt(r["ee_disp_var_K"]) / max(r["ee_disp_mean_norm_K"], 1e-9) for r in rs])
        d["ee_rel_spread_K"] = dict(mean=float(rel.mean()), median=float(np.median(rel)))
        relH = np.array([np.sqrt(r["ee_disp_var_H"]) / max(r["ee_disp_mean_norm_H"], 1e-9) for r in rs])
        d["ee_rel_spread_H"] = dict(mean=float(relH.mean()), median=float(np.median(relH)))
        div[ph] = d
    out["diversity"] = div
    # ---------- oracle ----------
    orc = [r for r in recs if "cand_success_once" in r]
    out["n_oracle_states"] = len(orc)

    def table(rs, field):
        res = {}
        for N in NS:
            per_ep = {}
            for r in rs:
                v = r[field]
                per_ep.setdefault(r["episode_id"], []).append(pass_at(sum(v), len(v), N))
            per_ep = {k: np.array(v) for k, v in per_ep.items()}
            allv = np.concatenate(list(per_ep.values())) if per_ep else np.array([])
            res[N] = dict(mean=float(allv.mean()) if len(allv) else float("nan"),
                          ci=boot(per_ep) if per_ep else None, n=int(len(allv)))
        return res

    subsets = {
        "all": orc,
        "pre_success": [r for r in orc if not r["main_success_before"]],
        "d0_initial": [r for r in orc if r["decision"] == 0],
        "main_failed_eps_pre": [r for r in orc if not r["main_success_once"] and not r["main_success_before"]],
    }
    for ph in ["approach", "transport", "post_grasp_lost"]:
        subsets[f"phase_{ph}_pre"] = [r for r in orc if r["phase"] == ph and not r["main_success_before"]]
    ot = {}
    for name, rs in subsets.items():
        if not rs:
            continue
        ot[name] = {"n_states": len(rs), "n_eps": len({r["episode_id"] for r in rs}),
                    "cand_once": table(rs, "cand_success_once"), "ctrl_once": table(rs, "ctrl_success_once"),
                    "cand_end": table(rs, "cand_success_end")}
        # paired diff cand@16 - ctrl@16 and cand@16 - cand@1 per episode
        for a, b, nm in (("cand_success_once", "cand_success_once", "gain16_vs_1"),
                         ("cand_success_once", "ctrl_success_once", "cand16_minus_ctrl16")):
            per_ep = {}
            for r in rs:
                x = pass_at(sum(r[a]), 16, 16)
                y = pass_at(sum(r[b]), 16, 1) if nm == "gain16_vs_1" else pass_at(sum(r[b]), 16, 16)
                per_ep.setdefault(r["episode_id"], []).append(x - y)
            per_ep = {k: np.array(v) for k, v in per_ep.items()}
            ot[name][nm] = dict(mean=float(np.concatenate(list(per_ep.values())).mean()), ci=boot(per_ep))
    out["oracle"] = ot
    # ---------- episode-level ----------
    ep = {}
    for e in eps:
        rs = [r for r in orc if r["episode_id"] == e]
        ep[e] = dict(main=bool(rs[0]["main_success_once"]),
                     any_cand=any(max(r["cand_success_once"]) for r in rs),
                     any_cand_pre=any(max(r["cand_success_once"]) for r in rs if not r["main_success_before"]),
                     d0_cand_any=bool(max(rs[0]["cand_success_once"])) if rs[0]["decision"] == 0 else None,
                     d0_cand_rate=float(np.mean(rs[0]["cand_success_once"])) if rs[0]["decision"] == 0 else None,
                     d0_ctrl_any=bool(max(rs[0]["ctrl_success_once"])) if rs[0]["decision"] == 0 else None)
    m = np.array([v["main"] for v in ep.values()])
    out["episode"] = dict(
        main_success_once=float(m.mean()),
        d0_cand_pass1=float(np.mean([v["d0_cand_rate"] for v in ep.values()])),
        d0_cand_pass16=float(np.mean([v["d0_cand_any"] for v in ep.values()])),
        d0_ctrl_pass16=float(np.mean([v["d0_ctrl_any"] for v in ep.values()])),
        any_decision_cand_success=float(np.mean([v["any_cand"] for v in ep.values()])),
        failed_eps=int((~m).sum()),
        failed_eps_rescuable_any_decision=float(np.mean([v["any_cand_pre"] for v in ep.values() if not v["main"]]))
        if (~m).any() else None,
    )
    out["per_episode"] = {str(k): v for k, v in ep.items()}
    print(json.dumps({k: v for k, v in out.items() if k != "per_episode"}, indent=1))
    return out


if __name__ == "__main__":
    o = main(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(o, f, indent=1)
