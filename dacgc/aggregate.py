"""Aggregation, pre-registered gate decision and validation selection."""
from __future__ import annotations

import numpy as np

KEYS = ["SR_seen", "SR_unseen", "seen_rare_sr", "seen_freq_sr", "SER_unseen", "SER_unseen_freq",
        "unseen_wt_other", "unseen_wrong_obj", "unseen_timeout", "seen_wt", "seen_rare_wt_freq"]
RANK_KEYS = ["rankacc_branch_seen", "rankacc_common_seen", "rankacc_branch_heldout", "rankacc_common_heldout",
             "rankacc_branch_holding_seen", "rankacc_branch_holding_heldout", "sap_seen", "sap_heldout",
             "shared_divergence_seen", "shared_divergence_heldout"]
SWITCH_KEYS = ["switch_approach_seen_follow", "switch_approach_unseen_follow", "switch_carry_seen_follow",
               "switch_carry_unseen_follow", "switch_carry_unseen_to_freq", "switch_approach_unseen_to_freq"]
GATE_KEYS = ["gate_rate", "gate_rate_handempty", "gate_rate_holding_shared_obj", "gate_rate_holding_goal_obj_diffobj",
             "gate_rate_holding_other", "oracle_gate_rate", "learned_gate_recall", "learned_gate_precision",
             "learned_gate_recall_rare_holding", "frac_oracle_gated_with_failed_learned_continuation",
             "learned_gate_on_shortcut_states", "learned_label_agree"]


def _stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.array(v, float)
    return dict(mean=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                per_seed=[float(x) for x in a])


def group(results, key_fn):
    G = {}
    for r in results:
        if "error" in r:
            continue
        G.setdefault(key_fn(r["cfg"]), []).append(r)
    out = {}
    for k, rs in G.items():
        rs = sorted(rs, key=lambda r: r["cfg"]["seed"])
        d = dict(n=len(rs), seeds=[r["cfg"]["seed"] for r in rs])
        for m in KEYS:
            d[m] = _stats([r["summary"][m] for r in rs])
        for m in RANK_KEYS:
            d[m] = _stats([r.get("rank", {}).get(m) for r in rs])
        for m in SWITCH_KEYS:
            d[m] = _stats([r.get("switch", {}).get(m) for r in rs])
        for m in GATE_KEYS:
            d[m] = _stats([(r.get("gate_summary") or {}).get(m) for r in rs])
        d["env_steps"] = rs[0]["env_steps"]
        d["branch_steps"] = _stats([r["branch_steps"] for r in rs])
        d["wall_total_s"] = _stats([r["wall_total_s"] for r in rs])
        d["leak_checks"] = int(sum(r["leak_checks"] for r in rs))
        # learning curve: mean over seeds at each eval point
        steps = [c["step"] for c in rs[0]["curve"]]
        d["curve"] = {
            "step": steps,
            **{m: [float(np.mean([r["curve"][i][m] for r in rs if i < len(r["curve"])])) for i in range(len(steps))]
               for m in ["SR_seen", "SR_unseen", "SR_seen_rare", "SER_unseen", "SER_unseen_freq"]}}
        out[k] = d
    return out


# ---------------- pre-registered Phase 0/1 decision ----------------
PHASE0_MIN_SR_SEEN = 0.7  # "B0 learns most of the 6 seen goals" (balanced)
GATE_ORDER = ["balanced", "skew0.8", "skew0.95", "skew0.95+biasedBC"]
GATE_MIN_SER_FREQ = 0.15  # held-out: goal object committed on the object's frequent target
GATE_MIN_RATIO = 2.0  # ... at least 2x as often as on the other wrong target (directed, not random)
GATE_MIN_SR_SEEN = 0.5  # the skills exist (otherwise failure is skill failure, not shortcut)


def gate_decision(agg):
    rows = []
    chosen = None
    p0 = agg.get("balanced", {}).get("SR_seen")
    phase0_go = bool(p0 and p0["mean"] >= PHASE0_MIN_SR_SEEN)
    for cond in GATE_ORDER:
        a = agg.get(cond)
        if a is None:
            continue
        ser_f = a["SER_unseen_freq"]["mean"]
        ser_o = a["unseen_wt_other"]["mean"]
        srs = a["SR_seen"]["mean"]
        ok = ser_f >= GATE_MIN_SER_FREQ and ser_f >= GATE_MIN_RATIO * ser_o and srs >= GATE_MIN_SR_SEEN
        rows.append(dict(condition=cond, SER_freq=ser_f, SER_other=ser_o, SR_seen=srs,
                         SR_unseen=a["SR_unseen"]["mean"], timeout_unseen=a["unseen_timeout"]["mean"],
                         wrong_obj_unseen=a["unseen_wrong_obj"]["mean"], meets=ok))
        if ok and chosen is None:
            chosen = cond
    return dict(phase0_go=phase0_go, phase0_sr_seen_balanced=p0["mean"] if p0 else None,
                rows=rows, chosen=chosen, go=bool(phase0_go and chosen is not None))


# ---------------- pre-registered validation selection (Phase 3) ----------------
def select(agg_by_cfg):
    """Validation seeds only, held-out goals never used.  Criterion: highest mean SR on the
    frequency-starved *train* goals (seen_rare_sr); ties within 0.02 broken by SR_seen, then by
    the smaller lambda."""
    items = []
    for k, a in agg_by_cfg.items():
        items.append((k, a["seen_rare_sr"]["mean"], a["SR_seen"]["mean"]))
    best = max(v[1] for v in items)
    cand = [v for v in items if v[1] >= best - 0.02]
    cand.sort(key=lambda v: (-v[2], v[0]))
    return cand[0][0], items
