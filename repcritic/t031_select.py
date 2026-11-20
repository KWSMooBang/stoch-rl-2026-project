"""T031 state selection (CPU). Pre-registration: repcritic/reports/t031_prereg.md.

48 decision states from the T017 buffer TRAIN split, 4 strata x 12, one state per episode, episodes not used by
T017 branch sets nor T026 (fresh), d <= 12, phase != placed. T022 held-out 128 (perm[500:900] of seed 20261119) is
disjoint from the T017 buffer (perm[:500]) by construction; asserted below.
  S0_init  : d = 0
  S1_pre   : approach, d >= 1, gripper-object < 8 cm  (just before contact)
  S2_post  : grasp_contact, or transport with no earlier grasp (first decision right after contact)
  S3_lowV  : d >= 1, V_hat = T023 vla@platt (= load_potential("vla","platt") on vlm17_img_mean) < 0.082, not in S0-S2
Within a stratum: half from successful episodes, half from failed (logged success_once), filling from the other
half if one side runs short. Rarest stratum picked first. Run order = interleaved across strata (seeded), so that a
budget cut leaves a stratum-balanced prefix.

usage: .venv/bin/python -m repcritic.t031_select
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SRC = Path("/workspace/stochrl_t017")
OUT = Path("/workspace/stochrl_t031")
T026 = Path("/workspace/stochrl_t026/state_select.json")
VHAT = Path("/outputs/StochRL/t023/vhat_all.npz")
TAU = 0.082
NEAR_M = 0.08
PER = 12
SEED = 31031
STRATA = ("S0_init", "S1_pre", "S2_post", "S3_lowV")


def heldout_ids():
    g = torch.Generator().manual_seed(20261119)
    perm = torch.randperm(802816, generator=g).tolist()
    return set(perm[500:900])


def main():
    held = heldout_ids()
    train = [int(l.strip().replace("ep", "")) for l in open(SRC / "split_train.txt") if l.strip()]
    assert not (set(train) & held), "train split overlaps T022 held-out"
    used = set()
    for f in ("state_select", "state_select_ext"):
        used |= {s["eid"] for s in json.loads((SRC / "branch" / f"{f}.json").read_text())["states"]}
    used |= {s["eid"] for s in json.loads(T026.read_text())["states"]}
    vz = np.load(VHAT, allow_pickle=True)
    vidx = {int(str(e).replace("ep", "")): i for i, e in enumerate(vz["episodes"])}
    V = vz["vla@platt"]
    succ = {}
    for r in (json.loads(l) for f in sorted(SRC.glob("decisions_*.jsonl")) for l in open(f)):
        if r["decision"] == 0:
            succ[r["eid"]] = bool(r["success_once"])
    pool = {k: [] for k in STRATA}
    for e in sorted(set(train) - used):
        z = np.load(SRC / "privileged" / f"ep{e:06d}.npz")
        ph, gd, gn = z["dec_phase"].astype(str), z["dec_gripper_carrot_dist"], z["dec_grasping_now"].astype(bool)
        for d in range(13):
            if ph[d] == "placed":
                continue
            v = float(V[vidx[e], d])
            k = None
            if d == 0:
                k = "S0_init"
            elif ph[d] == "approach" and gd[d] < NEAR_M:
                k = "S1_pre"
            elif ph[d] == "grasp_contact" or (ph[d] == "transport" and not gn[:d].any()):
                k = "S2_post"
            elif v < TAU:
                k = "S3_lowV"
            if k:
                pool[k].append(dict(eid=e, decision=d, t=d * 5, stratum=k, phase=str(ph[d]),
                                    gripper_obj_dist=float(gd[d]), grasping_now=bool(gn[d]), vhat=v,
                                    episode_success_once=succ[e], split="train", state_id=f"ep{e:06d}_d{d:02d}"))
    rng = np.random.RandomState(SEED)
    taken, states = set(), []
    for k in sorted(STRATA, key=lambda k: len({x["eid"] for x in pool[k]})):
        got = []
        for want in (True, False):
            c = [x for x in pool[k] if x["episode_success_once"] == want]
            for i in rng.permutation(len(c)):
                if sum(g["episode_success_once"] == want for g in got) >= PER // 2:
                    break
                if c[i]["eid"] in taken:
                    continue
                got.append(c[i]); taken.add(c[i]["eid"])
        if len(got) < PER:  # fill from either side
            c = list(pool[k])
            for i in rng.permutation(len(c)):
                if len(got) >= PER:
                    break
                if c[i]["eid"] in taken:
                    continue
                got.append(c[i]); taken.add(c[i]["eid"])
        states += got
    by = {k: [s for s in states if s["stratum"] == k] for k in STRATA}
    for k in STRATA:
        by[k] = [by[k][i] for i in rng.permutation(len(by[k]))]
    order = [by[k][j] for j in range(PER) for k in STRATA if j < len(by[k])]
    for i, s in enumerate(order):
        s["run_order"] = i
    pool_stats = {k: dict(n=len(v), n_eps=len({x["eid"] for x in v}),
                          n_succ_ep=len({x["eid"] for x in v if x["episode_success_once"]})) for k, v in pool.items()}
    meta = dict(seed=SEED, tau=TAU, near_m=NEAR_M, per=PER, n_excluded_eids=len(used), pool=pool_stats,
                heldout_overlap=0, states=order)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_select.json").write_text(json.dumps(meta, indent=0))
    print(json.dumps(pool_stats))
    for k in STRATA:
        ss = [s for s in order if s["stratum"] == k]
        print(k, len(ss), "succ_ep", sum(s["episode_success_once"] for s in ss), "d", sorted(s["decision"] for s in ss),
              "phases", sorted({s["phase"] for s in ss}), "vhat_med %.3f" % np.median([s["vhat"] for s in ss]))


if __name__ == "__main__":
    main()
