"""E04 (T019) label-cost-inclusive concept inputs: refit the C-Goal/C-Geo/C-Phase ridge probes on the SAME train
episode subset the critic sees (10/25/50%), alpha per target re-chosen on VAL, 5-fold episode cross-fit on the
subset's train rows (same protocol as T018 fit_subspace). Output: z_crossfit [8000, 21] per (frac, seed).

  python -m repcritic.e04_lc            # CPU, writes /workspace/stochrl_t019/lc/C-FusionLC_f{frac}_s{seed}.npy
"""
import os

import numpy as np

from repcritic.e03_probe import build_concepts, episode_subsample, load_feat
from repcritic.e03_stage2 import fit_subspace

OUT = "/workspace/stochrl_t019/lc"
FRACS = (0.1, 0.25, 0.5)
SEEDS = (0, 1, 2)


def targets(L, C):
    sone = L["success_once"].astype(float)
    succ_by15 = (L["success_before"] | L["succ_k15"]).astype(float)
    return {
        "Goal": ("vlm17_img_mean", np.stack([sone, L["value"], succ_by15], 1), ["bin", "reg", "bin"], None),
        "Geo": ("vlm9_img_grid4", C["geo_rel"][1], ["reg"] * 12, C["geo_rel"][0]),
        "Phase": ("vlm9_txt_mean", np.concatenate([C["phase4"][1], C["grasp"][1], C["contact"][1]], 1), ["bin"] * 6,
                  None),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    L = dict(np.load("/workspace/stochrl_t018/cache/labels.npz"))
    C = build_concepts(L)
    T = targets(L, C)
    parts = {(f, s): {} for f in FRACS for s in SEEDS}
    # sanity: 100% refit reproduces the T018 file
    for c, (key, Y, kinds, rm) in T.items():
        X = load_feat(key)
        if c == "Goal":
            _, Z = fit_subspace(X, L["split"], Y.astype(float), kinds, rowmask=rm, episodes=L["episode"])
            ref = np.load(f"/workspace/stochrl_t018/results/e04_inputs/C-{c}.npz")["z"]
            print("100% refit vs T018 z max|d|:", float(np.abs(Z - ref).max()), flush=True)
        for f in FRACS:
            for s in SEEDS:
                sub = episode_subsample(L, f, s)
                sp = L["split"].copy()
                sp[(sp == "train") & ~sub] = "drop"
                info, _ = fit_subspace(X, sp, Y.astype(float), kinds, rowmask=rm, episodes=L["episode"])
                parts[(f, s)][c] = info["z_crossfit"]
                print("[lc]", c, f, s, flush=True)
        del X
    for (f, s), d in parts.items():
        np.save(f"{OUT}/C-FusionLC_f{f}_s{s}.npy", np.concatenate([d["Goal"], d["Geo"], d["Phase"]], 1).astype(np.float32))


if __name__ == "__main__":
    main()
