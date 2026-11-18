"""E04 (T019) diagnostic: within-state concordance of critic scores (3-seed z-avg, test) with short-horizon after-K
progress labels, and of those labels with the branch success outcome.

  python -m repcritic.e04_afterk
"""
import glob
import json

import numpy as np

from repcritic.e04_critic import conc, conc_parts

NAMES = ["Geo_q_buf+br_f1.0", "Fusion_q_buf+br_f1.0", "Fusion_nce_buf+br_f1.0", "Full_bce_buf+br_f1.0",
         "Raw_q_buf+br_f1.0", "Fusion_pair_buf+br_f1.0"]


def main():
    D = dict(np.load("/workspace/stochrl_t019/data.npz"))
    runs = {}
    for f in glob.glob("/workspace/stochrl_t019/runs/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["frac"] == 1.0 and r["name"] in NAMES:
                runs.setdefault(r["name"], []).append(r)
    m = D["b_split"] == "test"
    st = D["b_state"][m]
    labs = {"grasp_after": D["b_grasp_after"][m], "neg_gc_after": -D["b_gc_after"][m],
            "neg_cp_after": -D["b_cp_after"][m]}
    out = {}
    for nm in NAMES:
        S = np.mean([(np.array(r["test_scores_br"]) - np.mean(r["test_scores_br"])) / np.std(r["test_scores_br"])
                     for r in runs[nm]], 0)
        out[nm] = {k: round(conc(conc_parts(S, v, st)), 3) for k, v in labs.items()}
    y = D["b_y"][m]
    out["label_vs_success"] = {k: round(conc(conc_parts(v, y, st)), 3) for k, v in labs.items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open("/outputs/StochRL/t019/afterK_concordance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
