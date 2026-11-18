"""E04 (T019) data assembly: decision buffer + candidate-level branch rows -> /workspace/stochrl_t019/data.npz

Rows of the decision buffer follow /workspace/stochrl_t018/cache/labels.npz (episode, decision) order (8000).
Branch rows follow cand_labels.npz order (397 states x 16 cand = 6352).

Action = frozen-VLA chunk in the MODEL's normalized space, H=8 x 7 real dims (chunk_model[..., :7]); this is the space
the flow sampler integrates in, so E07 gradients come out in the right units. Dims 7..31 are padding (|.|<0.04).

  python -m repcritic.e04_data
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

SRC = "/workspace/stochrl_t017"
CACHE = "/workspace/stochrl_t018/cache"
OUT = "/workspace/stochrl_t019"
GAMMA = 0.98
K = 5
ADIM = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    L = dict(np.load(f"{CACHE}/labels.npz"))
    ep, dec, split = L["episode"], L["decision"], L["split"]
    N = len(ep)
    row_of = {(e, int(d)): i for i, (e, d) in enumerate(zip(ep, dec))}

    # ---------------- decision buffer
    A = np.zeros((N, 8, ADIM), np.float32)
    R = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    nxt = np.full(N, -1, np.int64)
    first_succ = {}
    for e in np.unique(ep):
        d = np.load(f"{SRC}/episodes/{e}.npz")
        cm = d["chunk_model"]
        assert np.abs(cm[..., ADIM:]).max() < 0.1
        ss = d["step_success"]
        tfs = int(np.argmax(ss)) if ss.any() else -1
        first_succ[e] = tfs
        for k in range(16):
            i = row_of[(e, k)]
            A[i] = cm[k, :, :ADIM]
            t = 5 * k
            if tfs >= 0 and t <= tfs < t + K:
                R[i], done[i] = GAMMA ** (tfs - t), True
            elif k == 15:
                done[i] = True          # fixed 80-step horizon
            if k < 15:
                nxt[i] = row_of[(e, k + 1)]
    pre = ~L["success_before"]
    # sanity: MC discounted return along logged trajectory == value label on pre-success rows
    mc = np.zeros(N, np.float32)
    for e in np.unique(ep):
        rows = [row_of[(e, k)] for k in range(16)]
        g = 0.0
        for i in reversed(rows):
            g = R[i] + (0.0 if done[i] else GAMMA ** K * g)
            mc[i] = g
    err = np.abs(mc[pre] - L["value"][pre]).max()
    print("MC vs value label max err (pre-success rows):", err)
    assert err < 1e-4

    # ---------------- branch rows
    CL = dict(np.load(f"{CACHE}/cand_labels.npz"))
    agg = defaultdict(lambda: dict(n=0, s=0.0, q=0.0, se=0.0, gc=0.0, cp=0.0, gr=0.0))
    for f in sorted(glob.glob(f"{SRC}/branch/branch_results_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            a = agg[(r["state_id"], r["cand"])]
            a["n"] += 1
            a["s"] += r["success_once"]
            a["se"] += r["success_at_end"]
            a["q"] += GAMMA ** (r["first_success_t"] - r["t"]) if r["success_once"] else 0.0
            a["gc"] += r["afterK"]["gripper_carrot_dist"]
            a["cp"] += r["afterK"]["carrot_plate_dist"]
            a["gr"] += float(r["afterK"]["grasping_now"])
    M = len(CL["state_id"])
    bA = np.zeros((M, 8, ADIM), np.float32)
    b = {k: np.zeros(M, np.float32) for k in ("y", "y_end", "q", "gc_after", "cp_after", "grasp_after")}
    b_ncont = np.zeros(M, np.int64)
    b_row = np.zeros(M, np.int64)
    cache_c = {}
    for j in range(M):
        sid, c = CL["state_id"][j], int(CL["cand"][j])
        if sid not in cache_c:
            cache_c[sid] = np.load(f"{SRC}/branch/candidates/{sid}.npz")["chunk_model"][..., :ADIM]
        bA[j] = cache_c[sid][c]
        a = agg[(sid, c)]
        n = a["n"]
        b_ncont[j] = n
        b["y"][j], b["y_end"][j], b["q"][j] = a["s"] / n, a["se"] / n, a["q"] / n
        b["gc_after"][j], b["cp_after"][j], b["grasp_after"][j] = a["gc"] / n, a["cp"] / n, a["gr"] / n
        b_row[j] = row_of[(CL["episode"][j], int(CL["decision"][j]))]
    assert np.allclose(b["y"], CL["succ_mean"]) and (b_ncont == CL["n_cont"]).all()
    assert (split[b_row] == CL["split"]).all()
    # cand 0 = logged executed chunk: the branch EXECUTED the logged action (candidates/*.npz holds a recompute that
    # differs slightly, rarely flipping the gripper), so use the logged one.
    c0 = CL["cand"] == 0
    print("cand0 recompute vs logged chunk max|d|:", np.abs(bA[c0] - A[b_row[c0]]).max())
    bA[c0] = A[b_row[c0]]
    np.savez(f"{OUT}/data.npz", A=A, R=R, done=done, nxt=nxt, pre=pre, split=split, episode=ep, decision=dec,
             obj_id=L["obj_id"], success_once=L["success_once"], success_before=L["success_before"],
             value=L["value"], b_A=bA, b_row=b_row, b_ncont=b_ncont, b_state=CL["state_id"], b_cand=CL["cand"],
             b_split=CL["split"], **{f"b_{k}": v for k, v in b.items()})
    for sp in ("train", "val", "test"):
        m = CL["split"] == sp
        print(sp, "buffer pre rows", int((pre & (split == sp)).sum()), "branch states", len(np.unique(CL["state_id"][m])),
              "oracle states", len(np.unique(CL["state_id"][m & (b_ncont >= 4)])))
    print("action std per dim (buffer):", A.reshape(-1, ADIM).std(0).round(3))


if __name__ == "__main__":
    main()
