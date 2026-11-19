"""E09 (T027) critic-hacking case study — CPU only, no new rollouts.

Reruns the T019 gradient-ascent probe (same protocol as e04_critic.ood_and_grad: 20 normalized ascent steps of
L2 size 0.05 over the chunk, on the T019 test branch candidates) for the T020-selected critic `Geo_q_buf+br`
(3 seeds), recording the WHOLE trajectory instead of only the endpoint, plus Gaussian action perturbations on a
finer sigma grid. Output: /outputs/StochRL/t027/hacking_case.json (+ .npz with per-state trajectories).

  python -m repcritic.e09_hacking_case
"""
import json
import os

import numpy as np
import torch

from repcritic.e04_critic import Ctx, Critic, load_z, score_rows

CK = "/outputs/StochRL/t019/critic_ckpt"
OUT = os.environ.get("E09_OUT", "/outputs/StochRL/t027")
NAME = "Geo_q_buf+br_f1.0"


def main():
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    ctx = Ctx(dev)
    D = ctx.D
    Zb_np, Zbr_np, kind = load_z("Geo", D)
    Zb = torch.from_numpy(Zb_np)
    _, brr = ctx.rows("test")
    br_t = torch.as_tensor(brr)
    st = D["b_state"][brr]
    y = D["b_y"][brr]
    A0 = ctx.bA[br_t]
    asd = ctx.A.reshape(-1, 7).std(0)
    groups = {g: np.where(st == g)[0] for g in np.unique(st)}
    res = {"critic": NAME, "n_rows": int(len(brr)), "n_states": len(groups), "seeds": {}}
    traj_all = []
    for seed in (0, 1, 2):
        ck = torch.load(f"{CK}/{NAME}/seed{seed}.pt", map_location="cpu", weights_only=False)
        m = Critic(ck["dz"], ck["kind"], ck["cfg"]["obj"] if "obj" in ck["cfg"] else "q")
        m.load_state_dict(ck["state_dict"])
        m.eval()
        s_in = score_rows(m, ctx, Zb, None, rows_br=br_t).numpy()
        smax, smu, ssd = np.zeros(len(brr)), np.zeros(len(brr)), np.zeros(len(brr))
        for g, ii in groups.items():
            smax[ii], smu[ii], ssd[ii] = s_in[ii].max(), s_in[ii].mean(), s_in[ii].std() + 1e-6
        # ascent trajectory
        a = A0.clone()
        zs, wins, clips, l2 = [], [], [], []
        for step in range(21):
            s = score_rows(m, ctx, Zb, None, rows_br=br_t, A=a).numpy()
            zs.append((s - smu) / ssd)
            wins.append(float((s > smax).mean()))
            clips.append(float((a.abs() > 1.0).float().mean()))
            l2.append(float((a - A0).flatten(1).norm(dim=1).mean()))
            if step == 20:
                break
            _, ga, _ = score_rows(m, ctx, Zb, None, rows_br=br_t, A=a, grad=True)
            a = a + 0.05 * ga / (ga.flatten(1).norm(dim=1)[:, None, None] + 1e-12)
        zs = np.stack(zs)  # [21, rows]
        traj_all.append(zs)
        # gaussian perturbation sweep
        gen = torch.Generator().manual_seed(7)
        sweep = {}
        for sig in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            ws = []
            for _ in range(4):
                Ap = A0 + sig * asd * torch.randn(A0.shape, generator=gen)
                so = score_rows(m, ctx, Zb, None, rows_br=br_t, A=Ap).numpy()
                ws.append(float((so > smax).mean()))
            sweep[str(sig)] = float(np.mean(ws))
        # within-state: does the critic's top in-dist candidate succeed more than average? (single continuation)
        pick = [ii[np.argmax(s_in[ii])] for ii in groups.values()]
        pg = float(np.mean([y[p] for p in pick]) - np.mean([y[ii].mean() for ii in groups.values()]))
        res["seeds"][seed] = {"ascent_mean_z": zs.mean(1).tolist(), "ascent_win": wins, "ascent_clip_frac": clips,
                              "ascent_l2_mean": l2, "gauss_win": sweep, "pick_gain_1c_recomputed": pg,
                              "in_dist_max_z_mean": float(np.mean([((s_in[ii].max() - s_in[ii].mean()) /
                                                                    (s_in[ii].std() + 1e-6)) for ii in groups.values()]))}
        print(seed, "final z", zs[-1].mean(), "win", wins[-1], "pick_gain", pg)
    T = np.stack(traj_all)  # [seed, 21, rows]
    # example states: those with mixed outcomes, pick 6 deterministically (first 6 in state order)
    mixed = [g for g, ii in groups.items() if 0 < y[ii].mean() < 1][:6]
    ex = {}
    for g in mixed:
        ii = groups[g]
        ex[str(g)] = {"y": y[ii].tolist(), "z_traj_seed0": T[0][:, ii].tolist()}
    res["examples"] = ex
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/hacking_case.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/hacking_case.npz", traj=T.astype(np.float32), state=st, y=y)
    print("wrote", f"{OUT}/hacking_case.json")


if __name__ == "__main__":
    main()
