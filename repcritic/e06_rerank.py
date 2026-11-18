"""E06 (T020) closed-loop action reranking, paired on identical initial states (episode_id).

Pre-registration: repcritic/reports/t020_prereg.md (committed before any closed-loop run).

Conditions (receding horizon: every decision re-samples, executes the first K=5 of the chosen H=8 chunk, rest discarded):
  B0        : frozen pi0.5, N=1 (noise = candidate 0 of the shared noise stream)
  Random8   : N=8, uniform random pick (separate seeded RNG)
  ActOnly8  : N=8, argmax T019 action-only ridge (state-free)
  Critic8   : N=8, argmax T019 Geo_q_buf+br (3-seed ensemble)
  CriticApp8: N=8, Critic8 pick only when the C-Phase probe says "approach", else candidate 0 (= frozen sample)
Noise: per batch one CPU generator; each decision draws [n, 8, H, 32] for every condition, so the d=0 candidate sets are
identical across N=8 conditions and B0 executes candidate 0.

  CUDA_VISIBLE_DEVICES=0 RC_POLICY_MB=40 python -m repcritic.e06_rerank --part 0 --nparts 2
  CUDA_VISIBLE_DEVICES=0 python -m repcritic.e06_rerank --bench
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from repcritic import rc_common as C

C.setup_env_vars()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from repcritic import e02_common as E  # noqa: E402
from repcritic.e06_scorers import ActionOnly, CriticScorer, GeoZ, PhaseGate  # noqa: E402

OUT = Path(os.environ.get("RC_E06_OUT", "/workspace/stochrl_t020"))
EP_SEED = 20261119           # T017 permutation; T017 used [:500]
EP_OFFSET = 500              # T020 test episodes = perm[500:], minus T016 ids -> never used for training/selection
T016_SEED, T016_N = 12345, 30
NOISE_BASE = 20200
N_CAND = 8
N_DEC = C.MAX_STEPS // C.K
CONDS = ["B0", "Random8", "ActOnly8", "Critic8", "CriticApp8"]


def test_episodes(total, n):
    g = torch.Generator().manual_seed(EP_SEED)
    perm = torch.randperm(total, generator=g).tolist()
    used = set(perm[:EP_OFFSET])
    g2 = torch.Generator().manual_seed(T016_SEED)
    used |= set(torch.randperm(total, generator=g2)[:T016_N].tolist())
    out = [e for e in perm[EP_OFFSET:EP_OFFSET + n + 64] if e not in used][:n]
    assert len(out) == n
    man = json.load(open("/workspace/stochrl_t017/split_manifest.json"))
    assert not (set(out) & {r["eid"] for r in man["episodes"]})
    return out


def obs_rep(obs, N):
    out = {}
    for k, v in obs.items():
        if k == "task_descriptions":
            out[k] = [x for x in v for _ in range(N)]
        elif v is None:
            out[k] = None
        else:
            out[k] = v.repeat_interleave(N, dim=0)
    return out


def sync():
    torch.cuda.synchronize()


def run_cond(env, pol, S, eids, cond, noise_seed, pick_seed):
    n = len(eids)
    obs, _ = env.reset(seed=0, options={"episode_id": torch.tensor(eids).to(env.device)})
    instr = list(obs["task_descriptions"])
    gen = torch.Generator().manual_seed(noise_seed)
    rng = np.random.RandomState(pick_seed)
    succ = np.zeros((n, C.MAX_STEPS), bool)
    lat_pol, lat_score, lat_tot = [], [], []
    dec = dict(sel=np.zeros((n, N_DEC), np.int8), gate=np.zeros((n, N_DEC), bool),
               phase=np.zeros((n, N_DEC), "U16"), div=np.zeros((n, N_DEC), np.float32),
               crit=np.full((n, N_DEC, N_CAND), np.nan, np.float32), ao=np.full((n, N_DEC, N_CAND), np.nan, np.float32),
               sel_eq_c0=np.zeros((n, N_DEC), bool), clip_frac=np.zeros((n, N_DEC), np.float32))
    lo = torch.as_tensor(env.env.action_space.low[0]).float()
    hi = torch.as_tensor(env.env.action_space.high[0]).float()
    t = 0
    for d in range(N_DEC):
        dec["phase"][:, d] = E.phase_from_priv(E.privileged(env)).astype("U16")  # label-only (analysis)
        nz = pol.noise(n * N_CAND, gen).view(n, N_CAND, *pol.noise_shape)
        sync(); t0 = C.now()
        if cond == "B0":
            acts, macts = pol.act(obs, nz[:, 0])
            sync(); t1 = C.now()
            chosen = acts[:, : C.K]
            sel = np.zeros(n, int)
        else:
            orp = obs_rep(obs, N_CAND)
            nzf = nz.reshape(n * N_CAND, *pol.noise_shape)
            need_feat = cond.startswith("Critic")
            if need_feat:
                acts, macts, F = pol.act_feats(orp, nzf, full_h=True)
            else:
                acts, macts = pol.act(orp, nzf, full_h=True)
            sync(); t1 = C.now()
            A = macts[..., :7].float().view(n, N_CAND, C.H, 7)
            if cond == "Random8":
                sel = rng.randint(0, N_CAND, size=n)
            elif cond == "ActOnly8":
                s = S["ao"](A)
                dec["ao"][:, d] = s.cpu().numpy()
                sel = s.argmax(1).cpu().numpy()
            else:
                hg = F["vlm9_img_grid4"][::N_CAND]
                ht = F["vlm9_txt_mean"][::N_CAND]
                z = S["geo"](hg)
                ens, _ = S["critic"](z, A)
                dec["crit"][:, d] = ens.cpu().numpy()
                sel = ens.argmax(1).cpu().numpy()
                if cond == "CriticApp8":
                    g = S["gate"](ht)
                    dec["gate"][:, d] = g
                    sel = np.where(g, sel, 0)
            sync()
            Af = A.reshape(n, N_CAND, -1)
            dec["div"][:, d] = torch.cdist(Af, Af).mean((1, 2)).cpu().numpy() * N_CAND / (N_CAND - 1)
            chosen = acts.view(n, N_CAND, C.H, 7)[torch.arange(n), torch.as_tensor(sel)][:, : C.K]
        t2 = C.now()
        lat_pol.append((t1 - t0) * 1000); lat_score.append((t2 - t1) * 1000); lat_tot.append((t2 - t0) * 1000)
        dec["sel"][:, d] = sel
        dec["sel_eq_c0"][:, d] = sel == 0
        ch = chosen.cpu()
        dec["clip_frac"][:, d] = ((ch < lo) | (ch > hi)).float().mean((1, 2)).numpy()
        for k in range(C.K):
            obs, r, te, tr, info = env.step(chosen[:, k], auto_reset=False)
            succ[:, t] = info["success"].cpu().numpy()
            t += 1
    s_once, s_end = succ.any(1), succ[:, -1]
    first = np.where(s_once, succ.argmax(1) + 1, C.MAX_STEPS)
    return dict(s_once=s_once, s_end=s_end, steps=first, instr=instr, dec=dec,
                lat=dict(pol_ms=float(np.mean(lat_pol)), score_ms=float(np.mean(lat_score)),
                         tot_ms=float(np.mean(lat_tot))))


def scorers(dev):
    return dict(geo=GeoZ(dev), gate=PhaseGate(dev), critic=CriticScorer(dev), ao=ActionOnly(dev))


def bench(pol, S, env, reps=10):
    """batch-1 single-env latency: N=1 plain vs N=8 (+features+critic)."""
    obs, _ = env.reset(seed=0, options={"episode_id": torch.tensor([0] * env.env.unwrapped.num_envs).to(env.device)})
    o1 = C.obs_select(obs, [0])
    gen = torch.Generator().manual_seed(0)
    res = {}
    for name in ("B0", "Random8", "ActOnly8", "Critic8"):
        ts = []
        for r in range(reps + 2):
            nz = pol.noise(N_CAND, gen)
            sync(); t0 = C.now()
            if name == "B0":
                pol.act(o1, nz[:1])
            else:
                orp = obs_rep(o1, N_CAND)
                if name == "Critic8":
                    _, macts, F = pol.act_feats(orp, nz, full_h=True)
                    z = S["geo"](F["vlm9_img_grid4"][:1])
                    S["critic"](z, macts[..., :7].float().view(1, N_CAND, C.H, 7))
                else:
                    _, macts = pol.act(orp, nz, full_h=True)
                    if name == "ActOnly8":
                        S["ao"](macts[..., :7].float().view(1, N_CAND, C.H, 7))
            sync()
            if r >= 2:
                ts.append((C.now() - t0) * 1000)
        res[name] = dict(mean_ms=float(np.mean(ts)), sd_ms=float(np.std(ts)))
        print(name, res[name], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=120)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=2)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "decisions").mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    pol = E.FeatPolicy(C.build_policy())
    pol.ctx_pass = False  # AE ctx pass not needed (Geo/Phase are VLM-prefix features)
    S = scorers(dev)
    if args.bench:
        env = C.build_env(1)
        res = bench(pol, S, env)
        res["gpu"] = os.environ["CUDA_VISIBLE_DEVICES"]
        (OUT / "bench_latency.json").write_text(json.dumps(res, indent=1))
        return
    env = C.build_env(args.batch)
    total = env.env.unwrapped.total_num_trials
    eps = test_episodes(total, args.n_total)
    per = len(eps) // args.nparts
    mine = eps[args.part * per:(args.part + 1) * per]
    if args.smoke:  # smoke on T017 TRAIN episodes (never on test episodes)
        man = json.load(open("/workspace/stochrl_t017/split_manifest.json"))
        mine = [r["eid"] for r in man["episodes"] if r["split"] == "train"][: args.batch]
    tag = "smoke" if args.smoke else f"p{args.part}"
    T0 = time.time()
    with open(OUT / f"episodes_{tag}.jsonl", "a") as fo:
        for bi in range(0, len(mine), args.batch):
            batch = mine[bi: bi + args.batch]
            assert len(batch) == args.batch
            gb = (args.part * per + bi) // args.batch  # global batch index -> noise stream
            for cond in args.conds:
                t0 = time.time()
                r = run_cond(env, pol, S, batch, cond, NOISE_BASE + gb, 777 + gb)
                np.savez_compressed(OUT / "decisions" / f"{tag}_b{gb}_{cond}.npz", eids=np.array(batch), **r["dec"])
                for i, e in enumerate(batch):
                    fo.write(json.dumps(dict(eid=int(e), batch=gb, cond=cond, instr=r["instr"][i],
                                             success_once=bool(r["s_once"][i]), success_at_end=bool(r["s_end"][i]),
                                             steps=int(r["steps"][i]), n_batch=len(batch), **r["lat"],
                                             frac_sel_c0=float(r["dec"]["sel_eq_c0"][i].mean()),
                                             frac_gate=float(r["dec"]["gate"][i].mean()))) + "\n")
                fo.flush()
                print(f"[{tag}] batch {gb} {cond:10s} once {r['s_once'].mean():.3f} end {r['s_end'].mean():.3f} "
                      f"| {time.time()-t0:.0f}s lat {json.dumps({k: round(v) for k, v in r['lat'].items()})} "
                      f"peak {C.gpu_mem_gb():.1f}GB | elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    print("done", (time.time() - T0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()
