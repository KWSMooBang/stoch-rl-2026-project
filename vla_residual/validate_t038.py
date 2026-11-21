"""T038 checks: committed residual (commit_k) in the env and grpo_commit in the GRPO loss.

  --mode cpu  (no GPU, RLinf functions only)
    C1 grpo_commit(k=1) through RLinf's calculate_adv_and_returns == adv_type grpo, bit for bit (advantages,
       loss_mask, loss_mask_sum), on a worker-like rollout batch (loss_mask from compute_loss_mask(dones)).
    C2 grpo_commit(k=3): loss_mask cleared exactly at t % 3 != 0 (in rollout_batch, in place), loss_mask_sum =
       sample decisions per episode, advantages on sample rows == grpo, episode scores unchanged.
    C3 RLinf actor loss (compute_ppo_actor_loss, masked_mean_ratio): d loss / d logprob is 0 at reuse decisions,
       non-zero at sample decisions with |A|>0; per-episode total weight (sum over steps of mask * 16 / n_s) = 16,
       the same as E2's (sum mask * 16 / L).
    C4 Hydra composition of run_t038.sh overrides: k=1 vs E2 (t022-grpo-s0 overrides) differ only in commit_k /
       adv_type / advantage_mode; k=3 adds obs_dim 2093.
  --mode gpu  (physical GPU 0; one shared ManiSkill scene, no Ray)
    G1 k=3 train (group 8, success terminates) and eval (held-out, ignore_terminations), random u every decision:
       u_exec = policy u at dec % 3 == 0, else bitwise the u_exec of the previous decision; every executed sim
       action == clip(a_base[k] + alpha * tanh(u_exec), lo, hi) bitwise; states dim 2093, last column = c_rem / 3.
    G2 k=1: u_exec == policy u bitwise at every decision, executed actions == E2 formula, states dim 2092 and
       == [feat | qpos | a_base / scale | t/T] bitwise (E2 obs).
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t038 --mode gpu --out /outputs/StochRL/t038/val
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

WT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- CPU
def worker_batch(n_steps=16, bsz=64, group=8, seed=0, p_succ=0.06):
    """Rollout batch as EmbodiedFSDPActor._process_received_rollout_batch leaves it (num_action_chunks 1)."""
    from rlinf.utils.metric_utils import compute_loss_mask

    g = torch.Generator().manual_seed(seed)
    succ = torch.rand(n_steps, bsz, generator=g) < p_succ
    first = succ & (succ.long().cumsum(0) == 1)
    rewards = first.float()[..., None]  # +1 at the first success
    dones = torch.zeros(n_steps + 1, bsz, 1, dtype=torch.bool)
    dones[1:, :, 0] = first
    dones[-1] = True  # truncation at T for episodes that did not succeed
    dones[-1, :, 0] |= True
    loss_mask, loss_mask_sum = compute_loss_mask(dones)
    return dict(rewards=rewards, dones=dones, loss_mask=loss_mask, loss_mask_sum=loss_mask_sum)


def run_adv(rb, adv_type, mode=None, group=8):
    """Mirror EmbodiedFSDPActor.compute_advantages_and_returns (kwargs dict -> rollout_batch update)."""
    from rlinf.algorithms.registry import calculate_adv_and_returns

    kwargs = dict(task_type="embodied", adv_type=adv_type, rewards=rb["rewards"], dones=rb["dones"], values=None,
                  prev_logprobs=None, teacher_logprobs=None, num_action_chunks=1, gamma=0.96, gae_lambda=0.95,
                  group_size=group, reward_type="action_level", loss_mask=rb["loss_mask"],
                  loss_mask_sum=rb["loss_mask_sum"], advantage_mode=mode)
    out = calculate_adv_and_returns(**kwargs)
    rb = dict(rb)
    rb.update(out)
    rb["loss_mask"] = kwargs["loss_mask"]
    rb["loss_mask_sum"] = kwargs["loss_mask_sum"]
    return rb


def episode_scores(rb):
    r = rb["rewards"][..., 0]
    m0 = (rb["dones"][:-1, :, 0].long().cumsum(0) == 0)  # before (incl.) the first done step
    return (r * m0).sum(0)


def cpu_checks():
    from vla_residual import rlinf_ext

    rlinf_ext._patch_commit()
    res = {}
    # C1
    a = worker_batch(seed=1)
    b = {k: v.clone() for k, v in a.items()}
    b["loss_mask_sum"] = b["loss_mask"].sum(dim=(0, 2), keepdim=True).expand_as(b["loss_mask"])  # same construction
    ra = run_adv(a, "grpo")
    rb = run_adv(b, "grpo_commit", "commit_k=1")
    res["C1_k1_equals_grpo"] = dict(
        advantages=bool(torch.equal(ra["advantages"], rb["advantages"])),
        loss_mask=bool(torch.equal(ra["loss_mask"], rb["loss_mask"])),
        loss_mask_sum=bool(torch.equal(ra["loss_mask_sum"], rb["loss_mask_sum"])),
        n_nonzero_adv=int((ra["advantages"] != 0).sum()))
    # C2
    a = worker_batch(seed=2)
    lm_ref = a["loss_mask"].clone()
    lm_obj = a["loss_mask"]
    ra = run_adv({k: (v.clone() if k != "loss_mask_sum" else v.clone()) for k, v in a.items()}, "grpo")
    rc = run_adv(a, "grpo_commit", "commit_k=3")
    t = torch.arange(16)
    reuse = (t % 3) != 0
    lm = rc["loss_mask"][..., 0]
    exp_lm = lm_ref[..., 0] & ~reuse[:, None]
    n_s = exp_lm.sum(0)
    res["C2_k3"] = dict(
        rollout_batch_mask_edited_in_place=bool(torch.equal(lm_obj[..., 0], exp_lm)),
        mask_equals_expected=bool(torch.equal(lm, exp_lm)),
        reuse_rows_all_false=bool((~lm[reuse]).all()),
        sample_rows_unchanged=bool(torch.equal(lm[~reuse], lm_ref[~reuse, :, 0])),
        loss_mask_sum_equals_n_sample=bool(torch.equal(rc["loss_mask_sum"][..., 0], n_s[None].expand(16, -1))),
        n_sample_min_max=[int(n_s.min()), int(n_s.max())],
        adv_sample_rows_equal_grpo=bool(torch.equal(rc["advantages"][~reuse], ra["advantages"][~reuse])),
        adv_reuse_rows_zero=bool((rc["advantages"][reuse] == 0).all()),
        episode_scores_unchanged=bool(torch.equal(episode_scores(rc), episode_scores(ra))))
    # C3: real RLinf actor loss
    from rlinf.algorithms.losses import compute_ppo_actor_loss
    from rlinf.utils.utils import masked_mean

    def grad_of(rbx):
        n, B = 16, rbx["advantages"].shape[1]
        lp = torch.zeros(n, B, 7, requires_grad=True)
        old = torch.zeros(n, B, 7)
        adv = rbx["advantages"].reshape(n, B, 1).expand(n, B, 7).float()
        mask = rbx["loss_mask"].reshape(n, B, 1).expand(n, B, 7)
        lms = rbx["loss_mask_sum"].reshape(n, B, 1).expand(n, B, 7)
        loss, _ = compute_ppo_actor_loss(logprobs=lp, old_logprobs=old, advantages=adv, clip_ratio_low=0.2,
                                         clip_ratio_high=0.2, clip_ratio_c=3.0, loss_mask=mask, loss_mask_sum=lms,
                                         max_episode_steps=16, loss_agg_func=masked_mean)
        loss.backward()
        return lp.grad[..., 0]

    g3 = grad_of(rc)
    g1 = grad_of(ra)
    nz = rc["advantages"][..., 0] != 0
    w3 = (rc["loss_mask"][..., 0].float() * 16 / rc["loss_mask_sum"][..., 0].float()).sum(0)
    w1 = (ra["loss_mask"][..., 0].float() * 16 / ra["loss_mask_sum"][..., 0].float()).sum(0)
    # gradient magnitude per sample step relative to E2: |g3| / |g1| = L / n_s
    ratio = (g3[~reuse].abs() / g1[~reuse].abs().clamp(min=1e-30))[nz[~reuse]]
    res["C3_actor_loss"] = dict(
        grad_zero_at_reuse=bool((g3[reuse] == 0).all()),
        grad_nonzero_at_sample_with_adv=bool((g3[nz] != 0).all()),
        episode_weight_k3=[float(w3.min()), float(w3.max())], episode_weight_e2=[float(w1.min()), float(w1.max())],
        grad_ratio_sample_vs_e2_min_max=[float(ratio.min()), float(ratio.max())] if len(ratio) else None)
    # C4: Hydra composition
    res["C4_config"] = config_diff()
    ok = (all(res["C1_k1_equals_grpo"][k] for k in ("advantages", "loss_mask", "loss_mask_sum"))
          and all(v for k, v in res["C2_k3"].items() if isinstance(v, bool))
          and res["C3_actor_loss"]["grad_zero_at_reuse"] and res["C3_actor_loss"]["grad_nonzero_at_sample_with_adv"]
          and abs(res["C3_actor_loss"]["episode_weight_k3"][0] - 16) < 1e-4
          and abs(res["C3_actor_loss"]["episode_weight_k3"][1] - 16) < 1e-4
          and res["C4_config"]["ok"])
    res["pass"] = bool(ok)
    return res


def compose(overrides):
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    os.environ.setdefault("EMBODIED_PATH", str(WT / "third_party/RLinf/examples/embodiment"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(WT / "vla_residual/configs"), version_base=None):
        cfg = hcompose(config_name="t022_grpo", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=False)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def config_diff():
    common = ["~cluster.component_placement", "+cluster.component_placement={actor:0,env:0,rollout:0}",
              "actor.seed=0", "env.train.seed=0", "env.eval.seed=1000"]

    def t038(k):
        return common + [f"env.train.vla_residual.commit_k={k}", f"env.eval.vla_residual.commit_k={k}",
                         f"actor.model.obs_dim={2093 if k > 1 else 2092}", "algorithm.adv_type=grpo_commit",
                         f"+algorithm.advantage_mode='commit_k={k}'"]

    e2 = flat(compose(common))
    out = {}
    for k in (1, 3):
        c = flat(compose(t038(k)))
        out[f"k{k}_vs_e2"] = {key: [e2.get(key), c.get(key)] for key in sorted(set(e2) | set(c))
                              if e2.get(key) != c.get(key)}
    allowed1 = {"algorithm.adv_type", "algorithm.advantage_mode"}
    allowed3 = allowed1 | {"env.train.vla_residual.commit_k", "env.eval.vla_residual.commit_k", "actor.model.obs_dim"}
    out["ok"] = bool(set(out["k1_vs_e2"]) <= allowed1 and set(out["k3_vs_e2"]) <= allowed3
                     and out["k3_vs_e2"].get("env.train.vla_residual.commit_k") == [1, 3])
    return out


# ---------------------------------------------------------------- GPU
def gpu_rollout(split, k, batch, seed=0, n_dec=16):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    kw = dict(group_size=8, auto_reset=False, ignore_term=False) if split == "train" else dict(ignore_term=True)
    cfg = make_cfg(split, n_eval=batch, commit_k=k, **kw)
    cfg.seed = seed
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    executed = []
    orig = env._sim_step

    def rec(a, render):
        executed.append(a.clone())
        return orig(a, render)

    env._sim_step = rec
    g = torch.Generator().manual_seed(4242 + k)
    obs, _ = env.reset()
    rows, prev_exec = [], None
    S = [obs["states"].clone()]
    feats_ok = []
    for d in range(n_dec):
        st = S[-1]
        e2_obs = torch.cat([env.feat, env.qpos, (env.a_base / V.BASE_SCALE.to(env.device)).reshape(batch, -1),
                            (env.dec.float() / env.max_dec)[:, None]], 1)
        feats_ok.append(bool(torch.equal(st[:, :V.OBS_DIM], e2_obs)))
        crem_exp = torch.minimum(k - env.dec % k, (env.max_dec - env.dec).clamp(min=1)).float() / k
        crem_ok = bool(torch.equal(st[:, -1], crem_exp)) if env.commit_obs else None
        a_base = env.a_base.clone()
        u = (0.7 * torch.randn(batch, 1, V.ACT_DIM, generator=g)).to(env.device)
        executed.clear()
        o, r, te, tr, inf = env.chunk_step(u)
        lc = env.last_commit
        sample = lc["sample"]
        ue = lc["u_exec"]
        exp_exec = torch.where(sample[:, None], u[:, 0], prev_exec if prev_exec is not None else u[:, 0])
        acts_ok = all(torch.equal(executed[j], (a_base[:, j] + env.alpha * torch.tanh(ue)).clamp(env.lo, env.hi))
                      for j in range(len(executed)))
        rows.append(dict(dec=d, n_sample=int(sample.sum()), sample_expected=bool((sample == (d % k == 0)).all()),
                         u_exec_rule=bool(torch.equal(ue, exp_exec)),
                         reuse_equals_prev=(bool(torch.equal(ue[~sample], prev_exec[~sample]))
                                            if prev_exec is not None and bool((~sample).any()) else None),
                         sample_equals_policy=bool(torch.equal(ue[sample], u[sample, 0])),
                         executed_actions_formula=bool(acts_ok), n_sim_steps=len(executed),
                         obs_dim=int(st.shape[1]), e2_part_bitwise=feats_ok[-1], crem_col=crem_ok,
                         policy_u_differs_at_reuse=(bool((u[~sample, 0] != ue[~sample]).any())
                                                    if bool((~sample).any()) else None)))
        prev_exec = ue.clone()
        S.append(o[0]["states"].clone())
    return dict(split=split, k=k, batch=batch, succ_once=float(env.success_once.float().mean()), rows=rows)


def gpu_checks(batch):
    res = {}
    t0 = time.time()
    for split in ("train", "eval"):
        for k in (3, 1):
            r = gpu_rollout(split, k, batch)
            rows = r["rows"]
            ok = all(x["sample_expected"] and x["u_exec_rule"] and x["sample_equals_policy"]
                     and x["executed_actions_formula"] and x["e2_part_bitwise"] and x["n_sim_steps"] == 5
                     and x["reuse_equals_prev"] is not False and x["crem_col"] is not False for x in rows)
            ok &= all(x["obs_dim"] == (2093 if k > 1 else 2092) for x in rows)
            if k == 1:
                ok &= all(x["n_sample"] == batch for x in rows)
            else:
                ok &= all(x["reuse_equals_prev"] for x in rows if x["dec"] % k)
            r["pass"] = bool(ok)
            res[f"{split}_k{k}"] = r
            print(f"[t038-val] {split} k={k} pass={ok} succ_once={r['succ_once']:.3f} {time.time()-t0:.0f}s", flush=True)
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = cpu_checks() if a.mode == "cpu" else gpu_checks(a.batch)
    with open(os.path.join(a.out, f"{a.mode}.json"), "w") as f:
        json.dump(res, f, indent=1)
    summ = {k: (v if not isinstance(v, dict) else v.get("pass", v)) for k, v in res.items()}
    print(json.dumps(summ if a.mode == "gpu" else res, indent=1, default=str))
    print(f"[t038-val] {a.mode} PASS={res['pass']}")
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
