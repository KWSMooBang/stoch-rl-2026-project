"""T036: single-process (no Ray) checks and evaluation-only rollouts for the competence gate. Physical GPU 0 only.

  --mode cpu    G2 gate head: same 7-D MLP as E2, z-mean == G1 logit at init, gradient / Adam step size (no GPU)
  --mode unit   executed action == clip(a_base + alpha * g * tanh(u)) bitwise for gate none / one / fixed / learned,
                g=1 path == E2 path bitwise, obs layout, order-swapped none-vs-one rollouts (train split, random u)
  --mode b0     held-out B0 (residual 0) rollout, 128 episodes: V_hat / g per decision with privileged success
                labels (analysis only) -> where would the G1 gate be open / closed
  --mode diag   diagnostic control: stored E2 checkpoints (t022-grpo-s0, t030-grpo-s1, t030-grpo-s2) evaluated with
                the G1 gate and, in the same process, without it (no training)

Evaluation settings are those of the RLinf held-out eval of the E2 runs (as in T034's eval_t034.py): 4 blocks x 32
slots, env seed 1000, ignore_terminations, 16 decisions, u = actor mean, pi0.5 noise per (episode id, decision).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("RC_POLICY_MB", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# (tau, temp) per gate predictor: vla_residual/gate.py select() on the T017 train split (gate_select*.json)
GATES = {"t023": (0.37, 0.1001), "end": (0.41, 0.1047)}
TAU, TEMP = GATES["end"]  # predictor used for training (L1 decision 2026-11-21)
E2 = {"e2-s0": "t022-grpo-s0", "e2-s1": "t030-grpo-s1", "e2-s2": "t030-grpo-s2"}
RUN_ROOT = Path("/outputs/StochRL/runs")
EVAL_SEED, BATCH = 1000, 32


# ------------------------------------------------------------------ cpu
def mode_cpu():
    from omegaconf import OmegaConf

    import rlinf.models.embodiment.mlp_policy as mp
    from vla_residual import rlinf_ext
    from vla_residual.gate import g_fixed

    rlinf_ext._patch_mlp()
    base = dict(model_type="mlp_policy", obs_dim=2092, action_dim=7, num_action_chunks=1, add_value_head=False,
                vr_init=dict(zero_last=False, init_logstd=-1.5))
    S = 100.0
    gcfg = dict(base, obs_dim=2093, action_dim=8, vr_gate=dict(tau=TAU, temp=TEMP, scale=S, init_logstd=0.0))
    torch.manual_seed(0); e2 = mp.get_model(OmegaConf.create(base))
    torch.manual_seed(0); g2 = mp.get_model(OmegaConf.create(gcfg))
    st = torch.randn(64, 2093)
    st[:, 2092] = torch.rand(64)  # V_hat
    st[:, 2091] = torch.randint(0, 16, (64,)).float() / 16
    m_e2, ls_e2 = e2._sample_actions(st[:, :2092])
    m_g2, ls_g2 = g2._sample_actions(st)
    out = dict(
        u_mean_bitwise_equal_e2=bool(torch.equal(m_e2, m_g2[:, :7])),
        z_mean_vs_g1_logit_max_abs=float((m_g2[:, 7] - (TAU - st[:, 2092]) / TEMP).abs().max()),
        g_init_vs_g1_max_abs=float((torch.sigmoid(m_g2[:, 7]) - g_fixed(st[:, 2092], TAU, TEMP)).abs().max()),
        logstd=[round(float(x), 3) for x in ls_g2[0]],
        n_params_e2=sum(p.numel() for p in e2.parameters()), n_params_g2=sum(p.numel() for p in g2.parameters()),
        state_dict_keys_g2=list(g2.state_dict().keys()))
    # eval mode = mean; log-prob over 8 dims through the stock default_forward
    ca, res = g2.predict_action_batch({"states": st}, mode="eval")
    out["eval_action_is_mean"] = bool(torch.equal(ca[:, 0], m_g2))
    fo = g2.default_forward(dict(states=st, action=ca[:, 0]), compute_values=False)
    out["logprob_shape"] = list(fo["logprobs"].shape)
    # gradient: d z / d p = (V_hat, t/16, 1) (not multiplied by S); one Adam step moves the coefficients by S * lr
    g2.zero_grad()
    g2._sample_actions(st)[0][:, 7].sum().backward()
    gp = g2.actor_mean.p.grad
    exp = torch.stack([st[:, 2092].sum(), st[:, 2091].sum(), torch.tensor(64.0)])
    out["grad_p_rel_err"] = float(((gp - exp).abs() / exp.abs()).max())
    c0 = g2.actor_mean.coef().detach().clone()
    opt = torch.optim.Adam(g2.parameters(), lr=1e-4)
    opt.step()
    out["adam_step_coef_delta"] = [float(x) for x in (g2.actor_mean.coef().detach() - c0)]
    out["coef_init"] = [float(x) for x in c0]
    # gate algebra used by the env: g = 1 multiplies exactly
    a, s = torch.rand(7), torch.tanh(torch.randn(4096, 7) * 3)
    out["g1_alpha_bitwise"] = bool(torch.equal(a * (torch.ones(4096)[:, None] * s), a * s))
    out["pass"] = bool(out["u_mean_bitwise_equal_e2"] and out["g_init_vs_g1_max_abs"] < 1e-5 and out["eval_action_is_mean"]
                       and out["logprob_shape"] == [64, 8] and out["grad_p_rel_err"] < 1e-5 and out["g1_alpha_bitwise"]
                       and all(abs(abs(d) - S * 1e-4) < 1e-4 for d in out["adam_step_coef_delta"])
                       and out["n_params_g2"] == out["n_params_e2"] + 4)
    return out


# ------------------------------------------------------------------ env helpers
def make_env(split, gate, batch=BATCH, seed=EVAL_SEED, group_size=1):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    extra = {} if gate is None else dict(gate=gate)
    if split == "eval":
        cfg = make_cfg("eval", n_eval=V.N_EVAL_DEFAULT, auto_reset=False, ignore_term=True, **extra)
    else:
        cfg = make_cfg("train", auto_reset=False, ignore_term=False, group_size=group_size, **extra)
    cfg.seed = seed
    return V.ResidualVLAEnv(cfg, batch, 0, 1, None)


def hook_steps(env):
    steps = []
    orig = env._sim_step

    def sim_step(a, render):
        out = orig(a, render)
        steps.append(a.detach().clone())
        return out

    env._sim_step = sim_step
    return steps


def gate_of(mode, vhat="end"):
    return None if mode == "none" else dict(mode=mode, vhat=vhat, tau=GATES[vhat][0], temp=GATES[vhat][1])


# ------------------------------------------------------------------ unit
@torch.no_grad()
def unit_rollout(mode, batch, n_dec, seed, u_seed):
    """Train split, random u (same stream for every mode). Returns per-decision obs / base chunk and the bitwise
    checks of the executed actions against the formula evaluated on the env's own pre-step state."""
    from vla_residual import core as V

    env = make_env("train", gate_of(mode), batch=batch, seed=seed, group_size=8)
    steps = hook_steps(env)
    g = torch.Generator().manual_seed(u_seed)
    obs, _ = env.reset()
    A = 8 if mode == "learned" else 7
    S, AB, chk, gs = [obs["states"].cpu()], [env.a_base.cpu().clone()], [], []
    for d in range(n_dec):
        u = (torch.randn(batch, A, generator=g) * 0.7).to(env.device)
        a_base, alpha = env.a_base.clone(), env.alpha
        v = env.gpot(feat_raw=env.feat_raw) if env.gate is not None else None
        st = obs["states"].to(env.device)
        e2_exec = torch.stack([(a_base[:, k] + alpha * torch.tanh(u[:, :7])).clamp(env.lo, env.hi) for k in range(5)], 1)
        if mode in ("none", "one"):
            want, gg = e2_exec, torch.ones(batch, device=env.device)
        else:
            gg = torch.sigmoid(u[:, 7]) if mode == "learned" else torch.sigmoid((TAU - v) / TEMP)
            want = torch.stack([(a_base[:, k] + alpha * (gg[:, None] * torch.tanh(u[:, :7]))).clamp(env.lo, env.hi)
                                for k in range(5)], 1)
        c = dict(obs_dim=st.shape[1])
        if mode == "learned":
            c["vhat_col_equal"] = bool(torch.equal(st[:, V.OBS_DIM], v))
            c["t_col"] = bool(torch.equal(st[:, V.OBS_DIM - 1], env.dec.float() / env.max_dec))
            c["e2_block_equal"] = bool(torch.equal(st[:, : V.OBS_DIM], torch.cat(
                [env.feat, env.qpos, (a_base / V.BASE_SCALE.to(env.device)).reshape(batch, -1),
                 (env.dec.float() / env.max_dec)[:, None]], 1).float()))
        steps.clear()
        o, r, te, tr, inf = env.chunk_step(u[:, None])
        obs = o[0]
        ex = torch.stack(steps, 1)
        c["exec_bitwise"] = bool(torch.equal(ex, want))
        c["exec_ne_e2_frac"] = float((ex != e2_exec).flatten(1).any(1).float().mean())
        if env.gate is not None:
            c["g_logged_equal"] = bool(torch.equal(env.last_gate["g"], gg))
        chk.append(c); gs.append(gg.cpu())
        S.append(obs["states"].cpu()); AB.append(env.a_base.cpu().clone())
    G = torch.stack(gs)
    return dict(s=torch.stack(S), a=torch.stack(AB), chk=chk, eids=env.eids.clone(),
                g_min=float(G.min()), g_max=float(G.max()), g_mean=float(G.mean()),
                succ=float(env.success_once.float().mean()))


def mode_unit(batch, n_dec):
    from vla_residual import core as V

    R, t0 = {}, time.time()
    for name, mode in (("none", "none"), ("one", "one"), ("none_b", "none"), ("fixed", "fixed"), ("learned", "learned")):
        R[name] = unit_rollout(mode, batch, n_dec, seed=0, u_seed=123)
        print(f"[unit] {name} succ {R[name]['succ']:.3f} g {R[name]['g_mean']:.3f} {time.time()-t0:.0f}s", flush=True)

    def allk(name, k):
        return all(c[k] for c in R[name]["chk"])

    def eqfrac(x, y):  # per (decision, env): obs bitwise equal
        return float((R[x]["s"][..., : V.OBS_DIM] == R[y]["s"][..., : V.OBS_DIM]).all(-1).float().mean())

    out = dict(
        batch=batch, n_dec=n_dec,
        exec_bitwise={k: allk(k, "exec_bitwise") for k in R},
        one_exec_equals_e2_formula=allk("one", "exec_bitwise"),
        obs_dim={k: R[k]["chk"][0]["obs_dim"] for k in R},
        g_logged_equal={k: allk(k, "g_logged_equal") for k in ("one", "fixed", "learned")},
        g_range={k: [R[k]["g_min"], R[k]["g_max"], R[k]["g_mean"]] for k in R},
        fixed_exec_differs_from_e2_frac=float(np.mean([c["exec_ne_e2_frac"] for c in R["fixed"]["chk"]])),
        learned_obs={k: allk("learned", k) for k in ("vhat_col_equal", "t_col", "e2_block_equal")},
        # whole-rollout equality, order-swapped (PhysX / pi0.5 run-to-run floor = none vs none_b)
        obs_equal_frac_none_vs_one=eqfrac("none", "one"), obs_equal_frac_one_vs_none_b=eqfrac("one", "none_b"),
        obs_equal_frac_floor_none_vs_none_b=eqfrac("none", "none_b"),
        dec0_obs_equal_none_vs_one=bool(torch.equal(R["none"]["s"][0], R["one"]["s"][0])),
        eids_equal=bool(torch.equal(R["none"]["eids"], R["one"]["eids"]) and torch.equal(R["none"]["eids"], R["fixed"]["eids"])),
        success_once={k: R[k]["succ"] for k in R}, wall_s=time.time() - t0,
        peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    out["pass"] = bool(all(out["exec_bitwise"].values()) and all(out["g_logged_equal"].values())
                       and all(out["learned_obs"].values()) and out["dec0_obs_equal_none_vs_one"] and out["eids_equal"]
                       and out["obs_dim"]["learned"] == V.OBS_DIM + 1 and out["obs_dim"]["fixed"] == V.OBS_DIM
                       and out["g_range"]["one"][:2] == [1.0, 1.0]
                       and 0 <= out["g_range"]["fixed"][0] and out["g_range"]["fixed"][1] <= 1)
    return out


# ------------------------------------------------------------------ held-out rollouts
def ckpt_file(run_id):
    cks = sorted((RUN_ROOT / run_id / "rlinf_logs" / run_id / "checkpoints").glob("global_step_*"),
                 key=lambda p: int(p.name.split("_")[-1]))
    return cks[-1] / "actor/model_state_dict/full_weights.pt"


def load_actor(run_id, device):
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    from vla_residual import core as V

    m = MLPPolicy(V.OBS_DIM, V.ACT_DIM, num_action_chunks=1, add_value_head=False, add_q_head=False)
    m.load_state_dict(torch.load(ckpt_file(run_id), map_location="cpu"), strict=True)
    return m.to(device).float().eval()


@torch.no_grad()
def heldout(tag, run_id, gated, n_blocks, n_dec):
    """128 held-out episodes. gated in {"end", "t023"} -> env gate mode fixed with that predictor; None/False -> gate
    off (g := 1). Both predictors are recorded at every decision: v / g1 = T023 V_hat, v_end / g1_end = V_hat_end."""
    from vla_residual import core as V
    from vla_residual.gate import PRED_ROOT
    from vla_residual.potential import Potential

    env = make_env("eval", gate_of("fixed", gated) if gated else None)
    pot = Potential("vla", device=env.device)
    pot_end = Potential("vla", device=env.device, root=PRED_ROOT["end"])
    actor = load_actor(run_id, env.device) if run_id else None
    rows = []
    B = BATCH
    for b in range(n_blocks):
        obs, _ = env.reset() if b == 0 else env.reset(options={"env_idx": torch.arange(B)})
        eids = env.eids.clone().numpy()
        now = np.zeros(B, bool); ever = np.zeros(B, bool)
        for d in range(n_dec):
            st = obs["states"].to(env.device).float()
            u = actor._sample_actions(st)[0] if actor is not None else torch.zeros(B, V.ACT_DIM, device=env.device)
            v = pot(feat_raw=env.feat_raw).cpu().numpy()
            ve = pot_end(feat_raw=env.feat_raw).cpu().numpy()
            o, r, te, tr, inf = env.chunk_step(u[:, None])
            obs = o[0]
            g = env.last_gate["g"].cpu().numpy() if gated else np.ones(B, np.float32)
            if gated:
                assert np.array_equal(env.last_gate["v"].cpu().numpy(), ve if gated == "end" else v)
            now2 = inf[0]["success"].cpu().numpy().astype(bool)
            ever2 = inf[0]["episode"]["success_once"].cpu().numpy().astype(bool)
            sa = torch.tanh(u).abs().mean(1).cpu().numpy()
            for i in range(B):
                rows.append(dict(tag=tag, eid=int(eids[i]), dec=d, v=float(v[i]), v_end=float(ve[i]), g=float(g[i]),
                                 g1=float(1 / (1 + np.exp(-(GATES["t023"][0] - v[i]) / GATES["t023"][1]))),
                                 g1_end=float(1 / (1 + np.exp(-(GATES["end"][0] - ve[i]) / GATES["end"][1]))),
                                 now=bool(now[i]), ever=bool(ever[i]),
                    now_after=bool(now2[i]), ever_after=bool(ever2[i]), s_abs=float(sa[i])))
            now, ever = now2, ever2
        print(f"[{tag}] block {b} once {ever.mean():.3f} end {now.mean():.3f}", flush=True)
    return rows


def _st(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), frac_lt_0p1=float((x < 0.1).mean()),
                frac_gt_0p9=float((x > 0.9).mean()), q10=float(np.percentile(x, 10)), q90=float(np.percentile(x, 90)))


def summarize(rows, key="g", vkey="v"):
    """State classes are defined at decision time (privileged, analysis only)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df["v"] = df[vkey]
    last = df[df.dec == df.dec.max()].set_index("eid")
    ep_once = df.groupby("eid").ever_after.max()
    df["ep_once"] = df.eid.map(ep_once)
    first_succ = df[df.ever_after & ~df.ever].set_index("eid").dec  # decision that produced the first success
    df["fs"] = df.eid.map(first_succ)
    pre = ~df.ever
    on = df.now
    out = dict(
        n_episodes=int(df.eid.nunique()), once=float(last.ever_after.mean()), end=float(last.now_after.mean()),
        end_given_once=float(last.now_after[last.ever_after].mean()),
        loss_per_success_decision=float((df.now & ~df.now_after).sum() / max(int(df.now.sum()), 1)),
        n_success_decisions=int(df.now.sum()),
        s_abs_mean=float(df.s_abs.mean()), gs_abs_mean=float((df.g * df.s_abs).mean()),
        cls={
            "pre_success_all": _st(df[key][pre]),
            "pre_success_eps_that_succeed": _st(df[key][pre & df.ep_once]),
            "pre_success_eps_that_fail": _st(df[key][pre & ~df.ep_once]),
            "right_after_first_success": _st(df[key][df.dec == df.fs + 1]),
            "success_flag_on": _st(df[key][on]),
            "success_lost": _st(df[key][df.ever & ~df.now]),
            "all": _st(df[key]),
        },
        vhat={
            "pre_success_eps_that_succeed": _st(df.v[pre & df.ep_once]),
            "pre_success_eps_that_fail": _st(df.v[pre & ~df.ep_once]),
            "right_after_first_success": _st(df.v[df.dec == df.fs + 1]),
            "success_flag_on": _st(df.v[on]), "success_lost": _st(df.v[df.ever & ~df.now]),
        },
        g_by_decision_mean=[float(x) for x in df.groupby("dec")[key].mean()],
        episodes=[dict(eid=int(e), success_once=bool(r.ever_after), success_at_end=bool(r.now_after))
                  for e, r in last.iterrows()])
    return out


def paired(a, b, n_boot=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), n=int(len(d)))


def mode_b0(out, blocks, n_dec):
    import pandas as pd

    t0 = time.time()
    rows = heldout("b0", None, "end", blocks, n_dec)  # u = 0 -> the gate multiplies 0: a B0 rollout with g logged
    pd.DataFrame(rows).to_parquet(out / "b0_gates.parquet")
    s = dict(end=summarize(rows, key="g1_end", vkey="v_end"), t023=summarize(rows, key="g1", vkey="v"))
    assert max(abs(r["g"] - r["g1_end"]) for r in rows) < 1e-5
    for k in s:
        s[k].pop("episodes")
    s.update(note="residual 0; g is what G1 would apply at each visited state, for both gate predictors (same rollout)",
             gates=GATES, once=s["end"]["once"], end_rate=s["end"]["end"],
             g_mean_success_flag_on_end=s["end"]["cls"]["success_flag_on"]["mean"],
             g_mean_right_after_first_success_end=s["end"]["cls"]["right_after_first_success"]["mean"],
             wall_s=time.time() - t0, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return s


def mode_diag(out, blocks, n_dec, names):
    import pandas as pd

    t0, res, eps = time.time(), {}, {}
    order = []
    arms = ["end", None, "t023"]
    for i, n in enumerate(names):  # rotate the arm order per checkpoint
        order += [(n, arms[(i + j) % 3]) for j in range(3)]
    for n, gated in order:
        tag = f"{n}-{'g1' + gated if gated else 'nogate'}"
        f = out / f"{tag}.parquet"
        rows = pd.read_parquet(f).to_dict("records") if f.exists() else heldout(tag, E2[n], gated, blocks, n_dec)
        if not f.exists():
            pd.DataFrame(rows).to_parquet(f)
        s = summarize(rows, key="g" if gated else "g1_end", vkey="v" if gated == "t023" else "v_end")
        eps[tag] = {e["eid"]: e for e in s.pop("episodes")}
        s["run_id"], s["ckpt"] = E2[n], str(ckpt_file(E2[n]))
        res[tag] = s
        print(json.dumps({k: s[k] for k in ("once", "end", "end_given_once", "loss_per_success_decision")}), flush=True)
    comp = {}
    for n in names:
        b = eps[f"{n}-nogate"]
        for arm in ("g1end", "g1t023"):
            a = eps[f"{n}-{arm}"]
            ids = sorted(set(a) & set(b))
            comp[f"{n}-{arm}"] = dict(
                once=paired([a[e]["success_once"] for e in ids], [b[e]["success_once"] for e in ids]),
                end=paired([a[e]["success_at_end"] for e in ids], [b[e]["success_at_end"] for e in ids]))
    mean = {arm: {k: float(np.mean([res[f"{n}-{arm}"][k] for n in names]))
                  for k in ("once", "end", "end_given_once", "loss_per_success_decision", "gs_abs_mean")}
            for arm in ("g1end", "g1t023", "nogate")}
    return dict(task="T036", kind="diagnostic_eval_only", gates=GATES, arms=res, paired_gate_minus_nogate=comp,
                mean_over_seeds=mean, heldout_success_once=mean["g1end"]["once"],
                heldout_success_at_end=mean["g1end"]["end"],
                nogate_success_once=mean["nogate"]["once"], nogate_success_at_end=mean["nogate"]["end"],
                wall_s=time.time() - t0, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                note="g1end / g1t023 = E2 checkpoint + G1 gate (V_hat_end / T023 V_hat) at evaluation only (no "
                     "training); nogate = same checkpoint, same process. cls/g for nogate arms is the V_hat_end G1 "
                     "value that would have applied (not executed). heldout_* at top level = g1end mean.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cpu", "unit", "b0", "diag"])
    ap.add_argument("--out", default="/outputs/StochRL/t036/val")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_dec", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--runs", default="e2-s0,e2-s1,e2-s2")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.mode != "cpu":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "T036: physical GPU 0 only"
    if a.mode == "cpu":
        r = mode_cpu()
    elif a.mode == "unit":
        r = mode_unit(a.batch, a.n_dec)
    elif a.mode == "b0":
        r = mode_b0(out, a.blocks, a.n_dec)
    else:
        r = mode_diag(out, a.blocks, a.n_dec, a.runs.split(","))
    name = a.json or f"{a.mode}.json"
    (out / name).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("episodes", "arms", "state_dict_keys_g2")}, indent=1)[:6000])
    if "pass" in r:
        raise SystemExit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
