"""T037 pre-training checks for E0v: obs = [0 x 2048 | qpos 8 | base chunk 35 | t/16 | V_hat_end], g = 1.

  --mode cpu   (no GPU)
     * predictor: vla_residual.potential.Potential(root = T036 predictors) and an independent numpy float64 formula on
       the cached T017 VLM17 features reproduce /outputs/StochRL/t036/vhat_end_all.npz (what T036 fit / selected on)
     * actor: the plain RLinf MLP with obs_dim 2093 reads column 2092; zero-initialised output layer -> mean 0
  --mode gpu   (single process, no Ray, physical GPU 0 only; one shared ManiSkill scene)
     A) per decision of an E0v rollout (train split, GRPO grouping, random u): obs dim 2093, first 2048 dims == 0,
        dims 2048:2092 == E0 composition of the env state (qpos | base chunk | t/16), column 2092 == Potential(feat_raw)
        == independent numpy formula; toggling vhat_obs off on the same state gives the E0 obs bit by bit; the executed
        sim actions == clip(a_base + alpha tanh(u)) (the E0 / E2 formula) bit by bit.
     B) separate env objects, same seed and u stream, order E0, E0v, E0', E0v' (order-swapped repeat): base chunk and
        obs[:, 2048:2092] compared bit by bit per decision. Decision 0 must be identical; later decisions are compared
        with the E0-vs-E0' run-to-run floor (PhysX / pi0.5 nondeterminism, T029 / T032).
     C) eval split, residual 0, held-out block 0 (32 episodes x 16 decisions): V_hat column vs the values T036 logged
        for the same (episode id, decision) in its B0 rollout (/outputs/StochRL/t036/val/b0_gates.parquet, v_end).
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t037 --mode gpu --out /outputs/StochRL/t037/val
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

PRED = Path("/outputs/StochRL/t036/predictors")
VHAT_ALL = Path("/outputs/StochRL/t036/vhat_end_all.npz")
B0_GATES = Path("/outputs/StochRL/t036/val/b0_gates.parquet")
CACHE = Path("/workspace/stochrl_t018/cache")
F, D0, D1 = 2048, 2092, 2093
GATE = dict(mode="one", vhat="end", tau=0.41, temp=0.1047)


def vhat_np(feat_raw):
    """Independent float64 reimplementation: standardize -> linear logit -> sigmoid -> Platt."""
    P = np.load(PRED / "vla.npz", allow_pickle=True)
    cal = json.load(open(PRED / "calibration.json"))["vla"]
    z = (np.asarray(feat_raw, np.float64) - P["mu"]) / P["sd"]
    p = 1 / (1 + np.exp(-(z @ P["W"] + float(P["b"]))))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(cal["platt_a"] * np.log(p / (1 - p)) + cal["platt_b"])))


# ------------------------------------------------------------------ cpu
def mode_cpu():
    from omegaconf import OmegaConf

    import rlinf.models.embodiment.mlp_policy as mp
    from vla_residual import rlinf_ext
    from vla_residual.potential import Potential

    out = {}
    d = np.load(VHAT_ALL, allow_pickle=True)
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    eps = lab["episode"].reshape(500, 16)[:, 0]
    X = np.load(CACHE / "feat_vlm17_img_mean.npy").astype(np.float32).reshape(500, 16, -1)
    row = {e: i for i, e in enumerate(eps.tolist())}
    X = X[[row[e] for e in d["episodes"].tolist()]]  # the 425 train + val episodes T036 kept
    ref = d["vhat_end"].astype(np.float64)
    v_pot = Potential("vla", device="cpu", root=PRED)(feat_raw=torch.as_tensor(X.reshape(-1, F))).numpy().reshape(ref.shape)
    v_np = vhat_np(X.reshape(-1, F)).reshape(ref.shape)
    out["predictor"] = dict(
        n_states=int(ref.size), potential_vs_t036_max_abs=float(np.abs(v_pot - ref).max()),
        numpy_vs_t036_max_abs=float(np.abs(v_np - ref).max()), potential_vs_numpy_max_abs=float(np.abs(v_pot - v_np).max()),
        vhat_min=float(v_pot.min()), vhat_max=float(v_pot.max()), vhat_mean=float(v_pot.mean()), vhat_std=float(v_pot.std()),
        post_success_median=float(np.median(v_pot[d["post"].astype(bool)])),
        pre_success_end_fail_median=float(np.median(v_pot[~d["post"].astype(bool) & (d["y_end"] == 0)[:, None]])))
    # actor MLP
    rlinf_ext._patch_mlp()
    base = dict(model_type="mlp_policy", obs_dim=D0, action_dim=7, num_action_chunks=1, add_value_head=False,
                vr_init=dict(zero_last=True, init_logstd=-1.5))
    torch.manual_seed(0); e0 = mp.get_model(OmegaConf.create(base))
    torch.manual_seed(0); ev = mp.get_model(OmegaConf.create(dict(base, obs_dim=D1)))
    st = torch.randn(64, D1); st[:, :F] = 0; st[:, D0] = torch.rand(64)
    st2 = st.clone(); st2[:, D0] = 1 - st2[:, D0]
    m, ls = ev._sample_actions(st)
    first = next(p for p in ev.backbone.parameters() if p.dim() == 2)
    out["actor"] = dict(
        first_layer_in=int(first.shape[1]), n_params_e0=sum(p.numel() for p in e0.parameters()),
        n_params_e0v=sum(p.numel() for p in ev.parameters()), hidden=int(first.shape[0]),
        init_mean_all_zero=bool((m == 0).all()), logstd=[round(float(x), 3) for x in ls[0]],
        backbone_reads_vhat_col=bool((ev.backbone(st) != ev.backbone(st2)).any()),
        vhat_col_weight_abs_mean=float(first[:, D0].abs().mean()), other_col_weight_abs_mean=float(first[:, F:D0].abs().mean()))
    a = out["actor"]
    out["pass"] = bool(out["predictor"]["potential_vs_t036_max_abs"] < 1e-5 and out["predictor"]["numpy_vs_t036_max_abs"] < 1e-5
                       and a["first_layer_in"] == D1 and a["init_mean_all_zero"] and a["backbone_reads_vhat_col"]
                       and a["n_params_e0v"] == a["n_params_e0"] + a["hidden"])
    return out


# ------------------------------------------------------------------ gpu
def make_env(split, e0v, batch, seed):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    extra = dict(zero_vlm=True)
    if e0v:
        extra.update(vhat_obs=True, gate=dict(GATE))
    if split == "eval":
        cfg = make_cfg("eval", n_eval=V.N_EVAL_DEFAULT, auto_reset=False, ignore_term=True, **extra)
    else:
        cfg = make_cfg("train", auto_reset=False, ignore_term=False, group_size=8, **extra)
    cfg.seed = seed
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    assert env.zero_vlm and env.vhat_obs == e0v
    return env


@torch.no_grad()
def rollout(split, e0v, batch, n_dec, seed, zero_u=False):
    from vla_residual import core as V

    env = make_env(split, e0v, batch, seed)
    steps, orig = [], env._sim_step

    def sim_step(a, render):
        steps.append(a.detach().clone())
        return orig(a, render)

    env._sim_step = sim_step
    g = torch.Generator().manual_seed(4242)
    U = torch.zeros(n_dec, batch, V.ACT_DIM) if zero_u else 0.5 * torch.randn(n_dec, batch, V.ACT_DIM, generator=g)
    obs, _ = env.reset()
    S, A, VH, DEC, chk = [obs["states"].cpu().clone()], [env.a_base.cpu().clone()], [], [], []

    def check(st):
        st = st.to(env.device)
        rest = torch.cat([env.qpos, (env.a_base / V.BASE_SCALE.to(env.device)).reshape(batch, -1),
                          (env.dec.float() / env.max_dec)[:, None]], 1).float()
        c = dict(dim=int(st.shape[1]), feat_zero=bool((st[:, :F] == 0).all()), rest_equal=bool(torch.equal(st[:, F:D0], rest)),
                 internal_feat_nonzero=bool((env.feat != 0).float().mean() > 0.99))
        if e0v:
            v = env.gpot(feat_raw=env.feat_raw)
            vn = vhat_np(env.feat_raw.cpu().numpy())
            c["vhat_col_equal_potential"] = bool(torch.equal(st[:, D0], v))
            c["vhat_col_vs_numpy_max_abs"] = float(np.abs(st[:, D0].cpu().numpy() - vn).max())
            c["vhat_in_01"] = bool(((st[:, D0] > 0) & (st[:, D0] < 1)).all())
            a0, f0 = env.a_base.clone(), env.feat_raw.clone()
            env.vhat_obs = False
            o0 = env._obs()["states"]
            env.vhat_obs = True
            c["toggle_off_is_e0_obs"] = bool(o0.shape[1] == D0 and torch.equal(o0, st[:, :D0]))
            c["state_untouched"] = bool(torch.equal(a0, env.a_base) and torch.equal(f0, env.feat_raw))
            VH.append(st[:, D0].cpu().clone()); DEC.append(env.dec.cpu().clone())
        return c

    chk.append(check(obs["states"]))
    for d in range(n_dec):
        u = U[d].to(env.device)
        want = torch.stack([(env.a_base[:, k] + env.alpha * torch.tanh(u)).clamp(env.lo, env.hi) for k in range(5)], 1)
        steps.clear()
        o, r, te, tr, inf = env.chunk_step(u[:, None])
        c_exec = bool(torch.equal(torch.stack(steps, 1), want))
        g_one = bool((env.last_gate["g"] == 1).all()) if e0v else True
        S.append(o[0]["states"].cpu().clone()); A.append(env.a_base.cpu().clone())
        c = check(o[0]["states"]); c["exec_bitwise_e0_formula"] = c_exec; c["g_is_one"] = g_one
        chk.append(c)
    return dict(s=torch.stack(S), a=torch.stack(A), chk=chk, eids=env.eids.clone(), vh=VH, dec=DEC,
                succ=float(env.success_once.float().mean()))


def cmp(x, y):
    ea = (x["a"] == y["a"]).flatten(2).all(-1)
    er = (x["s"][..., F:D0] == y["s"][..., F:D0]).all(-1)
    return dict(base_chunk_equal_frac=float(ea.float().mean()), base_chunk_equal_dec0=bool(ea[0].all()),
                rest_equal_frac=float(er.float().mean()), rest_equal_dec0=bool(er[0].all()),
                base_chunk_equal_by_dec=[round(float(v), 3) for v in ea.float().mean(1)],
                eids_equal=bool(torch.equal(x["eids"], y["eids"])))


def allk(R, k):
    return all(c[k] for c in R["chk"] if k in c)


def mode_gpu(batch, n_dec):
    import pandas as pd

    t0, out = time.time(), dict(gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=batch, n_dec=n_dec)
    R = {}
    for name, e0v in (("E0", False), ("E0v", True), ("E0b", False), ("E0vb", True)):
        R[name] = rollout("train", e0v, batch, n_dec, seed=0)
        print(f"[train] {name} succ_once {R[name]['succ']:.3f} {time.time()-t0:.0f}s", flush=True)
    keys = ("feat_zero", "rest_equal", "internal_feat_nonzero", "vhat_col_equal_potential", "vhat_in_01",
            "toggle_off_is_e0_obs", "state_untouched", "exec_bitwise_e0_formula", "g_is_one")
    A = {k: bool(allk(R["E0v"], k) and allk(R["E0vb"], k)) for k in keys}
    A["obs_dim"] = sorted({c["dim"] for c in R["E0v"]["chk"]})
    A["vhat_col_vs_numpy_max_abs"] = max(c["vhat_col_vs_numpy_max_abs"] for c in R["E0v"]["chk"] + R["E0vb"]["chk"])
    vh = torch.stack(R["E0v"]["vh"])
    A["vhat_stats"] = dict(min=float(vh.min()), max=float(vh.max()), mean=float(vh.mean()), std=float(vh.std()))
    A["n_checks"] = len(R["E0v"]["chk"]) + len(R["E0vb"]["chk"])
    E0chk = {k: bool(allk(R["E0"], k) and allk(R["E0b"], k)) for k in ("feat_zero", "rest_equal", "exec_bitwise_e0_formula")}
    E0chk["obs_dim"] = sorted({c["dim"] for c in R["E0"]["chk"]})
    Bc = dict(E0v_vs_E0=cmp(R["E0v"], R["E0"]), E0v_vs_E0b=cmp(R["E0v"], R["E0b"]), E0vb_vs_E0b=cmp(R["E0vb"], R["E0b"]),
              floor_E0_vs_E0b=cmp(R["E0"], R["E0b"]), floor_E0v_vs_E0vb=cmp(R["E0v"], R["E0vb"]))
    cross = min(Bc[k]["base_chunk_equal_frac"] for k in ("E0v_vs_E0", "E0v_vs_E0b", "E0vb_vs_E0b"))
    floor = min(Bc[k]["base_chunk_equal_frac"] for k in ("floor_E0_vs_E0b", "floor_E0v_vs_E0vb"))
    out.update(A_e0v_obs=A, A_e0_obs=E0chk, B_base_chunk=Bc, cross_min_equal_frac=cross, floor_min_equal_frac=floor,
               train_success_once={k: v["succ"] for k, v in R.items()})
    # C) eval split, residual 0: V_hat column vs T036's B0 rollout log
    ev = rollout("eval", True, batch, 16, seed=1000, zero_u=True)
    e0 = rollout("eval", False, batch, 16, seed=1000, zero_u=True)
    print(f"[eval] E0v / E0 residual 0 once {ev['succ']:.3f} / {e0['succ']:.3f} {time.time()-t0:.0f}s", flush=True)
    ref = pd.read_parquet(B0_GATES)
    ref = {(int(a), int(b)): float(c) for a, b, c in zip(ref["eid"], ref["dec"], ref["v_end"])}
    eids = ev["eids"].tolist()
    dif = np.array([[abs(float(ev["vh"][d][i]) - ref[(eids[i], d)]) for i in range(batch)] for d in range(16)])
    out["C_eval"] = dict(
        n_episodes=batch, obs_checks={k: bool(allk(ev, k)) for k in keys}, eids_in_t036_b0=True,
        vhat_vs_t036_b0_dec0_max_abs=float(dif[0].max()), vhat_vs_t036_b0_all_max_abs=float(dif.max()),
        vhat_vs_t036_b0_frac_lt_1e4=float((dif < 1e-4).mean()), vhat_vs_t036_b0_frac_lt_1e4_by_dec=[round(float(x), 3) for x in (dif < 1e-4).mean(1)],
        vhat_vs_t036_b0_median_abs=float(np.median(dif)), base_chunk_E0v_vs_E0=cmp(ev, e0),
        success_once=dict(E0v=ev["succ"], E0=e0["succ"]))
    out["pass"] = bool(
        all(v for k, v in A.items() if isinstance(v, bool)) and A["obs_dim"] == [D1] and A["vhat_col_vs_numpy_max_abs"] < 1e-5
        and all(v for k, v in E0chk.items() if isinstance(v, bool)) and E0chk["obs_dim"] == [D0]
        and all(Bc[k]["base_chunk_equal_dec0"] and Bc[k]["rest_equal_dec0"] and Bc[k]["eids_equal"]
                for k in ("E0v_vs_E0", "E0v_vs_E0b", "E0vb_vs_E0b"))
        and all(out["C_eval"]["obs_checks"].values()) and out["C_eval"]["vhat_vs_t036_b0_dec0_max_abs"] < 1e-4
        and out["C_eval"]["base_chunk_E0v_vs_E0"]["base_chunk_equal_dec0"])
    out["pass_all_decisions_bitwise"] = bool(cross == 1.0)
    out["wall_s"] = time.time() - t0
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_dec", type=int, default=16)
    ap.add_argument("--out", default="/outputs/StochRL/t037/val")
    a = ap.parse_args()
    res = mode_cpu() if a.mode == "cpu" else mode_gpu(a.batch, a.n_dec)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
