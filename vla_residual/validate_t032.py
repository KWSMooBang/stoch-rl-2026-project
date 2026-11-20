"""T032 unit test: vla_residual.zero_vlm (E0) changes only the first 2048 dims of "states".

Single process, no Ray. One shared ManiSkill scene (same batch size for every env wrapper).
  A) in-situ toggle (exact, no physics rerun): during one E2 rollout, at every decision build the obs twice from the
     same env state (zero_vlm False / True). E0 obs must be [0 x 2048 | identical 44 dims], base chunk untouched.
  B) separate env objects, same seed, same residual sequence, order E2, E0, E2', E0' (order-swapped repeat):
     base chunk and states[:, 2048:] compared bit by bit per decision. E2 vs E2' is the run-to-run noise floor
     (PhysX / pi0.5 nondeterminism, T029). Decision 0 (right after reset) must be bitwise identical.
     Train split (GRPO setting: group 8, success terminates) and eval split (held-out ids, fixed noise).
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t032 --out /outputs/StochRL/t032/val
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from vla_residual import core as V
from vla_residual.validate import make_cfg

import torch  # noqa: E402

F = V.FEAT_DIM


def rollout(split, zero_vlm, batch, n_dec, seed, toggle=False):
    kw = dict(group_size=8, auto_reset=False, ignore_term=False) if split == "train" else dict(ignore_term=True)
    cfg = make_cfg(split, n_eval=batch, zero_vlm=zero_vlm, **kw)
    cfg.seed = seed
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    assert env.zero_vlm == zero_vlm
    g = torch.Generator().manual_seed(4242)
    U = 0.5 * torch.randn(n_dec, batch, 1, V.ACT_DIM, generator=g)  # same residual sequence for every rollout
    obs, _ = env.reset()
    A, S, FE, tog = [env.a_base.clone().cpu()], [obs["states"].clone().cpu()], [env.feat.clone().cpu()], []

    def check_toggle():
        a0, f0 = env.a_base.clone(), env.feat.clone()
        env.zero_vlm = False
        o2 = env._obs()["states"]
        env.zero_vlm = True
        o0 = env._obs()["states"]
        env.zero_vlm = zero_vlm
        tog.append(dict(
            e0_feat_all_zero=bool((o0[:, :F] == 0).all()), e2_feat_nonzero_frac=float((o2[:, :F] != 0).float().mean()),
            e2_feat_eq_env_feat=bool(torch.equal(o2[:, :F], env.feat)), rest_bitwise_equal=bool(torch.equal(o0[:, F:], o2[:, F:])),
            state_untouched=bool(torch.equal(a0, env.a_base) and torch.equal(f0, env.feat)), shape=list(o0.shape)))

    if toggle:
        check_toggle()
    for d in range(n_dec):
        o, r, te, tr, inf = env.chunk_step(U[d].to(env.device))
        A.append(env.a_base.clone().cpu()); S.append(o[0]["states"].clone().cpu()); FE.append(env.feat.clone().cpu())
        if toggle:
            check_toggle()
    return dict(a=torch.stack(A), s=torch.stack(S), f=torch.stack(FE), eids=env.eids.clone(), tog=tog,
                succ=float(env.success_once.float().mean()))


def cmp(x, y):
    """Bitwise comparison per (decision, env)."""
    ea = (x["a"] == y["a"]).flatten(2).all(-1)  # [D+1, B] base chunk
    er = (x["s"][..., F:] == y["s"][..., F:]).all(-1)  # qpos | base chunk | t
    ef = (x["f"] == y["f"]).all(-1)  # internal (unzeroed) VLM feature
    return dict(base_chunk_equal_frac=float(ea.float().mean()), base_chunk_equal_dec0=bool(ea[0].all()),
                rest_equal_frac=float(er.float().mean()), rest_equal_dec0=bool(er[0].all()),
                internal_feat_equal_frac=float(ef.float().mean()), internal_feat_equal_dec0=bool(ef[0].all()),
                base_chunk_equal_by_dec=[round(float(v), 3) for v in ea.float().mean(1)],
                base_chunk_max_abs_diff=float((x["a"] - y["a"]).abs().max()), eids_equal=bool(torch.equal(x["eids"], y["eids"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_dec", type=int, default=16)
    ap.add_argument("--n_dec_eval", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/outputs/StochRL/t032/val")
    a = ap.parse_args()
    t0 = time.time()
    res = dict(gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=a.batch, seed=a.seed, obs_dim=V.OBS_DIM)
    for split, nd in (("train", a.n_dec), ("eval", a.n_dec_eval)):
        R = {}
        for name, z in (("E2", False), ("E0", True), ("E2b", False), ("E0b", True)):
            R[name] = rollout(split, z, a.batch, nd, a.seed, toggle=(name == "E2"))
            print(f"[{split}] {name} done succ_once {R[name]['succ']:.3f} {time.time()-t0:.0f}s", flush=True)
        tog = R["E2"]["tog"]
        out = dict(
            n_dec=nd,
            A_toggle=dict(n_checks=len(tog), **{k: all(t[k] for t in tog) for k in
                                                ("e0_feat_all_zero", "e2_feat_eq_env_feat", "rest_bitwise_equal", "state_untouched")},
                          e2_feat_nonzero_frac_min=min(t["e2_feat_nonzero_frac"] for t in tog), shape=tog[0]["shape"]),
            B_e0_feat_block_all_zero=bool((R["E0"]["s"][..., :F] == 0).all() and (R["E0b"]["s"][..., :F] == 0).all()),
            B_e2_feat_block_nonzero_frac=float((R["E2"]["s"][..., :F] != 0).float().mean()),
            B_e2_feat_block_eq_internal=bool(torch.equal(R["E2"]["s"][..., :F], R["E2"]["f"])),
            B_E0_vs_E2=cmp(R["E0"], R["E2"]), B_E0_vs_E2b=cmp(R["E0"], R["E2b"]), B_E0b_vs_E2b=cmp(R["E0b"], R["E2b"]),
            B_floor_E2_vs_E2b=cmp(R["E2"], R["E2b"]), B_floor_E0_vs_E0b=cmp(R["E0"], R["E0b"]),
            success_once={k: v["succ"] for k, v in R.items()})
        A = out["A_toggle"]
        cross = min(out[k]["base_chunk_equal_frac"] for k in ("B_E0_vs_E2", "B_E0_vs_E2b", "B_E0b_vs_E2b"))
        floor = min(out[k]["base_chunk_equal_frac"] for k in ("B_floor_E2_vs_E2b", "B_floor_E0_vs_E0b"))
        out["cross_min_equal_frac"], out["floor_min_equal_frac"] = cross, floor
        out["pass_strict"] = bool(
            A["e0_feat_all_zero"] and A["e2_feat_eq_env_feat"] and A["rest_bitwise_equal"] and A["state_untouched"]
            and A["shape"] == [a.batch, V.OBS_DIM] and A["e2_feat_nonzero_frac_min"] > 0.99
            and out["B_e0_feat_block_all_zero"] and out["B_e2_feat_block_eq_internal"]
            and all(out[k]["base_chunk_equal_dec0"] and out[k]["rest_equal_dec0"] and out[k]["internal_feat_equal_dec0"]
                    and out[k]["eids_equal"] for k in ("B_E0_vs_E2", "B_E0_vs_E2b", "B_E0b_vs_E2b")))
        out["pass_all_decisions_bitwise"] = bool(cross == 1.0)
        res[split] = out
    res["wall_s"] = time.time() - t0
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["pass"] = bool(res["train"]["pass_strict"] and res["eval"]["pass_strict"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "units_t032.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
