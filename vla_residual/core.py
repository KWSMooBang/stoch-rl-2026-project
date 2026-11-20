"""T022: frozen pi0.5 inside an env wrapper -> "MLP policy env" for RLinf residual RL.

One RL step = one pi0.5 decision (K=5 sim steps at 5 Hz, H=8 generated, first K executed).
    a_exec[k] = clip(a_base[k] + alpha * s(u), action_space)   k = 0..K-1   (same residual for the whole chunk)
    s(u) = tanh(u)  (PPO / GRPO / DAgger: raw-Gaussian MLP)   or   clip(u, -1, 1)  (SAC / CrossQ: policy already tanh-squashed)
    alpha = (a_pos x3 [m/step], a_rot x3 [rad/step], a_grip)  -- identical for every algorithm.
Observation "states" (float32, dim OBS_DIM):
    [ standardized VLM layer-17 image-token mean (2048, clipped +-5) | robot qpos (8) | base chunk K x 7 (scaled) | t / T ]
    The VLM feature is computed in the same pi0.5 forward that produced the base chunk (hook, observe-only).
    T032 E0: cfg.vla_residual.zero_vlm=True replaces the 2048 feature dims of "states" by constant 0 (same OBS_DIM,
    same MLP). Only the returned obs changes: the pi0.5 forward, base chunk, self.feat / feat_raw stay untouched.
    Extra key "expert_u" (privileged scripted-expert residual label, in the same space as u) is present only when
    cfg.vla_residual.expert_label is True; the student MLP reads only "states".
Reward: sparse success. Train: +1 on the decision in which success first becomes true, terminated=True (no bootstrap).
        Truncation after T=16 decisions (80 sim steps) -> truncated=True (bootstrap where the algorithm does).
Eval:   ignore_terminations=True -> every episode runs 16 decisions; success_once / success_at_end recorded.
Episode ids: train ids are drawn uniformly from all 802,816 ids minus the held-out set; eval ids are the fixed
        held-out list (T020 test episodes, first N_EVAL), assigned to env slots block by block.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from repcritic import rc_common as C  # noqa: E402

FEAT_LAYER = 17
FEAT_DIM = 2048
QPOS_DIM = 8
ACT_DIM = 7
T_DEC = C.MAX_STEPS // C.K  # 16 decisions
OBS_DIM = FEAT_DIM + QPOS_DIM + C.K * ACT_DIM + 1  # 2092
BASE_SCALE = torch.tensor([0.03, 0.03, 0.03, 0.1, 0.1, 0.1, 1.0])  # scaling of base actions in the obs
N_EVAL_DEFAULT = 128

# ---------------------------------------------------------------- episode splits
EP_SEED, EP_OFFSET, T016_SEED, T016_N = 20261119, 500, 12345, 30


def heldout_episodes(total=802816, n=N_EVAL_DEFAULT):
    """T020 test episodes (never used for training/selection in T016-T020); first n are T022's held-out set."""
    g = torch.Generator().manual_seed(EP_SEED)
    perm = torch.randperm(total, generator=g).tolist()
    used = set(perm[:EP_OFFSET])
    g2 = torch.Generator().manual_seed(T016_SEED)
    used |= set(torch.randperm(total, generator=g2)[:T016_N].tolist())
    out = [e for e in perm[EP_OFFSET:EP_OFFSET + 400] if e not in used]
    return out[:n], out  # (held-out, everything excluded from training = first 150 T020 test ids and more)


# ---------------------------------------------------------------- frozen pi0.5 + feature hook
class VLABase:
    """Frozen pi0.5 (C.Policy semantics) that also returns the pooled VLM layer-17 image feature."""

    def __init__(self):
        C.setup_env_vars()
        self.pol = C.Policy(C.build_policy())
        m = self.pol.m
        self._rec = False
        self._x = None
        self._pm = None
        self._prompt_mask = None
        m.llm.layers[FEAT_LAYER].register_forward_hook(self._hook)
        orig = m.build_prefix_cache

        def bpc(observation):
            out = orig(observation)
            if self._rec:
                self._pm = out[1].detach()
                self._prompt_mask = observation.tokenized_prompt_mask.detach()
            return out

        m.build_prefix_cache = bpc
        st = np.load(REPO / "vla_residual/feat_stats_vlm17_img_mean.npz")
        dev = self.pol.device
        self.f_mean = torch.as_tensor(st["mean"], device=dev)
        self.f_std = torch.as_tensor(st["std"], device=dev)

    def _hook(self, mod, inp, out):
        if self._rec and out[0][0] is not None:
            self._x = out[0][0].detach()
        return None

    def _pool(self):
        pm, x = self._pm, self._x
        B = pm.shape[0]
        n_slots = (pm.shape[1] - self._prompt_mask.shape[1]) // 256
        valid = pm[:, : n_slots * 256].reshape(B, n_slots, 256).all(-1)
        slot = int(valid[0].float().argmax())
        assert bool(valid[:, slot].all()), "image slot differs across batch"
        return x[:, slot * 256:(slot + 1) * 256].float().mean(1)

    @torch.no_grad()
    def act(self, obs, noise):
        """obs: wrapped ManiskillEnv obs. Returns (env actions [B,K,7] float32, standardized feature [B,2048])."""
        from rlinf.models.embodiment.openpi.transforms.apply import apply_output_transform
        import dataclasses
        from torch.utils._pytree import tree_map

        m = self.pol.m
        observation = m.env_obs_to_observation(obs)
        B = observation.state.shape[0]
        mb = int(os.environ.get("RC_POLICY_MB", "32"))
        outs, feats = [], []
        for s in range(0, B, mb):
            sl = slice(s, min(B, s + mb))
            sub = dataclasses.replace(observation, **{
                f.name: tree_map(lambda x: x[sl] if isinstance(x, torch.Tensor) else x, getattr(observation, f.name))
                for f in dataclasses.fields(observation)})
            self._rec = True
            try:
                outs.append(m.sample_actions(sub, num_steps=m.num_steps, noise=noise[sl]))
            finally:
                self._rec = False
            feats.append(self._pool())
        macts = torch.cat(outs, 0)
        acts = apply_output_transform(m._output_transform_fn, {"actions": macts, "state": observation.state},
                                      action_chunk=None)["actions"].to(torch.float32)[:, : C.K]
        f = torch.cat(feats, 0)
        self.last_feat_raw = f  # unstandardized (T018 cache convention; T024 potential input)
        f = ((f - self.f_mean) / self.f_std).clamp(-5, 5)
        return acts, f


_VLA = None
_INNER = {}


def get_vla():
    global _VLA
    if _VLA is None:
        _VLA = VLABase()
    return _VLA


# ---------------------------------------------------------------- privileged state + scripted expert
def priv_state(env):
    """Lean privileged state (label/expert only; never in "states"). Positions in the robot base frame."""
    base = env.env.unwrapped
    E = base.num_envs
    dev = base.device
    cp = torch.zeros(E, 3, device=dev)
    grasp = torch.zeros(E, dtype=torch.bool, device=dev)
    for idx, name in enumerate(base.model_db_carrot):
        sel = base.select_carrot_ids == idx
        if not bool(sel.any()):
            continue
        obj = base.objs_carrot[name]
        cp = torch.where(sel[:, None], obj.pose.p, cp)
        grasp = torch.where(sel, base.agent.is_grasping(obj), grasp)
    cq = torch.zeros(E, 4, device=dev)
    lsz = torch.zeros(E, 3, device=dev)
    for idx, name in enumerate(base.model_db_carrot):
        sel = base.select_carrot_ids == idx
        if bool(sel.any()):
            cq = torch.where(sel[:, None], base.objs_carrot[name].pose.q, cq)
            lsz = torch.where(sel[:, None], torch.as_tensor(base.model_bbox_sizes[name], device=dev).float(), lsz)
    from mani_skill.utils.geometry import rotation_conversions as rc

    R = rc.quaternion_to_matrix(cq)  # [E,3,3] local->world
    long_local = lsz[:, :2].argmax(1)  # long horizontal local axis
    long_w = R[torch.arange(E), :, long_local]  # column = local axis in world
    th_long = torch.atan2(long_w[:, 1], long_w[:, 0])
    fd = base.agent.finger1_link.pose.p - base.agent.finger2_link.pose.p
    th_close = torch.atan2(fd[:, 1], fd[:, 0])
    yaw_err = th_long + torch.pi / 2 - th_close  # want closing direction perpendicular to the long axis
    yaw_err = torch.remainder(yaw_err + torch.pi / 2, torch.pi) - torch.pi / 2  # axis symmetry -> [-pi/2, pi/2)
    elong = lsz[:, :2].max(1).values / lsz[:, :2].min(1).values.clamp(min=1e-3)
    pp = base.objs_plate[base.plate_names[0]].pose.p
    root = base.agent.robot.pose
    from mani_skill.utils.structs.pose import Pose

    inv = root.inv()
    cb = (inv * Pose.create_from_pq(p=cp)).p
    pb = (inv * Pose.create_from_pq(p=pp)).p
    ee = base.agent.controller.controllers["arm"].ee_pose_at_base.p
    qpos = base.agent.robot.get_qpos()
    fingers = qpos[:, -2:].mean(1)
    on_target = base.episode_stats["src_on_target"] if "src_on_target" in base.episode_stats else torch.zeros_like(grasp)
    return dict(carrot=cb, plate=pb, ee=ee, grasp=grasp, fingers=fingers, on_target=on_target.bool(),
                carrot_w=cp, plate_w=pp, yaw_err=yaw_err, elong=elong)


class ScriptedExpert:
    """Privileged decision-level pick-and-place controller with a small per-env phase memory.

    One call per decision; the per-step delta is constant over the K=5 sim steps (same granularity as the residual).
    Phases: APPROACH (hover above object) -> DESCEND -> CLOSE (2 decisions) -> LIFT (slow; retry if the object does
    not follow) -> CARRY (over plate) -> LOWER -> RELEASE -> DONE (rise, open). RETRY = open + rise -> APPROACH.
    Yaw (rot z) turns the finger closing direction perpendicular to the object's long axis while approaching. Positions in the robot base frame; output in env action units [B,7].
    """
    APPROACH, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, DONE, RETRY = range(9)
    NAMES = ["approach", "descend", "close", "lift", "carry", "lower", "release", "done", "retry"]

    def __init__(self, n, device, vmax=0.02, vslow=0.01, z_hover=0.07, z_grasp=0.01, z_carry=0.10, z_place=0.03,
                 xy_tol=0.01, z_tol=0.008, n_close=2, wmax=0.1, yaw_sign=1.0):
        self.vmax, self.vslow, self.z_hover, self.z_carry = vmax, vslow, z_hover, z_carry
        self.z_grasp = float(os.environ.get("VR_ZGRASP", z_grasp))
        self.z_place, self.xy_tol, self.z_tol, self.n_close = z_place, xy_tol, z_tol, n_close
        self.wmax, self.yaw_sign = wmax, float(os.environ.get("VR_YAW_SIGN", yaw_sign))
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.cnt = torch.zeros(n, dtype=torch.long, device=device)
        self.cz0 = torch.zeros(n, device=device)
        self.ez0 = torch.zeros(n, device=device)
        self.off = torch.zeros(n, 3, device=device)
        self.last = torch.zeros(n, ACT_DIM, device=device)

    def reset(self, mask):
        self.phase[mask] = self.APPROACH
        self.cnt[mask] = 0

    def __call__(self, ps, mask=None):
        P = self
        ee, cb, pb = ps["ee"], ps["carrot"], ps["plate"]
        B = ee.shape[0]
        dev = ee.device
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=dev)
        ph = self.phase.clone(); cnt = self.cnt.clone(); cz0 = self.cz0.clone(); ez0 = self.ez0.clone()
        off = self.off.clone()
        e_xy = torch.linalg.norm(ee[:, :2] - cb[:, :2], dim=1)
        hover_z = cb[:, 2] + self.z_hover
        grasp_z = cb[:, 2] + self.z_grasp
        followed = (cb[:, 2] - cz0) > 0.01
        c_xy = torch.linalg.norm(cb[:, :2] - pb[:, :2], dim=1)
        z_carry = pb[:, 2] + self.z_carry + off[:, 2]
        place_z = pb[:, 2] + self.z_place + off[:, 2]
        # ---- transitions (one pass, in phase order)
        t = ph == P.APPROACH
        ph = torch.where(t & (e_xy < self.xy_tol), P.DESCEND, ph)
        t = ph == P.DESCEND
        ph = torch.where(t & (e_xy > 2.5 * self.xy_tol), P.APPROACH, ph)
        go_close = (ph == P.DESCEND) & ((ee[:, 2] - grasp_z).abs() < self.z_tol)
        ph = torch.where(go_close, P.CLOSE, ph); cnt = torch.where(go_close, 0, cnt)
        t = (ph == P.CLOSE) & (cnt >= self.n_close)
        cz0 = torch.where(t, cb[:, 2], cz0); ez0 = torch.where(t, ee[:, 2], ez0)
        off = torch.where(t[:, None], ee - cb, off)
        ph = torch.where(t, P.LIFT, ph)
        z_carry = pb[:, 2] + self.z_carry + off[:, 2]
        place_z = pb[:, 2] + self.z_place + off[:, 2]
        t = ph == P.LIFT
        fail = t & ((ee[:, 2] - ez0) > 0.03) & ~followed
        ph = torch.where(fail, P.RETRY, ph)
        ph = torch.where((ph == P.LIFT) & (ee[:, 2] > z_carry - 0.015) & followed, P.CARRY, ph)
        t = ph == P.CARRY
        ph = torch.where(t & ~followed & ~ps["grasp"], P.RETRY, ph)
        ph = torch.where((ph == P.CARRY) & (c_xy < self.xy_tol), P.LOWER, ph)
        t = ph == P.LOWER
        go_rel = t & (ee[:, 2] < place_z + 0.008)
        ph = torch.where(go_rel, P.RELEASE, ph); cnt = torch.where(go_rel, 0, cnt)
        ph = torch.where((ph == P.RELEASE) & (cnt >= 1), P.DONE, ph)
        t = ph == P.RETRY
        ph = torch.where(t & (ee[:, 2] > hover_z - 0.02) & (ps["fingers"] > 0.03), P.APPROACH, ph)
        # ---- actions
        tgt = ee.clone()
        grip = torch.ones(B, device=dev)
        vmax = torch.full((B,), self.vmax, device=dev)
        rise = torch.stack([ee[:, 0], ee[:, 1], torch.maximum(hover_z, ee[:, 2])], 1)
        app = torch.stack([cb[:, 0], cb[:, 1], hover_z], 1)
        app = torch.where(((ee[:, 2] < hover_z - 0.03) & (e_xy > 2.5 * self.xy_tol))[:, None], rise, app)
        tgt = torch.where((ph == P.APPROACH)[:, None], app, tgt)
        tgt = torch.where((ph == P.DESCEND)[:, None], torch.stack([cb[:, 0], cb[:, 1], grasp_z], 1), tgt)
        closing = ph == P.CLOSE
        grip = torch.where(closing, -1.0, grip)
        lift = ph == P.LIFT
        tgt = torch.where(lift[:, None], torch.stack([ee[:, 0], ee[:, 1], z_carry], 1), tgt)
        carry = ph == P.CARRY
        ctgt = torch.stack([pb[:, 0] + off[:, 0], pb[:, 1] + off[:, 1], z_carry], 1)
        tgt = torch.where(carry[:, None], ctgt, tgt)
        lower = ph == P.LOWER
        tgt = torch.where(lower[:, None], torch.stack([ctgt[:, 0], ctgt[:, 1], place_z], 1), tgt)
        grip = torch.where(lift | carry | lower, -1.0, grip)
        vmax = torch.where(lift | lower, self.vslow, vmax)
        done = (ph == P.DONE) | (ph == P.RETRY)
        tgt = torch.where(done[:, None], torch.stack([ee[:, 0], ee[:, 1], ee[:, 2] + 0.05], 1), tgt)
        tgt = torch.where((ph == P.RETRY)[:, None], rise, tgt)
        d = ((tgt - ee) / C.K)
        d = torch.maximum(torch.minimum(d, vmax[:, None]), -vmax[:, None])
        a = torch.zeros(B, ACT_DIM, device=dev)
        a[:, :3] = d
        # yaw alignment while approaching / descending (only for elongated objects)
        rot = (ph == P.APPROACH) | (ph == P.DESCEND)
        yz = (self.yaw_sign * ps["yaw_err"] / C.K).clamp(-self.wmax, self.wmax)
        a[:, 5] = torch.where(rot & (ps["elong"] > 1.2), yz, torch.zeros_like(yz))
        a[:, 6] = grip
        cnt = torch.where(closing | (ph == P.RELEASE), cnt + 1, cnt)
        # ---- commit only masked rows
        m = mask
        self.phase = torch.where(m, ph, self.phase); self.cnt = torch.where(m, cnt, self.cnt)
        self.cz0 = torch.where(m, cz0, self.cz0); self.ez0 = torch.where(m, ez0, self.ez0)
        self.off = torch.where(m[:, None], off, self.off)
        self.last = torch.where(m[:, None], a, self.last)
        return self.last.clone()


# ---------------------------------------------------------------- RLinf-compatible env
class ResidualVLAEnv:
    """RLinf env (ManiskillEnv interface subset used by EnvWorker): reset / chunk_step / properties."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics=True):
        C.setup_env_vars()
        self.cfg = cfg
        vr = cfg.vla_residual
        self.vr = vr
        self.split = str(vr.split)  # train | eval
        self.seed = int(cfg.seed) + seed_offset
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.group_size = int(cfg.get("group_size", 1))
        self.video_cfg = cfg.video_cfg
        self.worker_info = worker_info
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics
        self.max_dec = int(vr.get("max_decisions", T_DEC))
        self.render_skip = bool(vr.get("render_skip", True))
        self.alpha = torch.tensor([vr.a_pos] * 3 + [vr.a_rot] * 3 + [vr.a_grip], dtype=torch.float32)
        self.squash = str(vr.squash)  # tanh | clip
        self.expert_label = bool(vr.get("expert_label", False))
        self.zero_vlm = bool(vr.get("zero_vlm", False))  # T032 E0: VLM feature block of "states" = 0
        if self.zero_vlm:
            print(f"[vla_residual] zero_vlm=True ({self.split}): states[:, :{FEAT_DIM}] = 0", flush=True)
        # T035 base-anchored GRPO (train split only): in every group the first anchor_n_base slots are frozen-base
        # members (executed residual forced to 0, policy output only recorded); anchor_crn shares the pi0.5 noise
        # of a decision between all members of a group (seeded by episode id, decision, env seed).
        self.anchor_n_base = int(vr.get("anchor_n_base", 0)) if self.split == "train" else 0
        self.anchor_crn = bool(vr.get("anchor_crn", False)) and self.split == "train"
        assert 0 <= self.anchor_n_base < max(self.group_size, 1) or self.anchor_n_base == 0, "anchor_n_base >= group_size"
        self.base_member = (torch.arange(num_envs) % max(self.group_size, 1)) < self.anchor_n_base
        if self.anchor_n_base or self.anchor_crn:
            print(f"[vla_residual] anchor ({self.split}): n_base={self.anchor_n_base}/group {self.group_size} "
                  f"crn={self.anchor_crn}", flush=True)
        self.expert_exec = float(vr.get("expert_exec", 0.0))  # 1.0 -> execute expert residual (validation only)
        ecfg = C.env_cfg(num_envs)
        ecfg.max_episode_steps = self.max_dec * C.K + 1  # we handle truncation ourselves
        ecfg.init_params.max_episode_steps = self.max_dec * C.K + 1
        ecfg.max_steps_per_rollout_epoch = self.max_dec * C.K + 1
        # One ManiSkill scene per process and size: RLinf runs train rollouts and evaluation sequentially, and a second
        # scene costs ~5-6 GB of renderer memory. Ownership is tracked so a wrapper notices when the other one used it.
        key = num_envs
        if key not in _INNER:
            _INNER[key] = self._build_inner(ecfg, num_envs)
            _INNER[key]._vr_owner = None
        self.inner = _INNER[key]
        self.n_interrupted = 0
        self.vla = get_vla()
        self.dev = self.inner.device
        self.alpha = self.alpha.to(self.dev)
        sp = self.inner.env.action_space
        self.lo = torch.as_tensor(sp.low[0], device=self.dev).float()
        self.hi = torch.as_tensor(sp.high[0], device=self.dev).float()
        self.total = int(self.inner.env.unwrapped.total_num_trials)
        held, excl = heldout_episodes(self.total, int(vr.get("n_eval", N_EVAL_DEFAULT)))
        self.heldout = torch.tensor(held)
        self.excluded = set(excl)
        self.eval_cursor = 0
        self.gen = torch.Generator().manual_seed(1000003 * self.seed + (7 if self.split == "eval" else 0))
        B = num_envs
        self.expert = ScriptedExpert(B, self.dev)
        self.eids = torch.zeros(B, dtype=torch.long)
        self.dec = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.ep_rep = torch.zeros(B, dtype=torch.long)  # eval noise stream
        self.eval_rep = int(vr.get("eval_noise_rep", 0))
        self._init_metrics()
        self._is_start = True
        self.a_base = None
        self.feat = None
        self.reset_log = []
        self.log_dir = os.environ.get("VR_LOG_DIR")
        self.eval_count = -1
        # T024 PBRS (train split only): r' = r + beta (gamma Phi(s') - Phi(s)), Phi(s')=phi_term on success.
        pb = vr.get("pbrs", None)
        self.pbrs = None
        if pb is not None and str(pb.get("potential", "none")) != "none" and self.split == "train":
            from vla_residual.potential import Potential

            self.pbrs = dict(potential=str(pb.potential), beta=float(pb.beta), gamma=float(pb.gamma),
                             phi_term=float(pb.get("phi_term", 1.0)))
            self.pot = Potential(self.pbrs["potential"], device=self.dev)
            self.phi = torch.zeros(B, device=self.dev)
            self.ended = torch.zeros(B, dtype=torch.bool, device=self.dev)
            self.shp_sum = torch.zeros(B, device=self.dev)
            self.shp_abs = torch.zeros(B, device=self.dev)
            self.phi0 = torch.zeros(B, device=self.dev)
            self.res_abs = torch.zeros(B, device=self.dev)
        self._t0 = __import__("time").time()

    def _log(self, name, rows):
        if not self.log_dir or not rows:
            return
        import json, time

        with open(os.path.join(self.log_dir, name), "a") as f:
            for r in rows:
                r["wall"] = round(time.time(), 1)
                f.write(json.dumps(r) + "\n")

    @staticmethod
    def _build_inner(ecfg, num_envs):
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

        return ManiskillEnv(ecfg, num_envs=num_envs, seed_offset=0, total_num_processes=1, worker_info=None,
                            record_metrics=True)

    # --- properties used by RLinf
    @property
    def num_envs(self):
        return self.inner.num_envs

    @property
    def device(self):
        return self.dev

    @property
    def elapsed_steps(self):
        return self.dec.clone()

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, v):
        self._is_start = v

    @property
    def total_num_group_envs(self):
        return self.total

    def update_reset_state_ids(self):
        pass

    # --- metrics
    def _init_metrics(self):
        B = self.num_envs
        self.success_once = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self._logged = torch.zeros(B, dtype=torch.bool)
        self.returns = torch.zeros(B, device=self.dev)
        self.u_abs_sum = torch.zeros(B, device=self.dev)  # T035: sum over decisions of mean |s(u)| (policy output)

    def _reset_metrics(self, mask):
        self._logged[mask.cpu()] = False
        self.success_once[mask] = False
        self.returns[mask] = 0
        self.u_abs_sum[mask] = 0
        self.dec[mask] = 0

    # --- episode ids
    def _draw_eids(self, n):
        if self.split == "eval":
            idx = (torch.arange(n) + self.eval_cursor) % len(self.heldout)
            self.eval_cursor = int((self.eval_cursor + n) % len(self.heldout))
            return self.heldout[idx]
        assert n % self.group_size == 0 or n < self.group_size
        ng = max(1, n // self.group_size)
        out = []
        while len(out) < ng:
            e = int(torch.randint(0, self.total, (1,), generator=self.gen))
            if e not in self.excluded:
                out.append(e)
        return torch.tensor(out).repeat_interleave(self.group_size)[:n] if n >= self.group_size else torch.tensor(out[:1] * n)

    def _noise(self, idx):
        """pi0.5 initial noise for env slots idx. Eval: deterministic per (episode id, decision, rep) -> paired."""
        shp = self.vla.pol.noise_shape
        if self.split == "eval":
            zs = []
            for i in idx.tolist():
                g = torch.Generator().manual_seed(int(self.eids[i]) * 64 + int(self.dec[i]) + 1_000_003 * self.eval_rep)
                zs.append(torch.randn(*shp, generator=g))
            return torch.stack(zs).to(self.vla.pol.device)
        if self.anchor_crn:  # same noise for every member of a group at a given decision
            cache, zs = {}, []
            for i in idx.tolist():
                k = (int(self.eids[i]), int(self.dec[i]))
                if k not in cache:
                    g = torch.Generator().manual_seed(k[0] * 64 + k[1] + 1_000_003 * (1000 + self.seed))
                    cache[k] = torch.randn(*shp, generator=g)
                zs.append(cache[k])
            return torch.stack(zs).to(self.vla.pol.device)
        return torch.randn(len(idx), *shp, generator=self.gen).to(self.vla.pol.device)

    # --- obs
    def _make_obs(self, wobs, idx=None):
        """Run pi0.5 on wrapped obs (all slots, or subset idx) and refresh base chunk / feature."""
        B = self.num_envs
        if idx is None:
            idx = torch.arange(B)
        sub = C.obs_select(wobs, idx.tolist()) if len(idx) < B else wobs
        acts, f = self.vla.act(sub, self._noise(idx))
        if self.a_base is None:
            self.a_base = torch.zeros(B, C.K, ACT_DIM, device=self.dev)
            self.feat = torch.zeros(B, FEAT_DIM, device=self.dev)
            self.feat_raw = torch.zeros(B, FEAT_DIM, device=self.dev)
            self.qpos = torch.zeros(B, QPOS_DIM, device=self.dev)
        di = idx.to(self.dev)
        self.a_base[di] = acts.to(self.dev)
        self.feat[di] = f.to(self.dev)
        self.feat_raw[di] = self.vla.last_feat_raw.to(self.dev)
        self.qpos[di] = wobs["states"][di].to(self.dev).float() if len(idx) < B else wobs["states"].to(self.dev).float()

    def _obs(self, emask=None):
        B = self.num_envs
        feat = torch.zeros_like(self.feat) if self.zero_vlm else self.feat
        st = torch.cat([feat, self.qpos, (self.a_base / BASE_SCALE.to(self.dev)).reshape(B, -1),
                        (self.dec.float() / self.max_dec)[:, None]], 1)
        out = {"states": st.float()}
        if self.expert_label or self.expert_exec > 0:
            self._a_exp = self.expert(priv_state(self.inner), emask)
            out["expert_u"] = self.residual_label(self._a_exp)
        return out

    def residual_label(self, a_exp):
        """Expert residual in the policy's u-space: s* = clip((a_exp - mean_k a_base) / alpha, -1, 1)."""
        s = ((a_exp - self.a_base.mean(1)) / self.alpha).clamp(-1, 1)
        if self.squash == "tanh":
            return torch.atanh(s.clamp(-0.995, 0.995))
        return s

    # --- gym API
    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        B = self.num_envs
        if "env_idx" in options:
            idx = torch.as_tensor(options["env_idx"]).cpu()
        else:
            idx = torch.arange(B)
            if self.split == "eval":  # every RLinf evaluation starts with an external full reset -> same id order
                self.eval_cursor = 0
                self.eval_count += 1
        e = self._draw_eids(len(idx))
        self.eids[idx] = e
        self.reset_log.append(e.tolist())
        mask = torch.zeros(B, dtype=torch.bool, device=self.dev)
        mask[idx.to(self.dev)] = True
        if len(idx) == B:
            wobs, infos = self.inner.reset(seed=self.seed, options={"episode_id": self.eids.to(self.dev)})
            self.inner._vr_owner = id(self)
        else:
            wobs, infos = self.inner.reset(options={"env_idx": idx.to(self.dev), "episode_id": e.to(self.dev)})
        self._reset_metrics(mask)
        self.expert.reset(mask)
        self._make_obs(wobs, None if len(idx) == B else idx)
        if self.pbrs is not None:
            phi = self.potential()
            self.phi = torch.where(mask, phi, self.phi)
            self.phi0 = torch.where(mask, phi, self.phi0)
            self.ended[mask] = False
            for t in (self.shp_sum, self.shp_abs, self.res_abs):
                t[mask] = 0
        self._obs_cache = self._obs(mask)
        return {k: v.clone() for k, v in self._obs_cache.items()}, {}

    def _sim_step(self, a, render):
        base = self.inner.env
        if render or not self.render_skip:
            return self.inner.step(a, auto_reset=False)
        ub = base.unwrapped
        mode = ub._obs_mode
        ub._obs_mode = "none"
        try:
            o, r, te, tr, info = base.step(a)
        finally:
            ub._obs_mode = mode
        return None, r, te, tr, info

    def chunk_step(self, chunk_actions):
        chunk_actions = torch.as_tensor(chunk_actions)
        if self.inner._vr_owner != id(self):
            return self._interrupted()
        u = chunk_actions[:, 0] if chunk_actions.dim() == 3 else chunk_actions
        u = u.to(self.dev).float()
        s = torch.tanh(u) if self.squash == "tanh" else u.clamp(-1, 1)
        delta = self.alpha * s
        if self.expert_exec > 0:  # validation: hand-coded residual (expert - base, clipped to alpha)
            delta = self.alpha * ((self._a_exp - self.a_base.mean(1)) / self.alpha).clamp(-1, 1)
        if self.vr.get("expert_only", False):  # validation: expert alone (no base)
            delta = None
        B = self.num_envs
        if self.anchor_n_base:
            bm = self.base_member.to(self.dev)
            self._u_abs = torch.where(self._ep_live(), s.abs().mean(1), torch.zeros(B, device=self.dev))
            self.u_abs_sum += self._u_abs
            delta = torch.where(bm[:, None], torch.zeros_like(delta), delta)
        self.last_a_exec = torch.zeros(B, C.K, ACT_DIM, device=self.dev)
        succ_any = torch.zeros(B, dtype=torch.bool, device=self.dev)
        wobs = info = None
        for k in range(C.K):
            if delta is None:
                a = self._a_exp.clamp(self.lo, self.hi)
            else:
                a = (self.a_base[:, k] + delta).clamp(self.lo, self.hi)
            self.last_a_exec[:, k] = a
            wobs, _, _, _, info = self._sim_step(a, render=(k == C.K - 1))
            succ_any |= info["success"].to(self.dev).bool()
        succ_end = info["success"].to(self.dev).bool()
        self.dec += 1
        newly = succ_any & ~self.success_once
        reward = newly.float()
        self.success_once |= succ_any
        self.returns += reward
        if self.ignore_terminations:
            term = torch.zeros(B, dtype=torch.bool, device=self.dev)
        else:
            term = newly.clone()
        trunc = (self.dec >= self.max_dec) & ~term
        dones = term | trunc
        # base chunk + features for the post-step obs of every slot (before shaping: Phi(s') needs them)
        self._make_obs(wobs)
        if self.pbrs is not None:
            reward, shp, rinfo = self._shape(reward, term, dones, s)
        infos = {"success": succ_end, "episode": {
            "success_once": self.success_once.clone(), "success_at_end": succ_end.clone(),
            "return": self.returns.clone(), "episode_len": (self.dec * C.K).clone(),
            "reward": self.returns / (self.dec * C.K).clamp(min=1)}}
        fin = torch.nonzero(dones).flatten().tolist()
        if self.split == "eval":
            self._log("eval_episodes.jsonl", [dict(eval=self.eval_count, eid=int(self.eids[i]),
                                                   success_once=bool(self.success_once[i]),
                                                   success_at_end=bool(succ_end[i])) for i in fin])
        else:
            ext = (lambda i: {}) if self.pbrs is None else (lambda i: dict(
                shp_sum=round(float(self.shp_sum[i]), 5), shp_abs=round(float(self.shp_abs[i]), 5),
                phi0=round(float(self.phi0[i]), 5), phi_prev_end=round(float(rinfo["phi_prev"][i]), 5),
                phi_next_end=round(float(rinfo["phi_next"][i]), 5), r_end=round(float(reward[i]), 5),
                term=bool(term[i]), res_abs=round(float(self.res_abs[i] / max(int(self.dec[i]), 1)), 5)))
            anc = (lambda i: {}) if not self.anchor_n_base else (lambda i: dict(
                base=bool(self.base_member[i]), slot=int(i),
                u_abs=round(float(self.u_abs_sum[i] / max(int(self.dec[i]), 1)), 5)))
            self._log("train_episodes.jsonl", [dict(eid=int(self.eids[i]), success_once=bool(self.success_once[i]),
                                                    dec=int(self.dec[i]), **ext(i), **anc(i))
                                               for i in fin if not self._logged[i]])
            for i in fin:
                self._logged[i] = True
        obs = self._obs()
        if dones.any() and self.auto_reset:
            final_obs = {k: v.clone() for k, v in obs.items()}
            final_info = {"success": succ_end.clone(), "episode": {k: v.clone() for k, v in infos["episode"].items()}}
            idx = torch.nonzero(dones).flatten().cpu()
            obs, _ = self.reset(options={"env_idx": idx})
            infos = dict(infos)
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
        self._obs_cache = {k: v.clone() for k, v in obs.items()}
        rew = reward[:, None]
        return [obs], rew, term[:, None], trunc[:, None], [infos]

    def _ep_live(self):
        """Slots whose current episode has not finished yet (auto_reset=False keeps stepping finished slots)."""
        return ~self._logged.to(self.dev)

    def potential(self):
        return self.pot(feat_raw=self.feat_raw, qpos=self.qpos, dec=self.dec)

    def _shape(self, reward, term, dones, s):
        """Called after the sim step and _make_obs(post-step obs), before auto-reset. Steps after an episode ended
        (auto_reset=False tail) get no shaping. beta == 0 returns the sparse reward object unchanged."""
        from vla_residual.potential import shaped_reward

        pb = self.pbrs
        active = ~self.ended
        phi_prev = self.phi.clone()
        phi_next = self.potential()
        r2, shp = shaped_reward(reward, phi_prev, phi_next, term, pb["beta"], pb["gamma"], pb["phi_term"], active)
        if pb["beta"] == 0.0:
            r2 = reward
        self.shp_sum += shp
        self.shp_abs += shp.abs()
        self.res_abs += torch.where(active, s.abs().mean(1), torch.zeros_like(shp))
        self.ended |= dones
        self.phi = phi_next
        self._last_rinfo = dict(phi_prev=phi_prev, phi_next=phi_next, sparse=reward, shaped=r2, shp=shp, active=active,
                                term=term.clone(), trunc=(dones & ~term).clone(), dec=self.dec.clone())
        return r2, shp, dict(phi_prev=phi_prev, phi_next=phi_next, sparse=reward, shp=shp, active=active)

    def _interrupted(self):
        """The shared scene was used by the other wrapper (evaluation) since our last step: the in-flight episodes are
        gone. Truncate all of them (reward 0, final obs = last obs we returned) and start new episodes."""
        B = self.num_envs
        self.n_interrupted += 1
        last = self._obs_cache
        trunc = torch.ones(B, dtype=torch.bool, device=self.dev)
        term = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self._log("train_interrupted.jsonl", [dict(n=self.n_interrupted, eids=self.eids.tolist(),
                                                     dec=self.dec.tolist())])
        self._logged[:] = True  # interrupted episodes are not counted as finished train episodes
        infos = {"success": torch.zeros(B, dtype=torch.bool, device=self.dev), "episode": {
            "success_once": self.success_once.clone(), "success_at_end": torch.zeros(B, dtype=torch.bool, device=self.dev),
            "return": self.returns.clone(), "episode_len": (self.dec * C.K).clone(),
            "reward": self.returns / (self.dec * C.K).clamp(min=1)}}
        obs, _ = self.reset()
        if self.auto_reset:
            infos["final_observation"] = last
            infos["final_info"] = {"success": infos["success"].clone(), "episode": dict(infos["episode"])}
            infos["_final_info"] = trunc
            infos["_final_observation"] = trunc
            infos["_elapsed_steps"] = trunc
        return [obs], torch.zeros(B, 1, device=self.dev), term[:, None], trunc[:, None], [infos]

    def step(self, actions=None, auto_reset=True):
        actions = torch.as_tensor(actions)
        o, r, te, tr, inf = self.chunk_step(actions[:, None] if actions.dim() == 2 else actions)
        return o[0], r[:, 0], te[:, 0], tr[:, 0], inf[0]

    # --- misc hooks some RLinf paths touch
    def render(self, info, rew=None):
        return None

    def offload(self):
        pass

    def onload(self):
        pass
