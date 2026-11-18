"""E02 shared pieces: frozen pi0.5 feature recorder, privileged labels, batched snapshots.

Feature recorder (inference only, hooks observe and never modify outputs):
  - VLM prefix (PaliGemma gemma_2b, width 2048) at llm.layers.{VLM_LAYERS}: candidate-INDEPENDENT z(o,l).
      pools: img_mean [2048], img_grid4 [16,2048] (16x16 SigLIP patch grid avg-pooled to 4x4, row-major),
             txt_mean [2048] (valid prompt tokens), txt_last [2048] (last valid prompt token).
  - SigLIP (So400m, width 1152) at img.encoder.layers.{SIGLIP_LAYERS}, valid camera only:
      sig_mean [1152], sig_grid4 [16,1152]. Candidate-INDEPENDENT.
  - Action expert (gemma_300m, width 1024) at llm.layers.{AE_LAYERS} (suffix stream):
      * ae_ctx: one extra suffix pass with x_t = 0 (mean of the initial-noise distribution), t = 1.
        Deterministic in (o,l) -> candidate-INDEPENDENT AE context. Not part of the sampling path
        (KV cache is read-only); action equality with/without the extra pass is tested.
        ae_ctx_mean [L,1024] (mean over H=8 action tokens), ae_ctx_tok_last [8,1024] (last AE layer).
      * ae_cand: hidden states of the actual Euler denoise passes for the sampled candidate
        (depends on the candidate's initial noise) -> candidate-DEPENDENT z(o,l,a).
        ae_cand_mean [4 steps, L, 1024], ae_cand_tok_last [8,1024] (last step, last AE layer).
All features are stored fp16.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from repcritic import rc_common as C

# gemma_2b / gemma_300m have 18 layers (0..17); SigLIP So400m has 27 (0..26).
VLM_LAYERS = (4, 9, 13, 17)
AE_LAYERS = (4, 9, 13, 17)
SIGLIP_LAYERS = (8, 17, 26)
N_IMG_TOK = 256
GRID = 16


def grid4(x):
    """x [B,256,D] row-major 16x16 patches -> [B,16,D] (4x4 avg pool)."""
    B, N, D = x.shape
    g = x.reshape(B, GRID, GRID, D).reshape(B, 4, 4, 4, 4, D)  # [B, gy, py, gx, px, D]
    return g.mean(dim=(2, 4)).reshape(B, 16, D)


class FeatPolicy(C.Policy):
    """C.Policy + optional feature capture. When record=False it is exactly C.Policy.act."""

    def __init__(self, model):
        super().__init__(model)
        m = self.m
        self.n_siglip = len(m.img.encoder.layers)
        self.n_llm = len(m.llm.layers)
        assert max(SIGLIP_LAYERS) < self.n_siglip and max(VLM_LAYERS) < self.n_llm
        self.record = False
        self.ctx_pass = True
        self._buf = None
        self._mode = None  # "prefix" | "ctx" | int denoise step
        self._step = 0
        self._handles = []
        for i in sorted(set(VLM_LAYERS) | set(AE_LAYERS)):
            self._handles.append(m.llm.layers[i].register_forward_hook(self._llm_hook(i)))
        for i in SIGLIP_LAYERS:
            self._handles.append(m.img.encoder.layers[i].register_forward_hook(self._sig_hook(i)))
        self._orig_bpc = m.build_prefix_cache
        self._orig_rs = m.run_suffix
        m.build_prefix_cache = self._bpc
        m.run_suffix = self._rs

    # ---- hooks (observe only; return None) ----
    def _llm_hook(self, i):
        def hook(mod, inp, out):
            if not self.record:
                return None
            xs = out[0]
            if xs[0] is not None and i in VLM_LAYERS:
                self._buf["vlm_raw"][i] = xs[0].detach()
            if xs[1] is not None and i in AE_LAYERS:
                h = xs[1][:, -C.H:].detach().float()
                self._buf["ae"].setdefault(self._mode, {})[i] = h
            return None
        return hook

    def _sig_hook(self, i):
        def hook(mod, inp, out):
            if not self.record:
                return None
            x = out
            while isinstance(x, (tuple, list)):
                x = x[0]
            self._buf["sig_raw"].setdefault(i, []).append(x.detach())
            return None
        return hook

    def _bpc(self, observation):
        if self.record:
            self._mode = "prefix"
        out = self._orig_bpc(observation)
        if self.record:
            self._buf["prefix_mask"] = out[1].detach()
            self._buf["prompt_mask"] = observation.tokenized_prompt_mask.detach()
            if self.ctx_pass:
                B = out[1].shape[0]
                self._mode = "ctx"
                x0 = torch.zeros(B, self.m.action_horizon, self.m.action_dim, device=out[1].device)
                t1 = torch.ones(B, device=out[1].device, dtype=torch.float32)
                self._orig_rs(observation, x0, t1, out[2], out[1])
            self._step = 0
        return out

    def _rs(self, observation, x_t, t_tensor, kv_cache, prefix_mask):
        if self.record:
            self._mode = self._step
            self._step += 1
        return self._orig_rs(observation, x_t, t_tensor, kv_cache, prefix_mask)

    # ---- pooling ----
    def _pool(self):
        b = self._buf
        pm = b["prefix_mask"]  # [B, 968]
        B = pm.shape[0]
        n_img_slots = (pm.shape[1] - b["prompt_mask"].shape[1]) // N_IMG_TOK
        slot_valid = pm[:, : n_img_slots * N_IMG_TOK].reshape(B, n_img_slots, N_IMG_TOK).all(-1)  # [B,slots]
        assert bool((slot_valid.sum(1) == 1).all()), slot_valid
        slot = int(slot_valid[0].float().argmax())
        assert bool(slot_valid[:, slot].all())
        tm = b["prompt_mask"].float()  # [B, T]
        T0 = n_img_slots * N_IMG_TOK
        last_idx = (tm.cumsum(1) * tm).argmax(1)  # index of last valid prompt token
        f = {}
        for i in VLM_LAYERS:
            x = b["vlm_raw"][i].float()
            img = x[:, slot * N_IMG_TOK:(slot + 1) * N_IMG_TOK]
            txt = x[:, T0:]
            f[f"vlm{i}_img_mean"] = img.mean(1)
            f[f"vlm{i}_img_grid4"] = grid4(img)
            f[f"vlm{i}_txt_mean"] = (txt * tm[..., None]).sum(1) / tm.sum(1, keepdim=True)
            f[f"vlm{i}_txt_last"] = txt[torch.arange(B), last_idx]
        for i in SIGLIP_LAYERS:
            x = b["sig_raw"][i][slot].float()
            f[f"sig{i}_mean"] = x.mean(1)
            f[f"sig{i}_grid4"] = grid4(x)
        ae = b["ae"]
        La = sorted(AE_LAYERS)
        if "ctx" in ae:
            f["ae_ctx_mean"] = torch.stack([ae["ctx"][i].mean(1) for i in La], 1)  # [B,L,1024]
            f["ae_ctx_tok_last"] = ae["ctx"][La[-1]]  # [B,8,1024]
        steps = sorted(k for k in ae if isinstance(k, int))
        f["ae_cand_mean"] = torch.stack([torch.stack([ae[s][i].mean(1) for i in La], 1) for s in steps], 1)
        f["ae_cand_tok_last"] = ae[steps[-1]][La[-1]]
        return {k: v.to(torch.float16).cpu() for k, v in f.items()}, slot

    def act_feats(self, obs, noise, full_h=True):
        """Like act(full_h) but also returns pooled features (dict of fp16 [B,...] CPU tensors)."""
        from rlinf.models.embodiment.openpi.transforms.apply import apply_output_transform
        import dataclasses
        from torch.utils._pytree import tree_map

        feats = []
        with torch.no_grad():
            observation = self.m.env_obs_to_observation(obs)
            B = observation.state.shape[0]
            mb = int(os.environ.get("RC_POLICY_MB", "0")) or B
            outs = []
            for s in range(0, B, mb):
                sl = slice(s, min(B, s + mb))
                sub = dataclasses.replace(observation, **{
                    fl.name: tree_map(lambda x: x[sl] if isinstance(x, torch.Tensor) else x,
                                      getattr(observation, fl.name))
                    for fl in dataclasses.fields(observation)})
                self._buf = {"vlm_raw": {}, "sig_raw": {}, "ae": {}}
                self.record = True
                try:
                    outs.append(self.m.sample_actions(sub, num_steps=self.m.num_steps, noise=noise[sl]))
                finally:
                    self.record = False
                fe, slot = self._pool()
                feats.append(fe)
                self._buf = None
            model_actions = torch.cat(outs, 0)
            out = apply_output_transform(
                self.m._output_transform_fn,
                {"actions": model_actions, "state": observation.state},
                action_chunk=None,
            )["actions"].to(torch.float32)
        acts = out if full_h else out[:, : C.K]
        F = {k: torch.cat([f[k] for f in feats], 0) for k in feats[0]}
        F["_img_slot"] = slot
        return acts, model_actions, F


# ---------------- privileged labels (LABEL-ONLY, never policy/critic input) ----------------
def privileged(env):
    """Per-env privileged sim labels at the current state (read-only queries; does NOT call evaluate(),
    which would advance the grasp counter). Returns dict of np arrays [E,...]."""
    base = env.env.unwrapped
    es = base.episode_stats
    E = base.num_envs
    dev = base.device
    sel_c = base.select_carrot_ids
    plate = base.objs_plate[base.plate_names[0]]
    assert len(base.plate_names) == 1
    cp = torch.zeros(E, 3, device=dev); cq = torch.zeros(E, 4, device=dev)
    grasp_now = torch.zeros(E, dtype=torch.bool, device=dev)
    f1 = torch.zeros(E, device=dev); f2 = torch.zeros(E, device=dev); fcp = torch.zeros(E, device=dev)
    for idx, name in enumerate(base.model_db_carrot):
        is_sel = sel_c == idx
        if not bool(is_sel.any()):
            continue
        obj = base.objs_carrot[name]
        cp = torch.where(is_sel[:, None], obj.pose.p, cp)
        cq = torch.where(is_sel[:, None], obj.pose.q, cq)
        grasp_now = torch.where(is_sel, base.agent.is_grasping(obj), grasp_now)
        a = torch.linalg.norm(base.scene.get_pairwise_contact_forces(base.agent.finger1_link, obj), dim=-1)
        b = torch.linalg.norm(base.scene.get_pairwise_contact_forces(base.agent.finger2_link, obj), dim=-1)
        c = torch.linalg.norm(base.scene.get_pairwise_contact_forces(obj, plate), dim=-1)
        f1 = torch.where(is_sel, a, f1); f2 = torch.where(is_sel, b, f2); fcp = torch.where(is_sel, c, fcp)
    pp = plate.pose.p; pq = plate.pose.q
    gp = (base.agent.finger1_link.pose.p + base.agent.finger2_link.pose.p) / 2
    gq = (base.agent.finger1_link.pose.q + base.agent.finger2_link.pose.q) / 2
    n = lambda t: t.float().cpu().numpy()  # noqa: E731
    out = dict(pos_carrot=n(cp), q_carrot=n(cq), pos_plate=n(pp), q_plate=n(pq), pos_gripper=n(gp), q_gripper=n(gq),
               gripper_carrot_dist=n(torch.linalg.norm(gp - cp, dim=1)),
               gripper_plate_dist=n(torch.linalg.norm(gp - pp, dim=1)),
               carrot_plate_dist=n(torch.linalg.norm(cp - pp, dim=1)),
               carrot_plate_rel=n(cp - pp), gripper_carrot_rel=n(cp - gp))
    out["ever_grasped"] = es["is_src_obj_grasped"].cpu().numpy()
    out["consecutive_grasp_flag"] = es["consecutive_grasp"].cpu().numpy()  # ever had >=5 consecutive grasp steps
    out["consec_grasp_count"] = base.consecutive_grasp.cpu().numpy().astype(np.int32)
    out["src_on_target"] = es["src_on_target"].cpu().numpy()  # = success at the last evaluate()
    out["grasping_now"] = grasp_now.cpu().numpy()
    out["finger1_obj_force"] = n(f1)
    out["finger2_obj_force"] = n(f2)
    out["obj_plate_force"] = n(fcp)
    out["contact_any_finger"] = ((f1 > 1e-3) | (f2 > 1e-3)).cpu().numpy()
    out["carrot_id"] = sel_c.cpu().numpy().astype(np.int32)
    out["overlay_id"] = base.select_overlay_ids.cpu().numpy().astype(np.int32)
    return out


def phase_from_priv(p):
    """Heuristic phase label (same priority order as T016 + contact):
    placed > transport (grasping now) > grasp_contact (finger contact, not grasped) > post_grasp_lost > approach."""
    n = len(p["src_on_target"])
    ph = np.empty(n, dtype=object)
    for i in range(n):
        if p["src_on_target"][i]:
            ph[i] = "placed"
        elif p["grasping_now"][i]:
            ph[i] = "transport"
        elif p["contact_any_finger"][i]:
            ph[i] = "grasp_contact"
        elif p["ever_grasped"][i]:
            ph[i] = "post_grasp_lost"
        else:
            ph[i] = "approach"
    return ph


PHASES = ["approach", "grasp_contact", "transport", "post_grasp_lost", "placed"]


# ---------------- batched snapshots ----------------
def snapshot_batch(env):
    """Snapshot every sub-env at once (CPU). Returns list of single-env snaps compatible with C.restore_all."""
    base = env.env.unwrapped
    sd = C._tree(lambda t: t.detach().cpu().clone(), base.get_state_dict())
    task = {a: getattr(base, a).detach().cpu().clone() for a in C.TASK_STATE_ATTRS}
    stats = C._tree(lambda t: t.detach().cpu().clone(), dict(base.episode_stats))
    el = base._elapsed_steps.detach().cpu().clone()
    pr = env.prev_step_reward.detach().cpu().clone()
    snaps = []
    for i in range(base.num_envs):
        sl = lambda t: t[i:i + 1].clone()  # noqa: E731
        snaps.append({"sim": C._tree(sl, sd), "task": {a: sl(v) for a, v in task.items()},
                      "episode_stats": C._tree(sl, stats), "elapsed": sl(el), "prev_rew": sl(pr)})
    return snaps


def snap_to_device(snap, device):
    return C._tree(lambda t: t.to(device), snap)
