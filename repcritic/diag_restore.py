"""Diagnose snapshot/restore fidelity: per-step, per-joint divergence and image diff after restore."""
from __future__ import annotations

import json
import sys

from repcritic import rc_common as C

C.setup_env_vars()

import torch  # noqa: E402

from repcritic.oracle_diversity import env_probe  # noqa: E402


def main():
    eid = int(sys.argv[1]) if len(sys.argv) > 1 else 266722
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    pol = C.Policy(C.build_policy())
    env = C.build_env(E)
    base = env.env.unwrapped
    print("controller", base.agent.controller.controllers if hasattr(base.agent.controller, "controllers") else base.agent.controller)
    for uid, c in base.agent.controller.controllers.items():
        print(uid, type(c).__name__, "use_target", getattr(c.config, "use_target", None),
              "interpolate", getattr(c.config, "interpolate", None), "use_delta", getattr(c.config, "use_delta", None))
    obs, _ = env.reset(seed=0, options={"episode_id": torch.full((E,), eid, device=env.device)})
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        a, _ = pol.act(C.obs_select(obs, [0]), pol.noise(1, g))
        for k in range(C.K):
            obs, *_ = env.step(a.expand(E, -1, -1)[:, k], auto_reset=False)
    snap = C.snapshot(env, 0)
    sd_before = base.get_state_dict()
    img_before = obs["main_images"].float().cpu()
    g2 = torch.Generator().manual_seed(1)
    seq = torch.cat([pol.act(C.obs_select(obs, [0]), pol.noise(1, g2))[0][0] for _ in range(4)], 0)

    def play():
        qs = []
        for k in range(seq.shape[0]):
            env.step(seq[k : k + 1].expand(E, -1), auto_reset=False)
            qs.append(env_probe(env)[0])
        return torch.stack(qs)  # [T,E,J]

    qn = play()
    obs_r = C.restore_all(env, snap)
    sd_after = base.get_state_dict()

    def cmp(a, b, path=""):
        if isinstance(a, dict):
            for k in a:
                cmp(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, torch.Tensor):
            d = (a[0:1].float() - b[0:1].float()).abs().max().item()
            if d > 0:
                print("state diff env0", path, d)

    cmp(sd_before, sd_after)
    im = obs_r["main_images"].float().cpu()
    dimg = (im[0] - img_before[0]).abs()
    print("img after restore vs before (env0): max", dimg.max().item(), "mean", dimg.mean().item(),
          "frac>10", (dimg > 10).float().mean().item())
    q1 = play()
    C.restore_all(env, snap)
    q2 = play()
    torch.set_printoptions(precision=4, sci_mode=True, linewidth=200)
    print("per-step max|qn-q1| env0 per joint:")
    print((qn[:, 0] - q1[:, 0]).abs())
    print("per-step max|q1-q2| env0:", (q1[:, 0] - q2[:, 0]).abs().max(1).values)
    print("per-step cross-env |q1[e]-q1[0]| max:", (q1 - q1[:, 0:1]).abs().amax((1, 2)))
    print("qpos traj env0 (natural) last step", qn[-1, 0])


if __name__ == "__main__":
    main()
