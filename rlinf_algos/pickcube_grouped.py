"""T021: PickCube variant whose initial state is a deterministic function of `episode_id`.

Why: RLinf GRPO groups `group_size` envs by giving them the same `reset_state_ids` and
passing them as `options["episode_id"]` (rlinf/envs/sim/maniskill/maniskill_env.py:284, 383).
Stock `PickCube-v1` ignores `episode_id` and draws cube/goal poses from the global
`torch.rand`, so a "group" on stock PickCube contains G *different* initial states and the
group-relative advantage degenerates to a batch-normalised REINFORCE signal.

This variant draws the same distribution as PickCube-v1 (cube xy, yaw, goal xyz) but from a
per-env generator seeded by `episode_id` when provided (falls back to stock behaviour
otherwise). Use with env `init_params.id=PickCubeGrouped-v1` and
`env.train.use_fixed_reset_state_ids=True`.

Registration: RLinf auto-imports every module in rlinf/envs/sim/maniskill/tasks/ inside the
env worker, so to use it in RLinf training, copy (or symlink) this file there. T021 did not
patch the submodule (scope reduced to smoke); see `python -m rlinf_algos.pickcube_grouped`
for a standalone check.
"""

import math

import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeGrouped-v1", max_episode_steps=50)
class PickCubeGroupedEnv(PickCubeEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        ep = options.get("episode_id") if options else None
        if ep is None:
            return
        ep = torch.as_tensor(ep).reshape(-1).cpu()
        b = len(env_idx)
        if len(ep) != b:  # RLinf passes ids aligned with env_idx; full reset passes all envs
            ep = ep[env_idx.cpu()] if len(ep) == self.num_envs else ep[:b]
        u = torch.empty((b, 6))
        for i, e in enumerate(ep.tolist()):
            g = torch.Generator().manual_seed(int(e))
            u[i] = torch.rand(6, generator=g)
        with torch.device(self.device):
            u = u.to(self.device)
            h = self.cube_spawn_half_size
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = u[:, 0:2] * h * 2 - h
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            yaw = u[:, 2] * 2 * math.pi
            qs = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            goal = torch.zeros((b, 3))
            goal[:, :2] = u[:, 3:5] * h * 2 - h
            goal[:, 0] += self.cube_spawn_center[0]
            goal[:, 1] += self.cube_spawn_center[1]
            goal[:, 2] = u[:, 5] * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal))


if __name__ == "__main__":
    import gymnasium as gym

    G, n = 4, 16
    env = gym.make("PickCubeGrouped-v1", num_envs=n, obs_mode="state", sim_backend="gpu")
    ids = torch.randint(0, 10_000, (n // G,)).repeat_interleave(G)
    env.reset(seed=0, options={"episode_id": ids})
    u = env.unwrapped
    cube, goal = u.cube.pose.p.cpu().view(n // G, G, 3), u.goal_site.pose.p.cpu().view(n // G, G, 3)
    within = (cube - cube[:, :1]).abs().max().item(), (goal - goal[:, :1]).abs().max().item()
    across = (cube[:, 0] - cube[:1, 0]).abs().max().item()
    # partial reset of env 1 must restore the same group state
    env.unwrapped.reset(options={"env_idx": torch.tensor([1]), "episode_id": ids[1:2]})
    again = (u.cube.pose.p.cpu()[1] - cube.view(n, 3)[1]).abs().max().item()
    print(f"within-group max|d| cube={within[0]:.2e} goal={within[1]:.2e}  across-group max|d|={across:.3f}  partial-reset |d|={again:.2e}")
    assert within[0] < 1e-5 and within[1] < 1e-5 and across > 1e-3 and again < 1e-5
    print("OK")
