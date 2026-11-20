"""T036 G2: learned competence gate head on RLinf's MLPPolicy (built from vla_residual.rlinf_ext get_model).

Policy action = [u (7) | z (1)], env executes alpha * sigmoid(z) * tanh(u):
    u-mean = E2 MLP(states[:, :n])                    n = cfg.obs_dim - 1 (= 2092, the E2 observation)
    z-mean = S * (p_v * V_hat + p_t * t/16 + p_b)     V_hat = states[:, n], t/16 = states[:, n - 1]
The stored parameter is p (coefficients / S). The forward value is S * p with the gradient of p (straight-through),
so Adam at the shared lr moves the logit coefficients by ~S * lr per step, and the gate adds an unscaled gradient
to the global-norm clip. Log-std: the E2 value for u, vr_gate.init_logstd for z (state-free, learnable).
Only `backbone` / `actor_mean` are wrapped, so every MLPPolicy forward path (sampling, eval mean, log-prob) is the
stock code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vla_residual.gate import linear_init


class GateBackbone(nn.Module):
    def __init__(self, mlp, n):
        super().__init__()
        self.mlp = mlp
        self.n = int(n)

    def forward(self, states):
        n = self.n
        return torch.cat([self.mlp(states[:, :n]), states[:, n:n + 1], states[:, n - 1:n]], 1)


class GateMean(nn.Module):
    def __init__(self, lin, w, scale):
        super().__init__()
        self.lin = lin
        self.scale = float(scale)
        self.p = nn.Parameter(torch.tensor(list(w), dtype=torch.float32) / self.scale)  # (v, t, bias) / S

    def coef(self):
        """Logit coefficients (w_v, w_t, b): value S * p, gradient d/dp = 1."""
        return self.p + (self.p * self.scale - self.p).detach()

    def forward(self, feat):
        c = self.coef()
        z = feat[:, -2] * c[0] + feat[:, -1] * c[1] + c[2]
        return torch.cat([self.lin(feat[:, :-2].contiguous()), z[:, None]], 1)  # contiguous: bitwise = plain MLP


def build_plain(cfg, orig_get, torch_dtype):
    """The E2 MLPPolicy (obs_dim - 1, action_dim - 1); vr_init is applied to it by the caller before finish()."""
    assert int(cfg.num_action_chunks) == 1 and not cfg.get("add_value_head", False)
    sub = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    sub.obs_dim, sub.action_dim = int(cfg.obs_dim) - 1, int(cfg.action_dim) - 1
    del sub["vr_gate"]
    model = orig_get(sub, torch_dtype)
    assert model.independent_std
    return model


def finish(cfg, model):
    gc = cfg.vr_gate
    with torch.no_grad():
        ls = torch.cat([model.actor_logstd.data, torch.full((1, 1), float(gc.init_logstd))], 1)
    model.backbone = GateBackbone(model.backbone, int(cfg.obs_dim) - 1)
    model.actor_mean = GateMean(model.actor_mean, linear_init(float(gc.tau), float(gc.temp)), float(gc.scale))
    model.actor_logstd = nn.Parameter(ls)
    model.obs_dim = model.critic_obs_dim = int(cfg.obs_dim)
    model.action_dim = int(cfg.action_dim)
    return model


def gate_coef(state_dict):
    """(w_v, w_t, b) from a checkpoint state dict; needs the scale -> stored alongside by make_metrics_t036."""
    return state_dict["actor_mean.p"]
