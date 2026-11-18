import torch
import torch.nn as nn

from .env import OBS_DIM, N_OBJ, N_TGT, N_ACT


class QNet(nn.Module):
    """Goal-conditioned Q(s, ., g).  g = (object id, target id).

    enc='concat': [s, onehot(o), onehot(t)] -> MLP.  (design doc default)
    enc='film'  : object and target each FiLM-modulate both hidden layers through separate
                  embedding tables, combined multiplicatively: h <- h*(1+γ_o)*(1+γ_t) + β_o + β_t.
                  No joint (o,t) parameters.
    """

    def __init__(self, enc: str = "concat", hidden: int = 256):
        super().__init__()
        self.enc = enc
        if enc == "concat":
            self.l1 = nn.Linear(OBS_DIM + N_OBJ + N_TGT, hidden)
        elif enc == "film":
            self.l1 = nn.Linear(OBS_DIM, hidden)
            self.fo = nn.Embedding(N_OBJ, 4 * hidden)
            self.ft = nn.Embedding(N_TGT, 4 * hidden)
            nn.init.normal_(self.fo.weight, std=0.1)
            nn.init.normal_(self.ft.weight, std=0.1)
        else:
            raise ValueError(enc)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, N_ACT)
        self.hidden = hidden

    def forward(self, obs, go, gt):
        if self.enc == "concat":
            g = torch.cat([nn.functional.one_hot(go, N_OBJ), nn.functional.one_hot(gt, N_TGT)], -1).float()
            h = torch.relu(self.l1(torch.cat([obs, g], -1)))
            h = torch.relu(self.l2(h))
        else:
            H = self.hidden
            mo = self.fo(go).view(-1, 4, H)
            mt = self.ft(gt).view(-1, 4, H)
            h = self.l1(obs) * (1 + mo[:, 0]) * (1 + mt[:, 0]) + mo[:, 1] + mt[:, 1]
            h = torch.relu(h)
            h = self.l2(h) * (1 + mo[:, 2]) * (1 + mt[:, 2]) + mo[:, 3] + mt[:, 3]
            h = torch.relu(h)
        return self.out(h)
