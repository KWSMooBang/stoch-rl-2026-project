"""T040 memory probe without the ManiSkill scene: frozen pi0.5 (VLM) + trainable action expert, observations built
from T017 logged frames (same wrapped-obs keys as ManiskillEnv). Peak = torch max_memory_allocated of this process.
  CUDA_VISIBLE_DEVICES=1 python -m qvgm.memprobe --out /workspace/stochrl_t040/closed_loop
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from repcritic import rc_common as C
from qvgm.eval_qascent import memprobe_model, smi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/stochrl_t040/closed_loop")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    C.setup_env_vars()
    pol = C.Policy(C.build_policy())
    m = pol.m
    load_alloc = torch.cuda.memory_allocated()
    T = Path("/workspace/stochrl_t017")
    d = [json.loads(l) for l in open(T / "decisions_0.jsonl")][:8]
    imgs, sts, txt = [], [], []
    for r in d:
        e = np.load(T / f"episodes/ep{r['eid']:06d}.npz")
        imgs.append(e["rgb"][r["decision"]]); sts.append(e["state"][r["decision"]]); txt.append(r["language_goal"])
    obs = dict(main_images=torch.as_tensor(np.stack(imgs)), states=torch.as_tensor(np.stack(sts)),
               task_descriptions=txt, wrist_images=None)
    observation = m.env_obs_to_observation(obs)
    memprobe_model(m, observation, Path(a.out), dict(scene_loaded=False, policy_alloc_GB=load_alloc / 2 ** 30,
                                                     smi_after_load=smi()))


if __name__ == "__main__":
    main()
