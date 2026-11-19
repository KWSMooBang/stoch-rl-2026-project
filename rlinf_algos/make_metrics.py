"""T021: build $RUN_DIR/metrics.json from RLinf tensorboard logs.

Eval success curve vs (epoch, env steps, wall-clock), final success, train time,
peak VRAM (whole-GPU memory.used sampled every 5 s minus baseline), and the
hydra overrides used (diff vs the RLinf default config).
"""

import argparse
import glob
import json
import os

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--rc", type=int, default=0)
    a = ap.parse_args()

    ov = read(os.path.join(a.run_dir, "t021_overrides.txt"), "").splitlines()[3:]
    ov_d = dict(x.split("=", 1) for x in ov if "=" in x)
    base = OmegaConf.load(a.cfg)

    def cfg_get(key, default=None):
        if key in ov_d:
            return type(default)(ov_d[key]) if default is not None else ov_d[key]
        v = OmegaConf.select(base, key)
        return default if v is None else v

    n_env = int(cfg_get("env.train.total_num_envs", 1))
    steps_per_epoch = int(cfg_get("env.train.max_steps_per_rollout_epoch", 1))
    rollout_epoch = int(cfg_get("env.train.rollout_epoch", 1))
    env_steps_per_epoch = n_env * steps_per_epoch * rollout_epoch

    t0 = float(read(os.path.join(a.run_dir, "t_start"), "0"))
    t1 = float(read(os.path.join(a.run_dir, "t_end"), "0"))
    curve, train_curve = [], []
    ev = glob.glob(os.path.join(a.run_dir, "rlinf_logs", "**", "events*"), recursive=True)
    if ev:
        ea = EventAccumulator(ev[0], size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        for tag, out in (("eval/success_once", curve), ("env/success_once", train_curve)):
            if tag not in tags:
                continue
            at_end = {}
            if tag.startswith("eval") and "eval/success_at_end" in tags:
                at_end = {e.step: e.value for e in ea.Scalars("eval/success_at_end")}
            for e in ea.Scalars(tag):
                r = {
                    "epoch": e.step,
                    # env/* is logged at step=epoch; env steps consumed after this epoch
                    "env_steps": (e.step + 1) * env_steps_per_epoch,
                    "wall_s": round(e.wall_time - t0, 1),
                    "success_once": e.value,
                }
                if e.step in at_end:
                    r["success_at_end"] = at_end[e.step]
                out.append(r)

    base_mib = 0
    b = read(os.path.join(a.run_dir, "gpu_baseline.txt"), "")
    if b:
        base_mib = int(b.split(",")[1].split()[0])
    vt = [int(x) for x in (read(os.path.join(a.run_dir, "vram_trace.txt"), "") or "").split() if x.isdigit()]
    m = {
        "algo": a.algo,
        "seed": a.seed,
        "return_code": a.rc,
        "config": os.path.basename(a.cfg),
        "config_overrides": ov,
        "env_steps_per_epoch": env_steps_per_epoch,
        "train_time_s": round(t1 - t0, 1) if t1 > 0 else None,
        "peak_vram_mib_gpu_total": max(vt) if vt else None,
        "peak_vram_mib_minus_baseline": (max(vt) - base_mib) if vt else None,
        "eval_curve": curve,
        "train_curve": train_curve,
        "final_eval_success_once": curve[-1]["success_once"] if curve else None,
        "final_eval_success_at_end": curve[-1].get("success_at_end") if curve else None,
        "best_eval_success_once": max((c["success_once"] for c in curve), default=None),
    }
    with open(os.path.join(a.run_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: v for k, v in m.items() if not k.endswith("curve")}, indent=1))


if __name__ == "__main__":
    main()
