"""Probe: which physical GPUs does env creation (SAPIEN/Vulkan) touch? Prints nvidia-smi memory before/after."""
import subprocess
import sys
import time

from repcritic import rc_common as C

C.setup_env_vars()
import torch  # noqa: E402


def mem():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.replace("\n", " | ")


print("before", mem(), flush=True)
if "--sapien_device" in sys.argv:
    import sapien

    sapien.render.set_device("cuda:0") if hasattr(sapien.render, "set_device") else None
env = C.build_env(4)
obs, _ = env.reset(seed=0, options={"episode_id": torch.zeros(4, dtype=torch.long, device=env.device)})
time.sleep(5)
print("after ", mem(), flush=True)
