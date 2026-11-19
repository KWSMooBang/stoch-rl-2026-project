"""T029 submission gate: exit 0 only if every pre-training check and the PPO smoke passed."""
import json
import sys

V = "/outputs/StochRL/t029/val"
fails = []
u = json.load(open(f"{V}/units.json"))["checks"]
fails += [f"units.{k}" for k, v in u.items() if not v]
t = json.load(open(f"{V}/train.json"))
for k, v in t["hold"].items():
    if isinstance(v, bool) and not v:
        fails.append(f"train.hold.{k}")
if t["hold"]["grpo_within_group_max_diff_m"] > 1e-4:
    fails.append("train.hold.grpo_within_group_max_diff_m")
for k, v in t["bitwise"]["t022_vs_t029_default"].items():
    if not v:
        fails.append(f"bitwise.{k}")
e = json.load(open(f"{V}/eval.json"))
for m in ("b0", "upper"):
    for k in ("h1_return_eq_success_at_end", "env_return_eq_h1"):
        if not e[m][k]:
            fails.append(f"eval.{m}.{k}")
s = json.load(open("/workspace/stochrl_t029/smoke/t029-smoke-ppo/metrics.json"))
if s["return_code"] != 0 or not s.get("hold") or not s["hold"]["return_eq_success_at_end"] or s["heldout_n"] < 32:
    fails.append("smoke")
print("GATE", "FAIL " + ",".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
