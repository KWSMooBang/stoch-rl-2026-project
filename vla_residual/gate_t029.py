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
# Default reward vs T022 code (47a4d76), 4 sequential runs (validate_t029 --skip_hold --tag _bitwise).
# PhysX/renderer runs are not bitwise reproducible run-to-run (a T022-vs-T022 control pair can differ on ~2% of
# decisions), so require: r/te/tr/eids equal in every pair, and at least one T022-vs-T029 pair bitwise equal in states.
tb = json.load(open(f"{V}/train_bitwise.json"))
for pair, eq in tb["bitwise"].items():
    for k in ("r", "te", "tr", "eids"):
        if not eq[k]:
            fails.append(f"bitwise.{pair}.{k}")
if not any(eq["states"] for pair, eq in tb["bitwise"].items() if "t022" in pair and "t029" in pair):
    fails.append("bitwise.no_cross_code_pair_states_equal")
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
