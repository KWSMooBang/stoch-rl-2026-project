"""Markdown table from a run's metrics.json.  usage: python -m aapr.report_runs RUN_DIR [tag]"""
import json
import sys


def main(run_dir, tag=None):
    m = json.load(open(f"{run_dir}/metrics.json"))
    print(f"jobs done {m['n_jobs_done']}, failed {m['n_jobs_failed']}; selected λ: {m.get('selected_lambda')}\n")
    print("| tag | 바람 | 방법 | λ | n | 최종 SR ± se | AUC ± se | seed별 SR | 정지반례 SR | sim tr. (M) | train s | \\|λ·sh\\| | MAE prog/aapr1/mh/mh_ind | SPR |")
    print("|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|---:|")
    for r in m["table"]:
        if tag and r["tag"] != tag:
            continue
        a = r.get("attribution") or {}
        mae = "/".join(f"{a.get(k, float('nan')):.3f}" for k in
                       ("mae_prog_vs_c1", "mae_aapr1_vs_c1", "mae_mh_vs_mh", "mae_mhind_vs_mh")) if a else "-"
        hp = f"{r['wind_helps_sr']:.2f}" if r.get("wind_helps_sr") is not None else "n/a"
        print(f"| {r['tag']} | {r['cond']} | {r['method']} | {r['lam']} | {r['n_seeds']} | "
              f"{r['sr_mean']:.2f} ± {r['sr_se']:.2f} | {r['auc_mean']:.2f} ± {r['auc_se']:.2f} | "
              f"{' '.join(f'{x:.2f}' for x in r['seeds_sr'])} | {hp} | {r['sim_transitions'] / 1e6:.2f} | "
              f"{r['wall_clock_train_s']:.0f} | {r['mean_abs_shaping']:.3f} | {mae} | {a.get('spr', float('nan')):.2f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
