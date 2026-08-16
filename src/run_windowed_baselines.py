"""Windowed metaheuristic/DRL baselines over the 60-instance benchmark.

Do GA, PSO, and DDQN also benefit from the batching horizon? Compares each
online baseline (from multiseed_results.json) with its windowed variant (W=40).
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from lib.instance_gen import make_instance
from lib.sim_core import make_edge_servers, preset_for
from lib.baselines_windowed import WINDOWED_BASELINES

SCENARIOS = [1, 2, 3]


def ci95(x):
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), stats.sem(x)
    return float(m), float(se * stats.t.ppf(0.975, len(x) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--window', type=int, default=40)
    ap.add_argument('--results-dir', default='results')
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    rec = {m: {'tc': [], 'ntc': [], 'wt': []} for m in WINDOWED_BASELINES}
    for sc in SCENARIOS:
        for sd in seeds:
            df = make_instance(sd, sc)
            preset = preset_for(df)
            for m, fn in WINDOWED_BASELINES.items():
                servers = make_edge_servers(preset=preset)
                s = fn(servers, df.copy(), window=args.window,
                       seed=42).summary(servers)
                rec[m]['tc'].append(s['tc_cr'])
                rec[m]['ntc'].append(s['ntc_cr'])
                rec[m]['wt'].append(s['weighted_completed'])
        print(f"scenario {sc}: done")

    out = {}
    for m in WINDOWED_BASELINES:
        tcm, tch = ci95(rec[m]['tc'])
        out[m] = {'tc_cr_mean': round(tcm, 4), 'tc_cr_ci95': round(tch, 4),
                  'ntc_cr_mean': round(float(np.mean(rec[m]['ntc'])), 4),
                  'weighted_mean': round(float(np.mean(rec[m]['wt'])), 2)}

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, 'windowed_baselines.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'method':<10}{'TC_CR':>9}{'+/-':>7}{'NTC_CR':>8}{'wt':>8}")
    for m in WINDOWED_BASELINES:
        e = out[m]
        print(f"{m:<10}{e['tc_cr_mean']:>9.3f}{e['tc_cr_ci95']:>7.3f}"
              f"{e['ntc_cr_mean']:>8.3f}{e['weighted_mean']:>8.1f}")
    print(f"\nsaved -> {path}")


if __name__ == '__main__':
    main()
