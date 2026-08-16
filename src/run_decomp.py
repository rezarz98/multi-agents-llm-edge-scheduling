"""Factorial decomposition: isolate the three factors of
the proposed scheduler -- {ordering} x {placement} x {horizon} -- with a
single windowed list-scheduler driver, over the same 60 instances as the
main comparison. No auction, no LLM. Verifies the windowed-heuristic points
used in the decomposition figure.

Causality: even in batch mode a task cannot start before it
physically arrives -- ExecutionEngine.estimate uses start = max(gpu_time,
release + comm), so batching is intra-window ORDERING lookahead only, never a
violation of release-time causality.
"""
import argparse
import json
import os
import random

import numpy as np
from scipy import stats

from lib.instance_gen import make_instance
from lib.sim_core import (make_edge_servers, preset_for, SimMetrics,
                          ExecutionEngine, release_ordered_records, iter_windows)

SCENARIOS = [1, 2, 3]

ORDERINGS = {
    'EDF':   lambda t: (t['Due Date'], t['Processing Time']),
    'TCEDF': lambda t: (0 if t['PriorityClass'] == 'TC' else 1,
                        t['Due Date'], t['Processing Time']),
}


def windowed_dispatch(edge_servers, tasks_df, order_key, place='eft',
                      window=40, rng=None):
    """List-schedule each release-ordered window: order by `order_key`,
    then place each task (EFT or random-feasible) and execute."""
    if rng is None:
        rng = random.Random(0)
    metrics = SimMetrics()
    records = release_ordered_records(tasks_df)
    for win in iter_windows(records, window):
        for task in sorted(win, key=order_key):
            if place == 'eft':
                server, gpu, _ = ExecutionEngine.place(edge_servers, task)
            else:
                feasible = [es for es in edge_servers
                            if ExecutionEngine.best_gpu(es, task)[1] <= task['Due Date']]
                server = rng.choice(feasible if feasible else edge_servers)
                gpu = ExecutionEngine.best_gpu(server, task)[0]
            if gpu is None:
                metrics.record(task, completed=False)
            else:
                ExecutionEngine.execute(server, gpu, task, metrics)
    return metrics


def ci95(x):
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return float(m), float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--results-dir', default='results')
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    # W=1 is strict release order (no intra-window lookahead).
    HORIZONS = [1, 40, 200]
    cells = {}
    for order in ORDERINGS:
        for place in ('eft', 'rand'):
            for W in HORIZONS:
                cells[(order, place, W)] = {'tc': [], 'ntc': [], 'wt': []}

    for sc in SCENARIOS:
        for sd in seeds:
            df = make_instance(sd, sc)
            preset = preset_for(df)
            for order in ORDERINGS:
                for place in ('eft', 'rand'):
                    for W in HORIZONS:
                        servers = make_edge_servers(preset=preset)
                        m = windowed_dispatch(servers, df.copy(),
                                              ORDERINGS[order], place=place,
                                              window=W, rng=random.Random(0))
                        s = m.summary(servers)
                        c = cells[(order, place, W)]
                        c['tc'].append(s['tc_cr'])
                        c['ntc'].append(s['ntc_cr'])
                        c['wt'].append(s['weighted_completed'])
        print(f"scenario {sc}: done")

    out = {}
    for k, v in cells.items():
        tcm, tch = ci95(v['tc'])
        out['|'.join(map(str, k))] = {
            'ordering': k[0], 'placement': k[1], 'window': k[2],
            'tc_cr_mean': round(tcm, 4), 'tc_cr_ci95': round(tch, 4),
            'ntc_cr_mean': round(float(np.mean(v['ntc'])), 4),
            'weighted_mean': round(float(np.mean(v['wt'])), 2),
        }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, 'decomp_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'ordering':<8}{'place':<6}{'W':>5}{'TC_CR':>9}{'+/-':>7}{'NTC_CR':>8}{'wt':>8}")
    for k in sorted(out, key=lambda x: (out[x]['ordering'], out[x]['placement'], out[x]['window'])):
        e = out[k]
        print(f"{e['ordering']:<8}{e['placement']:<6}{e['window']:>5}"
              f"{e['tc_cr_mean']:>9.3f}{e['tc_cr_ci95']:>7.3f}"
              f"{e['ntc_cr_mean']:>8.3f}{e['weighted_mean']:>8.1f}")
    print(f"\nsaved -> {path}")


if __name__ == '__main__':
    main()
