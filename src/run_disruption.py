"""Disruption / non-stationary evaluation: does adaptation help when the
workload changes? Compares the STATIC auction (CNP-rule, = the windowed
heuristic) against adaptive variants (UCB1 bandit, LLM broker/edge agents,
and the full HAS with LLM monitor) under a mid-run time-critical burst and
a mid-run server failure. Under these disruptions the static heuristic is
no longer near-optimal (measured ~10-point headroom vs the offline
optimum), so there is room for adaptation to help.
"""
import argparse
import json
import os
import random

import numpy as np
from scipy import stats

from lib.instance_gen import make_instance
from lib.sim_core import make_edge_servers, preset_for
from schedulers.cnp_core import ContractNetScheduler

CONDITIONS = {
    'nominal': dict(disruption='none', kw={}),
    'TC-burst': dict(disruption='burst', kw={}),
    'failure': dict(disruption='none',
                    kw=dict(failure_at_window=3, failed_server_ids=[1])),
}
METHODS = {  # name -> (consult_llm, monitor_kind, needs_api)
    'static': (False, None, False),
    'bandit': (False, 'bandit', False),
    'CNP-LLM': (True, None, True),
    'HAS': (True, 'llm', True),
}


def run(name, df, preset, kw, llm_mode):
    consult, mon, _ = METHODS[name]
    servers = make_edge_servers(preset=preset)
    sch = ContractNetScheduler(servers, df.copy(), window_size=40,
                               consult_llm=consult, monitor_kind=mon,
                               llm_mode=llm_mode, review_every=2, seed=42, **kw)
    s = sch.run().summary(servers)
    return s['tc_cr'], s['ntc_cr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--conditions', default='TC-burst,failure')
    ap.add_argument('--results-dir', default='results')
    args = ap.parse_args()
    conds = args.conditions.split(',')
    seeds = list(range(1, args.seeds + 1))

    data = {c: {m: [] for m in METHODS} for c in conds}
    data_ntc = {c: {m: [] for m in METHODS} for c in conds}
    for c in conds:
        cfg = CONDITIONS[c]
        for sc in (1, 2, 3):
            for sd in seeds:
                df = make_instance(sd, sc, disruption=cfg['disruption'])
                p = preset_for(df)
                for m in METHODS:
                    random.seed(42)
                    mode = 'api' if METHODS[m][2] else 'mock'
                    tc, ntc = run(m, df, p, cfg['kw'], mode)
                    data[c][m].append(tc)
                    data_ntc[c][m].append(ntc)
        print(f"[{c}] done {len(seeds)} seeds x 3 scenarios")

    out = {'config': vars(args), 'conditions': {}}
    for c in conds:
        ref = np.array(data[c]['static'])
        row = {}
        for m in METHODS:
            v = np.array(data[c][m])
            vn = np.array(data_ntc[c][m])
            entry = {'tc_cr_mean': round(float(v.mean()), 4),
                     'tc_cr_ci95': round(float(1.96 * v.std() / len(v) ** 0.5), 4),
                     'ntc_cr_mean': round(float(vn.mean()), 4),
                     'ntc_cr_ci95': round(float(1.96 * vn.std() / len(vn) ** 0.5), 4)}
            if m != 'static':
                entry['delta_vs_static'] = round(float((v - ref).mean()), 4)
                if np.any(v - ref != 0):
                    entry['wilcoxon_p'] = round(float(stats.wilcoxon(v, ref).pvalue), 4)
                else:
                    entry['wilcoxon_p'] = 1.0
            row[m] = entry
        out['conditions'][c] = row

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, 'disruption_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    for c in conds:
        print(f"\n=== {c} ===")
        print(f"{'method':<9}{'TCR_TC':>9}{'+/-':>7}{'TCR_NTC':>9}{'Δ vs static':>13}{'p':>8}")
        for m in METHODS:
            e = out['conditions'][c][m]
            print(f"{m:<9}{e['tc_cr_mean']:>9.4f}{e['tc_cr_ci95']:>7.4f}"
                  f"{e['ntc_cr_mean']:>9.4f}"
                  f"{e.get('delta_vs_static', 0):>13.4f}{e.get('wilcoxon_p', float('nan')):>8.3f}")
    print(f"\nsaved -> {path}")


if __name__ == '__main__':
    main()
