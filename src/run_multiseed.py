"""Multi-seed statistical evaluation.

Generates `--seeds` independent instances per scenario, runs every
non-LLM scheduler (dispatching rules, mapping heuristics,
metaheuristics/DRL, the three decomposition controls, and the two auction
variants), computes the per-instance offline optimum and feasibility
ceiling, and reports mean +/- 95% CI plus paired Wilcoxon signed-rank
tests (Holm-Bonferroni corrected) with Cliff's-delta effect sizes,
referenced to the contract-net auction (CNP).

LLM variants (CNP-LLM, HAS) are evaluated separately (run_llm_seeds.py)
because they require model calls.
"""

import argparse
import json
import os
import random

import numpy as np
from scipy import stats

from lib.instance_gen import make_instance
from lib.sim_core import make_edge_servers, preset_for
from lib.dispatch_rules import DISPATCH_RULES
from lib.baselines import LITERATURE_BASELINES
from lib.baselines_recent import RECENT_BASELINES
from lib.controls import CONTROLS
from lib.optimum import feasibility_ceiling, offline_optimum
from schedulers.cnp_core import ContractNetScheduler

SCENARIOS = [1, 2, 3]
DISPLAY = (['FCFS', 'EDF', 'EDD', 'EFDF', 'CR', 'COVERT', 'ERA',
            'Min-Min', 'Max-Min', 'Sufferage', 'HEFT', 'LLF',
            'PSO', 'GA', 'DDQN', 'EDF+EFT', 'TCEDF+EFT', 'EDF+rand',
            'CNP', 'CNP-bandit'])
KEY = {'FCFS': 'fcfs', 'EDF': 'edf', 'EDD': 'edd', 'EFDF': 'efdf', 'CR': 'cr',
       'COVERT': 'covert', 'ERA': 'era', 'Min-Min': 'min_min',
       'Max-Min': 'max_min', 'Sufferage': 'sufferage', 'HEFT': 'heft',
       'LLF': 'llf', 'PSO': 'pso', 'GA': 'ga', 'DDQN': 'ddqn',
       'EDF+EFT': 'EDF+EFT', 'TCEDF+EFT': 'TCEDF+EFT', 'EDF+rand': 'EDF+rand'}


def run_one(name, servers, df, window):
    if name in ('CNP', 'CNP-bandit'):
        sched = ContractNetScheduler(
            servers, df.copy(), window_size=window, consult_llm=False,
            monitor_kind=('bandit' if name == 'CNP-bandit' else None),
            llm_mode='mock', review_every=2, seed=42)
        return sched.run().summary(servers)
    if name in CONTROLS:
        return CONTROLS[name](servers, df.copy()).summary(servers)
    if KEY[name] in DISPATCH_RULES:
        return DISPATCH_RULES[KEY[name]](servers, df.copy()).summary(servers)
    if KEY[name] in LITERATURE_BASELINES:
        return LITERATURE_BASELINES[KEY[name]](servers, df.copy()).summary(servers)
    return RECENT_BASELINES[KEY[name]](servers, df.copy()).summary(servers)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()), 0.0)
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return float(m), float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--window', type=int, default=40)
    ap.add_argument('--opt-time', type=float, default=15.0)
    ap.add_argument('--results-dir', default='results')
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    # per-method per-instance records
    rec = {m: {'tc_cr': [], 'ntc_cr': [], 'unweighted': [], 'weighted': [],
               'makespan': [], 'gap': [], 'tc_over_feas': []} for m in DISPLAY}
    ceilings = {'tc_feasible_rate': [], 'weighted_opt': [], 'proven': []}

    for sc in SCENARIOS:
        for sd in seeds:
            random.seed(42)
            df = make_instance(sd, sc)
            preset = preset_for(df)
            base = make_edge_servers(preset=preset)
            ceil = feasibility_ceiling(base, df)
            opt = offline_optimum(base, df, time_limit=args.opt_time)
            ceilings['tc_feasible_rate'].append(ceil['tc_feasible'] / ceil['tc_total'])
            ceilings['weighted_opt'].append(opt['weighted_opt'])
            ceilings['proven'].append(opt['proven_optimal'])
            opt_denom = opt['weighted_bound'] or opt['weighted_opt']  # rigorous UB
            for m in DISPLAY:
                servers = make_edge_servers(preset=preset)
                s = run_one(m, servers, df, args.window)
                rec[m]['tc_cr'].append(s['tc_cr'])
                rec[m]['ntc_cr'].append(s['ntc_cr'])
                rec[m]['unweighted'].append(s['tcr'])
                rec[m]['weighted'].append(s['weighted_completed'])
                rec[m]['makespan'].append(s['makespan'])
                rec[m]['gap'].append(s['weighted_completed'] / opt_denom
                                     if opt_denom > 0 else 0.0)
                rec[m]['tc_over_feas'].append(
                    s['tc_completed'] / ceil['tc_feasible'] if ceil['tc_feasible'] else 0.0)
        print(f"scenario {sc}: done {len(seeds)} seeds")

    # aggregate + paired tests vs CNP
    ref = np.array(rec['CNP']['tc_cr'])
    raw_p, agg = {}, {}
    for m in DISPLAY:
        tc = np.array(rec[m]['tc_cr'])
        mmean, mh = ci95(tc)
        agg[m] = {
            'tc_cr_mean': round(mmean, 4), 'tc_cr_ci95': round(mh, 4),
            'ntc_cr_mean': round(float(np.mean(rec[m]['ntc_cr'])), 4),
            'unweighted_mean': round(float(np.mean(rec[m]['unweighted'])), 4),
            'weighted_mean': round(float(np.mean(rec[m]['weighted'])), 2),
            'makespan_mean': round(float(np.mean(rec[m]['makespan'])), 2),
            'opt_gap_mean': round(float(np.mean(rec[m]['gap'])), 4),
            'tc_over_feasible_mean': round(float(np.mean(rec[m]['tc_over_feas'])), 4),
        }
        if m != 'CNP':
            d = tc - ref
            if np.any(d != 0):
                raw_p[m] = stats.wilcoxon(tc, ref, zero_method='wilcox').pvalue
            else:
                raw_p[m] = 1.0
            agg[m]['cliffs_delta_vs_CNP'] = round(cliffs_delta(tc, ref), 3)
            agg[m]['mean_diff_vs_CNP'] = round(float(np.mean(d)), 4)

    # Holm-Bonferroni over the comparisons
    order = sorted(raw_p, key=lambda k: raw_p[k])
    k = len(order)
    for i, m in enumerate(order):
        agg[m]['p_holm_vs_CNP'] = round(min(1.0, raw_p[m] * (k - i)), 5)

    out = {
        'config': vars(args), 'n_instances': len(seeds) * len(SCENARIOS),
        'ceiling': {
            'tc_feasible_rate_mean': round(float(np.mean(ceilings['tc_feasible_rate'])), 4),
            'weighted_opt_mean': round(float(np.mean(ceilings['weighted_opt'])), 2),
            'fraction_proven_optimal': round(float(np.mean(ceilings['proven'])), 3),
        },
        'methods': agg,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, 'multiseed_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nCeiling: mean TC-feasible rate={out['ceiling']['tc_feasible_rate_mean']}, "
          f"fraction proven optimal={out['ceiling']['fraction_proven_optimal']}")
    print(f"\n{'method':<11}{'TCR_TC':>9}{'+/-':>7}{'unwt':>7}{'opt-gap':>8}"
          f"{'d_vs_CNP':>9}{'p_holm':>8}")
    for m in sorted(DISPLAY, key=lambda x: -agg[x]['tc_cr_mean']):
        a = agg[m]
        print(f"{m:<11}{a['tc_cr_mean']:>9.3f}{a['tc_cr_ci95']:>7.3f}"
              f"{a['unweighted_mean']:>7.3f}{a['opt_gap_mean']:>8.3f}"
              f"{a.get('cliffs_delta_vs_CNP', 0):>9.2f}"
              f"{a.get('p_holm_vs_CNP', float('nan')):>8.3f}")
    print(f"\nsaved -> {path}")


if __name__ == '__main__':
    main()
