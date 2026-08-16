"""Single-benchmark runner for the baselines and the agent schedulers.

Examples
--------
# rule-based contract-net auction (no-LLM ablation) on the small benchmark
python run_experiment.py --scheduler cnp_rule

# LLM-CNP with mock agents (offline, deterministic)
python run_experiment.py --scheduler cnp_llm --llm mock

# HAS (LLM-CNP + monitor agent) with real light LLM agents
python run_experiment.py --scheduler has --llm api

# a dispatching-rule baseline for comparison
python run_experiment.py --scheduler edf

# everything on one benchmark
python run_experiment.py --scheduler all --data ../data/refined_bench/scenario1/P29-2v3v4v5.csv

For the multi-seed results reported in the paper use run_multiseed.py,
run_decomp.py, run_disruption.py, and run_windowed_baselines.py instead.
"""

import argparse
import json
import os
import random

from lib.data_loader_1 import read_csv_to_dataframe
from lib.sim_core import make_edge_servers
from lib.dispatch_rules import DISPATCH_RULES
from lib.baselines import LITERATURE_BASELINES
from lib.baselines_recent import RECENT_BASELINES
from schedulers.cnp_core import ContractNetScheduler

# Dispatching rules (lib/dispatch_rules.py): fcfs, edf, edd, efdf, cr,
# covert, era. Classical mapping heuristics (lib/baselines.py): min_min,
# max_min, sufferage, heft, llf. Metaheuristic/DRL (lib/baselines_recent.py):
# pso, ga, ddqn. See each file for citations.
LIT_BASELINES = {**DISPATCH_RULES, **LITERATURE_BASELINES, **RECENT_BASELINES}
CNP_SCHEDULERS = ('cnp_rule', 'cnp_llm', 'has')
ALL_SCHEDULERS = list(LIT_BASELINES) + list(CNP_SCHEDULERS)


def run_lit_baseline(name, tasks_df, args):
    random.seed(args.seed)
    servers = make_edge_servers(preset=args.servers)
    metrics = LIT_BASELINES[name](servers, tasks_df.copy())
    return metrics.summary(servers)


def run_cnp(name, tasks_df, args):
    random.seed(args.seed)
    servers = make_edge_servers(preset=args.servers)
    scheduler = ContractNetScheduler(
        servers, tasks_df.copy(),
        window_size=args.window,
        consult_llm=(name != 'cnp_rule'),
        use_monitor=(name == 'has'),
        llm_mode=(args.llm if name != 'cnp_rule' else 'mock'),
        llm_model=args.model,
        review_every=args.review_every,
        seed=args.seed,
    )
    metrics = scheduler.run()

    if args.save_trace:
        os.makedirs(args.results_dir, exist_ok=True)
        trace_path = os.path.join(
            args.results_dir,
            f'{name}_{os.path.basename(args.data).replace(".csv", "")}'
            f'_seed{args.seed}_trace.jsonl',
        )
        scheduler.bus.save(trace_path)
        print(f"  message trace -> {trace_path}")
    return metrics.summary(servers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scheduler', default='cnp_rule',
                        choices=ALL_SCHEDULERS + ['all'])
    parser.add_argument('--data',
                        default='../data/refined_bench/scenario1/P29-2v3v4v5.csv')
    parser.add_argument('--servers', default='small', choices=['small', 'large'])
    parser.add_argument('--llm', default='mock', choices=['mock', 'api'],
                        help='mock = deterministic offline agents; '
                             'api = light Claude agents (claude-haiku-4-5)')
    parser.add_argument('--model', default='claude-haiku-4-5')
    parser.add_argument('--window', type=int, default=20,
                        help='tasks per auction window')
    parser.add_argument('--review-every', type=int, default=5,
                        help='monitor agent review period (windows)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--save-trace', action='store_true', default=True)
    parser.add_argument('--no-save-trace', dest='save_trace',
                        action='store_false')
    args = parser.parse_args()

    tasks_df = read_csv_to_dataframe(args.data, seed=args.seed)
    print(f"Benchmark: {args.data}  ({len(tasks_df)} tasks, "
          f"servers={args.servers}, seed={args.seed})\n")

    names = ALL_SCHEDULERS if args.scheduler == 'all' else [args.scheduler]
    results = {}
    for name in names:
        if name in LIT_BASELINES:
            results[name] = run_lit_baseline(name, tasks_df, args)
        else:
            results[name] = run_cnp(name, tasks_df, args)

    # ---- report ----------------------------------------------------------
    cols = ['tc_cr', 'ntc_cr', 'tc_completed', 'ntc_completed',
            'tc_killed', 'ntc_killed', 'weighted_completed', 'makespan',
            'llm_calls', 'messages_exchanged']
    width = max(len(c) for c in cols) + 2
    header = f"{'scheduler':<14}" + ''.join(f"{c:>{width}}" for c in cols)
    print(header)
    print('-' * len(header))
    for name, summary in results.items():
        row = f"{name:<14}" + ''.join(f"{summary[c]:>{width}}" for c in cols)
        print(row)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f'results_{os.path.basename(args.data).replace(".csv", "")}'
        f'_seed{args.seed}.json',
    )
    with open(out_path, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == '__main__':
    main()
