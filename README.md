# When Do LLM Agents Help Autonomous-Vehicle Edge Scheduling?

Code and data for the paper. The framework simulates deadline-aware task
scheduling on heterogeneous, GPU-equipped mobile edge computing (MEC) servers
with two task classes: time-critical (TC) and non-time-critical (NTC).

The proposed scheduler is a **windowed contract-net auction**. A broker
announces a release-ordered window of tasks, per-server edge agents bid with
their earliest finish time, and the broker awards each task time-critical-first
by earliest deadline. An optional **LLM control plane** (broker, edge, and
monitor agents) sets scheduling policy once per window; the per-task data plane
stays deterministic, so no model call is ever on the critical path.

| Scheduler    | Description |
|--------------|-------------|
| `cnp_rule`   | Auction with deterministic policies only (no-LLM ablation) |
| `cnp_llm`    | Broker and edge agents consult a light LLM once per window |
| `has`        | `cnp_llm` plus a monitor agent that adapts fast-path parameters every *R* windows |
| `cnp_bandit` | Same parameter set as the monitor, tuned by a UCB1 bandit instead of an LLM |

## Install

```bash
pip install -r ../requirements.txt
```

## Reproducing the paper

Run from `src/`. Results are written to `results/`.

```bash
python run_multiseed.py --seeds 20           # main comparison: 60 instances, 15 baselines, stats
python run_decomp.py --seeds 20              # decomposition: ordering x placement x horizon
python run_windowed_baselines.py --seeds 20  # windowed GA / PSO / DDQN
python run_disruption.py --seeds 20 --conditions TC-burst   # non-stationary surge
python run_disruption.py --seeds 6  --conditions failure    # server outage
python llm_analysis.py                       # LLM latency, decisions, rationales
```

A single benchmark can also be run directly:

```bash
python run_experiment.py --scheduler cnp_rule            # offline, deterministic
python run_experiment.py --scheduler has --llm api       # real LLM agents
python run_experiment.py --scheduler all

# any of the four benchmarks
python run_experiment.py --scheduler all \
    --data ../data/refined_bench/scenario1/P31-11v13v12v15.csv
```

## LLM access

The agents use `claude-haiku-4-5`. Authentication is via an existing Claude
subscription login (read from `~/.claude/.credentials.json`), so no key is
needed. To use a pay-per-token key instead, put `ANTHROPIC_API_KEY=...` in a
`.env` file at the project root; an explicit key overrides the subscription.

Model responses are cached in `results/llm_cache/`, so re-runs are free and
reproduce the reported numbers exactly. Runs with `--llm mock` need no
credentials at all.

## Baselines

All baselines run through one shared online dispatch engine that exposes only
released tasks, so no method sees the future. Each file carries the full
citation for the methods it implements.

| Family | Methods | File |
|--------|---------|------|
| Dispatching rules | FCFS, EDF, EDD, EFDF, CR, COVERT, ERA, LLF | `lib/dispatch_rules.py` |
| Mapping heuristics | Min-Min, Max-Min, Sufferage, HEFT | `lib/baselines.py` |
| Metaheuristic / DRL | PSO, GA, DDQN | `lib/baselines_recent.py` |
| Windowed variants | PSO, GA, DDQN at *W*=40 | `lib/baselines_windowed.py` |
| Decomposition controls | EDF+EFT, TC-EDF+EFT, EDF+random | `lib/controls.py` |

## Benchmarks

`data/refined_bench/scenario1/` holds four benchmarks of 200 tasks each. They
differ in vehicle population and, most importantly, in deadline slack: P29 and
P30 are tight, P31 and P32 are loose, so the first two are the contended cases
where scheduling quality matters most.

| Benchmark | Tasks | Mean processing time | Mean slack |
|-----------|-------|----------------------|------------|
| `P29-2v3v4v5` (reference) | 200 | 5.37 | 4.54 |
| `P30-6v7v8v10` | 200 | 5.83 | 4.20 |
| `P31-11v13v12v15` | 200 | 5.71 | 9.24 |
| `P32-16v17v18v20` | 200 | 5.83 | 9.90 |

The multi-seed experiments resample every instance from the empirical marginals
of the reference benchmark, `P29-2v3v4v5.csv`, which is the contended case the
paper reports. The three MEC topologies are generated in code by
`data_loader_{1,2,3}` from that same task core, so they share the workload and
differ only in the network. The other three benchmarks are included so the
schedulers can be run on different workloads via `run_experiment.py --data`.

## Layout

```
src/
  lib/
    sim_core.py           execution engine, metrics, windows, server presets
    instance_gen.py       seeded instance generator (3 topologies)
    optimum.py            CP-SAT offline optimum and feasibility ceiling
    dispatch_rules.py     dispatching-rule baselines
    baselines.py          mapping heuristics
    baselines_recent.py   PSO, GA, DDQN
    baselines_windowed.py windowed PSO, GA, DDQN
    controls.py           decomposition controls
    data_loader_*.py      benchmark loaders, one per topology
    edge_servers.py, gpu.py, task_processor.py, task_distributor.py
  agents/                 message bus, LLM client, broker/edge/monitor agents
  schedulers/cnp_core.py  the windowed contract-net scheduler
  results/                run outputs and cached model responses (not tracked)
```

The completion-time model is defined once, in `sim_core.ExecutionEngine.estimate`:
communication overlaps queueing, so a task arrives at `release + comm` and only
then contends for a processing unit. Every scheduler and baseline uses it.
