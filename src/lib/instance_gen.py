"""Seeded instance generator for multi-seed evaluation.

Each instance is drawn by independently resampling the release time,
processing time, slack (deadline-release-processing), and benefit from
the empirical marginals of the reference contended benchmark
(P29-2v3v4v5), so every instance has the same statistical character but a
different realisation. The task core is generated once per seed and then
passed through each scenario's topology generator (the existing
data_loader_{1,2,3} modules) so the three scenarios share the same
workload and differ only in the network, exactly as in the original
benchmark.
"""

import os
import random

import pandas as pd

from . import data_loader_1, data_loader_2, data_loader_3

_LOADERS = {1: data_loader_1, 2: data_loader_2, 3: data_loader_3}
_CORE = None


def _core_pool():
    global _CORE
    if _CORE is None:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                            'refined_bench', 'scenario1', 'P29-2v3v4v5.csv')
        df = pd.read_csv(path)
        _CORE = {
            'release': df['Release Time'].tolist(),
            'proc': df['Processing Time'].tolist(),
            'slack': (df['Due Date'] - df['Release Time']
                      - df['Processing Time']).tolist(),
            'benefit': df['Benefit'].tolist(),
        }
    return _CORE


def gen_core(seed, n=200):
    pool = _core_pool()
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rel = rng.choice(pool['release'])
        proc = rng.choice(pool['proc'])
        slack = rng.choice(pool['slack'])
        ben = rng.choice(pool['benefit'])
        due = max(rel + 1, rel + proc + slack)
        rows.append({'taskid': i + 1, 'Processing Time': proc,
                     'Release Time': rel, 'Due Date': int(due), 'Benefit': ben})
    return pd.DataFrame(rows)


def make_instance(seed, scenario, disruption='none'):
    """Return a fully-populated task DataFrame for the given seed and
    scenario topology (1, 2, or 3).

    disruption='burst' injects a mid-run surge of ~40 time-critical tasks
    (released in a narrow window with tight deadlines), a non-stationary
    load spike that a static policy handles poorly.
    """
    core = gen_core(seed)
    burst_ids = set()
    if disruption == 'burst':
        rng = random.Random(seed * 7 + 11)
        nid = len(core) + 1
        extra = []
        for _ in range(40):
            rel = rng.randint(18, 22)
            proc = rng.randint(3, 8)
            extra.append({'taskid': nid, 'Processing Time': proc,
                          'Release Time': rel, 'Due Date': rel + proc + rng.randint(0, 3),
                          'Benefit': rng.randint(1, 20)})
            burst_ids.add(nid)
            nid += 1
        core = pd.concat([core, pd.DataFrame(extra)], ignore_index=True)

    random.seed(seed * 97 + scenario)          # topology comm randomness
    loader = _LOADERS[scenario]
    df = loader.add_additional_columns(core.copy())
    df = loader.assign_priority_classes(df, seed=seed)
    if burst_ids:
        df.loc[df['taskid'].isin(burst_ids), 'PriorityClass'] = 'TC'
    return df
