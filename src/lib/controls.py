"""Decomposition controls: isolate the sources of the proposed method's
advantage.

The proposed contract-net scheduler combines (i) a wide batching horizon,
(ii) time-critical-first ordering, (iii) earliest-finish-time (EFT)
placement across heterogeneous servers, and (iv) the auction / LLM
machinery. These controls remove factors one at a time so the
contribution of each can be measured:

  EDF+EFT    : batch earliest-deadline-first ordering, EFT placement
               (no criticality ordering, no auction/LLM).
  TCEDF+EFT  : time-critical-first, EDF-within-class ordering, EFT
               placement (the two-line heuristic at the heart of the
               auction, WITHOUT the auction or any LLM).
  EDF+rand   : batch EDF ordering, random deadline-feasible placement
               (removes EFT, to isolate its effect).

All run through the shared online dispatch driver, so they see exactly
the same released-task information as every other scheduler.
"""

import random

from .sim_core import online_dispatch


def edf_eft(edge_servers, tasks_df):
    return online_dispatch(
        edge_servers, tasks_df,
        lambda ready, now, srv: min(ready, key=lambda t: (t['Due Date'], t['Processing Time'])))


def tcedf_eft(edge_servers, tasks_df):
    def choose(ready, now, srv):
        return min(ready, key=lambda t: (0 if t['PriorityClass'] == 'TC' else 1,
                                         t['Due Date'], t['Processing Time']))
    return online_dispatch(edge_servers, tasks_df, choose)


def edf_random(edge_servers, tasks_df, seed=0):
    return online_dispatch(
        edge_servers, tasks_df,
        lambda ready, now, srv: min(ready, key=lambda t: (t['Due Date'], t['Processing Time'])),
        place='random', rng=random.Random(seed))


CONTROLS = {
    'EDF+EFT': edf_eft,
    'TCEDF+EFT': tcedf_eft,
    'EDF+rand': edf_random,
}
