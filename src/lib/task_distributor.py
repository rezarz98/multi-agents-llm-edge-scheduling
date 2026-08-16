"""GPU/node selection for task placement.

Strategies
----------
'eft'    : greedy Earliest Finish Time — place the task on the GPU that
           completes it earliest, accounting for per-server communication
           time and GPU speed. This is the standard placement rule used
           by list-scheduling heuristics for heterogeneous systems
           (e.g. the processor-selection phase of HEFT, Topcuoglu et al.,
           IEEE TPDS 2002).
'random' : uniform random GPU choice (ablation baseline).
"""

import random


class TaskDistributor:
    def __init__(self, edge_servers, tasks, strategy='eft', rng=None):
        self.edge_servers = edge_servers
        self.tasks = tasks
        self.strategy = strategy
        self.rng = rng if rng is not None else random.Random()

    def _end_time(self, server, gpu, task):
        arrival = (task['Release Time']
                   + task['Access Point Communication Time']
                   + task['Broker Communication Time']
                   + task[f'Base Station {server.id} Communication Time'])
        return max(gpu.current_time, arrival) + task['Processing Time'] / gpu.speed

    def _candidates(self):
        for server in self.edge_servers:
            for gpu in server.gpus:
                yield server, gpu

    def select(self, task):
        """Return (server, gpu) for the task under the configured strategy."""
        if self.strategy == 'random':
            pool = list(self._candidates())
            if not pool:
                return None, None
            return self.rng.choice(pool)

        return min(
            self._candidates(),
            key=lambda sg: self._end_time(sg[0], sg[1], task),
            default=(None, None),
        )
