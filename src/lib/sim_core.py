"""Event-driven simulation core for the MEC scheduling experiments.

Single source of truth for:
  * completion-time computation (comm times + processing / GPU speed),
  * task execution against GPU clocks (complete vs killed),
  * per-class (TC/NTC) metrics,
  * release-ordered task windows,
  * edge-server presets.

Every baseline and the contract-net schedulers execute tasks under
identical timing rules, so results are directly comparable.
"""

import random
from dataclasses import dataclass, field

from .edge_servers import EdgeServer

# ---------------------------------------------------------------------------
# Objective weights. Time-critical (TC) completions are worth more than
# non-time-critical (NTC) completions; every scheduler and the reported
# score use these same weights so the comparison is consistent.
# ---------------------------------------------------------------------------
W_TC = 2.0
W_NTC = 1.0


# ---------------------------------------------------------------------------
# Server presets
# ---------------------------------------------------------------------------
_SMALL_SPEEDS = [
    [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.1],
    [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.2],
    [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.1],
    [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.2],
]

SERVER_PRESETS = {
    # 4 servers x 9 GPUs — for the 4-base-station scenarios (scenario 1, 2)
    'small': _SMALL_SPEEDS,
    # 6 servers x 9 GPUs — for the 6-base-station scenario (scenario 3)
    'small6': _SMALL_SPEEDS + [
        [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.1],
        [0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 1, 1, 1.2],
    ],
    # 4 servers x 28 GPUs — matches the large_data scenarios of prior work
    'large': [
        [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 1, 1, 1, 1.1, 1.2, 1.2, 1.2,
         1.2, 1.3, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.6, 1.7, 1.7],
        [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 1, 1, 1, 1.2, 1.2, 1.2, 1.2,
         1.3, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.6, 1.7, 1.7],
        [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 1, 1, 1, 1.1, 1.2, 1.2, 1.2,
         1.2, 1.3, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.6, 1.7, 1.7],
        [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 1, 1, 1, 1.2, 1.2, 1.2, 1.2,
         1.3, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.7, 1.6, 1.6, 1.7, 1.7],
    ],
}


def make_edge_servers(preset='small'):
    """Instantiate edge servers from a preset."""
    return [
        EdgeServer(id=i + 1, gpu_speeds=list(speeds))
        for i, speeds in enumerate(SERVER_PRESETS[preset])
    ]


def n_base_stations(tasks_df):
    """Number of 'Base Station N Communication Time' columns in the data."""
    return sum(1 for c in tasks_df.columns
               if c.startswith('Base Station') and c.endswith('Communication Time'))


def preset_for(tasks_df, size='small'):
    """Pick the server preset that matches the data's base-station count.
    4 stations -> 'small'/'large', 6 stations -> 'small6'."""
    if size == 'large':
        return 'large'
    return 'small6' if n_base_stations(tasks_df) >= 6 else 'small'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class SimMetrics:
    completed_tc: list = field(default_factory=list)
    completed_ntc: list = field(default_factory=list)
    killed_tc: list = field(default_factory=list)
    killed_ntc: list = field(default_factory=list)
    total_benefit: float = 0.0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    messages_exchanged: int = 0

    @property
    def completed(self):
        return self.completed_tc + self.completed_ntc

    @property
    def killed(self):
        return self.killed_tc + self.killed_ntc

    def record(self, task, completed):
        is_tc = task['PriorityClass'] == 'TC'
        if completed:
            (self.completed_tc if is_tc else self.completed_ntc).append(task)
            self.total_benefit += task['Benefit']
        else:
            (self.killed_tc if is_tc else self.killed_ntc).append(task)

    def summary(self, edge_servers):
        makespan = max(
            (g.current_time for es in edge_servers for g in es.gpus),
            default=0.0,
        )
        n_tc_c, n_ntc_c = len(self.completed_tc), len(self.completed_ntc)
        n_tc_k, n_ntc_k = len(self.killed_tc), len(self.killed_ntc)
        n_completed = n_tc_c + n_ntc_c
        n_total = n_completed + n_tc_k + n_ntc_k
        total_tc = n_tc_c + n_tc_k
        total_ntc = n_ntc_c + n_ntc_k

        # Weighted throughput: TC completions worth W_TC, NTC worth W_NTC.
        weighted_completed = W_TC * n_tc_c + W_NTC * n_ntc_c

        return {
            'tc_completed': n_tc_c,
            'ntc_completed': n_ntc_c,
            'tc_killed': n_tc_k,
            'ntc_killed': n_ntc_k,
            'total_completed': n_completed,
            'total_killed': n_tc_k + n_ntc_k,
            # Completion rates (the SARS paper's TCR metric family).
            'tcr': round(n_completed / n_total, 4) if n_total else 0.0,
            'tc_cr': round(n_tc_c / total_tc, 4) if total_tc else 0.0,
            'ntc_cr': round(n_ntc_c / total_ntc, 4) if total_ntc else 0.0,
            # Weighted objective (TC worth W_TC, NTC worth W_NTC).
            'weighted_completed': round(weighted_completed, 2),
            'weighted_score': round(weighted_completed - 0.5 * makespan, 3),
            'total_benefit': self.total_benefit,
            'makespan': round(makespan, 3),
            'llm_calls': self.llm_calls,
            'llm_input_tokens': self.llm_input_tokens,
            'llm_output_tokens': self.llm_output_tokens,
            'messages_exchanged': self.messages_exchanged,
        }


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """Executes tasks on GPUs under the timing model shared with prior work."""

    @staticmethod
    def estimate(server, gpu, task):
        """Return (start_time, end_time) for running `task` on `gpu`.

        Communication delay is incurred in transit (it overlaps any
        queueing at the PU): the task ARRIVES at the server at
        release + comm, and only then contends for the PU, which it
        occupies for its processing time alone. This avoids the
        double-counting of the earlier additive form under contention.
        """
        arrival = (task['Release Time']
                   + task['Access Point Communication Time']
                   + task['Broker Communication Time']
                   + task[f'Base Station {server.id} Communication Time'])
        start = max(gpu.current_time, arrival)
        end = start + task['Processing Time'] / gpu.speed
        return start, end

    @classmethod
    def best_gpu(cls, server, task):
        """Earliest-finishing GPU on one server. Returns (gpu, end) or (None, inf)."""
        best_gpu, best_end = None, float('inf')
        for gpu in server.gpus:
            _, end = cls.estimate(server, gpu, task)
            if end < best_end:
                best_gpu, best_end = gpu, end
        return best_gpu, best_end

    @classmethod
    def place(cls, edge_servers, task):
        """Earliest-finish-time placement across ALL servers.
        Returns (server, gpu, end)."""
        best = (None, None, float('inf'))
        for es in edge_servers:
            gpu, end = cls.best_gpu(es, task)
            if gpu is not None and end < best[2]:
                best = (es, gpu, end)
        return best

    @classmethod
    def completion_times(cls, edge_servers, task):
        """Sorted list of finish times for `task` over every GPU (ascending).
        Used by assignment heuristics (Min-Min, Sufferage, ...)."""
        ends = [cls.estimate(es, gpu, task)[1]
                for es in edge_servers for gpu in es.gpus]
        ends.sort()
        return ends

    @classmethod
    def execute(cls, server, gpu, task, metrics):
        """Run the task on the given GPU. Advances the clock only on success.

        Returns True if completed before its due date, False if killed.
        A killed task never occupies the GPU (same rule as prior work).
        """
        _, end = cls.estimate(server, gpu, task)
        if end <= task['Due Date']:
            gpu.current_time = end
            gpu.completed_tasks.append(task)
            metrics.record(task, completed=True)
            return True
        metrics.record(task, completed=False)
        return False


# ---------------------------------------------------------------------------
# Online dispatch driver
#
# The problem is ONLINE: tasks become visible only at their release time,
# and a decision is taken whenever a server is free. Every non-agent
# scheduler runs through this driver, so all of them see exactly the same
# information (the set of released-but-unscheduled tasks) — none can look
# into the future. A scheduler supplies `choose(ready, current, servers)`
# which returns the next task to dispatch from the ready set; placement is
# always earliest-finish-time.
# ---------------------------------------------------------------------------
def online_dispatch(edge_servers, tasks_df, choose, place='eft', rng=None):
    """`place`: 'eft' = earliest-finish-time across all servers;
    'random' = a random deadline-feasible server (isolates the EFT effect)."""
    metrics = SimMetrics()
    records = tasks_df.sort_values('Release Time').to_dict('records')
    n, i, ready = len(records), 0, []
    if rng is None:
        rng = random.Random(0)

    def now():
        return min((g.current_time for es in edge_servers for g in es.gpus),
                   default=0.0)

    def place_task(task):
        if place == 'eft':
            server, gpu, _ = ExecutionEngine.place(edge_servers, task)
            return server, gpu
        feasible = [es for es in edge_servers
                    if ExecutionEngine.best_gpu(es, task)[1] <= task['Due Date']]
        server = rng.choice(feasible if feasible else edge_servers)
        return server, ExecutionEngine.best_gpu(server, task)[0]

    while i < n or ready:
        current = now()
        while i < n and records[i]['Release Time'] <= current:
            ready.append(records[i])
            i += 1
        if not ready:                       # idle: jump to next arrival
            current = records[i]['Release Time']
            while i < n and records[i]['Release Time'] <= current:
                ready.append(records[i])
                i += 1

        task = choose(ready, current, edge_servers)
        ready.remove(task)
        server, gpu = place_task(task)
        if gpu is None:
            metrics.record(task, completed=False)
        else:
            ExecutionEngine.execute(server, gpu, task, metrics)

    return metrics


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def release_ordered_records(tasks_df):
    """Task dict records sorted by release time (broker admission order)."""
    return tasks_df.sort_values('Release Time').to_dict('records')


def iter_windows(records, window_size):
    """Yield consecutive windows of tasks in release order."""
    for i in range(0, len(records), window_size):
        yield records[i:i + window_size]
