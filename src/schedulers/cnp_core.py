"""Contract-net-protocol (CNP) scheduling loop.

One scheduler core, three configurations (see run_experiment.py):

  cnp_rule : auction only — agents use their deterministic fast paths,
             no policy consultation at all (the no-LLM ablation).
  cnp_llm  : LLM-CNP — broker and edge agents consult a light LLM once
             per window to set auction / bidding policy.
  has      : Hierarchical Agentic Scheduler — LLM-CNP plus a monitor
             agent that adjusts fast-path parameters every K windows.

The per-task data plane is always deterministic and fast; LLM calls
happen only on the per-window control plane.
"""

import random

from agents.message_bus import MessageBus
from agents.llm_client import LLMClient
from agents.edge_agent import EdgeAgent
from agents.broker_agent import BrokerAgent
from agents.monitor_agent import MonitorAgent, FastPathParams
from lib.sim_core import SimMetrics, release_ordered_records, iter_windows


class ContractNetScheduler:
    def __init__(self, edge_servers, tasks_df, window_size=20,
                 consult_llm=False, use_monitor=False,
                 llm_mode='mock', llm_model='claude-haiku-4-5',
                 review_every=5, seed=42, monitor_kind=None,
                 failure_at_window=None, failed_server_ids=()):
        self.edge_servers = edge_servers
        self.tasks_df = tasks_df
        self.window_size = window_size
        self.consult_llm = consult_llm
        # disruption: from `failure_at_window` on, the listed servers are lost
        self.failure_at_window = failure_at_window
        self.failed_server_ids = set(failed_server_ids)
        # monitor_kind: 'llm' (LLM monitor), 'bandit' (UCB1), or None.
        # `use_monitor` is retained for backward compatibility.
        if monitor_kind is None:
            monitor_kind = 'llm' if use_monitor else None
        self.monitor_kind = monitor_kind
        self.use_monitor = monitor_kind is not None
        self.review_every = review_every
        self.rng = random.Random(seed)

        self.bus = MessageBus()
        self.llm = LLMClient(mode=llm_mode, model=llm_model)
        self.metrics = SimMetrics()
        self.params = FastPathParams()

        self.edge_agents = [EdgeAgent(es, self.llm, self.bus)
                            for es in edge_servers]
        self.broker = BrokerAgent(self.edge_agents, self.llm, self.bus, self.rng)
        self.monitor = (MonitorAgent(self.llm, self.bus, review_every)
                        if monitor_kind == 'llm' else None)
        # UCB1 bandit over the same fast-path parameter set the LLM monitor
        # tunes (ordering policy x NTC-deferral), as a non-LLM control.
        self._bandit_arms = [
            FastPathParams('edf', (0.5, 0.3, 0.2), 0),
            FastPathParams('edf', (0.5, 0.3, 0.2), 1),
            FastPathParams('semi_greedy', (0.5, 0.3, 0.2), 0),
            FastPathParams('semi_greedy', (0.5, 0.3, 0.2), 1),
        ]
        self._bandit_n = [0] * len(self._bandit_arms)
        self._bandit_v = [0.0] * len(self._bandit_arms)
        self._bandit_t = 0
        self._bandit_last = 0

    def _bandit_review(self, reward):
        """UCB1: credit the last-used arm with `reward`, pick the next."""
        import math
        a = self._bandit_last
        self._bandit_n[a] += 1
        self._bandit_v[a] += (reward - self._bandit_v[a]) / self._bandit_n[a]
        self._bandit_t += 1
        # pick unplayed arms first, then by UCB score
        if 0 in self._bandit_n:
            nxt = self._bandit_n.index(0)
        else:
            nxt = max(range(len(self._bandit_arms)),
                      key=lambda i: self._bandit_v[i]
                      + math.sqrt(2 * math.log(self._bandit_t) / self._bandit_n[i]))
        self._bandit_last = nxt
        self.params = self._bandit_arms[nxt]

    # ------------------------------------------------------------------
    def _window_summary(self, window_id, window):
        procs = [t['Processing Time'] for t in window]
        return {
            'window_id': window_id,
            'n_tc': sum(1 for t in window if t['PriorityClass'] == 'TC'),
            'n_ntc': sum(1 for t in window if t['PriorityClass'] != 'TC'),
            'mean_processing_time': round(sum(procs) / len(procs), 2),
            'min_due': min(t['Due Date'] for t in window),
            'max_due': max(t['Due Date'] for t in window),
        }

    def _cluster_mean_load(self):
        times = [g.current_time for es in self.edge_servers for g in es.gpus]
        return sum(times) / len(times) if times else 0.0

    def _should_defer(self, task, window_summary):
        """Defer NTC tasks whose slack comfortably exceeds the window's work."""
        if self.params.ntc_defer_windows <= 0:
            return False
        if task['PriorityClass'] == 'TC':
            return False
        if task.get('_deferrals', 0) >= self.params.ntc_defer_windows:
            return False
        now = min((g.current_time for es in self.edge_servers for g in es.gpus),
                  default=0.0)
        slack = task['Due Date'] - max(now, task['Release Time']) \
            - task['Processing Time']
        return slack > 2.0 * window_summary['mean_processing_time']

    # ------------------------------------------------------------------
    def run(self):
        records = release_ordered_records(self.tasks_df)
        windows = list(iter_windows(records, self.window_size))
        carry = []            # NTC tasks deferred into the next window
        window_id = 0
        last_review_counts = self._counts()

        for base_window in windows:
            window_id += 1
            window = carry + list(base_window)
            carry = []
            summary = self._window_summary(window_id, window)

            # ---- disruption: drop failed servers from this window on ---
            if (self.failure_at_window is not None
                    and window_id >= self.failure_at_window):
                self.broker.edge_agents = [
                    a for a in self.edge_agents
                    if a.server.id not in self.failed_server_ids]
            active = self.broker.edge_agents

            # ---- control plane (per window) --------------------------
            self.broker.announce(window_id, window)
            reports = [a.report(window_id) for a in active]
            if self.consult_llm:
                mean_load = self._cluster_mean_load()
                for agent in active:
                    agent.decide_window_policy(window_id, summary, mean_load)
                self.broker.decide_window_policy(window_id, summary, reports)

            # ---- data plane (per task) -------------------------------
            for task in self.broker.order_tasks(window, self.params):
                if self._should_defer(task, summary):
                    task = dict(task)
                    task['_deferrals'] = task.get('_deferrals', 0) + 1
                    carry.append(task)
                    continue
                agent = self.broker.award(task)
                if agent is None:
                    self.metrics.record(task, completed=False)  # early reject
                    continue
                agent.execute(task, self.metrics)

            # ---- meta layer (every K windows) ------------------------
            if self.use_monitor and window_id % self.review_every == 0:
                counts = self._counts()
                rolling = {k: counts[k] - last_review_counts[k] for k in counts}
                last_review_counts = counts
                if self.monitor_kind == 'llm':
                    self.params = self.monitor.review(window_id, rolling, self.params)
                else:  # bandit
                    reward = (2 * rolling['tc_completed'] + rolling['ntc_completed'])
                    self._bandit_review(reward)

        # flush any tasks still deferred at the end
        for task in carry:
            agent = self.broker.award(task)
            if agent is None:
                self.metrics.record(task, completed=False)
                continue
            agent.execute(task, self.metrics)

        self.metrics.llm_calls = self.llm.calls
        self.metrics.llm_input_tokens = self.llm.input_tokens
        self.metrics.llm_output_tokens = self.llm.output_tokens
        self.metrics.messages_exchanged = self.bus.count
        return self.metrics

    def _counts(self):
        return {
            'tc_completed': len(self.metrics.completed_tc),
            'ntc_completed': len(self.metrics.completed_ntc),
            'tc_killed': len(self.metrics.killed_tc),
            'ntc_killed': len(self.metrics.killed_ntc),
        }
