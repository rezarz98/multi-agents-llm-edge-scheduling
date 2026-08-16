"""Edge-server agent: bids on announced tasks, manages its local policy.

Fast path (every task): deterministic bid computation from GPU clocks.
Slow path (once per window): a light LLM call that sets the agent's
bidding policy — whether to accept NTC work and how conservative to bid.
"""

from .messages import Bid, LoadReport, PolicyDecision
from lib.sim_core import ExecutionEngine

POLICY_SCHEMA = {
    'type': 'object',
    'properties': {
        'accept_ntc': {'type': 'boolean'},
        'bid_margin': {'type': 'number'},
        'rationale': {'type': 'string'},
    },
    'required': ['accept_ntc', 'bid_margin', 'rationale'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = (
    'You are the resource-management agent of edge server {sid} in a '
    'multi-access edge computing cluster. Tasks are time-critical (TC) or '
    'non-time-critical (NTC). You bid on tasks in an auction run by a '
    'broker. The cluster objective is a weighted throughput score: each '
    'completed TC task is worth 2 points, each completed NTC task 1 point. '
    'Decide your bidding policy for the next window: accept_ntc and '
    'bid_margin (>=0, inflate completion estimates when your load is '
    'volatile; usually 0). Set accept_ntc=false ONLY when your server is so '
    'overloaded that accepting NTC would push an otherwise-feasible TC task '
    'past its deadline -- declining NTC that would not have displaced any TC '
    'just throws away points. Keep the rationale to one sentence.'
)


class EdgeAgent:
    def __init__(self, server, llm, bus):
        self.server = server
        self.llm = llm
        self.bus = bus
        self.name = f'edge-{server.id}'
        self.policy = {'accept_ntc': True, 'bid_margin': 0.0}

    # ------------------------------------------------------------------
    def local_load(self):
        times = [g.current_time for g in self.server.gpus]
        return sum(times) / len(times) if times else 0.0

    def report(self, window_id):
        times = [g.current_time for g in self.server.gpus]
        msg = LoadReport(
            sender=self.name, recipient='broker', window_id=window_id,
            mean_gpu_time=round(self.local_load(), 3),
            max_gpu_time=round(max(times, default=0.0), 3),
        )
        return self.bus.send(msg)

    # ------------------------------------------------------------------
    # Slow path: per-window policy (LLM or deterministic mock)
    # ------------------------------------------------------------------
    def decide_window_policy(self, window_id, window_summary, cluster_mean_load):
        my_load = self.local_load()

        def mock():
            # Under the weighted objective, declining NTC rarely pays off, so
            # the mock keeps NTC and lets TC-first ordering do the prioritising.
            return {
                'accept_ntc': True,
                'bid_margin': 0.0,
                'rationale': 'mock heuristic: accept NTC; TC priority is '
                             'handled by broker ordering',
            }

        state = {
            'window': window_summary,
            'my_mean_gpu_time': round(my_load, 2),
            'cluster_mean_gpu_time': round(cluster_mean_load, 2),
            'current_policy': self.policy,
        }
        decision = self.llm.query(
            system=SYSTEM_PROMPT.format(sid=self.server.id),
            user=str(state),
            schema=POLICY_SCHEMA,
            mock_fn=mock,
        )
        self.policy = {k: decision[k] for k in ('accept_ntc', 'bid_margin')}
        self.bus.send(PolicyDecision(
            sender=self.name, recipient='broker', window_id=window_id,
            decision=self.policy, rationale=decision.get('rationale', ''),
        ))

    # ------------------------------------------------------------------
    # Fast path: per-task bid
    # ------------------------------------------------------------------
    def bid(self, task):
        # Under the weighted objective, refusing a feasible NTC almost never
        # pays off (it trades many NTC points for at most one TC). Edge agents
        # therefore bid on all feasible work; TC priority is enforced by the
        # broker's TC-first ordering, not by admission refusal. `accept_ntc`
        # is retained in the policy record for the trace but is advisory:
        # a decline is only honoured when the server is genuinely saturated
        # (its lightest GPU is already past this task's due date).
        is_tc = task['PriorityClass'] == 'TC'
        gpu, end = ExecutionEngine.best_gpu(self.server, task)

        if (not is_tc and not self.policy['accept_ntc']
                and gpu is not None and gpu.current_time > task['Due Date']):
            return self.bus.send(Bid(
                sender=self.name, recipient='broker', task_id=task['taskid'],
                feasible=False, est_end=float('inf'),
                load=round(self.local_load(), 3), declined=True,
            ))

        est_end = end + self.policy['bid_margin']
        feasible = gpu is not None and est_end <= task['Due Date']

        return self.bus.send(Bid(
            sender=self.name, recipient='broker', task_id=task['taskid'],
            feasible=feasible, est_end=round(est_end, 3),
            load=round(self.local_load(), 3),
        ))

    # ------------------------------------------------------------------
    def execute(self, task, metrics):
        """Run an awarded task on the locally best GPU."""
        gpu, _ = ExecutionEngine.best_gpu(self.server, task)
        if gpu is None:
            metrics.record(task, completed=False)
            return False
        return ExecutionEngine.execute(self.server, gpu, task, metrics)
