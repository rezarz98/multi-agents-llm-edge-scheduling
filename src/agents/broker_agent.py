"""Broker agent: announces task windows, orders them, and awards bids.

Fast path (every task): sequential award — fresh bids are collected per
task so estimates never go stale, TC-first EDF ordering, best feasible
bid wins.
Slow path (once per window): a light LLM call that sets the window's
auction policy — the tie-break rule for near-equal bids and whether to
early-reject infeasible NTC tasks.
"""

import heapq

from .messages import TaskAnnounce, Award, PolicyDecision

POLICY_SCHEMA = {
    'type': 'object',
    'properties': {
        'tie_break': {'type': 'string', 'enum': ['fastest', 'load_balance']},
        'reject_infeasible_ntc': {'type': 'boolean'},
        'rationale': {'type': 'string'},
    },
    'required': ['tie_break', 'reject_infeasible_ntc', 'rationale'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = (
    'You are the broker agent of a multi-access edge computing cluster. Each '
    'window you auction released tasks to edge-server agents. Tasks are '
    'time-critical (TC) or non-time-critical (NTC). The objective is a '
    'weighted throughput score: each completed TC task is worth 2 points, '
    'each completed NTC task 1 point, minus a small makespan penalty. Decide '
    'the auction policy for the next window: tie_break ("fastest" awards '
    'near-equal bids to the earliest finisher, "load_balance" to the '
    'least-loaded server -- prefer load_balance when server loads have '
    'drifted apart) and reject_infeasible_ntc (drop NTC tasks that no server '
    'can finish on time instead of dispatching them). One-sentence rationale.'
)

TIE_EPSILON = 0.5  # bids within this end-time gap count as a tie


class BrokerAgent:
    def __init__(self, edge_agents, llm, bus, rng):
        self.edge_agents = edge_agents
        self.llm = llm
        self.bus = bus
        self.rng = rng
        self.name = 'broker'
        self.policy = {'tie_break': 'fastest', 'reject_infeasible_ntc': False}

    # ------------------------------------------------------------------
    def announce(self, window_id, window):
        self.bus.send(TaskAnnounce(
            sender=self.name, recipient='all-edges', window_id=window_id,
            task_ids=[t['taskid'] for t in window],
        ))

    # ------------------------------------------------------------------
    def decide_window_policy(self, window_id, window_summary, load_reports):
        loads = [r.mean_gpu_time for r in load_reports]
        spread = (max(loads) - min(loads)) if loads else 0.0
        mean = sum(loads) / len(loads) if loads else 0.0

        def mock():
            drifted = mean > 0 and spread > 0.3 * mean
            return {
                'tie_break': 'load_balance' if drifted else 'fastest',
                'reject_infeasible_ntc': False,
                'rationale': 'mock heuristic: balance loads once the spread '
                             'exceeds 30% of the mean',
            }

        state = {
            'window': window_summary,
            'server_loads': loads,
            'load_spread': round(spread, 2),
            'current_policy': self.policy,
        }
        decision = self.llm.query(
            system=SYSTEM_PROMPT, user=str(state),
            schema=POLICY_SCHEMA, mock_fn=mock,
        )
        self.policy = {k: decision[k] for k in ('tie_break', 'reject_infeasible_ntc')}
        self.bus.send(PolicyDecision(
            sender=self.name, recipient='all-edges', window_id=window_id,
            decision=self.policy, rationale=decision.get('rationale', ''),
        ))

    # ------------------------------------------------------------------
    def order_tasks(self, window, params):
        """TC before NTC; EDF within class.

        With params.order_policy == 'semi_greedy', the within-class order is
        randomised over the top-3 earliest deadlines with params.pick_probs
        (the semi-greedy mechanism from prior work, here as a fast-path
        policy the monitor agent can toggle).
        """
        tc = sorted((t for t in window if t['PriorityClass'] == 'TC'),
                    key=lambda t: t['Due Date'])
        ntc = sorted((t for t in window if t['PriorityClass'] != 'TC'),
                     key=lambda t: t['Due Date'])

        if params.order_policy == 'semi_greedy':
            tc = self._semi_greedy_shuffle(tc, params.pick_probs)
            ntc = self._semi_greedy_shuffle(ntc, params.pick_probs)
        return tc + ntc

    def _semi_greedy_shuffle(self, ordered, probs):
        pool = [(t['Due Date'], i, t) for i, t in enumerate(ordered)]
        heapq.heapify(pool)
        result = []
        while pool:
            k = min(3, len(pool))
            cands = heapq.nsmallest(k, pool)
            weights = list(probs[:k])
            pick = self.rng.choices(range(k), weights=weights, k=1)[0]
            chosen = cands[pick]
            pool.remove(chosen)
            heapq.heapify(pool)
            result.append(chosen[2])
        return result

    # ------------------------------------------------------------------
    def award(self, task):
        """Collect fresh bids for one task and award it. Returns the
        winning edge agent, or None if the task is rejected."""
        bids = [agent.bid(task) for agent in self.edge_agents]
        feasible = [b for b in bids if b.feasible]

        if not feasible:
            if (task['PriorityClass'] != 'TC'
                    and self.policy['reject_infeasible_ntc']):
                return None  # early reject: killed without dispatch
            # Dispatch to the fastest server anyway (it will be killed there,
            # same accounting as prior work).
            candidates = [b for b in bids if not b.declined] or bids
            best = min(candidates, key=lambda b: b.est_end)
        else:
            best_end = min(b.est_end for b in feasible)
            tied = [b for b in feasible if b.est_end <= best_end + TIE_EPSILON]
            if self.policy['tie_break'] == 'load_balance' and len(tied) > 1:
                best = min(tied, key=lambda b: b.load)
            else:
                best = min(tied, key=lambda b: b.est_end)

        agent = next(a for a in self.edge_agents if a.name == best.sender)
        self.bus.send(Award(
            sender=self.name, recipient=best.sender, task_id=task['taskid'],
            server_id=agent.server.id,
        ))
        return agent
