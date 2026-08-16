"""Monitor agent: the meta-scheduling layer of HAS.

Every K windows it reviews rolling outcome statistics and adjusts the
fast-path parameters the broker uses — ordering policy (strict EDF vs a
semi-greedy randomised top-3 order) and NTC deferral. This gives an
online, explainable adjustment loop over the scheduler's dispatch
policy.
"""

from dataclasses import dataclass, asdict

from .messages import PolicyAdjust

ADJUST_SCHEMA = {
    'type': 'object',
    'properties': {
        'order_policy': {'type': 'string', 'enum': ['edf', 'semi_greedy']},
        'p1': {'type': 'number'},
        'p2': {'type': 'number'},
        'p3': {'type': 'number'},
        'ntc_defer_windows': {'type': 'integer', 'enum': [0, 1, 2]},
        'rationale': {'type': 'string'},
    },
    'required': ['order_policy', 'p1', 'p2', 'p3', 'ntc_defer_windows',
                 'rationale'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = (
    'You are the monitoring agent of a multi-access edge computing cluster '
    'running an auction-based scheduler. The objective is a weighted '
    'throughput score: each completed TC task is worth 2 points, each NTC '
    'task 1 point. You periodically tune the fast-path parameters. '
    'order_policy: "edf" (strict earliest-deadline-first) or "semi_greedy" '
    '(randomised top-3 EDF with pick probabilities p1,p2,p3 summing to ~1; '
    'diversifies placement under contention). ntc_defer_windows: defer NTC '
    'tasks with large slack by 0-2 windows to protect TC deadlines. If the '
    'TC kill rate is rising, protect TC (strict EDF, defer NTC). If TC is '
    'safe but NTC kills are high, relax to recover NTC points. '
    'One-sentence rationale.'
)


@dataclass
class FastPathParams:
    order_policy: str = 'edf'
    pick_probs: tuple = (0.5, 0.3, 0.2)
    ntc_defer_windows: int = 0

    def to_dict(self):
        return asdict(self)


class MonitorAgent:
    def __init__(self, llm, bus, review_every=5):
        self.llm = llm
        self.bus = bus
        self.review_every = review_every
        self.name = 'monitor'

    def review(self, window_id, rolling_stats, params):
        """Return (possibly updated) FastPathParams."""
        tc_total = rolling_stats['tc_completed'] + rolling_stats['tc_killed']
        ntc_total = rolling_stats['ntc_completed'] + rolling_stats['ntc_killed']
        tc_kill_rate = rolling_stats['tc_killed'] / tc_total if tc_total else 0.0
        ntc_kill_rate = rolling_stats['ntc_killed'] / ntc_total if ntc_total else 0.0

        def mock():
            if tc_kill_rate > 0.10:
                return {'order_policy': 'edf', 'p1': 0.5, 'p2': 0.3, 'p3': 0.2,
                        'ntc_defer_windows': 1,
                        'rationale': 'mock heuristic: TC kill rate above 10%, '
                                     'protect TC work'}
            if ntc_kill_rate > 0.25:
                return {'order_policy': 'semi_greedy', 'p1': 0.5, 'p2': 0.3,
                        'p3': 0.2, 'ntc_defer_windows': 0,
                        'rationale': 'mock heuristic: TC safe but NTC kills '
                                     'high, diversify placement'}
            return {'order_policy': params.order_policy,
                    'p1': params.pick_probs[0], 'p2': params.pick_probs[1],
                    'p3': params.pick_probs[2],
                    'ntc_defer_windows': params.ntc_defer_windows,
                    'rationale': 'mock heuristic: kill rates nominal, keep '
                                 'current parameters'}

        state = {
            'tc_kill_rate': round(tc_kill_rate, 3),
            'ntc_kill_rate': round(ntc_kill_rate, 3),
            'rolling_stats': rolling_stats,
            'current_params': params.to_dict(),
        }
        decision = self.llm.query(
            system=SYSTEM_PROMPT, user=str(state),
            schema=ADJUST_SCHEMA, mock_fn=mock,
        )

        probs = [max(0.0, decision['p1']), max(0.0, decision['p2']),
                 max(0.0, decision['p3'])]
        total = sum(probs) or 1.0
        new_params = FastPathParams(
            order_policy=decision['order_policy'],
            pick_probs=tuple(p / total for p in probs),
            ntc_defer_windows=int(decision['ntc_defer_windows']),
        )
        self.bus.send(PolicyAdjust(
            sender=self.name, recipient='broker', window_id=window_id,
            params=new_params.to_dict(),
            rationale=decision.get('rationale', ''),
        ))
        return new_params
