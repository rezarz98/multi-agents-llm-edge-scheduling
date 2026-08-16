"""Typed messages exchanged between the scheduling agents.

Every message is a plain dataclass; the MessageBus serialises them to a
JSONL trace so a full negotiation transcript is available per run.
"""

from dataclasses import dataclass, asdict


@dataclass
class Message:
    sender: str
    recipient: str

    def to_dict(self):
        d = asdict(self)
        d['type'] = type(self).__name__
        return d


@dataclass
class TaskAnnounce(Message):
    """Broker -> edge agents: a window of released tasks is up for auction."""
    window_id: int
    task_ids: list


@dataclass
class Bid(Message):
    """Edge agent -> broker: offer to run one task."""
    task_id: int
    feasible: bool
    est_end: float
    load: float
    declined: bool = False


@dataclass
class Award(Message):
    """Broker -> edge agent: task assigned."""
    task_id: int
    server_id: int


@dataclass
class LoadReport(Message):
    """Edge agent -> broker/monitor: local state summary."""
    window_id: int
    mean_gpu_time: float
    max_gpu_time: float


@dataclass
class PolicyDecision(Message):
    """An LLM (or mock) policy decision taken by an agent."""
    window_id: int
    decision: dict
    rationale: str = ''


@dataclass
class PolicyAdjust(Message):
    """Monitor -> broker/edge agents: adjust fast-path parameters."""
    window_id: int
    params: dict
    rationale: str = ''
