"""Classic priority dispatching rules used as baselines in our prior work
(the SARS paper) and in the wider real-time / job-shop scheduling
literature.

All rules run through the shared ONLINE dispatch driver
(sim_core.online_dispatch): at each decision they see only the released,
not-yet-scheduled tasks, exactly like every other scheduler in this
framework (including our agent methods). None can use knowledge of tasks
that have not yet arrived. Placement onto a GPU is greedy
earliest-finish-time. Rules are class-agnostic (they do not special-case
TC/NTC) except ERA, which categorises tasks into priority tiers; the
TC/NTC weighting lives in the reported objective (W_TC, W_NTC).

Rules
-----
FCFS  : First-Come-First-Served — earliest release time first.
EDF   : Earliest Deadline First — earliest due date among ready tasks.
EDD   : Earliest Due Date — earliest due date among ready tasks. In this
        non-preemptive online setting EDF and EDD coincide; both are
        reported because both appear in our prior work's baseline set.
EFDF  : Earliest Feasible Deadline First — among ready tasks that can
        still meet their deadline, earliest due date; if none is
        feasible, shortest processing time.
CR    : Critical Ratio — smallest (due - now) / processing time first.
COVERT: Cost OVER Time — largest c/t priority first, where
        c_i = max(0, 1 - max(0, slack_i)/(k * p_i)) / p_i with
        slack_i = due_i - now - p_i (look-ahead k).
ERA   : Efficient Resource Allocation — priority-tier categorisation
        (high/medium/low), adapted from Choudhari et al. [17] as cited in
        our SARS paper: time-critical tasks are high priority, urgent
        non-time-critical tasks medium, the rest low; earliest deadline
        within a tier.

References
----------
Standard dispatching rules (FCFS/EDF/EDD/CR/COVERT) are covered in, e.g.,
M. L. Pinedo, "Scheduling: Theory, Algorithms, and Systems," Springer.
COVERT: Carroll, D. C., "Heuristic sequencing of single and multiple
component jobs," PhD thesis, MIT, 1965. EDF: Liu & Layland, JACM 1973.
ERA (priority categorisation): Choudhari et al. (ref [17] in the SARS
paper). These are the same baselines listed in our prior work.
"""

from .sim_core import online_dispatch, ExecutionEngine

COVERT_K = 2.0  # COVERT look-ahead parameter


def fcfs(edge_servers, tasks_df):
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t: t['Release Time']))


def edf(edge_servers, tasks_df):
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t: (t['Due Date'], t['Processing Time'])))


def edd(edge_servers, tasks_df):
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t: (t['Due Date'], t['Release Time'])))


def efdf(edge_servers, tasks_df):
    def choose(ready, now, srv):
        feasible = [t for t in ready
                    if ExecutionEngine.place(srv, t)[2] <= t['Due Date']]
        if feasible:
            return min(feasible, key=lambda t: t['Due Date'])
        return min(ready, key=lambda t: t['Processing Time'])
    return online_dispatch(edge_servers, tasks_df, choose)


def cr(edge_servers, tasks_df):
    def choose(ready, now, srv):
        return min(ready, key=lambda t:
                   (t['Due Date'] - now) / max(1e-9, t['Processing Time']))
    return online_dispatch(edge_servers, tasks_df, choose)


def covert(edge_servers, tasks_df):
    def priority(t, now):
        p = max(1e-9, t['Processing Time'])
        slack = t['Due Date'] - now - p
        return max(0.0, 1.0 - max(0.0, slack) / (COVERT_K * p)) / p
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: max(ready, key=lambda t: priority(t, now)))


def era(edge_servers, tasks_df):
    """Priority-tier categorisation adapted from Choudhari et al. [17].
    Tier 0 (high): TC tasks. Tier 1 (medium): urgent NTC (laxity <= its
    processing time). Tier 2 (low): remaining NTC. Earliest deadline
    within a tier."""
    def tier(t, now):
        if t['PriorityClass'] == 'TC':
            return 0
        laxity = t['Due Date'] - now - t['Processing Time']
        return 1 if laxity <= t['Processing Time'] else 2
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t: (tier(t, now), t['Due Date'])))


DISPATCH_RULES = {
    'fcfs': fcfs,
    'edf': edf,
    'edd': edd,
    'efdf': efdf,
    'cr': cr,
    'covert': covert,
    'era': era,
}
