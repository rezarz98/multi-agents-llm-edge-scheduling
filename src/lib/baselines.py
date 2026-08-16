"""Classical literature baseline schedulers for heterogeneous edge computing.

All baselines run through the shared ONLINE dispatch driver
(sim_core.online_dispatch): at each decision they see only the released,
not-yet-scheduled tasks — the online/dynamic form of these heuristics.
This is the canonical "immediate / batch-mode dynamic mapping" of
Maheswaran et al. (1999), not the fully-offline static form, so the
comparison against our online agent method is fair (no method may look
into the future). Placement is earliest-finish-time.

Completion time of task i on GPU j of server s:

    CT(i, j) = max(t_j, release_i) + comm_AP(i) + comm_broker(i)
               + comm_BS_s(i) + proc_i / speed_j

References (verify exact bibliographic details before citing):

[1] O. H. Ibarra and C. E. Kim, "Heuristic Algorithms for Scheduling
    Independent Tasks on Nonidentical Processors," Journal of the ACM,
    24(2):280-289, 1977.                                  (Min-Min origin)
[2] T. D. Braun et al., "A Comparison of Eleven Static Heuristics for
    Mapping a Class of Independent Tasks onto Heterogeneous Distributed
    Computing Systems," Journal of Parallel and Distributed Computing,
    61(6):810-837, 2001.                (Min-Min / Max-Min benchmark study)
[3] M. Maheswaran, S. Ali, H. J. Siegel, D. Hensgen, and R. F. Freund,
    "Dynamic Mapping of a Class of Independent Tasks onto Heterogeneous
    Computing Systems," Journal of Parallel and Distributed Computing,
    59(2):107-131, 1999.        (Sufferage; immediate/batch dynamic modes)
[4] H. Topcuoglu, S. Hariri, and M.-Y. Wu, "Performance-Effective and
    Low-Complexity Task Scheduling for Heterogeneous Computing," IEEE
    Transactions on Parallel and Distributed Systems, 13(3):260-274,
    2002.                                                        (HEFT)
[5] M. L. Dertouzos and A. K. Mok, "Multiprocessor On-Line Scheduling
    of Hard-Real-Time Tasks," IEEE Transactions on Software
    Engineering, 15(12):1497-1506, 1989.        (Least Laxity First, LLF)

Notes for the paper:
* Min-Min / Max-Min / Sufferage are makespan-oriented mapping heuristics;
  they are deadline-unaware and are included to show that generic
  task-mapping heuristics underperform on a deadline-driven objective.
* HEFT is defined for DAGs; for independent tasks its upward rank reduces
  to average execution cost, so the online form dispatches the ready task
  with the largest average cost first.
* LLF is a native online real-time rule (least laxity first).
"""

from .sim_core import online_dispatch, ExecutionEngine


def _avg_cost(edge_servers, task):
    ends = [ExecutionEngine.estimate(es, gpu, task)[1]
            for es in edge_servers for gpu in es.gpus]
    return sum(ends) / len(ends)


def min_min(edge_servers, tasks_df):
    """Min-Min [1,2], online form: dispatch the ready task with the
    smallest earliest completion time."""
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t: ExecutionEngine.place(srv, t)[2]))


def max_min(edge_servers, tasks_df):
    """Max-Min [1,2], online form: dispatch the ready task with the
    largest earliest completion time."""
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: max(ready, key=lambda t: ExecutionEngine.place(srv, t)[2]))


def sufferage(edge_servers, tasks_df):
    """Sufferage [3], online form: dispatch the ready task that would
    suffer most (largest gap between its best and second-best completion
    time) if not placed on its best GPU now."""
    def suff(t, srv):
        ct = ExecutionEngine.completion_times(srv, t)
        return (ct[1] - ct[0]) if len(ct) >= 2 else ct[0]
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: max(ready, key=lambda t: suff(t, srv)))


def heft(edge_servers, tasks_df):
    """HEFT [4], independent-task online form: dispatch the ready task
    with the largest average execution cost (its upward rank reduces to
    average cost for independent tasks)."""
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: max(ready, key=lambda t: _avg_cost(srv, t)))


def llf(edge_servers, tasks_df):
    """LLF [5], online: dispatch the ready task with least laxity
    (due - now - processing)."""
    return online_dispatch(edge_servers, tasks_df,
                           lambda ready, now, srv: min(ready, key=lambda t:
                                                       t['Due Date'] - now - t['Processing Time']))


LITERATURE_BASELINES = {
    'min_min': min_min,
    'max_min': max_min,
    'sufferage': sufferage,
    'heft': heft,
    'llf': llf,
}
