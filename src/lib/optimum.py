"""Offline optimum and feasibility ceiling.

`feasibility_ceiling` counts the tasks that COULD complete on some PU if
that PU were free at the task's arrival (an upper bound that ignores
contention). `offline_optimum` solves the true contention-aware maximum
weighted on-time throughput with CP-SAT: each task may run on at most one
PU, non-preemptively, in a window that starts no earlier than its arrival
(release + communication) and ends no later than its deadline, with no
two tasks overlapping on a PU. These establish the achievable ceiling
against which every scheduler's optimality gap is reported.
"""

import math

from ortools.sat.python import cp_model

from .sim_core import W_TC, W_NTC

SCALE = 100  # seconds -> integer centi-units for CP-SAT


def _machines(edge_servers):
    return [(es, gpu) for es in edge_servers for gpu in es.gpus]


def _comm(task, server):
    return (task['Access Point Communication Time']
            + task['Broker Communication Time']
            + task[f'Base Station {server.id} Communication Time'])


def feasibility_ceiling(edge_servers, tasks_df):
    """Per-class count of tasks feasible on at least one PU (ignoring
    contention)."""
    machines = _machines(edge_servers)
    tc = ntc = 0
    for _, t in tasks_df.iterrows():
        ok = any(t['Release Time'] + _comm(t, es) + t['Processing Time'] / g.speed
                 <= t['Due Date'] for es, g in machines)
        if ok:
            if t['PriorityClass'] == 'TC':
                tc += 1
            else:
                ntc += 1
    total_tc = int((tasks_df['PriorityClass'] == 'TC').sum())
    total_ntc = len(tasks_df) - total_tc
    return {'tc_feasible': tc, 'ntc_feasible': ntc,
            'tc_total': total_tc, 'ntc_total': total_ntc}


def offline_optimum(edge_servers, tasks_df, time_limit=25.0, workers=8):
    machines = _machines(edge_servers)
    records = tasks_df.to_dict('records')
    model = cp_model.CpModel()

    present_by_task = []
    per_machine = {m: [] for m in range(len(machines))}
    weights, is_tc = [], []

    for t in records:
        rel = t['Release Time']
        due = t['Due Date']
        lits = []
        for mi, (es, g) in enumerate(machines):
            e = int(math.ceil((rel + _comm(t, es)) * SCALE))
            p = int(math.ceil((t['Processing Time'] / g.speed) * SCALE))
            latest = int(math.floor(due * SCALE))
            if e + p > latest:
                continue
            lit = model.NewBoolVar(f'x_{t["taskid"]}_{mi}')
            start = model.NewIntVar(e, latest - p, f's_{t["taskid"]}_{mi}')
            iv = model.NewOptionalIntervalVar(start, p, start + p, lit,
                                              f'i_{t["taskid"]}_{mi}')
            per_machine[mi].append(iv)
            lits.append(lit)
        assigned = model.NewBoolVar(f'a_{t["taskid"]}')
        if lits:
            model.Add(sum(lits) == assigned)
        else:
            model.Add(assigned == 0)
        present_by_task.append(assigned)
        weights.append(W_TC if t['PriorityClass'] == 'TC' else W_NTC)
        is_tc.append(t['PriorityClass'] == 'TC')

    for mi in per_machine:
        if per_machine[mi]:
            model.AddNoOverlap(per_machine[mi])

    model.Maximize(sum(int(w) * a for w, a in zip(weights, present_by_task)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)

    tc_opt = sum(1 for a, tc in zip(present_by_task, is_tc)
                 if tc and solver.Value(a) == 1)
    ntc_opt = sum(1 for a, tc in zip(present_by_task, is_tc)
                  if (not tc) and solver.Value(a) == 1)
    return {
        'tc_opt': tc_opt, 'ntc_opt': ntc_opt,
        'weighted_opt': solver.ObjectiveValue(),
        'weighted_bound': solver.BestObjectiveBound(),
        'proven_optimal': status == cp_model.OPTIMAL,
    }
