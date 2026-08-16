"""Windowed variants of the metaheuristic and DRL baselines.

The baselines in baselines_recent.py dispatch online (event-driven over
released tasks). Their windowed counterparts here batch each release-ordered
admission window of size W and optimise / act within it, carrying GPU state
across windows -- giving the learning baselines the same batching lookahead
that the proposed windowed scheduler enjoys. This isolates how much of the
proposed method's margin over GA/PSO/DDQN is the batching horizon rather than
the search itself.

PSO/GA: optimise per-window priority keys, each window's dispatch starting from
the carried GPU state. DDQN: trained identically on full-instance episodes
(unchanged offline advantage), then evaluated greedily window-by-window with the
full-instance feature scales, so only the dispatch structure is windowed.
"""
import numpy as np
import pandas as pd

from .sim_core import SimMetrics
from .baselines_recent import (_Workload, _fitness, _QNet, _ddqn_episode,
                               _K)


def _gpu_times(edge_servers):
    return np.array([g.current_time for es in edge_servers for g in es.gpus],
                    dtype=float)


def _merge(master, m):
    master.completed_tc.extend(m.completed_tc)
    master.completed_ntc.extend(m.completed_ntc)
    master.killed_tc.extend(m.killed_tc)
    master.killed_ntc.extend(m.killed_ntc)
    master.total_benefit += m.total_benefit


def _windows(tasks_df, window):
    recs = tasks_df.sort_values('Release Time').to_dict('records')
    for i in range(0, len(recs), window):
        yield pd.DataFrame(recs[i:i + window])


# ---------------------------------------------------------------------------
def pso_windowed(edge_servers, tasks_df, window=40, swarm=24, iters=30,
                 w=0.7, c1=1.5, c2=1.5, seed=42):
    rng = np.random.default_rng(seed)
    master = SimMetrics()
    for wdf in _windows(tasks_df, window):
        wl = _Workload(edge_servers, wdf)
        n = wl.n
        if n == 0:
            continue
        start = _gpu_times(edge_servers)

        def fit(p):
            return _fitness(*wl.run_keyed(p, init_gpu_time=start))

        pos = rng.uniform(0, 1, size=(swarm, n))
        vel = rng.uniform(-0.1, 0.1, size=(swarm, n))
        pbest = pos.copy()
        pbest_fit = np.array([fit(p) for p in pos])
        g_idx = int(pbest_fit.argmax())
        gbest, gbest_fit = pbest[g_idx].copy(), float(pbest_fit[g_idx])
        for _ in range(iters):
            r1 = rng.uniform(0, 1, size=(swarm, n))
            r2 = rng.uniform(0, 1, size=(swarm, n))
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, 0.0, 1.0)
            for s in range(swarm):
                f = fit(pos[s])
                if f > pbest_fit[s]:
                    pbest_fit[s], pbest[s] = f, pos[s].copy()
                    if f > gbest_fit:
                        gbest_fit, gbest = f, pos[s].copy()
        met, _ = wl.run_keyed(gbest, commit=True, init_gpu_time=start)
        _merge(master, met)
    return master


# ---------------------------------------------------------------------------
def ga_windowed(edge_servers, tasks_df, window=40, pop=30, gens=30, elite=2,
                mut_rate=0.1, mut_scale=0.15, tournament=3, seed=42):
    rng = np.random.default_rng(seed)
    master = SimMetrics()
    for wdf in _windows(tasks_df, window):
        wl = _Workload(edge_servers, wdf)
        n = wl.n
        if n == 0:
            continue
        start = _gpu_times(edge_servers)

        def fit(ind):
            return _fitness(*wl.run_keyed(ind, init_gpu_time=start))

        population = rng.uniform(0, 1, size=(pop, n))
        fitness = np.array([fit(ind) for ind in population])

        def select():
            idx = rng.integers(0, pop, size=tournament)
            return population[idx[np.argmax(fitness[idx])]]

        for _ in range(gens):
            order = np.argsort(-fitness)
            new_pop = [population[order[k]].copy() for k in range(elite)]
            while len(new_pop) < pop:
                p1, p2 = select(), select()
                alpha = rng.uniform(0, 1, size=n)
                child = alpha * p1 + (1 - alpha) * p2
                mask = rng.uniform(0, 1, size=n) < mut_rate
                child[mask] += rng.normal(0, mut_scale, size=int(mask.sum()))
                new_pop.append(np.clip(child, 0.0, 1.0))
            population = np.array(new_pop)
            fitness = np.array([fit(ind) for ind in population])
        best = population[int(fitness.argmax())]
        met, _ = wl.run_keyed(best, commit=True, init_gpu_time=start)
        _merge(master, met)
    return master


# ---------------------------------------------------------------------------
def ddqn_windowed(edge_servers, tasks_df, window=40, episodes=60, hidden=64,
                  gamma=0.99, lr=1e-3, batch=32, buffer=5000, target_every=500,
                  eps_start=1.0, eps_end=0.05, seed=42):
    """Train exactly like ddqn() on full-instance episodes, then evaluate the
    greedy policy window-by-window, carrying GPU state and using the
    full-instance feature scales so only dispatch is windowed."""
    full = _Workload(edge_servers, tasks_df)
    if full.n == 0:
        return SimMetrics()
    rng = np.random.default_rng(seed)
    in_dim = _K * 5 + 2
    online = _QNet(in_dim, hidden, _K, rng)
    target = _QNet(in_dim, hidden, _K, rng)
    target.copy_from(online)

    replay, steps = [], [0]
    total_train_steps = episodes * full.n

    def masked_q(net, state, n_valid):
        q = net.predict(state).copy()
        if n_valid < _K:
            q[n_valid:] = -np.inf
        return q

    def train_policy(state, n_valid):
        e = eps_end + (eps_start - eps_end) * max(
            0.0, 1 - steps[0] / max(1, 0.8 * total_train_steps))
        steps[0] += 1
        if rng.uniform() < e:
            return int(rng.integers(0, n_valid))
        return int(np.argmax(masked_q(online, state, n_valid)))

    def learn():
        if len(replay) < batch:
            return
        idx = rng.integers(0, len(replay), size=batch)
        S = np.array([replay[k][0] for k in idx])
        A = np.array([replay[k][1] for k in idx])
        R = np.array([replay[k][2] for k in idx])
        S2 = np.array([replay[k][3] for k in idx])
        D = np.array([replay[k][4] for k in idx], dtype=float)
        q, cache = online.forward(S)
        q_next_online, _ = online.forward(S2)
        q_next_target, _ = target.forward(S2)
        a_star = q_next_online.argmax(axis=1)
        double_q = q_next_target[np.arange(batch), a_star]
        y = R + gamma * (1 - D) * double_q
        dq = np.zeros_like(q)
        pred = q[np.arange(batch), A]
        dq[np.arange(batch), A] = np.clip(pred - y, -1.0, 1.0)
        online.sgd_step(cache, dq, lr=lr)

    for _ in range(episodes):
        _, _, transitions = _ddqn_episode(full, train_policy)
        for tr in transitions:
            replay.append(tr)
            if len(replay) > buffer:
                replay.pop(0)
            learn()
            if steps[0] % target_every == 0:
                target.copy_from(online)

    def greedy(state, n_valid):
        return int(np.argmax(masked_q(online, state, n_valid)))

    # Windowed greedy evaluation with full-instance feature scales.
    master = SimMetrics()
    for wdf in _windows(tasks_df, window):
        wl = _Workload(edge_servers, wdf)
        if wl.n == 0:
            continue
        wl.proc_scale, wl.due_scale = full.proc_scale, full.due_scale
        gpu_time = _gpu_times(edge_servers)
        met, _, _ = _ddqn_episode(wl, greedy, gpu_time=gpu_time)
        for g, t in zip(wl.gpus_flat, gpu_time):
            g.current_time = float(t)
        _merge(master, met)
    return master


WINDOWED_BASELINES = {
    'PSO-win': pso_windowed,
    'GA-win': ga_windowed,
    'DDQN-win': ddqn_windowed,
}
