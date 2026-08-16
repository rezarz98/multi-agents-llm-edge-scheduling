"""Recent (2021-2024) baseline schedulers for heterogeneous edge computing.

These complement the classical heuristics in lib/baselines.py with the
two families that dominate the current MEC task-scheduling literature:
metaheuristics (PSO, GA) and deep reinforcement learning (Double DQN).

All three schedule the same workload model as the rest of the framework
(independent tasks, release times, hard due dates, per-server
communication times, heterogeneous GPU speeds) and place each dispatched
task on its earliest-finish-time GPU. They differ only in the ORDER in
which ready tasks are dispatched:

  pso  : each task gets a real-valued priority key; particles search key
         space to maximise weighted on-time completions.
  ga   : same random-key encoding, optimised by a genetic algorithm.
  ddqn : a Double Deep Q-Network selects, at each step, which of the
         three earliest-deadline ready tasks to dispatch next.

Implemented with numpy only (no torch), fully seeded and reproducible.

References (the algorithm families are standard; verify the exact
bibliographic details of the recent-application citations before use):

[1] J. Kennedy and R. Eberhart, "Particle Swarm Optimization,"
    Proc. IEEE Int. Conf. on Neural Networks (ICNN), 1995, pp. 1942-1948.
                                                        (PSO, origin)
[2] B. Wang, J. Cheng, J. Cao, C. Wang, and W. Huang, "Integer Particle
    Swarm Optimization Based Task Scheduling for Device-Edge-Cloud
    Cooperative Computing to Improve SLA Satisfaction," PeerJ Computer
    Science, 8:e893, 2022, doi:10.7717/peerj-cs.893.
                                      (recent PSO-for-MEC application)
[3] J. H. Holland, "Adaptation in Natural and Artificial Systems,"
    Univ. of Michigan Press, 1975; D. E. Goldberg, "Genetic Algorithms
    in Search, Optimization, and Machine Learning," Addison-Wesley,
    1989.                                                 (GA, origin)
[4] Y. Wang, P. Zhang, B. Wang, Z. Zhang, Y. Xu, and B. Lv, "A Hybrid
    PSO and GA Algorithm with Rescheduling for Task Offloading in
    Device-Edge-Cloud Collaborative Computing," Cluster Computing, 2024,
    doi:10.1007/s10586-024-04851-3.
                                       (recent GA-for-MEC application)
[5] H. van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning
    with Double Q-Learning," Proc. AAAI, 2016, pp. 2094-2100.
                                                        (Double DQN)
[6] L. Zeng, Q. Liu, S. Shen, and X. Liu, "Improved Double Deep Q
    Network-Based Task Scheduling Algorithm in Edge Computing for
    Makespan Optimization," Tsinghua Science and Technology,
    29(3):806-817, 2024, doi:10.26599/TST.2023.9010058.
                                       (recent DDQN-for-edge application)
"""

import heapq

import numpy as np

from .sim_core import SimMetrics, W_TC, W_NTC


# ---------------------------------------------------------------------------
# Vectorised workload + shared dispatch primitives
# ---------------------------------------------------------------------------
class _Workload:
    """Array view of a task set + edge cluster with an earliest-finish-time
    dispatch driver. Ordering policies decide *which* ready task runs next;
    placement is always earliest-finish-time."""

    def __init__(self, edge_servers, tasks_df):
        self.edge_servers = edge_servers
        self.records = tasks_df.to_dict('records')
        self.n = len(self.records)

        self.gpus_flat = [g for es in edge_servers for g in es.gpus]
        self.m = len(self.gpus_flat)
        self.speed = np.array([g.speed for g in self.gpus_flat], dtype=float)
        self.server_of = np.array(
            [si for si, es in enumerate(edge_servers) for _ in es.gpus],
            dtype=int)

        self.rel = np.array([t['Release Time'] for t in self.records], dtype=float)
        self.proc = np.array([t['Processing Time'] for t in self.records], dtype=float)
        self.due = np.array([t['Due Date'] for t in self.records], dtype=float)
        self.is_tc = np.array([t['PriorityClass'] == 'TC' for t in self.records])
        self.comm = np.array([
            [t['Access Point Communication Time']
             + t['Broker Communication Time']
             + t[f'Base Station {es.id} Communication Time']
             for es in edge_servers]
            for t in self.records
        ], dtype=float)

        self.release_order = list(np.argsort(self.rel, kind='stable'))
        self.proc_scale = max(1.0, float(self.proc.mean()))
        self.due_scale = max(1.0, float(self.due.mean()))

    # -- earliest-finish-time placement of one task -----------------------
    def eft(self, i, gpu_time):
        # communication overlaps queueing: arrival = release + comm
        arrival = self.rel[i] + self.comm[i][self.server_of]
        end_all = np.maximum(gpu_time, arrival) + self.proc[i] / self.speed
        j = int(end_all.argmin())
        return j, float(end_all[j])

    # -- dispatch under a fixed real-valued priority key per task ---------
    def run_keyed(self, keys, commit=False, init_gpu_time=None):
        gpu_time = (np.zeros(self.m) if init_gpu_time is None
                    else np.asarray(init_gpu_time, dtype=float).copy())
        metrics = SimMetrics()
        order, ptr, ready = self.release_order, 0, []

        while ptr < self.n or ready:
            current = gpu_time.min()
            while ptr < self.n and self.rel[order[ptr]] <= current:
                i = order[ptr]
                heapq.heappush(ready, (keys[i], i))
                ptr += 1
            if not ready:
                current = self.rel[order[ptr]]
                while ptr < self.n and self.rel[order[ptr]] <= current:
                    i = order[ptr]
                    heapq.heappush(ready, (keys[i], i))
                    ptr += 1

            _, i = heapq.heappop(ready)
            j, end = self.eft(i, gpu_time)
            if end <= self.due[i]:
                gpu_time[j] = end
                metrics.record(self.records[i], completed=True)
            else:
                metrics.record(self.records[i], completed=False)

        if commit:
            for g, t in zip(self.gpus_flat, gpu_time):
                g.current_time = float(t)
        return metrics, float(gpu_time.max())


def _fitness(metrics, makespan):
    """Weighted on-time completions minus a light makespan penalty.
    Uses the shared objective weights (TC worth W_TC, NTC worth W_NTC)."""
    return (W_TC * len(metrics.completed_tc) + W_NTC * len(metrics.completed_ntc)
            - 0.05 * makespan)


# ---------------------------------------------------------------------------
# Particle Swarm Optimization scheduler [1, 2]
# ---------------------------------------------------------------------------
def pso(edge_servers, tasks_df, swarm=24, iters=30, w=0.7, c1=1.5, c2=1.5,
        seed=42, history=None):
    """PSO over per-task priority keys. Each particle is a key vector;
    lower key = higher dispatch priority. Fitness maximises weighted
    on-time completions (TC weighted double) minus a makespan penalty."""
    wl = _Workload(edge_servers, tasks_df)
    n = wl.n
    if n == 0:
        return SimMetrics()

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, size=(swarm, n))
    vel = rng.uniform(-0.1, 0.1, size=(swarm, n))

    pbest = pos.copy()
    pbest_fit = np.array([_fitness(*wl.run_keyed(p)) for p in pos])
    g_idx = int(pbest_fit.argmax())
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    for _ in range(iters):
        r1 = rng.uniform(0, 1, size=(swarm, n))
        r2 = rng.uniform(0, 1, size=(swarm, n))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0.0, 1.0)
        for s in range(swarm):
            fit = _fitness(*wl.run_keyed(pos[s]))
            if fit > pbest_fit[s]:
                pbest_fit[s], pbest[s] = fit, pos[s].copy()
                if fit > gbest_fit:
                    gbest_fit, gbest = fit, pos[s].copy()
        if history is not None:
            history.append(float(gbest_fit))

    metrics, _ = wl.run_keyed(gbest, commit=True)
    return metrics


# ---------------------------------------------------------------------------
# Genetic Algorithm scheduler [3, 4]
# ---------------------------------------------------------------------------
def ga(edge_servers, tasks_df, pop=30, gens=30, elite=2, mut_rate=0.1,
       mut_scale=0.15, tournament=3, seed=42, history=None):
    """GA over per-task priority keys (random-key encoding). Tournament
    selection, blend crossover, Gaussian mutation, elitism. Same fitness
    as the PSO baseline."""
    wl = _Workload(edge_servers, tasks_df)
    n = wl.n
    if n == 0:
        return SimMetrics()

    rng = np.random.default_rng(seed)
    population = rng.uniform(0, 1, size=(pop, n))
    fitness = np.array([_fitness(*wl.run_keyed(ind)) for ind in population])

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
        fitness = np.array([_fitness(*wl.run_keyed(ind)) for ind in population])
        if history is not None:
            history.append(float(fitness.max()))

    best = population[int(fitness.argmax())]
    metrics, _ = wl.run_keyed(best, commit=True)
    return metrics


# ---------------------------------------------------------------------------
# Double Deep Q-Network scheduler [5, 6]
# ---------------------------------------------------------------------------
_K = 3  # action space: choose among the 3 earliest-deadline ready tasks


class _QNet:
    """Tiny 2-layer MLP (ReLU) with an Adam optimiser, numpy only."""

    def __init__(self, in_dim, hidden, out_dim, rng):
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, s1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, size=(hidden, out_dim))
        self.b2 = np.zeros(out_dim)
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k))
                   for k in ('W1', 'b1', 'W2', 'b2')}
        self._v = {k: np.zeros_like(getattr(self, k))
                   for k in ('W1', 'b1', 'W2', 'b2')}
        self._t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        q = a1 @ self.W2 + self.b2
        return q, (X, z1, a1)

    def predict(self, x):
        return self.forward(x[None, :])[0][0]

    def sgd_step(self, cache, dq, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        X, z1, a1 = cache
        b = X.shape[0]
        grads = {}
        grads['W2'] = a1.T @ dq / b
        grads['b2'] = dq.mean(axis=0)
        da1 = dq @ self.W2.T
        dz1 = da1 * (z1 > 0)
        grads['W1'] = X.T @ dz1 / b
        grads['b1'] = dz1.mean(axis=0)

        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def copy_from(self, other):
        for k in ('W1', 'b1', 'W2', 'b2'):
            setattr(self, k, getattr(other, k).copy())


def _ddqn_state(wl, cands, gpu_time, current, remaining):
    """Fixed-length feature vector for up to _K candidate tasks + globals."""
    feats = []
    for slot in range(_K):
        if slot < len(cands):
            _, i = cands[slot]
            _, end = wl.eft(i, gpu_time)
            feats += [
                (wl.due[i] - current) / wl.due_scale,
                wl.proc[i] / wl.proc_scale,
                1.0 if wl.is_tc[i] else 0.0,
                (wl.due[i] - current - wl.proc[i]) / wl.due_scale,  # laxity
                (end - current) / wl.due_scale,
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0]
    feats += [gpu_time.mean() / wl.due_scale, remaining / max(1, wl.n)]
    return np.array(feats, dtype=float)


def _ddqn_reward(wl, i, completed):
    if completed:
        return W_TC if wl.is_tc[i] else W_NTC
    return -W_TC / 2.0 if wl.is_tc[i] else -W_NTC / 2.0


def _ddqn_episode(wl, policy, gpu_time=None):
    """Run one full scheduling pass. `policy(state, n_valid)` returns an
    action in [0, n_valid). Yields (state, action, reward, next_state,
    done) transitions and records completions into a fresh SimMetrics.
    Returns (metrics, makespan, transitions)."""
    gpu_time = np.zeros(wl.m) if gpu_time is None else gpu_time
    metrics = SimMetrics()
    order, ptr, ready = wl.release_order, 0, []
    transitions = []
    remaining = wl.n
    prev = None  # (state, action)

    while ptr < wl.n or ready:
        current = gpu_time.min()
        while ptr < wl.n and wl.rel[order[ptr]] <= current:
            i = order[ptr]
            heapq.heappush(ready, (wl.due[i], i))
            ptr += 1
        if not ready:
            current = wl.rel[order[ptr]]
            while ptr < wl.n and wl.rel[order[ptr]] <= current:
                i = order[ptr]
                heapq.heappush(ready, (wl.due[i], i))
                ptr += 1

        cands = [heapq.heappop(ready) for _ in range(min(_K, len(ready)))]
        state = _ddqn_state(wl, cands, gpu_time, current, remaining)
        a = policy(state, len(cands))
        chosen = cands[a]
        for k, c in enumerate(cands):
            if k != a:
                heapq.heappush(ready, c)

        i = chosen[1]
        j, end = wl.eft(i, gpu_time)
        completed = end <= wl.due[i]
        if completed:
            gpu_time[j] = end
        metrics.record(wl.records[i], completed=completed)
        remaining -= 1
        r = _ddqn_reward(wl, i, completed)

        if prev is not None:
            transitions.append((prev[0], prev[1], prev[2], state, False))
        prev = (state, a, r)

    if prev is not None:
        transitions.append((prev[0], prev[1], prev[2], prev[0], True))
    return metrics, float(gpu_time.max()), transitions


def ddqn(edge_servers, tasks_df, episodes=60, hidden=64, gamma=0.99,
         lr=1e-3, batch=32, buffer=5000, target_every=500,
         eps_start=1.0, eps_end=0.05, seed=42, history=None):
    """Double DQN dispatcher [5, 6]. At each step the agent chooses which
    of the three earliest-deadline ready tasks to dispatch (placed on its
    earliest-finish-time GPU). Trained on repeated passes over the task
    set, then evaluated greedily."""
    wl = _Workload(edge_servers, tasks_df)
    if wl.n == 0:
        return SimMetrics()

    rng = np.random.default_rng(seed)
    in_dim = _K * 5 + 2
    online = _QNet(in_dim, hidden, _K, rng)
    target = _QNet(in_dim, hidden, _K, rng)
    target.copy_from(online)

    replay = []
    steps = [0]
    total_train_steps = episodes * wl.n
    eps = [eps_start]

    def masked_q(net, state, n_valid):
        q = net.predict(state).copy()
        if n_valid < _K:
            q[n_valid:] = -np.inf
        return q

    def train_policy(state, n_valid):
        # epsilon-greedy over valid actions
        e = eps_end + (eps_start - eps_end) * max(
            0.0, 1 - steps[0] / max(1, 0.8 * total_train_steps))
        eps[0] = e
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
        td = np.clip(pred - y, -1.0, 1.0)  # gradient clip via Huber-ish
        dq[np.arange(batch), A] = td
        online.sgd_step(cache, dq, lr=lr)

    for _ in range(episodes):
        _, _, transitions = _ddqn_episode(wl, train_policy)
        for tr in transitions:
            replay.append(tr)
            if len(replay) > buffer:
                replay.pop(0)
            learn()
            if steps[0] % target_every == 0:
                target.copy_from(online)
        if history is not None:
            history.append(float(sum(tr[2] for tr in transitions)))

    def greedy_policy(state, n_valid):
        return int(np.argmax(masked_q(online, state, n_valid)))

    gpu_time = np.zeros(wl.m)
    metrics, _, _ = _ddqn_episode(wl, greedy_policy, gpu_time=gpu_time)
    for g, t in zip(wl.gpus_flat, gpu_time):
        g.current_time = float(t)
    return metrics


RECENT_BASELINES = {
    'pso': pso,
    'ga': ga,
    'ddqn': ddqn,
}
