"""MILP-based large neighbourhood search (LNS) for WG / WGE partitioning.

The search state is the cut assignment x (one bit per graph edge); fragments,
widths and the best excision selection follow from it. One iteration:

  destroy  choose a vertex region R (generic operators: graph ball, cut-seeded
           ball, fragment union, excision hotspot, time window);
  repair   solve the exact WG/WGE MILP (`milp.build_milp`) with x fixed on
           every edge whose endpoints both lie outside R. Fragment labels z
           stay free everywhere, so fragments may split or merge through R;
           x-fixed retained edges make Gurobi's presolve collapse the rest;
  accept   keep the repaired solution if it is no worse.

The incumbent is always feasible for the sub-MILP (it is passed as a MIP start),
so repairs never worsen it. Operator choice uses adaptive weights; each
operator's region size adapts to keep repairs solvable within the per-repair
time limit. If a repair covers the whole graph, is solved to optimality and uses
K >= K_safe labels, the result is a certified optimum and the search stops.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .annealing import AnnealConfig, seed_states
from .cover_cuts import cover_cut_pool
from .hybrid import HybridPlan
from .milp import (
    MilpModel,
    build_milp,
    cheap_feasible_upper_bound,
    compute_k_safe,
    excision_selection_from_solution,
    milp_plan,
)
from .partition_graph import PartitionGraph

OPERATORS = ("ball", "cut_ball", "fragments", "excision", "time_window", "qubits")


def qubit_partition_seeds(graph: PartitionGraph, max_qubits: int, rng, *, restarts: int = 12):
    """Static seeds: k-way partitions of the qubit interaction graph (weights = gate-cut
    log costs, group sizes <= max_qubits), refined by qubit moves and swaps. Gates
    between groups are cut and every wire is kept. Returns a list of graph states."""
    n = graph.problem.circuit.num_qubits
    if n <= max_qubits:
        return []
    weight = np.zeros((n, n))
    gate_vars = []
    for e, var in enumerate(graph.variables):
        if var.kind != "gate" or len(var.options) < 2:
            continue
        a = graph.nodes[var.ports[0]]["qubit"]
        b = graph.nodes[var.ports[1]]["qubit"]
        cost = var.options[1].log_overhead
        weight[a, b] += cost
        weight[b, a] += cost
        gate_vars.append((e, a, b))
    # Qubit orders that keep strongly interacting qubits together.
    order_bfs = []
    seen = set()
    for start in np.argsort(-weight.sum(1)):
        if start in seen:
            continue
        queue = deque([int(start)])
        seen.add(int(start))
        while queue:
            q = queue.popleft()
            order_bfs.append(q)
            for r in np.argsort(-weight[q]):
                if weight[q, r] > 0 and int(r) not in seen:
                    seen.add(int(r))
                    queue.append(int(r))
    orders = [list(range(n)), order_bfs]
    orders += [list(rng.permutation(n)) for _ in range(max(0, restarts - len(orders)))]
    seeds, scores = [], []
    for k in sorted({-(-n // max_qubits), -(-n // max_qubits) + 1}):
        for order in orders:
            group = np.zeros(n, dtype=int)
            for pos, q in enumerate(order):
                group[q] = min(pos * k // n, k - 1)
            sizes = np.bincount(group, minlength=k)
            if sizes.max() > max_qubits:
                continue
            # attach[q, g] = weight from q to group g
            attach = np.stack([weight[:, group == g].sum(1) for g in range(k)], axis=1)
            improved = True
            while improved:
                improved = False
                for q in rng.permutation(n):
                    g = group[q]
                    gains = attach[q] - attach[q, g]
                    for h in np.argsort(-gains):
                        if h == g or gains[h] <= 1e-12:
                            break
                        if sizes[h] < max_qubits:  # move q to h
                            group[q] = h
                            sizes[g] -= 1
                            sizes[h] += 1
                            attach[:, g] -= weight[:, q]
                            attach[:, h] += weight[:, q]
                            improved = True
                            break
                        # swap q with the best r in h
                        members = np.flatnonzero(group == h)
                        swap_gain = gains[h] + (attach[members, g] - attach[members, h]) \
                            - 2 * weight[q, members]
                        r = members[int(np.argmax(swap_gain))]
                        if swap_gain.max() > 1e-12:
                            group[q], group[r] = h, g
                            attach[:, g] += weight[:, r] - weight[:, q]
                            attach[:, h] += weight[:, q] - weight[:, r]
                            improved = True
                            break
            state = [0] * len(graph.variables)
            cost = 0.0
            for e, a, b in gate_vars:
                if group[a] != group[b]:
                    state[e] = 1
                    cost += graph.variables[e].options[1].log_overhead
            seeds.append(tuple(state))
            scores.append(cost)
    return [seeds[i] for i in np.argsort(scores)[:4]]


@dataclass(frozen=True)
class LNSConfig:
    max_qubits: int
    excision: bool = True
    time_limit: float = 60.0
    repair_time_limit: float = 5.0
    initial_region: int = 60
    min_region: int = 16
    max_region: int = 400
    extra_labels: int = 3
    seed: int = 0
    threads: int = 1
    warm_start_seconds: float = 10.0
    operators: tuple[str, ...] = OPERATORS
    reaction: float = 0.2
    cover_cuts: bool = True
    cover_cut_seconds: float = 20.0
    mip_focus: int = 0
    stall_seconds: float | None = 1.0
    initial_state: tuple[int, ...] | None = None
    certify_seconds: float = 10.0
    certify_max_size: int = 60000
    seed_seconds: float = 30.0
    use_cutoff: bool = False
    idle_seconds: float | None = None


@dataclass
class LNSResult:
    objective: float
    state: tuple[int, ...]
    excisions: tuple[str, ...]
    plan: HybridPlan
    proven_optimal: bool
    iterations: int
    seconds: float
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    operator_stats: dict = field(default_factory=dict)
    start_objective: float | None = None
    start_source: str | None = None


def _components(graph: PartitionGraph, state) -> tuple[list[int], int]:
    parent = list(range(len(graph.nodes)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for left, right, _role, e in graph.edges:
        if state[e] == 0:
            ra, rb = root(left), root(right)
            if ra != rb:
                parent[ra] = rb
    ids, labels = {}, []
    for v in range(len(graph.nodes)):
        labels.append(ids.setdefault(root(v), len(ids)))
    return labels, len(ids)


class _Repairer:
    """One Gurobi copy of the MILP for a given label count K, re-bounded per repair.

    With `symmetry_breaking=False` (iteration repairs) a repair also fixes labels:
    every incumbent fragment that does not touch R or its neighbours keeps its own
    label, and the fragments in play may use only their own labels plus the spare
    ones. This is exact for the repair (those fragments' x is fixed anyway) and
    removes the symmetric re-packing of fixed fragments into labels.
    """

    def __init__(self, graph, config: LNSConfig, K: int, env, cuts=(), symmetry_breaking=True):
        import gurobipy as gp
        from gurobipy import GRB

        self.graph = graph
        self.K = K
        self.fix_labels = not symmetry_breaking
        self.model: MilpModel = build_milp(
            graph, max_qubits=config.max_qubits, K=K, excision=config.excision, cover_cuts=cuts,
            symmetry_breaking=symmetry_breaking,
        )
        model = self.model
        m = gp.Model(env=env)
        vtype = np.where(model.integrality == 0, GRB.CONTINUOUS, GRB.BINARY)
        v = m.addMVar(model.n_vars, lb=model.var_lb, ub=model.var_ub, vtype=vtype, obj=model.c)
        a = model.constraints.A.tocsr()
        lb, ub = model.constraints.lb, model.constraints.ub
        eq = lb == ub
        le = np.isneginf(lb) & ~eq
        ge = np.isposinf(ub) & ~eq
        if eq.any():
            m.addMConstr(a[eq], v, "=", lb[eq])
        if le.any():
            m.addMConstr(a[le], v, "<", ub[le])
        if ge.any():
            m.addMConstr(a[ge], v, ">", lb[ge])
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = 1e-6
        m.Params.IntFeasTol = 1e-9
        m.Params.FeasibilityTol = 1e-9
        m.Params.Threads = config.threads
        m.Params.MIPFocus = config.mip_focus
        self.m, self.v = m, v
        self.stall_seconds = config.stall_seconds
        self.use_cutoff = config.use_cutoff
        self.idle_seconds = config.idle_seconds
        edges = graph.edges
        self.left = np.array([e[0] for e in edges])
        self.right = np.array([e[1] for e in edges])
        self.edge_var = np.array([e[3] for e in edges])
        self.x_cols = model.x_offset + self.edge_var

    def start_vector(self, state, labels) -> np.ndarray:
        model = self.model
        x0 = np.zeros(model.n_vars)
        for vtx, k in enumerate(labels):
            sigma = (
                1 if model.incoming_wire[vtx] is None else int(state[model.incoming_wire[vtx]])
            )
            x0[model.z_col(vtx, k)] = 1.0
            x0[model.t_col(vtx, k)] = sigma
        x0[model.x_offset : model.x_offset + model.n_edges] = state
        return x0

    def solve(self, state, labels, region_mask: np.ndarray, time_limit: float, ncomp=None,
              incumbent=None):
        model = self.model
        lb, ub = model.var_lb.copy(), model.var_ub.copy()
        fixed = ~region_mask[self.left] & ~region_mask[self.right]
        values = np.asarray(state, dtype=float)[self.edge_var[fixed]]
        lb[self.x_cols[fixed]] = values
        ub[self.x_cols[fixed]] = values
        if self.fix_labels:
            n, K = model.n_vertices, self.K
            lab = np.asarray(labels)
            touch = region_mask.copy()
            touch[self.left[region_mask[self.right]]] = True
            touch[self.right[region_mask[self.left]]] = True
            active = np.unique(lab[touch])
            in_play = np.isin(lab, active)
            allowed = np.zeros(K, dtype=bool)
            allowed[active] = True
            allowed[ncomp:] = True
            zlb = lb[model.z_offset : model.z_offset + n * K].reshape(n, K)
            zub = ub[model.z_offset : model.z_offset + n * K].reshape(n, K)
            out = np.flatnonzero(~in_play)
            zub[out, :] = 0.0
            zlb[out, lab[out]] = 1.0
            zub[out, lab[out]] = 1.0
            zub[np.ix_(np.flatnonzero(in_play), np.flatnonzero(~allowed))] = 0.0
        self.v.setAttr("LB", lb)
        self.v.setAttr("UB", ub)
        self.m.reset()
        self.v.setAttr("Start", self.start_vector(state, labels))
        self.m.Params.TimeLimit = max(time_limit, 0.05)
        # Cutoff: only strictly better solutions are searched for, so a repair that
        # cannot improve ends with a quick proof instead of running to its limit.
        cutoff = self.use_cutoff and incumbent is not None
        self.m.Params.Cutoff = incumbent - 1e-7 if cutoff else float("inf")
        if (self.stall_seconds is None and self.idle_seconds is None) or incumbent is None:
            self.m.optimize()
        else:
            import gurobipy as gp

            first, last = [incumbent], [None]

            def stop_when_stalled(model_, where):
                # End the repair once it has improved on the start and then stalled:
                # improvements are what LNS needs, not optimality proofs.
                if where == gp.GRB.Callback.MIPSOL:
                    value = model_.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                    now = model_.cbGet(gp.GRB.Callback.RUNTIME)
                    if value < first[0] - 1e-9:
                        last[0] = now
                elif where == gp.GRB.Callback.MIP and last[0] is not None:
                    if self.stall_seconds is not None and \
                            model_.cbGet(gp.GRB.Callback.RUNTIME) - last[0] > self.stall_seconds:
                        model_.terminate()
                elif where == gp.GRB.Callback.MIP and last[0] is None \
                        and self.idle_seconds is not None \
                        and model_.cbGet(gp.GRB.Callback.RUNTIME) > self.idle_seconds:
                    model_.terminate()

            self.m.optimize(stop_when_stalled)
        if self.m.SolCount == 0:
            if cutoff:
                # No strictly better solution: proved (CUTOFF/INFEASIBLE) or not found.
                return dict(objective=incumbent, raw=None,
                            optimal=self.m.Status in (6, 3), runtime=float(self.m.Runtime))
            return None
        raw = np.array(self.v.X)
        return dict(
            objective=float(self.m.ObjVal), raw=raw, optimal=self.m.Status == 2,
            runtime=float(self.m.Runtime),
        )


class MilpLNS:
    def __init__(self, graph: PartitionGraph, config: LNSConfig):
        if config.excision and graph.grouping is None:
            raise ValueError("WGE search needs an excision-enabled PartitionGraph.")
        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.n_vertices = len(graph.nodes)
        self.adj = [[] for _ in graph.nodes]
        for left, right, _role, e in graph.edges:
            self.adj[left].append((right, e))
            self.adj[right].append((left, e))
        self.costs = np.array([
            var.options[1].log_overhead if len(var.options) > 1 else 0.0
            for var in graph.variables
        ])
        self.layers = self._layers()
        self.n_qubits = graph.problem.circuit.num_qubits
        self.hotspots = self._excision_hotspots()
        self._repairers: dict[int, _Repairer] = {}
        self.cuts: list[tuple[int, ...]] = []

    # ------------------------------------------------------------ structure

    def _layers(self):
        problem = self.graph.problem
        qubit_layer = [0] * problem.circuit.num_qubits
        op_layer = {}
        for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if item.operation.num_qubits != 2:
                continue
            qs = [problem.circuit.find_bit(q).index for q in item.qubits]
            layer = 1 + max(qubit_layer[q] for q in qs)
            for q in qs:
                qubit_layer[q] = layer
            op_layer[op_id] = layer
        return np.array([op_layer.get(n.get("operation_id"), 0) for n in self.graph.nodes])

    def _excision_hotspots(self):
        if not self.config.excision:
            return None
        best = {}
        for cand in self.graph.grouping.candidates:
            saving = cand.independent_log_overhead - 2 * math.log(cand.gamma)
            best[cand.anchor] = max(best.get(cand.anchor, 0.0), saving)
        if not best:
            return None
        anchors = list(best)
        weights = np.array([best[a] for a in anchors])
        return anchors, weights / weights.sum()

    # ------------------------------------------------------------ regions

    def _ball(self, seeds, size):
        region, seen = set(), set(seeds)
        queue = deque(seeds)
        while queue and len(region) < size:
            v = queue.popleft()
            region.add(v)
            for u, _e in self.adj[v]:
                if u not in seen:
                    seen.add(u)
                    queue.append(u)
        return region

    def _region(self, op, size, state, labels):
        rng = self.rng
        if op == "cut_ball":
            cut = np.flatnonzero(np.asarray(state) == 1)
            if len(cut):
                w = self.costs[cut]
                e = cut[rng.choice(len(cut), p=w / w.sum())]
                return self._ball(list(self.graph.variables[e].ports), size)
        if op == "excision" and self.hotspots is not None:
            anchors, p = self.hotspots
            anchor = anchors[rng.choice(len(anchors), p=p)]
            ports = self.graph.variables[self.graph.gate_variables[anchor]].ports
            return self._ball(list(ports), size)
        if op == "fragments":
            members = {}
            for v, k in enumerate(labels):
                members.setdefault(k, []).append(v)
            link = {}
            for left, right, _role, e in self.graph.edges:
                a, b = labels[left], labels[right]
                if a != b:
                    link.setdefault(a, {}).setdefault(b, 0.0)
                    link[a][b] += self.costs[e]
                    link.setdefault(b, {}).setdefault(a, 0.0)
                    link[b][a] += self.costs[e]
            first = int(rng.integers(len(members)))
            if len(members[first]) <= 2 * size:
                chosen, region = {first}, set(members[first])
                while len(region) < size:
                    frontier = {}
                    for a in chosen:
                        for b, w in link.get(a, {}).items():
                            if b not in chosen:
                                frontier[b] = frontier.get(b, 0.0) + w
                    if not frontier:
                        break
                    best = max(frontier, key=frontier.get)
                    if len(region) + len(members[best]) > 2 * size:
                        break
                    chosen.add(best)
                    region.update(members[best])
                return region
        if op == "qubits":
            # All vertices of 1-3 qubit wires (weighted by their cut cost), so a repair
            # can move whole qubits between fragments.
            wire_cost = np.zeros(self.n_qubits)
            for e in np.flatnonzero(np.asarray(state) == 1):
                for port in self.graph.variables[e].ports:
                    wire_cost[self.graph.nodes[port]["qubit"]] += self.costs[e]
            p = wire_cost + 1e-3
            region = set()
            for q in rng.choice(self.n_qubits, size=min(3, self.n_qubits), replace=False,
                                p=p / p.sum()):
                if region and len(region) + len(self.graph.wire_nodes[q]) > size:
                    break
                region.update(self.graph.wire_nodes[q])
            return region
        if op == "time_window":
            keys = np.unique(self.layers)
            center = int(rng.integers(len(keys)))
            lo = hi = center
            region = set(np.flatnonzero(self.layers == keys[center]).tolist())
            while len(region) < size and (lo > 0 or hi < len(keys) - 1):
                if hi < len(keys) - 1 and (lo == 0 or rng.random() < 0.5):
                    hi += 1
                    region.update(np.flatnonzero(self.layers == keys[hi]).tolist())
                else:
                    lo -= 1
                    region.update(np.flatnonzero(self.layers == keys[lo]).tolist())
            return region
        return self._ball([int(rng.integers(self.n_vertices))], size)

    # ------------------------------------------------------------ helpers

    def _repairer(self, n_components, env, symmetry_breaking=False):
        K = n_components + self.config.extra_labels
        key = (K, symmetry_breaking)
        if key not in self._repairers:
            if len(self._repairers) >= 4:
                self._repairers.pop(next(iter(self._repairers)))
            self._repairers[key] = _Repairer(
                self.graph, self.config, K, env, self.cuts, symmetry_breaking=symmetry_breaking
            )
        return self._repairers[key]

    def _seed_candidates(self):
        cfg = self.config
        graph = self.graph
        candidates = []
        if cfg.initial_state is not None:
            candidates.append(("given", tuple(int(v) for v in cfg.initial_state)))
        candidates += [("all_gate_cut", tuple(
            1 if var.kind == "gate" and len(var.options) > 1 else 0 for var in graph.variables
        ))]
        try:
            candidates += [("qubit_partition", st) for st in qubit_partition_seeds(
                graph, cfg.max_qubits, np.random.default_rng(cfg.seed))]
        except Exception:  # noqa: BLE001 -- seeds are optional
            pass
        try:
            # Seeding is bounded so it cannot eat the search budget on large graphs.
            deadline = perf_counter() + min(cfg.seed_seconds, 0.1 * cfg.time_limit)
            pool = seed_states(graph, AnnealConfig(cfg.max_qubits, partition_mode="multiway"),
                               count=8, seed=cfg.seed, deadline=deadline)
            candidates += [("spatial_seed", tuple(s)) for s in pool]
        except Exception:  # noqa: BLE001 -- seeds are optional
            pass
        feasible = []
        for name, state in candidates:
            ev = graph.evaluate(state, max_qubits=cfg.max_qubits, partition_mode="multiway",
                                forced_groups=() if cfg.excision else None)
            if ev.feasible:
                feasible.append((ev.log_overhead, name, state))
        feasible.sort(key=lambda item: item[0])
        return feasible

    # ------------------------------------------------------------ main loop

    def run(self) -> LNSResult:
        import gurobipy as gp

        cfg = self.config
        started = perf_counter()

        def elapsed():
            return perf_counter() - started

        all_vertices = np.ones(self.n_vertices, dtype=bool)
        if cfg.cover_cuts:
            self.cuts = cover_cut_pool(
                self.graph, max_qubits=cfg.max_qubits, excision=cfg.excision,
                time_budget=min(cfg.cover_cut_seconds, cfg.time_limit / 4),
            )
        with gp.Env(params={"OutputFlag": 0}) as env:
            seeds = self._seed_candidates()
            if not seeds:
                raise RuntimeError("No feasible seed (max_qubits too small for any gate cut).")
            _, source, state = seeds[0]
            labels, ncomp = _components(self.graph, state)
            rep = self._repairer(ncomp, env, symmetry_breaking=True)
            base = rep.solve(state, labels, np.zeros(self.n_vertices, bool), 30.0)
            incumbent, raw, inc_model = base["objective"], base["raw"], rep.model
            start_objective = incumbent
            trajectory = [(elapsed(), incumbent)]
            proven = False
            if cfg.warm_start_seconds > 0:
                res = rep.solve(state, labels, all_vertices,
                                min(cfg.warm_start_seconds, cfg.time_limit / 4))
                if res is not None and res["objective"] < incumbent - 1e-9:
                    incumbent, raw, inc_model = res["objective"], res["raw"], rep.model
                    state = tuple(int(round(raw[rep.model.x_col(e)]))
                                  for e in range(rep.model.n_edges))
                    source += "+milp_warm_start"
                    trajectory.append((elapsed(), incumbent))
                proven = bool(res and res["optimal"]) and self._certified(rep.K, incumbent)
            if not proven and cfg.certify_seconds > 0:
                # Exact certification attempt: the full MILP with K_safe labels, started
                # from the incumbent. Small instances finish here with a proof.
                labels, ncomp = _components(self.graph, state)
                bound = min(incumbent, cheap_feasible_upper_bound(self.graph, cfg.max_qubits))
                k_safe = max(compute_k_safe(self.graph, cfg.max_qubits, bound), ncomp)
                if k_safe * self.n_vertices <= cfg.certify_max_size:
                    cert = _Repairer(self.graph, cfg, k_safe, env, self.cuts,
                                     symmetry_breaking=True)
                    res = cert.solve(state, labels, all_vertices,
                                     min(cfg.certify_seconds, cfg.time_limit / 4))
                    if res is not None and res["objective"] < incumbent - 1e-9:
                        incumbent, raw, inc_model = res["objective"], res["raw"], cert.model
                        state = tuple(int(round(raw[cert.model.x_col(e)]))
                                      for e in range(cert.model.n_edges))
                        source += "+certify"
                        trajectory.append((elapsed(), incumbent))
                    proven = bool(res and res["optimal"])

            sizes = {op: float(cfg.initial_region) for op in cfg.operators}
            weights = {op: 1.0 for op in cfg.operators}
            stats = {op: dict(calls=0, improved=0, seconds=0.0) for op in cfg.operators}
            iterations = 0
            while not proven and elapsed() < cfg.time_limit:
                iterations += 1
                labels, ncomp = _components(self.graph, state)
                rep = self._repairer(ncomp, env)
                ops = list(cfg.operators)
                p = np.array([weights[o] for o in ops])
                op = ops[self.rng.choice(len(ops), p=p / p.sum())]
                size = int(min(max(sizes[op], cfg.min_region), cfg.max_region, self.n_vertices))
                region = self._region(op, size, state, labels)
                mask = np.zeros(self.n_vertices, dtype=bool)
                mask[list(region)] = True
                budget = min(cfg.repair_time_limit, cfg.time_limit - elapsed())
                if budget <= 0.05:
                    break
                res = rep.solve(state, labels, mask, budget, ncomp=ncomp, incumbent=incumbent)
                stats[op]["calls"] += 1
                if res is None:
                    sizes[op] *= 0.7
                    continue
                stats[op]["seconds"] += res["runtime"]
                reward = 0.0
                if res["raw"] is not None and res["objective"] <= incumbent + 1e-9:
                    if res["objective"] < incumbent - 1e-9:
                        reward = 1.0
                        stats[op]["improved"] += 1
                        trajectory.append((elapsed(), res["objective"]))
                    incumbent, raw, inc_model = res["objective"], res["raw"], rep.model
                    state = tuple(int(round(raw[rep.model.x_col(e)]))
                                  for e in range(rep.model.n_edges))
                # Grow when repairs are proved fast; shrink only unproductive time-outs.
                if res["optimal"] and res["runtime"] < 0.25 * budget:
                    sizes[op] = min(sizes[op] * 1.15, cfg.max_region)
                elif not res["optimal"] and reward == 0.0:
                    sizes[op] = max(sizes[op] * 0.8, cfg.min_region)
                weights[op] = max(
                    0.05,
                    (1 - cfg.reaction) * weights[op]
                    + cfg.reaction * reward / max(res["runtime"], 0.05),
                )
                if mask.all() and res["optimal"]:
                    proven = self._certified(rep.K, incumbent)

            excisions = excision_selection_from_solution(inc_model, raw) if cfg.excision else ()
            plan = milp_plan(inc_model, raw)
        return LNSResult(
            objective=incumbent, state=state, excisions=tuple(excisions), plan=plan,
            proven_optimal=proven, iterations=iterations, seconds=elapsed(),
            trajectory=trajectory,
            operator_stats=dict(stats, final_sizes=sizes, final_weights=weights,
                                cover_cuts=len(self.cuts)),
            start_objective=start_objective, start_source=source,
        )

    def _certified(self, K, objective):
        bound = min(objective, cheap_feasible_upper_bound(self.graph, self.config.max_qubits))
        return K >= compute_k_safe(self.graph, self.config.max_qubits, bound)


def run_lns(graph: PartitionGraph, config: LNSConfig) -> LNSResult:
    return MilpLNS(graph, config).run()
