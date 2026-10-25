"""Exact MILP formulation of the WG/WGE multiway partition problem.

Implements the model specified in `docs/MILP_prompt.txt`: an integer program
over the existing contracted temporal incidence graph (`PartitionGraph`) whose
proven-optimal objective certifies whether two-qubit gate excision can
strictly lower the minimum sampling overhead under this project's resource
model. This module adds no new graph, cost, or excision primitive — it only
exposes the existing vertex/edge structure (`PartitionGraph`), independent cut
norms (`TemplateRegistry`), and excision candidate catalogue
(`grouping.GroupLibrary`) to an off-the-shelf MILP solver, and converts a
solved assignment back to the existing `HybridPlan` format via
`GroupLibrary.forced_selection` so it can be recompiled and re-evaluated by
the existing pipeline exactly as any other plan.

Variables (Section 4 of the spec):
  z[v,k]  vertex v assigned to fragment label k               (binary)
  x[e]    edge e (== one existing ActionVariable) is cut      (binary)
  t[v,k]  linearization of sigma[v] * z[v,k]                  (continuous)
  y[c]    excision candidate c selected (WGE only)            (binary)
sigma[v] = 1 if v is the first incidence on its qubit, else x[in(v)].
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .grouping import GroupCandidate
from .hybrid import HybridPlan
from .partition_graph import PartitionGraph

_INF = float("inf")


def cheap_feasible_upper_bound(graph: PartitionGraph, max_qubits: int) -> float:
    """A trivial, always-feasible WG upper bound: cut every two-qubit gate.

    Cutting every gate edge disconnects each qubit into its own single-qubit
    component, so this is feasible for any max_qubits >= 1. Used as the
    default C_UB of Lemma 2 (Section 8) when a tighter SA/heuristic bound is
    not supplied; a tighter C_UB only shrinks K_safe, never changes validity.
    """
    if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 1:
        raise ValueError("max_qubits must be a positive integer.")
    state = tuple(1 if variable.kind == "gate" else 0 for variable in graph.variables)
    evaluation = graph.evaluate(state, max_qubits=max_qubits, partition_mode="multiway")
    if not evaluation.feasible:
        raise RuntimeError("The all-gate-cut state must always be feasible.")
    return evaluation.log_overhead


def compute_k_safe(graph: PartitionGraph, max_qubits: int, upper_bound_log_overhead: float) -> int:
    """A provably sufficient fragment-label count (Section 8, Lemmas 1-2).

    `upper_bound_log_overhead` must be the objective of ANY feasible solution
    (WG or WGE; a looser model's feasible solution is also feasible here)
    under the SAME graph and max_qubits — e.g. `cheap_feasible_upper_bound`,
    an SA result, or a simple constructive heuristic. A tighter bound only
    shrinks K (faster solves); it never affects whether K is sufficient.
    """
    if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 1:
        raise ValueError("max_qubits must be a positive integer.")
    if not math.isfinite(upper_bound_log_overhead) or upper_bound_log_overhead < 0:
        raise ValueError("upper_bound_log_overhead must be a finite nonnegative number.")
    n = graph.problem.circuit.num_qubits
    wire_log_cost = graph.registry.wire.log_overhead  # 2*ln(4), the independent wire norm.
    # Lemma 2: every cut wire edge costs at least 2*ln(4) in both models (an
    # excision covering w wire edges costs >= 2*ln(16^w) = w * 2 * 2*ln(4)).
    w_max = math.floor(upper_bound_log_overhead / wire_log_cost + 1e-9)
    s = n + w_max
    k_lemma1 = 2 * math.ceil(s / max_qubits) - 1
    k_min = math.ceil(n / max_qubits)
    return max(k_lemma1, k_min, 1)


@dataclass
class MilpModel:
    graph: PartitionGraph
    max_qubits: int
    K: int
    excision: bool
    partition_mode: str
    n_vertices: int
    n_edges: int
    candidates: tuple[GroupCandidate, ...]
    z_offset: int
    t_offset: int
    x_offset: int
    y_offset: int
    n_vars: int
    c: np.ndarray
    constraints: LinearConstraint
    var_lb: np.ndarray
    var_ub: np.ndarray
    integrality: np.ndarray
    incoming_wire: tuple[int | None, ...]
    build_seconds: float
    n_constraints: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_constraints = int(self.constraints.lb.shape[0])

    def z_col(self, v: int, k: int) -> int:
        return self.z_offset + v * self.K + k

    def t_col(self, v: int, k: int) -> int:
        return self.t_offset + v * self.K + k

    def x_col(self, e: int) -> int:
        return self.x_offset + e

    def y_col(self, candidate_index: int) -> int:
        if not self.excision:
            raise ValueError("The WG model has no excision candidates.")
        return self.y_offset + candidate_index


class _RowBuilder:
    """Accumulates sparse constraint rows as (row, col, value) blocks."""

    def __init__(self) -> None:
        self._row_blocks: list[np.ndarray] = []
        self._col_blocks: list[np.ndarray] = []
        self._val_blocks: list[np.ndarray] = []
        self._lb: list[float] = []
        self._ub: list[float] = []
        self._next_row = 0

    def add_row(self, entries: dict[int, float], lb: float, ub: float) -> None:
        row = self._next_row
        self._next_row += 1
        if entries:
            cols = np.fromiter(entries.keys(), dtype=np.int64, count=len(entries))
            vals = np.fromiter(entries.values(), dtype=np.float64, count=len(entries))
            self._row_blocks.append(np.full(len(entries), row, dtype=np.int64))
            self._col_blocks.append(cols)
            self._val_blocks.append(vals)
        self._lb.append(lb)
        self._ub.append(ub)

    def add_vectorized(
        self,
        row_local: np.ndarray,
        cols: np.ndarray,
        vals: np.ndarray,
        n_rows: int,
        lb: float,
        ub: float,
    ) -> None:
        base = self._next_row
        self._next_row += n_rows
        self._row_blocks.append(row_local + base)
        self._col_blocks.append(cols)
        self._val_blocks.append(vals)
        self._lb.extend([lb] * n_rows)
        self._ub.extend([ub] * n_rows)

    def finish(self, n_vars: int) -> LinearConstraint:
        rows = (
            np.concatenate(self._row_blocks) if self._row_blocks else np.array([], dtype=np.int64)
        )
        cols = (
            np.concatenate(self._col_blocks) if self._col_blocks else np.array([], dtype=np.int64)
        )
        vals = (
            np.concatenate(self._val_blocks) if self._val_blocks else np.array([], dtype=np.float64)
        )
        a = sparse.coo_matrix(
            (vals, (rows, cols)), shape=(self._next_row, n_vars), dtype=np.float64
        ).tocsr()
        return LinearConstraint(a, np.array(self._lb), np.array(self._ub))


def build_milp(
    graph: PartitionGraph,
    *,
    max_qubits: int,
    K: int,
    excision: bool,
    partition_mode: str = "multiway",
) -> MilpModel:
    """Build the WG (`excision=False`) or WGE (`excision=True`) MILP (Sections 6-7).

    `graph` must be the SAME excision-enabled `PartitionGraph` for both calls
    (build the WGE graph once; the WG model simply omits y/E1/E2). Reuses the
    existing `graph.grouping.candidates` excision LUT verbatim for `y`/E1/E2.
    """
    started = perf_counter()
    if partition_mode != "multiway":
        raise ValueError("The MILP model implements multiway semantics only (Section 2).")
    if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 1:
        raise ValueError("max_qubits must be a positive integer.")
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer.")
    if excision:
        if graph.grouping is None or set(graph.grouping.config.families) != {"excision"}:
            raise ValueError(
                "The WGE model requires a graph built with "
                'grouping=GroupingConfig(families=("excision",)); no other grouping family.'
            )
    graph.problem.validate_snapshot()
    n_vertices = len(graph.nodes)
    n_edges = len(graph.variables)
    n = graph.problem.circuit.num_qubits
    candidates = tuple(graph.grouping.candidates) if excision else ()

    z_offset = 0
    t_offset = z_offset + n_vertices * K
    x_offset = t_offset + n_vertices * K
    y_offset = x_offset + n_edges
    n_vars = y_offset + len(candidates)

    def z_col(v: int, k: int) -> int:
        return z_offset + v * K + k

    def t_col(v: int, k: int) -> int:
        return t_offset + v * K + k

    def x_col(e: int) -> int:
        return x_offset + e

    var_lb = np.zeros(n_vars)
    var_ub = np.ones(n_vars)
    var_ub[t_offset : t_offset + n_vertices * K] = _INF
    integrality = np.ones(n_vars)
    integrality[t_offset : t_offset + n_vertices * K] = 0

    # (C3) Disabled cuts: no independent decomposition registered for this edge.
    for e, variable in enumerate(graph.variables):
        if len(variable.options) < 2:
            var_ub[x_col(e)] = 0.0

    incoming_wire: list[int | None] = [None] * n_vertices
    for _left, right, role, variable in graph.edges:
        if role == "wire":
            incoming_wire[right] = variable

    builder = _RowBuilder()

    # (C1) Assignment: sum_k z[v,k] = 1, for all v.
    for v in range(n_vertices):
        builder.add_row({z_col(v, k): 1.0 for k in range(K)}, 1.0, 1.0)

    # (C2) Cut definition: x[e] >= z[u,k]-z[w,k] and x[e] >= z[w,k]-z[u,k].
    for left, right, _role, e in graph.edges:
        for k in range(K):
            builder.add_row(
                {x_col(e): 1.0, z_col(left, k): -1.0, z_col(right, k): 1.0}, 0.0, _INF
            )
            builder.add_row(
                {x_col(e): 1.0, z_col(left, k): 1.0, z_col(right, k): -1.0}, 0.0, _INF
            )

    # (C4) Width linearization: t[v,k] >= sigma[v] + z[v,k] - 1.
    for v in range(n_vertices):
        for k in range(K):
            if incoming_wire[v] is None:
                # sigma[v] == 1 (first incidence on its qubit): t[v,k] >= z[v,k].
                builder.add_row({t_col(v, k): 1.0, z_col(v, k): -1.0}, 0.0, _INF)
            else:
                # sigma[v] == x[in(v)]: t[v,k] - x[in(v)] - z[v,k] >= -1.
                builder.add_row(
                    {
                        t_col(v, k): 1.0,
                        x_col(incoming_wire[v]): -1.0,
                        z_col(v, k): -1.0,
                    },
                    -1.0,
                    _INF,
                )

    # (C5) Capacity: sum_v t[v,k] <= M, for all k.
    for k in range(K):
        builder.add_row({t_col(v, k): 1.0 for v in range(n_vertices)}, -_INF, float(max_qubits))

    # (C6) Segment conservation: sum_{v,k} t[v,k] - sum_{e in wire} x[e] == N.
    wire_edges = tuple(e for e, variable in enumerate(graph.variables) if variable.kind == "wire")
    entries = {t_col(v, k): 1.0 for v in range(n_vertices) for k in range(K)}
    for e in wire_edges:
        entries[x_col(e)] = entries.get(x_col(e), 0.0) - 1.0
    builder.add_row(entries, float(n), float(n))

    # (C7) Symmetry breaking on fragment labels (vertex order == node index order).
    var_lb[z_col(0, 0)] = 1.0
    if K > 1:
        tri_i, tri_j = np.tril_indices(n_vertices, k=-1)  # all pairs with tri_j < tri_i
        for k in range(1, K):
            n_rows = n_vertices - 1
            row_for_i = np.arange(1, n_vertices)  # local row index i-1 for vertex i, i=1..nV-1
            diag_rows = row_for_i - 1
            diag_cols = np.fromiter((z_col(i, k) for i in row_for_i), dtype=np.int64, count=n_rows)
            diag_vals = np.ones(n_rows)
            tri_rows = tri_i - 1
            tri_cols = np.fromiter(
                (z_col(j, k - 1) for j in tri_j), dtype=np.int64, count=tri_j.shape[0]
            )
            tri_vals = -np.ones(tri_j.shape[0])
            builder.add_vectorized(
                np.concatenate([diag_rows, tri_rows]),
                np.concatenate([diag_cols, tri_cols]),
                np.concatenate([diag_vals, tri_vals]),
                n_rows,
                -_INF,
                0.0,
            )

    c = np.zeros(n_vars)
    for e, variable in enumerate(graph.variables):
        if len(variable.options) >= 2:
            c[x_col(e)] = variable.options[1].log_overhead

    if excision:
        # (E1) sum_{c: e in B_c} y[c] <= x[e], for every edge e covered by >=1 candidate.
        covering: dict[int, list[int]] = {}
        for i, candidate in enumerate(candidates):
            for e in candidate.variables:
                covering.setdefault(e, []).append(i)
        for e, indices in covering.items():
            entries = {(y_offset + i): 1.0 for i in indices}
            entries[x_col(e)] = entries.get(x_col(e), 0.0) - 1.0
            builder.add_row(entries, -_INF, 0.0)

        # (E2) Retained-gate candidates: y[c] <= 1 - x[e_g(c)], and at most one
        # retained excision per gate (non-retained candidates share this limit
        # automatically via E1, since they cover their own gate edge).
        retained_by_gate: dict[str, list[int]] = {}
        for i, candidate in enumerate(candidates):
            if candidate.guards:
                gate_edge = candidate.guards[0]
                builder.add_row(
                    {
                        (y_offset + i): 1.0,
                        x_col(gate_edge): 1.0,
                    },
                    -_INF,
                    1.0,
                )
                retained_by_gate.setdefault(candidate.anchor, []).append(i)
        for indices in retained_by_gate.values():
            builder.add_row(
                {(y_offset + i): 1.0 for i in indices}, -_INF, 1.0
            )

        for i, candidate in enumerate(candidates):
            c[y_offset + i] = -(candidate.independent_log_overhead - 2 * math.log(candidate.gamma))

    constraints = builder.finish(n_vars)
    return MilpModel(
        graph,
        max_qubits,
        K,
        excision,
        partition_mode,
        n_vertices,
        n_edges,
        candidates,
        z_offset,
        t_offset,
        x_offset,
        y_offset,
        n_vars,
        c,
        constraints,
        var_lb,
        var_ub,
        integrality,
        tuple(incoming_wire),
        perf_counter() - started,
    )


@dataclass
class MilpSolution:
    status: str
    objective: float | None  # certified upper bound (best incumbent), log units
    bound: float | None  # certified lower bound, log units
    gap_abs: float | None
    gap_rel: float | None
    proven_optimal: bool
    runtime_seconds: float
    node_count: int | None
    raw_x: np.ndarray | None
    solver: str
    solver_options: dict


def _relaxation_bound(model: MilpModel) -> float | None:
    """A valid, if weak, certified lower bound: the LP relaxation (integrality dropped)."""
    result = milp(
        model.c,
        integrality=np.zeros(model.n_vars),
        bounds=Bounds(model.var_lb, model.var_ub),
        constraints=model.constraints,
        options={"disp": False},
    )
    return float(result.fun) if result.x is not None else None


def _solve_highs(
    model: MilpModel,
    *,
    time_limit: float | None,
    mip_gap_abs: float,
    mip_gap_rel: float,
    extra_options: dict,
) -> MilpSolution:
    options = {
        "disp": False,
        "mip_rel_gap": mip_gap_rel,
        "mip_abs_gap": mip_gap_abs,
        **extra_options,
    }
    if time_limit is not None:
        options["time_limit"] = time_limit
    started = perf_counter()
    with warnings.catch_warnings():
        # mip_abs_gap is a real HiGHS option that scipy's wrapper does not
        # recognize by name; it still passes it through verbatim.
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(
            model.c,
            integrality=model.integrality,
            bounds=Bounds(model.var_lb, model.var_ub),
            constraints=model.constraints,
            options=options,
        )
    elapsed = perf_counter() - started
    node_count = getattr(result, "mip_node_count", None)
    if result.x is None:
        bound = _relaxation_bound(model)
        return MilpSolution(
            "no_incumbent_time_limit" if result.status == 1 else "infeasible_or_error",
            None,
            bound,
            None,
            None,
            False,
            elapsed,
            node_count,
            None,
            "highs",
            dict(options),
        )
    objective = float(result.fun)
    bound = getattr(result, "mip_dual_bound", None)
    bound = float(bound) if bound is not None else _relaxation_bound(model)
    gap_abs = None if bound is None else abs(objective - bound)
    gap_rel = getattr(result, "mip_gap", None)
    gap_rel = float(gap_rel) if gap_rel is not None else None
    proven_optimal = result.status == 0
    return MilpSolution(
        "optimal" if proven_optimal else "time_limit_with_incumbent",
        objective,
        bound,
        gap_abs,
        gap_rel,
        proven_optimal,
        elapsed,
        node_count,
        np.asarray(result.x),
        "highs",
        dict(options),
    )


def _solve_gurobi(
    model: MilpModel,
    *,
    time_limit: float | None,
    mip_gap_abs: float,
    mip_gap_rel: float,
    mip_start: np.ndarray | None,
    extra_options: dict,
) -> MilpSolution:
    import gurobipy as gp
    from gurobipy import GRB

    started = perf_counter()
    env_params = extra_options.pop("env_params", {})
    with gp.Env(params=env_params) as env, gp.Model(env=env) as m:
        m.Params.MIPGap = mip_gap_rel
        m.Params.MIPGapAbs = mip_gap_abs
        m.Params.IntFeasTol = extra_options.pop("IntFeasTol", 1e-9)
        m.Params.FeasibilityTol = extra_options.pop("FeasibilityTol", 1e-9)
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
        for key, value in extra_options.items():
            setattr(m.Params, key, value)
        vtypes = [GRB.CONTINUOUS if integer == 0 else GRB.BINARY for integer in model.integrality]
        variables = m.addVars(
            model.n_vars,
            lb=model.var_lb,
            ub=model.var_ub,
            vtype=vtypes,
            obj=model.c,
            name="v",
        )
        a = model.constraints.A.tocsr()
        for row in range(a.shape[0]):
            start, end = a.indptr[row], a.indptr[row + 1]
            expr = gp.LinExpr(
                a.data[start:end].tolist(), [variables[int(i)] for i in a.indices[start:end]]
            )
            lb, ub = model.constraints.lb[row], model.constraints.ub[row]
            if lb == ub:
                m.addLConstr(expr, GRB.EQUAL, lb)
            elif np.isneginf(lb):
                m.addLConstr(expr, GRB.LESS_EQUAL, ub)
            elif np.isposinf(ub):
                m.addLConstr(expr, GRB.GREATER_EQUAL, lb)
            else:
                m.addRange(expr, lb, ub)
        if mip_start is not None:
            for i in range(model.n_vars):
                variables[i].Start = float(mip_start[i])
        m.optimize()
        elapsed = perf_counter() - started
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time_limit_with_incumbent",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        }
        status = status_map.get(m.Status, f"gurobi_status_{m.Status}")
        has_incumbent = m.SolCount > 0
        objective = float(m.ObjVal) if has_incumbent else None
        bound = float(m.ObjBound) if m.Status != GRB.INFEASIBLE else None
        if status == "time_limit_with_incumbent" and not has_incumbent:
            status = "no_incumbent_time_limit"
        gap_abs = None if bound is None or objective is None else abs(objective - bound)
        gap_rel = None if bound is None or objective is None else float(m.MIPGap)
        raw_x = (
            np.array([variables[i].X for i in range(model.n_vars)]) if has_incumbent else None
        )
        return MilpSolution(
            status,
            objective,
            bound,
            gap_abs,
            gap_rel,
            status == "optimal",
            elapsed,
            int(m.NodeCount),
            raw_x,
            "gurobi",
            {"MIPGap": mip_gap_rel, "MIPGapAbs": mip_gap_abs, "TimeLimit": time_limit},
        )


def solve_milp(
    model: MilpModel,
    *,
    solver: str = "highs",
    time_limit: float | None = None,
    mip_gap_abs: float = 1e-6,
    mip_gap_rel: float = 0.0,
    mip_start: np.ndarray | None = None,
    **extra_options,
) -> MilpSolution:
    """Solve `model` to certified optimality (or a certified bound under a time limit).

    `mip_start`, when given, only affects runtime (Section 9) — it is never
    required for, and never changes, the certificate. HiGHS (via
    `scipy.optimize.milp`) ignores it; the Gurobi backend applies it as a
    warm start.
    """
    if solver == "highs":
        if mip_start is not None:
            extra_options.setdefault("_mip_start_ignored", True)
            extra_options.pop("_mip_start_ignored")
        return _solve_highs(
            model,
            time_limit=time_limit,
            mip_gap_abs=mip_gap_abs,
            mip_gap_rel=mip_gap_rel,
            extra_options=extra_options,
        )
    if solver == "gurobi":
        return _solve_gurobi(
            model,
            time_limit=time_limit,
            mip_gap_abs=mip_gap_abs,
            mip_gap_rel=mip_gap_rel,
            mip_start=mip_start,
            extra_options=extra_options,
        )
    raise ValueError(f"Unknown MILP solver backend: {solver!r}")


def labels_from_solution(model: MilpModel, raw_x: np.ndarray) -> tuple[int, ...]:
    """Decode z[v,k] into one fragment label per vertex.

    Diagnostic only -- do NOT reconstruct `state` by feeding this through
    `graph.state_from_labels` (see `state_from_solution`'s docstring for why
    that loses information C2 deliberately allows).
    """
    labels = []
    for v in range(model.n_vertices):
        chosen = [k for k in range(model.K) if raw_x[model.z_col(v, k)] > 0.5]
        if len(chosen) != 1:
            raise ValueError(f"Vertex {v} has {len(chosen)} assigned fragment labels; expected 1.")
        labels.append(chosen[0])
    return tuple(labels)


def state_from_solution(model: MilpModel, raw_x: np.ndarray) -> tuple[int, ...]:
    """Decode x[e] directly into a `graph.evaluate`-compatible state.

    This must read x[e] itself, NOT re-derive it from z-labels via
    `graph.state_from_labels(labels_from_solution(...))`. C2 only lower-bounds
    x[e] by the endpoints' label disagreement (`x[e] >= |z_u - z_v|`); it does
    not force equality. An optimal WGE solution can legitimately set x[e] = 1
    even when z_u == z_v (same fragment) -- physically, an excision's
    measure/reset/prepare can reuse the same register (Section 5.3: "Local
    measurement followed by reset/preparation can use the same static
    register"), so paying for a cut that doesn't itself split components can
    still be worth it purely to unlock that excision's discount. Reconstructing
    x from labels instead would silently drop that cut, and the subsequent
    `forced_groups` selection would then look inconsistent with the state.
    """
    return tuple(
        1 if raw_x[model.x_col(e)] > 0.5 else 0 for e in range(model.n_edges)
    )


def excision_selection_from_solution(model: MilpModel, raw_x: np.ndarray) -> tuple[str, ...]:
    """Decode y[c] into the selected excision candidates' `group_id`s."""
    if not model.excision:
        return ()
    return tuple(
        candidate.group_id
        for i, candidate in enumerate(model.candidates)
        if raw_x[model.y_col(i)] > 0.5
    )


def milp_plan(model: MilpModel, raw_x: np.ndarray) -> HybridPlan:
    """Convert a solved assignment into the existing, independently compilable plan format."""
    state = state_from_solution(model, raw_x)
    # model.graph is the SAME excision-enabled graph for both WG and WGE
    # (the spec requires both models share one graph). For the WG model,
    # `forced_groups` must be `()` (force zero selections), not `None` (which
    # would fall back to the graph's own unrestricted grouping.select() and
    # could silently discount a cut pattern the WG model never had as a
    # choice, whenever that pattern happens to also satisfy some excision
    # candidate's requirements).
    forced = excision_selection_from_solution(model, raw_x) if model.excision else ()
    return HybridPlan.create(
        model.graph,
        state,
        max_qubits=model.max_qubits,
        partition_mode=model.partition_mode,
        forced_groups=forced,
    )
