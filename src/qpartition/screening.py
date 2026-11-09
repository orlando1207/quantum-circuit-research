"""Phase 2a breadth-first screening pipeline (docs/second_stage_cont_prompt.txt
Part 2). Keeps the existing graph formulation, MILP, cost registry and plan
format unchanged -- this module only orchestrates existing pieces.

Mandatory preprocessing order (identical for WG and WGE, per the prompt):
    logical circuit -> SWAP relabeling (`preprocessing.peel_swaps`)
                     -> two-qubit gate fusion (`preprocessing.fuse_two_qubit_gates`)
                     -> KAK-optimal gate-cut costs with SWAP peeling (already
                        applied by `peel_swaps`, so a post-fusion gate's
                        `gamma_cut_gate` IS its SWAP-peeled cost)
                     -> LP excision costs (`excision_lp.gate_excision_cost`).

Screening metric: solve WG only (excision disabled) with a short SA budget,
then check the WG-optimal cut state for excision-ELIGIBLE junctions (a cut
gate edge with both adjacent wire edges also cut, or any other pattern the
existing `GroupLibrary` catalogue recognizes) -- i.e. does excision's
conditional discount already apply "for free" to the WG-optimal structure
`select()` found, without re-running the search with excision turned on.
`_eligible_candidates` below is the exact eligibility test from
`GroupLibrary.select()` (grouping.py), lifted out standalone since no public
function already exposes it independent of the disjointness/packing search
that follows it there. Per CLAUDE.md's documented invariant, a WG state's
cut-choice indices are directly valid against a shared excision-enabled
graph built from the same `CircuitProblem` (this is the same fact that makes
`forced_groups=()` meaningful for MILP recompilation).

This is a cheap, NOT certified estimate: `predicted_gain` is the product of
`per_junction_gain` over the eligible junctions found in one WG-SA state,
not a search over all WG-optimal (or better) structures, and not a solved
WGE optimum. Phase 2b re-certifies any family this flags (junction_count>0)
to proven optimality via the existing `milp.build_milp`/`solve_milp`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Operator

from .annealing import AnnealConfig, anneal
from .circuits import CircuitProblem
from .excision_lp import gamma_cut_gate, gate_excision_cost, per_junction_gain
from .partition_graph import PartitionGraph
from .preprocessing import fuse_two_qubit_gates, peel_swaps, two_qubit_gate_count


@dataclass(frozen=True)
class ScreeningResult:
    family: str
    source: str
    version: str
    params: str
    n_qubits: int
    fused_two_qubit_gates: int
    max_qubits: int
    wg_log_overhead: float  # natural log, matching GraphEvaluation.log_overhead
    wg_feasible: bool
    junction_count: int
    predicted_gain: float
    predicted_gain_is_lower_bound_only: bool
    status: str  # "screened" | "not_certifiable_at_milp_scale" | "error"
    note: str = ""


def prepare_for_screening(problem: CircuitProblem) -> CircuitProblem:
    """Apply the mandatory SWAP-relabel-then-fuse order and rebuild a problem."""
    circuit, observable = peel_swaps(problem.circuit, problem.observable)
    fused = fuse_two_qubit_gates(circuit)
    return CircuitProblem.create(fused, observable)


def _eligible_candidates(graph: PartitionGraph, state: tuple[int, ...]) -> list:
    """Candidates eligible under `state` -- the same test as
    `GroupLibrary.select()` (grouping.py), without its disjointness/packing
    search (screening only needs the raw eligible set, not a chosen subset)."""
    return [
        c
        for c in graph.grouping.candidates
        if all(state[v] for v in c.variables) and not any(state[v] for v in c.guards)
    ]


def screen_problem(
    problem: CircuitProblem,
    *,
    max_qubits: int,
    family: str,
    source: str,
    version: str,
    params: str,
    max_evaluations: int = 1200,
    seed: int = 0,
) -> ScreeningResult:
    try:
        prepared = prepare_for_screening(problem)
    except Exception as exc:  # noqa: BLE001 -- a screening sweep must not crash on one family
        return ScreeningResult(
            family, source, version, params, problem.circuit.num_qubits, 0, max_qubits,
            float("nan"), False, 0, 1.0, False, "error", f"preprocessing: {exc}",
        )
    n_qubits = prepared.circuit.num_qubits
    fused_g2 = two_qubit_gate_count(prepared.circuit)

    wg_graph = PartitionGraph(prepared, excision=False)
    cfg = AnnealConfig(
        max_qubits,
        partition_mode="multiway",
        max_evaluations=max_evaluations,
        steps=3 * max_evaluations // 5,
        pilot_steps=48,
    )
    try:
        result = anneal(wg_graph, cfg, seed=seed)
    except Exception as exc:  # noqa: BLE001
        return ScreeningResult(
            family, source, version, params, n_qubits, fused_g2, max_qubits,
            float("nan"), False, 0, 1.0, False, "error", f"anneal: {exc}",
        )
    best = result.best
    if best is None or not best.feasible:
        return ScreeningResult(
            family, source, version, params, n_qubits, fused_g2, max_qubits,
            float("nan"), False, 0, 1.0, False,
            "not_certifiable_at_milp_scale", "SA found no feasible WG state",
        )

    exc_graph = PartitionGraph(prepared, excision=True)
    junctions = _eligible_candidates(exc_graph, best.state)

    predicted_gain = 1.0
    any_upper_bound = False
    for candidate in junctions:
        index = prepared.index(candidate.anchor)
        instruction = prepared.circuit.data[index].operation
        U = np.asarray(Operator(instruction).data, dtype=complex)
        gamma_eff = gamma_cut_gate(U)  # already SWAP-peeled, by `peel_swaps` upstream
        cost = gate_excision_cost(U)
        predicted_gain *= per_junction_gain(cost.gamma_ex, gamma_eff)
        any_upper_bound = any_upper_bound or cost.status != "EXACT"

    return ScreeningResult(
        family, source, version, params, n_qubits, fused_g2, max_qubits,
        best.log_overhead, best.feasible, len(junctions), predicted_gain,
        any_upper_bound, "screened",
    )
