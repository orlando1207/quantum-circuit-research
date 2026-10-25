"""Section 11 verification for docs/MILP_prompt.txt: tiny-instance correctness.

Checks:
  1. MILP optima match exhaustive enumeration (existing exact small-instance
     tools) on tiny circuits, including the two named reference instances.
  2. Compiled register counts equal MILP widths.
  3. Every selected excision has all its covered edges cut and a recipe norm
     matching gamma_exc(c).
  4. K_safe and a much larger K give the same optimum.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    GroupingConfig,
    PartitionGraph,
    build_milp,
    cheap_feasible_upper_bound,
    compile_hybrid,
    compute_k_safe,
    exact_search,
    excision_selection_from_solution,
    labels_from_solution,
    milp_plan,
    solve_milp,
    state_from_solution,
)
from qpartition.large_benchmarks import large_benchmark


def _graphs(problem):
    wg = PartitionGraph(problem)
    wge = PartitionGraph(
        problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    return wg, wge


def _solve_both(problem, max_qubits, *, k_slack=0, time_limit=60):
    wg_graph, wge_graph = _graphs(problem)
    ub = cheap_feasible_upper_bound(wge_graph, max_qubits)
    k = compute_k_safe(wge_graph, max_qubits, ub) + k_slack
    wg_model = build_milp(wg_graph, max_qubits=max_qubits, K=k, excision=False)
    wge_model = build_milp(wge_graph, max_qubits=max_qubits, K=k, excision=True)
    wg_sol = solve_milp(wg_model, time_limit=time_limit)
    wge_sol = solve_milp(wge_model, time_limit=time_limit)
    return wg_graph, wge_graph, wg_model, wge_model, wg_sol, wge_sol, k


def _brute_force_minimum(graph, max_qubits):
    best = None
    for state in product((0, 1), repeat=len(graph.variables)):
        evaluation = graph.evaluate(
            state, max_qubits=max_qubits, partition_mode="multiway", validate=False
        )
        if evaluation.feasible and (best is None or evaluation.log_overhead < best):
            best = evaluation.log_overhead
    return best


# --- Reference instance A (Section 11): six-qubit toy, M = 2. ---------------
# WG: three independent CX gate cuts, Gamma = 3^3 = 27, Omega = 729.
# WGE: excise the middle CX(1,2) only; four width-2 fragments, Omega = 256.
def _reference_a_problem():
    c = QuantumCircuit(6)
    c.cx(0, 1)
    c.cx(3, 2)
    c.cx(1, 2)
    c.cx(1, 4)
    c.cx(2, 5)
    return CircuitProblem.create(c, "Z" * 6)


def test_reference_instance_a_matches_prompt_section_11():
    problem = _reference_a_problem()
    *_, wg_sol, wge_sol, _ = _solve_both(problem, max_qubits=2)
    assert wg_sol.proven_optimal and wg_sol.objective == pytest.approx(np.log(729), abs=1e-8)
    assert wge_sol.proven_optimal and wge_sol.objective == pytest.approx(np.log(256), abs=1e-8)


# --- Reference instance B: the project's own excision-separator control. ---
# This is `large_benchmarks.large_benchmark("excision_butterfly", 6)` at its
# DEFAULTS (layers=3, seed=19) -- the same circuit as the existing validated
# `excision_butterfly_6_L3_s19` entry in
# validation/incidence_excision_v3/SUMMARY.md, which reports WG=2304, WGE=256
# from best-of-3 SA starts. That table is a heuristic upper bound (SA), never
# proven optimal. This MILP independently CERTIFIES those are the true global
# optima, cross-checked against the existing `exact_search` oracle.
def test_reference_instance_b_separator_matches_known_validated_result():
    benchmark = large_benchmark("excision_butterfly", 6)
    max_qubits = benchmark.max_qubits
    wg_graph, wge_graph, _, _, wg_sol, wge_sol, _ = _solve_both(
        benchmark.problem, max_qubits
    )
    assert wg_sol.proven_optimal and wg_sol.objective == pytest.approx(np.log(2304), abs=1e-8)
    assert wge_sol.proven_optimal and wge_sol.objective == pytest.approx(np.log(256), abs=1e-8)
    cfg = AnnealConfig(max_qubits, partition_mode="multiway", steps=40, max_evaluations=200)
    wg_exact = exact_search(wg_graph, cfg)
    assert wg_exact.complete and wg_exact.best.log_overhead == pytest.approx(
        wg_sol.objective, abs=1e-8
    )
    # The WGE branch-and-bound oracle does not finish within a practical node
    # budget on this instance; its best-found value still corroborates the
    # certified MILP optimum (a weaker, non-exhaustive cross-check here).
    wge_exact = exact_search(wge_graph, cfg, max_nodes=200_000)
    assert wge_exact.best.log_overhead == pytest.approx(wge_sol.objective, abs=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_small_random_circuits_match_exhaustive_enumeration(seed):
    """Section 11 check 1: MILP optima vs. exhaustive enumeration of cut states."""
    rng = np.random.default_rng(seed)
    n = 5
    circuit = QuantumCircuit(n)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    chosen = rng.choice(len(pairs), size=5, replace=True)
    for index in chosen:
        a, b = pairs[index]
        circuit.cx(a, b)
    problem = CircuitProblem.create(circuit, "Z" * n)
    max_qubits = 2
    wg_graph, wge_graph, _, _, wg_sol, wge_sol, _ = _solve_both(problem, max_qubits)
    assert wg_sol.proven_optimal
    assert wge_sol.proven_optimal
    assert _brute_force_minimum(wg_graph, max_qubits) == pytest.approx(wg_sol.objective, abs=1e-8)
    assert _brute_force_minimum(wge_graph, max_qubits) == pytest.approx(
        wge_sol.objective, abs=1e-8
    )
    # Sanity property (Section 7): the WGE optimum never exceeds the WG optimum.
    assert wge_sol.objective <= wg_sol.objective + 1e-8


def test_width_check_compiled_registers_equal_milp_widths():
    """Section 11 check 2."""
    problem = _reference_a_problem()
    _, wge_graph, _, wge_model, _, wge_sol, _ = _solve_both(problem, max_qubits=2)
    plan = milp_plan(wge_model, wge_sol.raw_x)
    compiled = compile_hybrid(wge_graph, plan)
    experiment = compiled.experiment(tuple(0 for _ in compiled.templates))
    for circuit, width in zip(experiment.circuits, compiled.evaluation.block_widths, strict=True):
        assert circuit.num_qubits == width
    assert max(compiled.evaluation.block_widths) <= wge_model.max_qubits


def test_excision_check_covered_edges_cut_and_norms_match_lut():
    """Section 11 check 3."""
    problem = _reference_a_problem()
    wge_graph = PartitionGraph(
        problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    ub = cheap_feasible_upper_bound(wge_graph, 2)
    k = compute_k_safe(wge_graph, 2, ub)
    wge_model = build_milp(wge_graph, max_qubits=2, K=k, excision=True)
    wge_sol = solve_milp(wge_model, time_limit=60)
    labels = labels_from_solution(wge_model, wge_sol.raw_x)
    state = wge_graph.state_from_labels(labels)
    selected_ids = excision_selection_from_solution(wge_model, wge_sol.raw_x)
    assert selected_ids, "Reference instance A's optimum selects exactly one excision."
    for group_id in selected_ids:
        candidate = wge_graph.grouping.by_id[group_id]
        assert all(state[v] for v in candidate.variables), "All B_c edges must be cut."
        assert not any(state[v] for v in candidate.guards), "A retained-gate guard must hold."
        template = wge_graph.registry.entries[candidate.template_id]
        assert candidate.gamma == pytest.approx(template.gamma, rel=1e-12)
        assert candidate.gamma == pytest.approx(16.0, rel=1e-9)  # CX excision recipe norm.


@pytest.mark.parametrize("k_slack", [0, 4, 10])
def test_k_check_k_safe_and_larger_k_give_the_same_optimum(k_slack):
    """Section 11 check 4."""
    problem = _reference_a_problem()
    *_, wge_sol, _k = _solve_both(problem, max_qubits=2, k_slack=k_slack)
    assert wge_sol.proven_optimal
    assert wge_sol.objective == pytest.approx(np.log(256), abs=1e-8)


def test_wg_recompilation_does_not_silently_apply_an_eligible_excision():
    """Regression test: WG and WGE must share one graph, but the WG plan must
    compile with ZERO excisions even when its optimal cut pattern happens to
    also satisfy some excision candidate's requirements (the separator
    family's defining property). A runner that builds separate WG/non-
    excision graphs can hide this; this test shares one excision-enabled
    graph for both models exactly as examples/run_milp_verification.py does.
    """
    from qpartition.large_benchmarks import large_benchmark

    benchmark = large_benchmark("excision_butterfly", 6, layers=1)
    max_qubits = benchmark.max_qubits
    graph = PartitionGraph(
        benchmark.problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    ub = cheap_feasible_upper_bound(graph, max_qubits)
    k = compute_k_safe(graph, max_qubits, ub)
    wg_model = build_milp(graph, max_qubits=max_qubits, K=k, excision=False)
    wg_solution = solve_milp(wg_model, time_limit=60)
    assert wg_solution.proven_optimal
    assert wg_solution.objective == pytest.approx(np.log(729), abs=1e-8)

    plan = milp_plan(wg_model, wg_solution.raw_x)
    assert plan.forced_groups == ()
    compiled = compile_hybrid(graph, plan)
    assert compiled.evaluation.log_overhead == pytest.approx(wg_solution.objective, abs=1e-8)
    assert not compiled.evaluation.groups


def test_state_must_be_read_from_x_not_rederived_from_labels():
    """Regression test: C2 (`x[e] >= |z_u - z_v|`) only LOWER-bounds x by the
    endpoints' label disagreement; it does not force equality. An optimal WGE
    solution can legitimately set x[e] = 1 even when z_u == z_v (the same
    fragment) -- physically, an excision's measure/reset/prepare can reuse the
    same register (Section 5.3), so paying for a cut that doesn't itself split
    components can still be worth it purely to unlock that excision's
    discount. `graph.state_from_labels(labels_from_solution(...))` cannot see
    this (it only looks at label disagreement); `state_from_solution` must
    read x[e] directly. This is a synthetic, hand-built solution -- not a
    solver's choice is needed to exhibit this, since which feasible solver
    optimum exhibits it is solver/tie-break dependent.
    """
    problem = _reference_a_problem()
    graph = PartitionGraph(
        problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    max_qubits = 2
    k = compute_k_safe(graph, max_qubits, cheap_feasible_upper_bound(graph, max_qubits))
    model = build_milp(graph, max_qubits=max_qubits, K=k, excision=True)
    assert model.candidates, "Reference instance A must have >=1 excision candidate."
    candidate = model.candidates[0]

    raw_x = np.zeros(model.n_vars)
    for v in range(model.n_vertices):
        raw_x[model.z_col(v, 0)] = 1.0  # Every vertex shares fragment label 0.
    for e in candidate.variables:
        raw_x[model.x_col(e)] = 1.0  # Cut, even though both endpoints share a label.
    raw_x[model.y_col(model.candidates.index(candidate))] = 1.0

    state = state_from_solution(model, raw_x)
    for e in candidate.variables:
        assert state[e] == 1, "state_from_solution must read x[e] directly."

    labels = labels_from_solution(model, raw_x)
    assert len(set(labels)) == 1, "All vertices were given the same label by construction."
    wrong_state = graph.state_from_labels(labels)
    for e in candidate.variables:
        assert wrong_state[e] == 0, "Demonstrates state_from_labels loses the x[e]=1 information."

    forced = excision_selection_from_solution(model, raw_x)
    assert candidate.group_id in forced
    # The correct decoding is consistent with the forced excision selection...
    graph.evaluate(state, max_qubits=max_qubits, partition_mode="multiway", forced_groups=forced)
    # ...while re-deriving x from labels would raise exactly the inconsistency
    # this bug caused in practice.
    with pytest.raises(ValueError, match="inconsistent with the state"):
        graph.evaluate(
            wrong_state, max_qubits=max_qubits, partition_mode="multiway", forced_groups=forced
        )

    # milp_plan must use the correct decoder, not merely have it available.
    assert milp_plan(model, raw_x).state == state
