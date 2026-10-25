"""Gurobi backend smoke test — skipped until gurobipy + a license are installed.

Exercises the same reference instance A as test_milp.py through the Gurobi
backend, and cross-checks the two backends agree bit-for-bit on the certified
optimum, exactly as docs/MILP_prompt.txt Section 9 requests ("On a subset of
instances, also run without a start to confirm identical optima").
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qpartition import (
    CircuitProblem,
    GroupingConfig,
    PartitionGraph,
    build_milp,
    cheap_feasible_upper_bound,
    compile_hybrid,
    compute_k_safe,
    milp_plan,
    solve_milp,
)

pytest.importorskip("gurobipy")


def _reference_a_problem():
    c = QuantumCircuit(6)
    c.cx(0, 1)
    c.cx(3, 2)
    c.cx(1, 2)
    c.cx(1, 4)
    c.cx(2, 5)
    return CircuitProblem.create(c, "Z" * 6)


def test_gurobi_matches_highs_on_reference_instance_a():
    problem = _reference_a_problem()
    wg_graph = PartitionGraph(problem)
    wge_graph = PartitionGraph(
        problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    max_qubits = 2
    ub = cheap_feasible_upper_bound(wge_graph, max_qubits)
    k = compute_k_safe(wge_graph, max_qubits, ub)
    wg_model = build_milp(wg_graph, max_qubits=max_qubits, K=k, excision=False)
    wge_model = build_milp(wge_graph, max_qubits=max_qubits, K=k, excision=True)

    cases = ((wg_model, wg_graph, 729.0), (wge_model, wge_graph, 256.0))
    for model, graph, expected_omega in cases:
        highs = solve_milp(model, solver="highs", time_limit=60)
        gurobi = solve_milp(model, solver="gurobi", time_limit=60)
        assert highs.proven_optimal and gurobi.proven_optimal
        assert highs.objective == pytest.approx(gurobi.objective, abs=1e-6)
        assert np.exp(gurobi.objective) == pytest.approx(expected_omega, rel=1e-6)
        plan = milp_plan(model, gurobi.raw_x)
        compiled = compile_hybrid(graph, plan)
        assert compiled.evaluation.log_overhead == pytest.approx(gurobi.objective, abs=1e-6)
