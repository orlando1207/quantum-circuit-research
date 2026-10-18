from itertools import combinations, product

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    GroupingConfig,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
)
from qpartition.group_heatbath import orbit, pair_probability, probabilities
from qpartition.label_moves import MultiwayLabelMoves


def test_group_orbit_heatbath_detailed_balance_including_birth_death():
    states = tuple(product(range(3), repeat=3))
    indices = {s: i for i, s in enumerate(states)}
    energies = np.array([0.7 * (s[0] != s[1]) + 1.3 * (s[1] == s[2]) for s in states])
    temperature, regions = 0.8, ((0, 1), (2,))
    matrix = np.zeros((len(states), len(states)))
    for i, source in enumerate(states):
        for pair in combinations(range(3), 2):
            q = pair_probability(source, *pair, 3)
            if not q:
                continue
            targets = orbit(source, regions, *pair)
            weights = probabilities(
                targets, [energies[indices[t]] for t in targets], temperature, pair, 3, False
            )
            for target, weight in zip(targets, weights, strict=True):
                matrix[i, indices[target]] += q * weight
    stationary = np.exp(-energies / temperature)
    flux = stationary[:, None] * matrix
    assert np.max(np.abs(matrix.sum(axis=1) - 1)) < 1e-12
    assert np.max(np.abs(flux - flux.T)) < 1e-12
    assert matrix[indices[(0, 0, 0)], indices[(0, 0, 1)]] > 0


def test_last_anchor_and_nonlocal_wire_windows_preserve_source_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ry(0.37, 1)
    qc.cp(0.7, 0, 1)
    qc.h(2)
    qc.cp(-0.4, 0, 2)
    problem = CircuitProblem.create(qc, "YXX")
    graph = PartitionGraph(problem, grouping=True)
    state = tuple(int(v.kind == "gate") for v in graph.variables)
    evaluation = graph.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert len(evaluation.groups) == 1
    candidate = graph.grouping.by_id[evaluation.groups[0].group_id]
    assert candidate.family == "rotations"
    assert candidate.anchor == candidate.operations[-1]
    assert not candidate.absorbed_operations
    compiled = compile_hybrid(
        graph, HybridPlan.create(graph, state, max_qubits=2, partition_mode="multiway")
    )
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-12)
    far = QuantumCircuit(7)
    far.cp(0.7, 0, 1)
    for _ in range(5):
        far.cx(4, 5)
    far.h(2)
    far.cp(0.4, 0, 2)
    p = CircuitProblem.create(far, "Z" * 7)
    g = PartitionGraph(p, grouping=GroupingConfig(max_gates=2, window_gates=2))
    assert any(
        c.operations == (p.operation_ids[0], p.operation_ids[-1])
        and c.anchor == p.operation_ids[-1]
        for c in g.grouping.candidates
    )
    local = PartitionGraph(
        p, grouping=GroupingConfig(max_gates=2, window_gates=2, wire_windows=False)
    )
    assert not any(
        c.operations == (p.operation_ids[0], p.operation_ids[-1]) for c in local.grouping.candidates
    )


def test_heatbath_budget_and_locked_action_closure():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 1)
    graph = PartitionGraph(CircuitProblem.create(qc, "ZZZZ"), grouping=True)
    config = AnnealConfig(
        2,
        partition_mode="multiway",
        steps=200,
        pilot_steps=0,
        allow_wire=False,
        group_heatbath_probability=1.0,
        temperature_calibration="feasible",
        max_evaluations=32,
    )
    state = tuple(int(v.kind == "gate") for v in graph.variables)
    result = anneal(graph, config, seed=31, initial_state=state)
    assert result.best.feasible
    assert result.evaluations <= 32
    assert result.heatbath_proposals > 0
    assert all(
        not b for v, b in zip(graph.variables, result.best.state, strict=True) if v.kind == "wire"
    )
    moves = MultiwayLabelMoves(
        graph, tuple((0, 1) if v.kind == "gate" else (0,) for v in graph.variables), 4
    )
    for left, right in moves.group_orbits:
        assert not set(left).intersection(right)
        for region in (left, right):
            assert all(
                not set(unit).intersection(region) or set(unit) <= set(region)
                for unit in moves.units
            )
