from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp

from qpartition import CircuitProblem
from qpartition.annealing import AnnealConfig, anneal, multi_start
from qpartition.exact_search import exact_search
from qpartition.hybrid import HybridPlan, compile_hybrid
from qpartition.partition_graph import PartitionGraph, pack_two
from qpartition.templates import TemplateRegistry


def select(graph, choices):
    state = [0] * len(graph.variables)
    for index, kind in choices.items():
        state[index] = next(
            i
            for i, option in enumerate(graph.variables[index].options)
            if option and option.kind == kind
        )
    return tuple(state)


def test_cx_modes_and_two_block_packing():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "ZZ")
    graph = PartitionGraph(problem)
    variable = graph.gate_variables[problem.operation_ids[0]]
    expected = {"gate": ([1, 1], 9), "excision": ([1, 1, 1, 1], 256)}
    for kind, (widths, overhead) in expected.items():
        evaluation = graph.evaluate(select(graph, {variable: kind}), max_qubits=2)
        assert sorted(evaluation.component_widths) == widths
        assert evaluation.sampling_overhead == pytest.approx(overhead)
        assert evaluation.feasible
    assert pack_two((2, 2, 2), 3)[0] > 0
    assert pack_two((2, 1, 3), 3)[0] == 0


@pytest.mark.parametrize("widths", [(2, 2, 2), (4, 1, 1), (1, 1, 1, 1), (3,), (2, 3, 4, 1)])
def test_two_block_packing_matches_bruteforce(widths):
    limit = 3
    expected = min(
        max(0, sum(w for w, color in zip(widths, colors) if color == 0) - limit) ** 2
        + max(0, sum(w for w, color in zip(widths, colors) if color == 1) - limit) ** 2
        for colors in product((0, 1), repeat=len(widths))
    )
    penalty, colors, blocks = pack_two(widths, limit)
    assert penalty == expected
    assert sum(blocks) == sum(widths)
    assert (
        tuple(sum(w for w, color in zip(widths, colors) if color == b) for b in range(len(blocks)))
        == blocks
    )


def test_wire_cut_rejoining_does_not_reuse_original_qubit():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "ZZ")
    graph = PartitionGraph(problem)
    wire = next(
        i
        for i, variable in enumerate(graph.variables)
        if variable.kind == "wire" and variable.operation_id == problem.operation_ids[2]
    )
    state = select(graph, {wire: "wire"})
    evaluation = graph.evaluate(state, max_qubits=2)
    assert evaluation.component_widths == (3,)
    assert evaluation.width_penalty > 0
    assert not evaluation.feasible


@pytest.mark.parametrize("kind", ["gate", "excision"])
@pytest.mark.parametrize("pauli", ["XX", "YY", "ZZ", "XY", "IZ"])
def test_two_qubit_channel_reconstruction(kind, pauli):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.37, 1)
    circuit.cx(0, 1)
    circuit.t(0)
    problem = CircuitProblem.create(circuit, pauli)
    graph = PartitionGraph(problem)
    state = select(graph, {graph.gate_variables[problem.operation_ids[2]]: kind})
    plan = HybridPlan.create(graph, state, max_qubits=3)
    result = compile_hybrid(graph, plan).run_exact()
    assert result.expectation == pytest.approx(problem.exact_expectation(), abs=1e-12)


def test_all_three_actions_with_correlations_and_pauli_sum(tmp_path):
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.h(2)
    circuit.cx(0, 1)
    circuit.t(0)
    circuit.cx(1, 2)
    query = SparsePauliOp.from_list([("XXX", 0.4), ("YXY", -0.2), ("ZIZ", 0.3)])
    problem = CircuitProblem.create(circuit, query)
    graph = PartitionGraph(problem)
    wire = next(
        i
        for i, variable in enumerate(graph.variables)
        if variable.kind == "wire"
        and variable.operation_id == problem.operation_ids[4]
        and variable.qubit == 1
    )
    state = select(
        graph,
        {
            wire: "wire",
            graph.gate_variables[problem.operation_ids[3]]: "excision",
            graph.gate_variables[problem.operation_ids[4]]: "gate",
        },
    )
    plan = HybridPlan.create(graph, state, max_qubits=3)
    assert graph.evaluate(state, max_qubits=3).action_counts == {
        "wire": 1,
        "gate": 1,
        "excision": 1,
    }
    plan.save(tmp_path / "plan.json")
    restored = HybridPlan.load(tmp_path / "plan.json")
    result = compile_hybrid(graph, restored).run_exact()
    assert result.expectation == pytest.approx(problem.exact_expectation(), abs=1e-12)
    graph.registry.save(tmp_path / "templates.json")


def test_exact_search_and_annealing_reproducibility():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "XX")
    graph = PartitionGraph(problem)
    config = AnnealConfig(1, steps=300, temperature_levels=30)
    exact = exact_search(graph, config)
    assert exact.complete
    assert exact.best.sampling_overhead == pytest.approx(9)
    first = anneal(graph, config, seed=5)
    repeated = anneal(graph, config, seed=5)
    assert first.best == repeated.best
    assert first.trace == repeated.trace
    assert first.best.sampling_overhead == pytest.approx(exact.best.sampling_overhead)
    results = multi_start(graph, config, seeds=(2, 3))
    assert all(result.best.feasible for result in results)
    limited = exact_search(graph, config, max_nodes=1)
    assert not limited.complete
    assert limited.best.feasible


def test_mode_validation_and_proven_infeasibility():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    graph = PartitionGraph(CircuitProblem.create(circuit, "ZZZ"))
    with pytest.raises(ValueError, match="infeasible"):
        anneal(graph, AnnealConfig(1))
    result = anneal(graph, AnnealConfig(1, partition_mode="components", steps=200), seed=1)
    assert result.best.feasible
    assert max(result.best.block_widths) == 1
    assert len(result.best.block_widths) == 3
    with pytest.raises(ValueError):
        anneal(graph, replace(AnnealConfig(2), cooling_rate=1))
    with pytest.raises(ValueError):
        graph.evaluate((0,), max_qubits=2)


def test_importance_sampling_trials_and_physical_shots():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "XX")
    graph = PartitionGraph(problem)
    state = select(graph, {graph.gate_variables[problem.operation_ids[1]]: "gate"})
    compiled = compile_hybrid(graph, HybridPlan.create(graph, state, max_qubits=1))
    result = compiled.run_sampled(trials=1000, seed=81)
    assert result.physical_shots == 2000
    assert result.reconstruction_trials == 1000
    assert abs(result.expectation - 1) <= 5 * result.standard_error
    assert np.isfinite(result.standard_error)


@pytest.mark.parametrize("gate", ["reverse_cx", "cp", "swap"])
@pytest.mark.parametrize("kind", ["gate", "excision"])
def test_general_gate_and_entangled_spectator(gate, kind):
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.ry(0.41, 1)
    if gate == "reverse_cx":
        circuit.cx(1, 0)
    elif gate == "cp":
        circuit.cp(0.72, 0, 1)
    else:
        circuit.swap(0, 1)
    circuit.t(1)
    query = SparsePauliOp.from_list([("XZX", 0.4), ("IZI", -0.2), ("ZZI", 0.3)])
    problem = CircuitProblem.create(circuit, query)
    graph = PartitionGraph(problem)
    state = select(graph, {graph.gate_variables[problem.operation_ids[3]]: kind})
    plan = HybridPlan.create(graph, state, max_qubits=4)
    result = compile_hybrid(graph, plan).run_exact(max_experiments=20000)
    assert result.expectation == pytest.approx(problem.exact_expectation(), abs=1e-12)


def test_custom_gate_name_cannot_corrupt_lookup(tmp_path):
    registry = TemplateRegistry()
    gate = UnitaryGate(SwapGate().to_matrix())
    gate.name = "cx"
    cut, excision = registry.for_gate(gate)
    assert cut.reconstruction_residual < 1e-10
    assert cut.gamma == pytest.approx(7)
    assert excision.reconstruction_residual < 1e-10
    registry.save(tmp_path / "lut.json")
