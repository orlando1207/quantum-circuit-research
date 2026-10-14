import json

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from qpartition.circuits import (
    CircuitProblem,
    interaction_graph,
    load_problem,
    save_problem,
    temporal_graph,
)
from qpartition.plans import CutPlan, WireCut


def test_query_signs_and_qubit_order():
    circuit = QuantumCircuit(2)
    circuit.x(0)
    problem = CircuitProblem.create(circuit, SparsePauliOp.from_list([("IZ", 2), ("ZI", -0.5)]))
    assert problem.exact_expectation() == pytest.approx(-2.5)


def test_qpy_problem_and_plan_round_trip(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "YY")
    plan = CutPlan.create(problem, gate_cuts=(problem.operation_ids[1],))
    save_problem(problem, tmp_path)
    plan.save(tmp_path / "plan.json")
    restored = load_problem(tmp_path)
    restored_plan = CutPlan.load(tmp_path / "plan.json")
    restored_plan.validate(restored)
    assert restored.exact_expectation() == pytest.approx(-1)


def test_temporal_graph_preserves_each_wire_and_dependencies():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.4, 1)
    circuit.cx(1, 2)
    problem = CircuitProblem.create(circuit, "ZZZ")
    graph = temporal_graph(problem)
    assert nx.is_directed_acyclic_graph(graph)
    assert nx.has_path(graph, problem.operation_ids[1], problem.operation_ids[3])
    assert not nx.has_path(graph, problem.operation_ids[3], problem.operation_ids[1])
    interaction = interaction_graph(problem)
    assert set(interaction.edges) == {(0, 1), (1, 2)}
    assert interaction[0][1]["operation_ids"] == [problem.operation_ids[1]]


@pytest.mark.parametrize("kind", ["parameter", "measurement", "reset", "nonhermitian", "width"])
def test_invalid_problem_is_rejected(kind):
    circuit = QuantumCircuit(1)
    observable = "Z"
    if kind == "parameter":
        circuit.rx(Parameter("theta"), 0)
    elif kind == "measurement":
        circuit.measure_all()
    elif kind == "reset":
        circuit.reset(0)
    elif kind == "nonhermitian":
        observable = SparsePauliOp("X", coeffs=[1j])
    else:
        observable = "ZZ"
    with pytest.raises(ValueError):
        CircuitProblem.create(circuit, observable)


def test_plan_rejects_mutated_snapshot_and_wrong_wire():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    problem = CircuitProblem.create(circuit, "IZ")
    with pytest.raises(ValueError, match="on that wire"):
        CutPlan.create(problem, wire_cuts=(WireCut(1, problem.operation_ids[0]),))
    plan = CutPlan.create(problem)
    problem.circuit.x(1)
    with pytest.raises(ValueError, match="snapshot changed"):
        plan.validate(problem)


def test_input_is_copied():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    observable = SparsePauliOp("Z")
    problem = CircuitProblem.create(circuit, observable)
    circuit.h(0)
    observable.coeffs[:] = np.array([3])
    assert problem.exact_expectation() == pytest.approx(-1)


def test_query_mutation_is_detected():
    problem = CircuitProblem.create(QuantumCircuit(1), "Z")
    problem.observable.coeffs[:] = 2
    with pytest.raises(ValueError, match="Observable snapshot changed"):
        problem.exact_expectation()


def test_problem_sidecar_detects_changed_query(tmp_path):
    problem = CircuitProblem.create(QuantumCircuit(1), "Z")
    save_problem(problem, tmp_path)
    path = tmp_path / "problem.json"
    sidecar = json.loads(path.read_text())
    sidecar["observable"][0][1] = 2
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="Observable does not match"):
        load_problem(tmp_path)


def test_composite_gate_validation_and_round_trip(tmp_path):
    block = QuantumCircuit(2)
    block.h(0)
    block.cx(0, 1)
    circuit = QuantumCircuit(2)
    circuit.append(block.to_gate(), [0, 1])
    problem = CircuitProblem.create(circuit, "YY")
    assert problem.exact_expectation() == pytest.approx(-1)
    problem.validate_snapshot()
    save_problem(problem, tmp_path)
    restored = load_problem(tmp_path)
    assert restored.circuit_fingerprint == problem.circuit_fingerprint
    assert restored.exact_expectation() == pytest.approx(-1)
