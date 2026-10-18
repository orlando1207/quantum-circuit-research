from math import exp, log

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qpartition import CircuitProblem
from qpartition.baseline_circuits import grover4_historical
from qpartition.paper_baseline import (
    GateGroup,
    PaperConfig,
    PaperTopology,
    run_paper_reconstruction,
)
from qpartition.qiskit_baseline import QiskitBaselineConfig, run_qiskit_baseline


def test_direct_qiskit_gate_and_wire_baselines():
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    problem = CircuitProblem.create(bell, "XX")
    _, result = run_qiskit_baseline(problem, QiskitBaselineConfig(1))
    assert result["sampling_overhead"] == pytest.approx(9)
    assert result["minimum_reached"]
    assert result["gate_cuts"] == [problem.operation_ids[1]]
    assert result["widths"] == (1, 1)
    chain = QuantumCircuit(3)
    chain.cx(0, 1)
    chain.cx(1, 2)
    problem = CircuitProblem.create(chain, "ZZZ")
    _, result = run_qiskit_baseline(problem, QiskitBaselineConfig(2, gate_lo=False))
    assert result["sampling_overhead"] == pytest.approx(16)
    assert sum(result["widths"]) == 4
    assert result["wire_cuts"] == [{"qubit": 1, "before_operation_id": problem.operation_ids[1]}]


def test_paper_graph_static_widths_and_swap_updates_against_connectivity():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    topology = PaperTopology(CircuitProblem.create(circuit, "ZZZ"))
    assert topology.size == 2 * 3 + 2 * 2
    rng = np.random.default_rng(9)
    for _ in range(100):
        labels = rng.integers(0, 2, topology.size)
        wires = nx.Graph()
        wires.add_nodes_from(range(topology.size))
        wires.add_edges_from((a, b) for a, b, _, _ in topology.wires if labels[a] == labels[b])
        expected = [0, 0]
        for segment in nx.connected_components(wires):
            colors = {labels[v] for v in segment}
            assert len(colors) == 1
            expected[colors.pop()] += 1
        assert topology.widths(labels).tolist() == expected
        deltas = topology.flip_width_deltas(labels)
        for a in np.flatnonzero(labels == 0):
            for b in np.flatnonzero(labels == 1):
                candidate = labels.copy()
                candidate[a], candidate[b] = 1, 0
                predicted = expected + deltas[a] + deltas[b] + topology.wire_adjacency[a, b]
                assert predicted.tolist() == topology.widths(candidate).tolist()


@pytest.mark.parametrize(
    "kind,pairs,colors,gamma",
    [
        ("parallel", [(0, 1), (2, 3)], (0, 1, 0, 1), 7),
        ("cascade_control", [(0, 1), (0, 2)], (0, 1, 1), 3),
        ("cascade_target", [(0, 2), (1, 2)], (0, 0, 1), 3),
    ],
)
def test_paper_explicit_group_rules_against_complete_unitary_schmidt_audit(
    kind, pairs, colors, gamma
):
    circuit = QuantumCircuit(len(colors))
    for a, b in pairs:
        circuit.cx(a, b)
    problem = CircuitProblem.create(circuit, "Z" * len(colors))
    group = GateGroup("figure3", kind, problem.operation_ids, "Paper Figure 3, explicit CX pattern")
    topology = PaperTopology(problem, (group,))
    labels = topology.spatial_labels(colors)
    assert exp(topology.cost(labels)) == pytest.approx(gamma)
    # Independent full-unitary operator-Schmidt check for these canonical patterns.
    n = len(colors)
    a, b = [q for q in range(n) if colors[q] == 0], [q for q in range(n) if colors[q] == 1]
    axes = [n - 1 - q for q in a] + [2 * n - 1 - q for q in a]
    axes += [n - 1 - q for q in b] + [2 * n - 1 - q for q in b]
    matrix = Operator(circuit).data.reshape([2] * (2 * n)).transpose(axes)
    schmidt = np.linalg.svd(
        matrix.reshape(4 ** len(a), 4 ** len(b)) / np.sqrt(2**n), compute_uv=False
    )
    assert 2 * sum(schmidt) ** 2 - 1 == pytest.approx(gamma)
    _, run = run_paper_reconstruction(problem, PaperConfig(max(len(a), len(b))), (group,))
    assert run["status"] == "feasible"
    assert "both stages" in run["grouped_status"]


def test_printed_group_equation_three_disagrees_with_objective_difference():
    # Both parallel CX gates cross. Swapping both endpoints of the first keeps
    # both cuts, so the group's true cost change is zero.
    current, single_a, single_b, swapped = log(7), log(3), log(3), log(7)
    printed = (current - single_a) + (current - single_b) + (swapped - current)
    actual = current - swapped
    assert actual == 0
    assert printed == pytest.approx(2 * log(7 / 3))
    assert printed > 0


def test_endpoint_surrogate_does_not_imply_physical_capacity():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    graph = PaperTopology(CircuitProblem.create(circuit, "ZZZZ"))
    labels = np.zeros(graph.size, dtype=int)
    labels[list(graph.outputs)] = 1
    endpoint = np.bincount(labels, weights=graph.endpoint_sizes)
    assert max(endpoint) <= 2 * 3
    assert graph.widths(labels).tolist() == [4, 4]


def test_invalid_or_overlapping_manual_group_manifests_are_rejected():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.cx(0, 2)
    problem = CircuitProblem.create(circuit, "ZZZ")
    group = GateGroup(
        "bad",
        "cascade_control",
        (problem.operation_ids[0], problem.operation_ids[2]),
        "Explicit invalid-interleaving diagnostic",
    )
    with pytest.raises(ValueError, match="Intervening"):
        PaperTopology(problem, (group,))
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    problem = CircuitProblem.create(circuit, "ZZZZ")
    group = GateGroup("parallel", "parallel", problem.operation_ids, "Paper canonical pattern")
    with pytest.raises(ValueError, match="Overlapping"):
        PaperTopology(problem, (group, group))


def test_historical_grover_uses_native_phase_gates_and_verified_unitary():
    problem, provenance = grover4_historical()
    assert problem.circuit.count_ops()["cx"] == 24
    assert problem.circuit.count_ops()["cp"] == 14
    assert provenance["complete_unitary_residual"] < 1e-12
    assert provenance["phase_unitary_residual"] < 1e-12
    assert max(item.operation.num_qubits for item in problem.circuit.data) <= 2
