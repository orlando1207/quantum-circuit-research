from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    GroupingConfig,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    exact_search,
)
from qpartition.annealing import allowed_options
from qpartition.label_moves import MultiwayLabelMoves


def all_gate_cuts(graph):
    return tuple(int(v.kind == "gate") for v in graph.variables)


def signed_map(matrix, steps):
    n = matrix.shape[0].bit_length() - 1
    for step in steps:
        if step.operation.name == "measure":
            qubit = step.qubits[0]
            one = (np.arange(2**n) & (1 << qubit)) != 0
            p0, p1 = np.diag(~one), np.diag(one)
            matrix = p0 @ matrix @ p0 + (-1 if step.scored else 1) * p1 @ matrix @ p1
        else:
            qc = QuantumCircuit(n)
            qc.append(step.operation, step.qubits)
            u = Operator(qc).data
            matrix = u @ matrix @ u.conj().T
    return matrix


@pytest.mark.parametrize(
    "kind",
    [
        "control",
        "target",
        "cz",
        "parallel",
        "phases",
        "swap",
        "chain",
        "interleaved_clifford",
        "generic",
        "large_star",
    ],
)
def test_complete_joint_channel_on_pauli_basis(kind):
    """Full process audit, including Y phases and multi-qubit parity coherence."""
    n = 4 if kind in {"parallel", "swap", "large_star"} else 3
    qc = QuantumCircuit(n)
    if kind == "control":
        qc.cx(0, 1)
        qc.cx(0, 2)
    elif kind == "target":
        qc.cx(0, 2)
        qc.cx(1, 2)
    elif kind == "cz":
        qc.cz(1, 0)
        qc.cz(2, 0)
    elif kind == "parallel":
        qc.cx(1, 0)
        qc.cx(2, 3)
    elif kind == "swap":
        qc.swap(0, 1)
        qc.cp(-0.71, 2, 3)
    elif kind == "chain":
        qc.cx(0, 1)
        qc.cx(1, 2)
    elif kind == "interleaved_clifford":
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(0, 2)
    elif kind == "generic":
        qc.cx(0, 1)
        qc.ry(0.1, 1)
        qc.cx(1, 2)
    elif kind == "large_star":
        qc.cp(0.7, 0, 1)
        qc.cp(-0.3, 0, 2)
        qc.cp(0.2, 0, 3)
    else:
        qc.cp(0.47, 0, 1)
        qc.rzz(-0.62, 0, 2)
    graph = PartitionGraph(CircuitProblem.create(qc, "Z" * n), grouping=True)
    evaluation = graph.evaluate(
        all_gate_cuts(graph), max_qubits=3 if kind == "large_star" else 2, partition_mode="multiway"
    )
    assert len(evaluation.groups) == 1
    selected = evaluation.groups[0]
    candidate = graph.grouping.by_id[selected.group_id]
    basis = graph.grouping.basis(selected)
    assert basis.gamma**2 == pytest.approx(evaluation.sampling_overhead)
    assert basis.unitary_residual < 1e-12
    sides = tuple(
        tuple(q for i, q in enumerate(candidate.qubits) if (selected.pattern >> i & 1) == side)
        for side in (0, 1)
    )
    new_order = sides[0] + sides[1]
    indices = [
        sum(((bits >> i) & 1) << candidate.qubits.index(q) for i, q in enumerate(new_order))
        for bits in range(2**n)
    ]
    target = Operator(qc).data[np.ix_(indices, indices)]
    paulis = [
        [Pauli("".join(p)).to_matrix() for p in product("IXYZ", repeat=len(side))] for side in sides
    ]
    local = [
        [np.array([signed_map(p, maps[side]) for p in paulis[side]]) for maps in basis.maps]
        for side in (0, 1)
    ]
    for a, pa in enumerate(paulis[0]):
        for b, pb in enumerate(paulis[1]):
            reconstructed = sum(
                w * np.kron(local[1][k][b], local[0][k][a])
                for k, w in enumerate(basis.coefficients)
            )
            rho = np.kron(pb, pa)
            assert np.max(np.abs(reconstructed - target @ rho @ target.conj().T)) < 2e-12


def test_canonical_costs_and_capacity_aware_local_bundling():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    graph = PartitionGraph(CircuitProblem.create(qc, "ZZZZ"), grouping=True)
    state = all_gate_cuts(graph)
    narrow = graph.evaluate(state, max_qubits=1, partition_mode="multiway")
    assert narrow.sampling_overhead == pytest.approx(81)
    assert narrow.block_widths == (1, 1, 1, 1)
    assert not narrow.groups  # Joint maps require two qubits on each party.
    wide = graph.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert wide.sampling_overhead == pytest.approx(49)
    assert wide.block_widths == (2, 2)
    assert sum(wide.block_widths) == 4
    binary = graph.evaluate(state, max_qubits=2, partition_mode="two_blocks")
    assert binary.sampling_overhead == pytest.approx(49)
    assert binary.block_widths == (2, 2)


def test_single_qubit_gate_validity_and_interior_wire_guards():
    def make(gate):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        getattr(qc, gate)(0)
        qc.cx(0, 2)
        return PartitionGraph(CircuitProblem.create(qc, "ZZZ"), grouping=True)

    blocked = make("h")
    assert all(c.family == "clifford" for c in blocked.grouping.candidates)
    assert blocked.evaluate(
        all_gate_cuts(blocked), max_qubits=2, partition_mode="multiway"
    ).sampling_overhead == pytest.approx(49)
    allowed = make("t")  # Diagonal on the common control, so it commutes.
    state = all_gate_cuts(allowed)
    grouped = allowed.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert grouped.sampling_overhead == pytest.approx(9)
    cuts = list(state)
    cuts[next(i for i, v in enumerate(allowed.variables) if v.kind == "wire")] = 1
    broken = allowed.evaluate(tuple(cuts), max_qubits=2, partition_mode="multiway")
    assert not broken.groups
    assert broken.sampling_overhead == pytest.approx(81 * 16)


def test_fusion_preserves_interleaved_single_qubit_gates_and_round_trip(tmp_path):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ry(0.37, 1)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.t(1)
    qc.ry(-0.4, 0)
    qc.cx(0, 1)
    qc.s(0)
    query = SparsePauliOp.from_list([("XZX", 0.4), ("YYI", -0.3), ("IZZ", 0.2)])
    problem = CircuitProblem.create(qc, query)
    graph = PartitionGraph(problem, grouping=True)
    state = list(all_gate_cuts(graph))
    state[graph.gate_variables[problem.operation_ids[2]]] = 0
    plan = HybridPlan.create(graph, tuple(state), max_qubits=2, partition_mode="multiway")
    assert plan.validate(graph).groups
    plan.save(tmp_path / "plan.json")
    restored = HybridPlan.load(tmp_path / "plan.json")
    assert restored == plan
    compiled = compile_hybrid(graph, restored)
    assert compiled.run_exact(max_experiments=50000).expectation == pytest.approx(
        problem.exact_expectation(), abs=1e-11
    )
    with pytest.raises(ValueError, match="grouping LUT"):
        compile_hybrid(graph, replace(plan, grouping_fingerprint="wrong"))
    other = PartitionGraph(problem, grouping=GroupingConfig(group_moves=False))
    with pytest.raises(ValueError, match="grouping LUT"):
        plan.validate(other)


def test_parallel_physical_reconstruction_with_nontrivial_query():
    qc = QuantumCircuit(4)
    qc.ry(0.4, 0)
    qc.h(1)
    qc.rx(-0.2, 2)
    qc.t(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.s(0)
    qc.ry(0.3, 3)
    problem = CircuitProblem.create(qc, SparsePauliOp.from_list([("XYYZ", 0.6), ("IZXX", -0.4)]))
    graph = PartitionGraph(problem, grouping=True)
    plan = HybridPlan.create(graph, all_gate_cuts(graph), max_qubits=2, partition_mode="multiway")
    compiled = compile_hybrid(graph, plan)
    assert compiled.run_exact(max_experiments=50000).expectation == pytest.approx(
        problem.exact_expectation(), abs=1e-12
    )
    experiment = compiled.experiment((0,))
    assert tuple(c.num_qubits for c in experiment.circuits) == (2, 2)
    sampled = compiled.run_sampled(trials=300, seed=78)
    assert sampled.physical_shots == 600
    assert abs(sampled.expectation - problem.exact_expectation()) < 5 * sampled.standard_error


def test_overlapping_groups_are_not_double_counted_and_exact_search_is_safe():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.cx(0, 1)
    graph = PartitionGraph(CircuitProblem.create(qc, "ZZ"), grouping=True)
    config = AnnealConfig(1, partition_mode="multiway", steps=100)
    evaluation = graph.evaluate(all_gate_cuts(graph), max_qubits=1, partition_mode="multiway")
    covered = [
        op
        for selected in evaluation.groups
        for op in graph.grouping.by_id[selected.group_id].operations
    ]
    assert len(covered) == len(set(covered))
    assert evaluation.sampling_overhead == pytest.approx(9)
    brute = []
    for state in product((0, 1), repeat=len(graph.variables)):
        e = graph.evaluate(state, max_qubits=1, partition_mode="multiway")
        if e.feasible:
            brute.append(e.sampling_overhead)
    exact = exact_search(graph, config)
    assert exact.complete
    assert exact.best.sampling_overhead == pytest.approx(min(brute))
    run = anneal(graph, config, seed=9)
    assert run.best.sampling_overhead == pytest.approx(min(brute))


def test_group_regions_respect_disabled_wire_actions():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    graph = PartitionGraph(CircuitProblem.create(qc, "ZZZ"), grouping=True)
    config = AnnealConfig(2, partition_mode="multiway", allow_wire=False)
    allowed = allowed_options(graph, config)
    moves = MultiwayLabelMoves(graph, allowed, 3)
    labels = (0, 1, 0, 2)
    rng = np.random.default_rng(42)
    for _ in range(50):
        labels, _, _ = moves.propose(labels, rng, 0.6)
        state = graph.state_from_labels(labels)
        assert all(v in choices for v, choices in zip(state, allowed))


def test_native_gate_name_spoofing_does_not_trigger_cascade():
    qc = QuantumCircuit(3)
    fake = UnitaryGate(Operator(Pauli("XX")).data)
    fake.name = "cx"
    qc.append(fake, [0, 1])
    qc.cx(0, 2)
    graph = PartitionGraph(CircuitProblem.create(qc, "ZZZ"), grouping=True)
    assert not graph.grouping.candidates


def test_moving_later_gates_only_respects_source_order():
    qc = QuantumCircuit(3)
    qc.ry(0.3, 0)
    qc.h(1)
    qc.cx(0, 1)
    qc.t(1)  # Does not commute with the first CX, but does commute with the later CX.
    qc.cx(0, 2)
    problem = CircuitProblem.create(qc, "XYY")
    graph = PartitionGraph(problem, grouping=True)
    e = graph.evaluate(all_gate_cuts(graph), max_qubits=2, partition_mode="multiway")
    assert e.sampling_overhead == pytest.approx(9)
    plan = HybridPlan.create(graph, e.state, max_qubits=2, partition_mode="multiway")
    assert compile_hybrid(graph, plan).run_exact().expectation == pytest.approx(
        problem.exact_expectation(), abs=1e-12
    )


def test_exact_search_does_not_certify_heuristic_group_selection():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    graph = PartitionGraph(
        CircuitProblem.create(qc, "ZZZ"), grouping=GroupingConfig(exact_selection_limit=0)
    )
    config = AnnealConfig(2, partition_mode="multiway", steps=20)
    result = exact_search(graph, config)
    assert not result.complete
    assert result.best.sampling_overhead == pytest.approx(9)
