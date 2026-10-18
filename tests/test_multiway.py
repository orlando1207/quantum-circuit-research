from dataclasses import replace
from itertools import combinations, product
from math import exp, log

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    exact_search,
    multi_start,
    refine_partition,
)
from qpartition.annealing import allowed_options, seed_states
from qpartition.benchmarks import chain_benchmark
from qpartition.label_moves import MultiwayLabelMoves
from qpartition.multiway import bridge_seed_state


def three_fragment_problem():
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.21, 1)
    circuit.cx(1, 2)
    circuit.rz(0.31, 2)
    circuit.cx(2, 3)
    circuit.h(3)
    query = SparsePauliOp.from_list([("XXXX", 0.6), ("IIZZ", 0.4)])
    return CircuitProblem.create(circuit, query)


def test_multiway_wire_only_search_and_physical_reconstruction(tmp_path):
    graph = PartitionGraph(three_fragment_problem())
    config = AnnealConfig(2, partition_mode="multiway", allow_gate=False, steps=300)
    exact = exact_search(graph, config)
    assert exact.complete
    assert exact.best.sampling_overhead == pytest.approx(256)
    assert exact.best.block_widths == (2, 2, 2)
    assert exact.best.action_counts == {"wire": 2, "gate": 0}
    binary = exact_search(graph, replace(config, partition_mode="two_blocks"))
    assert binary.complete and binary.best is None
    runs = multi_start(graph, config, seeds=(3, 4))
    assert all(run.best.log_overhead == pytest.approx(exact.best.log_overhead) for run in runs)
    best = runs[0].best
    plan = HybridPlan.create(graph, best.state, max_qubits=2, partition_mode="multiway")
    plan.save(tmp_path / "plan.json")
    restored = HybridPlan.load(tmp_path / "plan.json")
    compiled = compile_hybrid(graph, restored)
    assert compiled.run_exact().expectation == pytest.approx(
        graph.problem.exact_expectation(), abs=1e-12
    )
    assert tuple(c.num_qubits for c in compiled.experiment((0, 0)).circuits) == (2, 2, 2)
    sampled = compiled.run_sampled(trials=200, seed=11)
    assert sampled.physical_shots == 600
    assert abs(sampled.expectation - graph.problem.exact_expectation()) < 5 * sampled.standard_error
    assert refine_partition(graph, config, best.state).best.log_overhead == pytest.approx(
        best.log_overhead
    )


def test_multiway_label_space_equals_edge_space_optimum():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    graph = PartitionGraph(CircuitProblem.create(circuit, "ZZZ"))
    config = AnnealConfig(1, partition_mode="multiway", steps=200)
    exact = exact_search(graph, config)
    assert exact.complete
    # Exhaust every label assignment on this four-vertex graph, including four colors.
    feasible = []
    for labels in product(range(len(graph.nodes)), repeat=len(graph.nodes)):
        evaluation = graph.evaluate(
            graph.state_from_labels(labels), max_qubits=1, partition_mode="multiway"
        )
        if evaluation.feasible:
            feasible.append(evaluation.log_overhead)
    assert min(feasible) == pytest.approx(exact.best.log_overhead)
    assert anneal(graph, config, seed=8).best.log_overhead == pytest.approx(exact.best.log_overhead)
    assert graph.evaluate(exact.best.state, max_qubits=1, partition_mode="components") == exact.best


def test_multiway_hastings_kernel_normalization_and_detailed_balance():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    graph = PartitionGraph(CircuitProblem.create(circuit, "ZZZ"))
    config = AnnealConfig(2, partition_mode="multiway")
    moves = MultiwayLabelMoves(graph, allowed_options(graph, config), 3)
    palette = range(moves.palette_size)
    region = (0, 1, 2)

    def energy(labels):
        evaluation = graph.evaluate(
            graph.state_from_labels(labels), max_qubits=2, partition_mode="multiway"
        )
        return evaluation.log_overhead + 20 * evaluation.width_penalty

    for labels in product(palette, repeat=len(graph.nodes)):
        probabilities = []
        for a, b in combinations(palette, 2):
            forward = moves.pair_probability(labels, region, a, b)
            probabilities.append(forward)
            if not forward:
                continue
            candidate = moves.swap(labels, region, a, b)
            assert moves.swap(candidate, region, a, b) == labels
            reverse = moves.pair_probability(candidate, region, a, b)
            difference = energy(candidate) - energy(labels)
            p_xy = forward * min(1, exp(-difference) * reverse / forward)
            p_yx = reverse * min(1, exp(difference) * forward / reverse)
            assert exp(-energy(labels)) * p_xy == pytest.approx(
                exp(-energy(candidate)) * p_yx, rel=1e-10, abs=1e-30
            )
        assert sum(probabilities) == pytest.approx(1)
    rng = np.random.default_rng(12)
    labels = (0, 0, 1, 2)
    for _ in range(100):
        candidate, _, correction = moves.propose(labels, rng, 0)
        changed = [i for i in range(len(labels)) if labels[i] != candidate[i]]
        assert len(changed) == 1
        a, b = labels[changed[0]], candidate[changed[0]]
        region = (changed[0],)
        assert correction == pytest.approx(
            log(moves.pair_probability(candidate, region, a, b))
            - log(moves.pair_probability(labels, region, a, b))
        )
        labels = candidate


@pytest.mark.parametrize("flags", [{"allow_gate": False}, {"allow_wire": False}])
def test_multiway_proposals_preserve_disabled_actions(flags):
    graph = PartitionGraph(three_fragment_problem())
    config = AnnealConfig(2, partition_mode="multiway", steps=100, **flags)
    allowed = allowed_options(graph, config)
    moves = MultiwayLabelMoves(graph, allowed, 4)
    initial = seed_states(graph, config, count=3)
    assert initial
    labels = graph.labels_from_evaluation(
        graph.evaluate(initial[0], max_qubits=2, partition_mode="multiway")
    )
    rng = np.random.default_rng(44)
    for _ in range(100):
        labels, _, _ = moves.propose(labels, rng, 0.6)
        state = graph.state_from_labels(labels)
        assert all(value in choices for value, choices in zip(state, allowed))


def test_multiway_fragment_cap_and_reproducibility():
    graph = PartitionGraph(three_fragment_problem())
    config = AnnealConfig(
        2, partition_mode="multiway", allow_gate=False, max_components=2, steps=150
    )
    exact = exact_search(graph, config)
    assert exact.complete and exact.best is None
    assert anneal(graph, config, seed=9).best is None
    feasible = replace(config, max_components=3, objective="fragment_shots")
    first = anneal(graph, feasible, seed=8)
    repeated = anneal(graph, feasible, seed=8)
    assert first.best == repeated.best and first.trace == repeated.trace
    assert len(first.best.block_widths) == 3
    for kwargs in ({"max_components": 0}, {"split_probability": 0}, {"split_probability": 1}):
        with pytest.raises(ValueError):
            replace(config, **kwargs).validate()


@pytest.mark.parametrize("n,limit,wire_cuts", [(10, 7, 1), (10, 6, 2), (20, 6, 5)])
def test_wire_bridge_tree_capacity_dp(n, limit, wire_cuts):
    graph = PartitionGraph(chain_benchmark(n).problem)
    config = AnnealConfig(limit, partition_mode="multiway")
    state = bridge_seed_state(graph, allowed_options(graph, config), max_qubits=limit)
    evaluation = graph.evaluate(state, max_qubits=limit, partition_mode="multiway")
    assert evaluation.feasible
    assert evaluation.action_counts == {"wire": wire_cuts, "gate": 0}
    assert evaluation.sampling_overhead == pytest.approx(16**wire_cuts)
    assert sum(evaluation.block_widths) == n + wire_cuts
    # Enumerate every subset of the cuttable wire bridges, independently of DP.
    import networkx as nx

    bridge_edges = {frozenset(e) for e in nx.bridges(graph.networkx_graph())}
    variables = [
        i
        for i, v in enumerate(graph.variables)
        if v.kind == "wire" and frozenset(v.ports) in bridge_edges
    ]
    costs = []
    for choices in product((0, 1), repeat=len(variables)):
        trial = [0] * len(state)
        for index, value in zip(variables, choices):
            trial[index] = value
        candidate = graph.evaluate(tuple(trial), max_qubits=limit, partition_mode="multiway")
        if candidate.feasible:
            costs.append(candidate.log_overhead)
    assert evaluation.log_overhead == pytest.approx(min(costs))
    assert bridge_seed_state(graph, allowed_options(graph, config), max_qubits=3) is None


def test_typed_bridge_dp_finds_mixed_cut_better_than_wire_only():
    import networkx as nx

    graph = PartitionGraph(chain_benchmark(20).problem)
    config = AnnealConfig(10, partition_mode="multiway")
    allowed = allowed_options(graph, config)
    state = bridge_seed_state(graph, allowed, max_qubits=10, include_gates=True)
    evaluation = graph.evaluate(state, max_qubits=10, partition_mode="multiway")
    assert evaluation.feasible
    assert evaluation.sampling_overhead == pytest.approx(144)
    assert evaluation.action_counts == {"wire": 1, "gate": 1}
    assert sorted(evaluation.block_widths) == [1, 10, 10]
    bridge_edges = {frozenset(e) for e in nx.bridges(graph.networkx_graph())}
    variables = [i for i, v in enumerate(graph.variables) if frozenset(v.ports) in bridge_edges]
    costs = []
    for choices in product((0, 1), repeat=len(variables)):
        trial = [0] * len(state)
        for index, value in zip(variables, choices):
            trial[index] = value
        candidate = graph.evaluate(tuple(trial), max_qubits=10, partition_mode="multiway")
        if candidate.feasible:
            costs.append(candidate.log_overhead)
    assert evaluation.log_overhead == pytest.approx(min(costs))
