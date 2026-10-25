from dataclasses import asdict
from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    GroupingConfig,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    exact_search,
    refine_partition,
)
from qpartition.annealing import allowed_options, seed_states
from qpartition.label_moves import MultiwayLabelMoves
from qpartition.large_benchmarks import large_benchmark


def middle_problem():
    c = QuantumCircuit(6)
    for q in range(6):
        c.ry(0.31 + 0.13 * q, q)
        c.rz(0.17 + 0.07 * q, q)
    c.cx(0, 2)
    c.cx(1, 3)
    c.rx(0.43, 0)
    c.ry(-0.37, 1)
    middle = f"op{len(c.data):06d}"
    c.cx(0, 1)
    c.rz(0.21, 0)
    c.rx(-0.57, 1)
    c.h(4)
    c.cx(0, 4)
    c.cx(1, 5)
    c.ry(0.22, 0)
    query = SparsePauliOp.from_list([("XIZIYX", 0.4), ("ZIIZII", 0.3)])
    return CircuitProblem.create(c, query), middle


def bundle_state(graph, op, directions):
    state = [0] * len(graph.variables)
    state[graph.gate_variables[op]] = 1
    pins = graph.variables[graph.gate_variables[op]].ports
    outgoing = {v.ports[0]: i for i, v in enumerate(graph.variables) if v.kind == "wire"}
    for i, direction in enumerate(directions):
        wire = graph.incoming_wires[op][i] if direction == "in" else outgoing[pins[i]]
        state[wire] = 1
    return tuple(state)


def test_graph_labels_variables_and_original_move_pools_are_identical():
    problem, _ = middle_problem()
    a, b = PartitionGraph(problem), PartitionGraph(problem, excision=True)
    assert a.nodes == b.nodes
    assert a.edges == b.edges
    assert a.variables == b.variables
    assert a.wire_nodes == b.wire_nodes
    assert a.gate_ports == b.gate_ports
    assert a.macro_neighborhoods == b.macro_neighborhoods
    cfg = AnnealConfig(3, partition_mode="multiway")
    ma = MultiwayLabelMoves(a, allowed_options(a, cfg), cfg.block_size)
    mb = MultiwayLabelMoves(b, allowed_options(b, cfg), cfg.block_size)
    for name in (
        "units",
        "regions",
        "polish_regions",
        "group_orbits",
        "spatial_orbits",
        "palette_size",
    ):
        assert getattr(ma, name) == getattr(mb, name)
    rng = np.random.default_rng(19)
    for _ in range(35):
        labels = tuple(int(v) for v in rng.integers(0, 6, len(a.nodes)))
        assert a.state_from_labels(labels) == b.state_from_labels(labels)
        state = a.state_from_labels(labels)
        ea = a.evaluate(state, max_qubits=3, partition_mode="multiway")
        eb = b.evaluate(state, max_qubits=3, partition_mode="multiway")
        for field in (
            "node_segments",
            "segment_components",
            "component_widths",
            "block_widths",
            "width_penalty",
        ):
            assert getattr(ea, field) == getattr(eb, field)
        assert eb.log_overhead <= ea.log_overhead + 1e-10


@pytest.mark.parametrize("directions", list(product(("in", "out"), repeat=2)))
def test_all_boundary_orientations_reconstruct_entangled_source_with_local_chains(
    tmp_path, directions
):
    problem, op = middle_problem()
    graph = PartitionGraph(problem, excision=True)
    state = bundle_state(graph, op, directions)
    before = PartitionGraph(problem).evaluate(state, max_qubits=2, partition_mode="multiway")
    ev = graph.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert before.sampling_overhead == pytest.approx(2304)
    assert ev.sampling_overhead == pytest.approx(256)
    assert ev.action_counts == {"gate": 0, "wire": 0, "excision": 1}
    assert ev.excision_wire_cuts == 2
    assert ev.segment_count == 8
    assert ev.block_widths == before.block_widths
    plan = HybridPlan.create(graph, state, max_qubits=2, partition_mode="multiway")
    plan.save(tmp_path / "plan.json")
    restored = HybridPlan.load(tmp_path / "plan.json")
    assert restored.graph_schema == "temporal-incidence-grouped-v3"
    compiled = compile_hybrid(graph, restored)
    result = compiled.run_exact()
    expected = problem.exact_expectation()
    assert abs(expected) > 0.01
    assert result.expectation == pytest.approx(expected, abs=1e-12)
    assert len(compiled.templates) == 1  # Covered wire QPDs are not executed again.
    assert compiled.gamma == pytest.approx(16)


@pytest.mark.parametrize("terminal", [False, True])
def test_first_and_last_interactions_use_existing_boundaries(terminal):
    c = QuantumCircuit(4)
    for q in range(4):
        c.ry(0.27 + 0.13 * q, q)
    if terminal:
        c.cx(0, 2)
        c.cx(1, 3)
    op = f"op{len(c.data):06d}"
    c.cx(0, 1)
    c.rx(0.47, 0)
    c.ry(-0.39, 1)
    if not terminal:
        c.cx(0, 2)
        c.cx(1, 3)
    p = CircuitProblem.create(c, "ZZZZ")
    g = PartitionGraph(p, excision=True)
    state = bundle_state(g, op, ("in", "in") if terminal else ("out", "out"))
    ev = g.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert ev.action_counts["excision"] == 1
    compiled = compile_hybrid(
        g, HybridPlan.create(g, state, max_qubits=2, partition_mode="multiway")
    )
    assert compiled.run_exact().expectation == pytest.approx(p.exact_expectation(), abs=1e-12)


def test_one_separated_operand_does_not_allocate_an_extra_local_register():
    c = QuantumCircuit(3)
    c.ry(0.31, 0)
    c.ry(0.83, 1)
    c.cx(0, 2)
    op = f"op{len(c.data):06d}"
    c.swap(0, 1)
    c.rx(0.23, 1)
    p = CircuitProblem.create(c, "ZZZ")
    g = PartitionGraph(p, excision=True)
    state = [0] * len(g.variables)
    state[g.gate_variables[op]] = 1
    state[g.incoming_wires[op][0]] = 1
    ev = g.evaluate(tuple(state), max_qubits=3, partition_mode="multiway")
    assert ev.action_counts == {"gate": 0, "wire": 0, "excision": 1}
    assert ev.excision_wire_cuts == 1
    assert ev.segment_count == 4  # Local measure/reset reuses its existing segment.
    compiled = compile_hybrid(
        g, HybridPlan.create(g, tuple(state), max_qubits=3, partition_mode="multiway")
    )
    assert compiled.run_exact().expectation == pytest.approx(p.exact_expectation(), abs=1e-12)


def test_neighboring_excision_factors_cannot_claim_a_wire_twice():
    c = QuantumCircuit(2)
    for _ in range(3):
        c.cx(0, 1)
    g = PartitionGraph(
        CircuitProblem.create(c, "ZZ"),
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    state = (1,) * len(g.variables)
    ev = g.evaluate(state, max_qubits=1, partition_mode="multiway")
    assert ev.grouping_selection_complete
    assert ev.action_counts == {"gate": 1, "wire": 0, "excision": 2}
    assert ev.sampling_overhead == pytest.approx(9 * 256**2)
    used = set()
    for selected in ev.groups:
        c = g.grouping.by_id[selected.group_id]
        assert not used.intersection(c.variables)
        used.update(c.variables)


def test_grouping_and_excision_share_selector_and_keep_group_move_pools():
    base, op = middle_problem()
    c = QuantumCircuit(10)
    c.compose(base.circuit, range(6), inplace=True)
    x = f"op{len(c.data):06d}"
    c.cx(6, 7)
    y = f"op{len(c.data):06d}"
    c.cx(8, 9)
    p = CircuitProblem.create(c, "Z" * 10)
    cfg = GroupingConfig(families=("parallel",))
    old = PartitionGraph(p, grouping=cfg)
    graph = PartitionGraph(p, grouping=cfg, excision=True)
    assert graph.group_regions == old.group_regions
    assert graph.grouping.orbits == old.grouping.orbits
    state = list(bundle_state(graph, op, ("in", "in")))
    state[graph.gate_variables[x]] = state[graph.gate_variables[y]] = 1
    ev = graph.evaluate(tuple(state), max_qubits=2, partition_mode="multiway")
    assert {graph.grouping.by_id[s.group_id].family for s in ev.groups} == {"excision", "parallel"}
    assert ev.sampling_overhead == pytest.approx(256 * 49)
    assert ev.segment_count == 12
    run = anneal(
        graph,
        AnnealConfig(
            3,
            partition_mode="multiway",
            steps=40,
            max_evaluations=100,
            group_heatbath_probability=0.5,
        ),
        seed=4,
        initial_state=tuple(state),
    )
    assert run.best is not None
    assert run.heatbath_proposals > 0


def test_assumed_costs_are_explicit_and_cannot_be_physically_compiled():
    p, op = middle_problem()
    g = PartitionGraph(p, excision=True, excision_gamma=4)
    state = bundle_state(g, op, ("in", "in"))
    ev = g.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert ev.sampling_overhead == pytest.approx(16)
    plan = HybridPlan.create(g, state, max_qubits=2, partition_mode="multiway")
    with pytest.raises(ValueError, match="cost study only"):
        compile_hybrid(g, plan)
    with pytest.raises(ValueError):
        PartitionGraph(p, excision_gamma=4)
    assert all(v.options[1].kind in {"gate", "wire"} for v in g.variables)


def test_excision_can_reprice_a_retained_gate_between_cut_input_wires():
    p, op = middle_problem()
    graph = PartitionGraph(p, excision=True, excision_gamma=4)
    state = list(bundle_state(graph, op, ("in", "in")))
    state[graph.gate_variables[op]] = 0
    ev = graph.evaluate(tuple(state), max_qubits=4, partition_mode="multiway")
    assert ev.action_counts == {"wire": 0, "gate": 0, "excision": 1}
    assert ev.sampling_overhead == pytest.approx(16)
    assert (
        ev.component_widths
        == PartitionGraph(p)
        .evaluate(tuple(state), max_qubits=4, partition_mode="multiway")
        .component_widths
    )


def test_exact_search_bound_does_not_charge_discounted_wires_as_unavoidable():
    b = large_benchmark("excision_butterfly", 6, layers=1)
    g = PartitionGraph(
        b.problem,
        excision=True,
        excision_gamma=4,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    cfg = AnnealConfig(b.max_qubits, partition_mode="multiway", steps=40, max_evaluations=100)
    values = []
    for state in product((0, 1), repeat=len(g.variables)):
        e = g.evaluate(state, max_qubits=b.max_qubits, partition_mode="multiway", validate=False)
        if e.feasible:
            values.append(e.log_overhead)
    exact = exact_search(g, cfg, max_nodes=100000)
    assert exact.complete
    assert exact.best.log_overhead == pytest.approx(min(values))


def test_existing_seeds_sa_and_polishing_choose_the_same_excision_realization():
    b = large_benchmark("excision_butterfly", 6)
    g = PartitionGraph(b.problem, excision=True, excision_gamma=16)
    cfg = AnnealConfig(b.max_qubits, partition_mode="multiway", steps=150, max_evaluations=150)
    initial = seed_states(g, cfg, count=3)[0]
    run = anneal(g, cfg, initial_state=initial, seed=2)
    assert run.config.search_space == "labels"
    assert run.best.sampling_overhead == pytest.approx(256)
    refined = refine_partition(g, cfg, run.best.state)
    assert refined.best.sampling_overhead == pytest.approx(256)
    assert run.evaluations <= 150
    assert "allow_excision" not in asdict(cfg)


def test_two_neighboring_excisions_reconstruct_without_sharing_cut_variables():
    c = QuantumCircuit(6)
    for q in range(6):
        c.ry(0.21 + 0.11 * q, q)
    c.cx(0, 2)
    c.cx(1, 3)
    first = f"op{len(c.data):06d}"
    c.cx(0, 1)
    c.rx(0.27, 0)
    second = f"op{len(c.data):06d}"
    c.cx(0, 1)
    c.ry(-0.39, 1)
    c.cx(0, 4)
    c.cx(1, 5)
    p = CircuitProblem.create(c, "ZZZZZZ")
    graph = PartitionGraph(p, excision=True)
    left = bundle_state(graph, first, ("in", "in"))
    right = bundle_state(graph, second, ("out", "out"))
    state = tuple(max(a, b) for a, b in zip(left, right))
    ev = graph.evaluate(state, max_qubits=2, partition_mode="multiway")
    assert ev.action_counts == {"gate": 0, "wire": 0, "excision": 2}
    assert ev.excision_wire_cuts == 4
    compiled = compile_hybrid(
        graph, HybridPlan.create(graph, state, max_qubits=2, partition_mode="multiway")
    )
    result = compiled.run_exact(max_experiments=30000)
    assert result.expectation == pytest.approx(p.exact_expectation(), abs=3e-12)


@pytest.mark.parametrize(
    "anchor,expected", [("first", {"parallel", "excision"}), ("last", {"excision"})]
)
def test_group_packets_cannot_move_shared_operations_across_a_selected_excision(anchor, expected):
    c = QuantumCircuit(7)
    for q in range(7):
        c.ry(0.31 + 0.07 * q, q)
    first = f"op{len(c.data):06d}"
    c.cz(0, 2)
    excised = f"op{len(c.data):06d}"
    c.cz(0, 1)
    last = f"op{len(c.data):06d}"
    c.cz(3, 4)
    c.cx(0, 5)
    c.cx(1, 6)
    graph = PartitionGraph(
        CircuitProblem.create(c, "Z" * 7),
        excision=True,
        grouping=GroupingConfig(families=("parallel",), anchor_strategy=anchor),
    )
    state = list(bundle_state(graph, excised, ("out", "out")))
    state[graph.gate_variables[first]] = state[graph.gate_variables[last]] = 1
    ev = graph.evaluate(tuple(state), max_qubits=2, partition_mode="multiway")
    families = {graph.grouping.by_id[g.group_id].family for g in ev.groups}
    assert families == expected
    assert ev.sampling_overhead == pytest.approx((49 if anchor == "first" else 81) * 256)
    if anchor == "last":
        assert graph.grouping.excision_temporal_conflicts
    compiled = compile_hybrid(
        graph, HybridPlan.create(graph, tuple(state), max_qubits=2, partition_mode="multiway")
    )
    assert 2 * np.log(compiled.gamma) == pytest.approx(ev.log_overhead)
