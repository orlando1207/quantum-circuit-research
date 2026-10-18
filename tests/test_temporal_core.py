from dataclasses import replace
from itertools import product
from math import pi, sin

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, CXGate, RZZGate, SwapGate
from qiskit.quantum_info import SparsePauliOp

from qpartition import (
    AnnealConfig,
    CircuitProblem,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    exact_search,
    migrate_legacy_plan,
    refine_partition,
)
from qpartition.annealing import seed_states
from qpartition.benchmarks import bridge_benchmark
from qpartition.legacy_graph import LegacyPartitionGraph
from qpartition.templates import TemplateRegistry


@pytest.mark.parametrize("angle", [0, 0.003, pi / 4, pi / 2, pi, -0.72, 2 * pi])
def test_physical_gate_costs_against_analytic_norms(angle):
    registry = TemplateRegistry()
    for gate, gamma in (
        (CPhaseGate(angle), 1 + 2 * abs(sin(angle / 2))),
        (RZZGate(angle), 1 + 2 * abs(sin(angle))),
    ):
        template, excision = registry.for_gate(gate)
        assert excision is None
        assert template.gamma == pytest.approx(gamma, abs=1e-12)
        assert template.gamma**2 == pytest.approx(sum(abs(c) for c in template.coefficients) ** 2)
        assert template.reconstruction_residual < 1e-10
    assert registry.wire.gamma**2 == pytest.approx(16)
    assert registry.for_gate(CXGate())[0].gamma ** 2 == pytest.approx(9)
    assert registry.for_gate(SwapGate())[0].gamma ** 2 == pytest.approx(49)
    assert all(t.kind != "excision" for t in registry.entries.values())


def problem_with_local_chains():
    c = QuantumCircuit(4)
    c.h(0)
    c.cx(0, 1)
    c.ry(0.42, 1)
    c.t(1)
    c.cp(0.63, 1, 2)
    c.rx(0.23, 2)
    c.cx(2, 0)
    c.h(3)  # An idle interaction wire still occupies a register.
    return CircuitProblem.create(c, SparsePauliOp.from_list([("XXIX", 0.4), ("IZZI", -0.2)]))


def test_contraction_preserves_costs_and_static_widths_for_random_cut_sets():
    problem = problem_with_local_chains()
    core, legacy = PartitionGraph(problem), LegacyPartitionGraph(problem)
    assert len(core.nodes) == 7  # 2*3 incidences + one local-only wire.
    assert len(core.gate_variables) == 3
    assert len(legacy.nodes) > len(core.nodes)
    rng = np.random.default_rng(5)
    for _ in range(50):
        state = tuple(int(rng.integers(len(v.options))) for v in core.variables)
        old_state = [0] * len(legacy.variables)
        for variable, value in zip(core.variables, state):
            if variable.kind == "gate":
                target = legacy.gate_variables[variable.operation_id]
            else:
                target = next(
                    i
                    for i, v in enumerate(legacy.variables)
                    if v.kind == "wire"
                    and v.operation_id == variable.operation_id
                    and v.qubit == variable.qubit
                )
            old_state[target] = value
        new = core.evaluate(state, max_qubits=3)
        old = legacy.evaluate(tuple(old_state), max_qubits=3)
        assert new.log_overhead == pytest.approx(old.log_overhead, abs=1e-12)
        assert sorted(new.component_widths) == sorted(old.component_widths)
        assert new.block_widths == old.block_widths
        assert new.width_penalty == old.width_penalty
        assert new.segment_count == 4 + new.action_counts["wire"]


def test_core_mixed_channel_with_contracted_local_chains(tmp_path):
    problem = problem_with_local_chains()
    graph = PartitionGraph(problem)
    state = [0] * len(graph.variables)
    state[graph.wire_variables["op000004", 1]] = 1
    state[graph.gate_variables["op000004"]] = 1
    plan = HybridPlan.create(graph, tuple(state), max_qubits=5)
    plan.save(tmp_path / "plan.json")
    restored = HybridPlan.load(tmp_path / "plan.json")
    assert restored.graph_schema == "temporal-incidence-v2"
    result = compile_hybrid(graph, restored).run_exact()
    assert result.expectation == pytest.approx(problem.exact_expectation(), abs=1e-12)
    assert graph.evaluate(tuple(state), max_qubits=5).sampling_overhead == pytest.approx(
        16 * (1 + 2 * abs(sin(0.63 / 2))) ** 2
    )


def test_no_excision_in_core_and_historical_schema_requires_migration():
    problem = problem_with_local_chains()
    core, old = PartitionGraph(problem), LegacyPartitionGraph(problem)
    assert all(o is None or o.kind in ("gate", "wire") for v in core.variables for o in v.options)
    with pytest.raises(TypeError):
        AnnealConfig(3, allow_excision=True)
    with pytest.raises(ValueError, match="gate/wire"):
        anneal(old, AnnealConfig(3, steps=5))
    state = [0] * len(old.variables)
    boundary = next(
        i
        for i, v in enumerate(old.variables)
        if v.kind == "wire" and v.operation_id == "op000002" and v.qubit == 1
    )
    state[boundary] = 1
    old_plan = HybridPlan.create(old, tuple(state), max_qubits=5)
    with pytest.raises(ValueError, match="migrate"):
        old_plan.validate(core)
    new_plan = migrate_legacy_plan(core, old_plan)
    assert new_plan.state[core.wire_variables["op000004", 1]] == 1
    before = compile_hybrid(old, old_plan).run_exact()
    after = compile_hybrid(core, new_plan).run_exact()
    assert after.expectation == pytest.approx(before.expectation, abs=1e-12)


def test_label_and_action_search_spaces_have_same_small_instance_optimum():
    c = QuantumCircuit(3)
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(2, 0)
    graph = PartitionGraph(CircuitProblem.create(c, "ZZZ"))
    cost = []
    for encoded in (
        product((0, 1), repeat=len(graph.variables)),
        (graph.state_from_labels(x) for x in product((0, 1), repeat=len(graph.nodes))),
    ):
        feasible = [graph.evaluate(tuple(s), max_qubits=2) for s in encoded]
        cost.append(min(e.log_overhead for e in feasible if e.feasible))
    assert cost[0] == pytest.approx(cost[1], abs=1e-12)
    for space in ("actions", "labels"):
        cfg = AnnealConfig(2, steps=500, search_space=space)
        exact = exact_search(graph, cfg)
        assert exact.complete
        result = anneal(graph, cfg, seed=8)
        assert result.best.log_overhead == pytest.approx(exact.best.log_overhead)
        assert set(result.accepted_action_changes) == {
            "gate_added",
            "gate_removed",
            "wire_added",
            "wire_removed",
        }


def test_interval_refinement_and_disabled_actions():
    c = QuantumCircuit(4)
    for a, b in ((0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (0, 1), (2, 3), (1, 2), (1, 2)):
        c.cz(a, b)
    graph = PartitionGraph(CircuitProblem.create(c, "XXXX"))
    # Use a direct spatial coloring to avoid relying on action-index conventions.
    labels = tuple(int(node["qubit"] >= 2) for node in graph.nodes)
    state = graph.state_from_labels(labels)
    cfg = AnnealConfig(3, steps=300)
    refinement = refine_partition(graph, cfg, state)
    assert graph.evaluate(state, max_qubits=3).sampling_overhead == pytest.approx(729)
    assert refinement.best.sampling_overhead == pytest.approx(144)
    assert refinement.best.action_counts == {"gate": 1, "wire": 1}
    gate_only = anneal(graph, replace(cfg, allow_wire=False), initial_state=state, seed=9)
    assert gate_only.best.action_counts["wire"] == 0


def test_noncommuting_strong_bridge_physical_wire_reconstruction():
    benchmark = bridge_benchmark(pi, "bridge_strong_test")
    graph = PartitionGraph(benchmark.problem)
    state = seed_states(graph, AnnealConfig(6))[0]
    evaluation = graph.evaluate(state, max_qubits=6)
    assert evaluation.action_counts == {"wire": 1, "gate": 0}
    assert evaluation.sampling_overhead == pytest.approx(16)
    expected = benchmark.problem.exact_expectation()
    assert abs(expected - 1) > 0.01  # The entangling layers no longer cancel to all H.
    compiled = compile_hybrid(graph, HybridPlan.create(graph, state, max_qubits=6))
    assert compiled.run_exact().expectation == pytest.approx(expected, abs=1e-12)
