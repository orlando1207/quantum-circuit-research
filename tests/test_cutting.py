import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, random_unitary

from qpartition.circuits import CircuitProblem
from qpartition.cutting import compile_cut
from qpartition.plans import CutPlan, WireCut
from qpartition.statistics import bootstrap_uncertainty


@pytest.mark.parametrize("pauli", ["XX", "YY", "ZZ", "IZ", "ZI"])
def test_bell_gate_cut_reconstruction(pauli):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, pauli)
    compiled = compile_cut(
        problem,
        CutPlan.create(problem, gate_cuts=(problem.operation_ids[1],), max_fragment_qubits=1),
    )
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)
    assert compiled.cost.gamma == pytest.approx(3)
    assert compiled.cost.sampling_overhead == pytest.approx(9)
    assert list(compiled.cost.fragment_widths.values()) == [1, 1]
    graph = compiled.reconstruction_graph()
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


@pytest.mark.parametrize("theta", [0, 0.17, np.pi / 2])
def test_rotation_gate_actual_norm(theta):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.4, 1)
    circuit.rzz(theta, 0, 1)
    problem = CircuitProblem.create(circuit, "YX")
    compiled = compile_cut(problem, CutPlan.create(problem, gate_cuts=(problem.operation_ids[2],)))
    assert compiled.cost.gamma == pytest.approx(1 + 2 * abs(np.sin(theta)))
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)


def test_wire_cut_and_mixed_cut_on_ghz():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    query = SparsePauliOp.from_list([("XXX", 0.3), ("ZIZ", 0.2), ("YXY", -0.1)])
    problem = CircuitProblem.create(circuit, query)
    wire = WireCut(1, problem.operation_ids[2])
    for gate_cuts, gamma in [((), 4), ((problem.operation_ids[1],), 12)]:
        compiled = compile_cut(
            problem,
            CutPlan.create(problem, gate_cuts=gate_cuts, wire_cuts=(wire,), max_fragment_qubits=2),
        )
        assert compiled.cost.gamma == pytest.approx(gamma)
        assert compiled.run_exact().expectation == pytest.approx(
            problem.exact_expectation(), abs=1e-10
        )
    with pytest.raises(ValueError, match="fragment widths"):
        compile_cut(problem, CutPlan.create(problem, wire_cuts=(wire,), max_fragment_qubits=1))


def test_multiple_cuts_on_one_original_wire():
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.23, 1)
    circuit.cx(1, 2)
    circuit.rz(0.19, 1)
    circuit.cx(1, 3)
    problem = CircuitProblem.create(circuit, "XXXX")
    plan = CutPlan.create(
        problem,
        wire_cuts=(WireCut(1, problem.operation_ids[3]), WireCut(1, problem.operation_ids[5])),
        max_fragment_qubits=2,
    )
    compiled = compile_cut(problem, plan)
    assert compiled.cost.sampling_overhead == pytest.approx(256)
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)


@pytest.mark.parametrize("pauli", ["XI", "IZ", "YX", "ZZ"])
def test_idle_wires_and_empty_circuits(pauli):
    problem = CircuitProblem.create(QuantumCircuit(2), pauli)
    compiled = compile_cut(problem, CutPlan.create(problem))
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)
    assert compiled.cost.gamma == 1


def test_generic_two_qubit_unitary():
    circuit = QuantumCircuit(2)
    circuit.ry(0.24, 0)
    circuit.unitary(random_unitary(4, seed=31), [0, 1])
    problem = CircuitProblem.create(circuit, "XY")
    compiled = compile_cut(problem, CutPlan.create(problem, gate_cuts=(problem.operation_ids[1],)))
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)


def test_finite_shots_bootstrap_and_accounting():
    circuit = QuantumCircuit(2)
    circuit.ry(0.73, 0)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, SparsePauliOp.from_list([("XX", 0.7), ("YY", -0.2)]))
    compiled = compile_cut(problem, CutPlan.create(problem, gate_cuts=(problem.operation_ids[1],)))
    result = compiled.run_shots(shots=2000, seed=18)
    uncertainty = bootstrap_uncertainty(compiled, result, replicates=100, seed=4)
    assert result.total_shots == 2000 * compiled.cost.unique_experiments
    assert uncertainty.standard_error > 0
    assert abs(result.expectation - problem.exact_expectation()) < 6 * uncertainty.standard_error
    assert compiled.cost.unique_experiments <= compiled.cost.generated_experiments
    assert compiled.run_shots(shots=2000, seed=18).expectation == result.expectation
    other = CircuitProblem.create(circuit, SparsePauliOp.from_list([("XX", 0.2), ("YY", -0.7)]))
    other_compiled = compile_cut(other, CutPlan.create(other, gate_cuts=(other.operation_ids[1],)))
    assert other_compiled.cost == compiled.cost
    with pytest.raises(ValueError, match="provenance"):
        bootstrap_uncertainty(other_compiled, result, replicates=2)


def test_expansion_guard_and_duplicate_cut_validation():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "ZZ")
    op_id = problem.operation_ids[0]
    with pytest.raises(ValueError, match="Duplicate gate"):
        CutPlan.create(problem, gate_cuts=(op_id, op_id))
    with pytest.raises(ValueError, match="Enumeration upper bound"):
        compile_cut(problem, CutPlan.create(problem, gate_cuts=(op_id,)), max_experiments=1)


def test_cut_whose_ports_remain_in_one_fragment():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    problem = CircuitProblem.create(circuit, "XZ")
    plan = CutPlan.create(problem, gate_cuts=(problem.operation_ids[1],))
    compiled = compile_cut(problem, plan)
    assert list(compiled.cost.fragment_widths.values()) == [2]
    graph = compiled.reconstruction_graph()
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 2
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)
    with pytest.raises(ValueError, match="fragment widths"):
        compile_cut(
            problem,
            CutPlan.create(problem, gate_cuts=plan.gate_cuts, max_fragment_qubits=1),
        )
