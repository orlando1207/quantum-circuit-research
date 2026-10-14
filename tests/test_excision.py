import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CCXGate, CXGate, IGate, TGate
from qiskit.quantum_info import Operator, random_unitary

from qpartition.circuits import CircuitProblem
from qpartition.excision import compile_excision, decompose_excision


@pytest.mark.parametrize(
    "gate,nonzero", [(IGate(), 4), (TGate(), 6), (CXGate(), 16), (CCXGate(), 232)]
)
def test_paper_sparse_counts_and_arbitrary_operator_identity(gate, nonzero):
    decomposition = decompose_excision(gate)
    assert decomposition.algebraic_terms == nonzero
    assert len(decomposition.terms) == nonzero * 2**gate.num_qubits
    rng = np.random.default_rng(52)
    dim = 2**gate.num_qubits
    arbitrary = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    matrix = Operator(gate).data
    assert decomposition.apply(arbitrary) == pytest.approx(matrix @ arbitrary @ matrix.conj().T)
    if isinstance(gate, IGate):
        assert decomposition.gamma == pytest.approx(4)
    if isinstance(gate, TGate):
        assert decomposition.gamma == pytest.approx(2 + 2 * np.sqrt(2))


@pytest.mark.parametrize("pauli", ["XX", "XY", "YX", "YY", "ZZ", "IZ"])
def test_t_excision_preserves_correlations_with_untouched_qubit(pauli):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(0)
    circuit.ry(0.31, 1)
    problem = CircuitProblem.create(circuit, pauli)
    compiled = compile_excision(problem, (problem.operation_ids[2],))
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)


def test_generic_two_qubit_excision_with_entangled_environment():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.ry(0.2, 1)
    circuit.unitary(random_unitary(4, seed=43), [1, 0])
    circuit.cx(1, 2)
    problem = CircuitProblem.create(circuit, "YXZ")
    compiled = compile_excision(problem, (problem.operation_ids[3],))
    assert compiled.run_exact().expectation == pytest.approx(problem.exact_expectation(), abs=1e-10)


def test_multiple_excisions_and_physical_sampling():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.tdg(0)
    problem = CircuitProblem.create(circuit, "X")
    compiled = compile_excision(problem, (problem.operation_ids[1], problem.operation_ids[2]))
    assert compiled.run_exact().expectation == pytest.approx(1, abs=1e-10)
    single = compile_excision(problem, (problem.operation_ids[1],))
    result = single.run_shots(shots=1500, seed=12)
    assert result.total_shots == 1500 * len(single.experiments)
    assert abs(result.expectation - 1) < 6 * result.standard_error
    sampled = single.run_sampled_terms(samples=12000, seed=31)
    assert sampled.total_shots == 12000
    assert abs(sampled.expectation - 1) < 6 * sampled.standard_error


def test_unitarity_width_and_expansion_guards():
    with pytest.raises(ValueError, match="unitary"):
        decompose_excision(np.array([[1, 1], [0, 1]]))
    with pytest.raises(ValueError, match="small-block"):
        decompose_excision(np.eye(16))
    circuit = QuantumCircuit(1)
    circuit.t(0)
    problem = CircuitProblem.create(circuit, "Z")
    with pytest.raises(ValueError, match="experiments"):
        compile_excision(problem, (problem.operation_ids[0],), max_experiments=1)


def test_importance_sampling_variance_calibration():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    problem = CircuitProblem.create(circuit, "X")
    compiled = compile_excision(problem, (problem.operation_ids[1],))
    truth = problem.exact_expectation()
    repetitions, samples = 24, 800
    values = np.array(
        [
            compiled.run_sampled_terms(samples=samples, seed=1000 + 97 * i).expectation
            for i in range(repetitions)
        ]
    )
    predicted_variance = (compiled.estimator_gamma**2 - truth**2) / samples
    # A calibration check of the full term-draw + physical measurement workflow.
    assert 0.3 < np.var(values, ddof=1) / predicted_variance < 2.5
    assert abs(np.mean(values) - truth) < 4 * np.sqrt(predicted_variance / repetitions)


def test_zero_observable_needs_no_excision_shots():
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit(1)
    circuit.t(0)
    problem = CircuitProblem.create(circuit, SparsePauliOp("Z", coeffs=[0]))
    compiled = compile_excision(problem, (problem.operation_ids[0],))
    assert compiled.run_shots(shots=100).total_shots == 0
    assert compiled.run_exact().expectation == 0
