"""Sparse Pauli-transfer gate excision and physical measure/prepare experiments.

This implements the algebra of Chen et al. (QCE 2025), then explicitly
expands output Paulis into signed product eigenstate preparations. The
resulting norm is that of this construction, not a claim of optimality.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator, Pauli

from .circuits import CircuitProblem
from .execution import ExecutionResults, execute_exact, execute_shots


@dataclass(frozen=True)
class ExcisionTerm:
    input_pauli: str
    output_pauli: str
    eigenbits: tuple[int, ...]  # Local qubit order, least significant qubit first.
    coefficient: float


@dataclass(frozen=True)
class ExcisionDecomposition:
    pauli_labels: tuple[str, ...]
    transfer_matrix: np.ndarray
    terms: tuple[ExcisionTerm, ...]
    algebraic_terms: int
    discarded_ptm_norm: float

    @property
    def gamma(self) -> float:
        return sum(abs(term.coefficient) for term in self.terms)

    def apply(self, operator: np.ndarray) -> np.ndarray:
        """Apply the sparse algebraic map to any matrix, including nonphysical ones."""
        dim = 2 ** len(self.pauli_labels[0])
        if np.shape(operator) != (dim, dim):
            raise ValueError("Operator shape does not match excision width.")
        paulis = [Pauli(label).to_matrix() for label in self.pauli_labels]
        result = np.zeros((dim, dim), dtype=complex)
        for output_index, input_index in zip(*np.nonzero(self.transfer_matrix), strict=True):
            result += (
                self.transfer_matrix[output_index, input_index]
                / dim
                * np.trace(paulis[input_index] @ operator)
                * paulis[output_index]
            )
        return result


def decompose_excision(
    gate: Instruction | np.ndarray, *, max_qubits: int = 3, tolerance: float = 1e-12
) -> ExcisionDecomposition:
    """Build R[Q,P] = Tr(Q U P U†)/2**n, then physical terms R[Q,P]/2**n."""
    matrix = Operator(gate).data if isinstance(gate, Instruction) else np.asarray(gate, complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Excision requires a square unitary of at least one qubit.")
    dim = matrix.shape[0]
    n = dim.bit_length() - 1
    if 2**n != dim or n > max_qubits:
        raise ValueError("Excision width exceeds the configured small-block limit.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix.conj().T @ matrix, np.eye(dim), atol=1e-10, rtol=0
    ):
        raise ValueError("Excision requires a unitary matrix.")
    labels = tuple("".join(chars) for chars in product("IXYZ", repeat=n))
    paulis = np.array([Pauli(label).to_matrix() for label in labels])
    rotated = matrix @ paulis @ matrix.conj().T
    full = np.einsum("qij,pji->qp", paulis, rotated).real / dim
    transfer = full.copy()
    transfer[np.abs(transfer) <= tolerance] = 0
    discarded = float(np.linalg.norm(full - transfer))
    terms = []
    for output_index, input_index in zip(*np.nonzero(transfer), strict=True):
        output = labels[output_index]
        for bits in product((0, 1), repeat=n):
            sign = prod(
                -1 if bit and axis != "I" else 1
                for axis, bit in zip(reversed(output), bits, strict=True)
            )
            terms.append(
                ExcisionTerm(
                    labels[input_index],
                    output,
                    bits,
                    float(sign * transfer[output_index, input_index] / dim),
                )
            )
    return ExcisionDecomposition(
        labels, transfer, tuple(terms), int(np.count_nonzero(transfer)), discarded
    )


def _measurement_basis(circuit: QuantumCircuit, axis: str, qubit: int) -> None:
    if axis == "X":
        circuit.h(qubit)
    elif axis == "Y":
        circuit.sdg(qubit)
        circuit.h(qubit)


def _prepare(circuit: QuantumCircuit, axis: str, qubit: int, eigenbit: int) -> None:
    circuit.reset(qubit)
    if eigenbit:
        circuit.x(qubit)
    if axis == "X":
        circuit.h(qubit)
    elif axis == "Y":
        circuit.h(qubit)
        circuit.s(qubit)


@dataclass(frozen=True)
class ScoredExperiment:
    circuit: QuantumCircuit
    coefficient: float
    score_mask: int

    def mean(self, distribution: dict[int, float]) -> float:
        return sum(
            probability * (-1 if (outcome & self.score_mask).bit_count() % 2 else 1)
            for outcome, probability in distribution.items()
        )


@dataclass(frozen=True)
class ExcisionResult:
    expectation: float
    standard_error: float | None
    total_shots: int
    executed_circuits: int
    execution_mode: str
    gamma: float


@dataclass
class CompiledExcision:
    problem: CircuitProblem
    operation_ids: tuple[str, ...]
    decompositions: tuple[ExcisionDecomposition, ...]
    experiments: list[ScoredExperiment]

    @property
    def gamma(self) -> float:
        """Physical decomposition norm excluding the observable coefficients."""
        return prod(decomposition.gamma for decomposition in self.decompositions)

    @property
    def estimator_gamma(self) -> float:
        return sum(abs(experiment.coefficient) for experiment in self.experiments)

    def _reconstruct(self, execution: ExecutionResults) -> ExcisionResult:
        means = [
            experiment.mean(distribution)
            for experiment, distribution in zip(
                self.experiments, execution.distributions, strict=True
            )
        ]
        coefficients = np.array([experiment.coefficient for experiment in self.experiments])
        expectation = float(coefficients @ means)
        se = None
        if execution.shots_per_circuit is not None and execution.shots_per_circuit > 1:
            # Every score is +/-1. Circuits are executed independently without data reuse.
            variance = float(
                coefficients**2 @ (1 - np.square(means)) / (execution.shots_per_circuit - 1)
            )
            se = float(np.sqrt(max(variance, 0)))
        return ExcisionResult(
            expectation, se, execution.total_shots, len(means), "enumerated", self.gamma
        )

    def run_exact(self, *, max_qubits: int = 12) -> ExcisionResult:
        self.problem.validate_snapshot()
        return self._reconstruct(
            execute_exact(
                [experiment.circuit for experiment in self.experiments], max_qubits=max_qubits
            )
        )

    def run_shots(self, *, shots: int, seed: int = 0, noise_model=None) -> ExcisionResult:
        self.problem.validate_snapshot()
        return self._reconstruct(
            execute_shots(
                [experiment.circuit for experiment in self.experiments],
                shots=shots,
                seed=seed,
                noise_model=noise_model,
            )
        )

    def run_sampled_terms(self, *, samples: int, seed: int = 0) -> ExcisionResult:
        """Importance-sample physical terms and take one score shot per draw.

        Repeated draws of one term are batched into a single circuit execution.
        The returned standard error includes both term and measurement randomness.
        """
        self.problem.validate_snapshot()
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError("samples must be an integer of at least 2.")
        coefficients = np.array([experiment.coefficient for experiment in self.experiments])
        gamma = float(np.abs(coefficients).sum())
        if gamma == 0:
            return ExcisionResult(0, 0, 0, 0, "sampled-terms", self.gamma)
        rng = np.random.default_rng(seed)
        allocation = rng.multinomial(samples, np.abs(coefficients) / gamma)
        total = 0.0
        executed = 0
        for index in np.flatnonzero(allocation):
            experiment = self.experiments[index]
            count = int(allocation[index])
            execution = execute_shots([experiment.circuit], shots=count, seed=seed + int(index) + 1)
            total += (
                count
                * gamma
                * np.sign(coefficients[index])
                * experiment.mean(execution.distributions[0])
            )
            executed += 1
        expectation = total / samples
        se = float(np.sqrt(max(gamma**2 - expectation**2, 0) / (samples - 1)))
        return ExcisionResult(expectation, se, samples, executed, "sampled-terms", self.gamma)


def compile_excision(
    problem: CircuitProblem,
    operation_ids: tuple[str, ...],
    *,
    max_experiments: int = 10000,
) -> CompiledExcision:
    """Excise small gates in situ; all untouched quantum correlations are retained."""
    problem.validate_snapshot()
    operation_ids = tuple(operation_ids)
    if not operation_ids or len(set(operation_ids)) != len(operation_ids):
        raise ValueError("Select at least one gate, with no duplicate excisions.")
    decompositions = []
    for op_id in operation_ids:
        item = problem.circuit.data[problem.index(op_id)]
        if item.operation.name == "barrier":
            raise ValueError("Cannot excise a barrier.")
        decompositions.append(decompose_excision(item.operation))
    count = prod(len(decomposition.terms) for decomposition in decompositions) * len(
        problem.observable
    )
    if count > max_experiments:
        raise ValueError(
            f"Excision needs {count} experiments; increase max_experiments explicitly."
        )
    experiments = []
    for selected in product(*(decomposition.terms for decomposition in decompositions)):
        replacement = dict(zip(operation_ids, selected, strict=True))
        input_bits = sum(sum(axis != "I" for axis in term.input_pauli) for term in selected)
        for observable, weight in problem.observable.to_list():
            if weight == 0:
                continue
            circuit = problem.circuit.copy_empty_like()
            circuit.add_register(ClassicalRegister(input_bits + circuit.num_qubits, "readout"))
            coefficient = float(weight.real * prod(term.coefficient for term in selected))
            next_bit = 0
            mask = 0
            for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
                if op_id not in replacement:
                    circuit.append(item.operation, item.qubits)
                    continue
                term = replacement[op_id]
                qubits = [problem.circuit.find_bit(q).index for q in item.qubits]
                for local, qubit in enumerate(qubits):
                    input_axis = term.input_pauli[-local - 1]
                    if input_axis != "I":
                        _measurement_basis(circuit, input_axis, qubit)
                        circuit.measure(qubit, next_bit)
                        mask |= 1 << next_bit
                        next_bit += 1
                    _prepare(circuit, term.output_pauli[-local - 1], qubit, term.eigenbits[local])
            for qubit, axis in enumerate(reversed(observable)):
                _measurement_basis(circuit, axis, qubit)
                circuit.measure(qubit, input_bits + qubit)
                if axis != "I":
                    mask |= 1 << (input_bits + qubit)
            experiments.append(ScoredExperiment(circuit, coefficient, mask))
    return CompiledExcision(problem, operation_ids, tuple(decompositions), experiments)
