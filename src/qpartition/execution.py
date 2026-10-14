"""Local execution with joint classical outcomes, including mid-circuit instruments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


@dataclass(frozen=True)
class ExecutionResults:
    distributions: list[dict[int, float]]
    shots_per_circuit: int | None
    elapsed_seconds: float
    backend: str
    seed: int | None = None

    @property
    def total_shots(self) -> int:
        return len(self.distributions) * (self.shots_per_circuit or 0)


def exact_distribution(
    circuit: QuantumCircuit, *, max_qubits: int = 12, max_branches: int = 65536
) -> dict[int, float]:
    """Enumerate unnormalized statevector branches for measurements and resets.

    This evaluates the joint classical register distribution, not merely final
    state probabilities. Reset outcomes are traced out, not postselected.
    """
    if circuit.num_qubits > max_qubits:
        raise ValueError("Exact execution exceeds the configured qubit limit.")
    if circuit.parameters:
        raise ValueError("Execution requires bound parameters.")
    initial = np.zeros(2**circuit.num_qubits, dtype=complex)
    initial[0] = 1
    branches = [(0, initial)]
    indices = np.arange(len(initial))
    for item in circuit.data:
        op = item.operation
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier":
            continue
        if hasattr(op, "blocks") or getattr(op, "condition", None) is not None:
            raise ValueError("Conditional/control-flow execution is not supported.")
        if op.name in {"measure", "reset"}:
            q = qubits[0]
            one = (indices & (1 << q)) != 0
            next_branches = []
            for memory, state in branches:
                for outcome in (0, 1):
                    projected = state.copy()
                    projected[one if outcome == 0 else ~one] = 0
                    if not np.any(projected):
                        continue
                    updated = memory
                    if op.name == "measure":
                        bit = circuit.find_bit(item.clbits[0]).index
                        updated = (memory & ~(1 << bit)) | (outcome << bit)
                    elif outcome:
                        projected = projected[indices ^ (1 << q)]
                    next_branches.append((updated, projected))
            if len(next_branches) > max_branches:
                raise ValueError("Exact execution exceeds the configured branch limit.")
            branches = next_branches
        else:
            branches = [
                (memory, Statevector(state).evolve(op, qargs=qubits).data)
                for memory, state in branches
            ]
    probabilities: dict[int, float] = {}
    for memory, state in branches:
        probabilities[memory] = probabilities.get(memory, 0.0) + float(np.vdot(state, state).real)
    if not np.isclose(sum(probabilities.values()), 1, atol=1e-10):
        raise RuntimeError("Exact branch probabilities did not conserve trace.")
    return probabilities


def execute_exact(circuits: list[QuantumCircuit], *, max_qubits: int = 12) -> ExecutionResults:
    start = perf_counter()
    distributions = [exact_distribution(circuit, max_qubits=max_qubits) for circuit in circuits]
    return ExecutionResults(
        distributions, None, perf_counter() - start, "exact-statevector-branches"
    )


def execute_shots(
    circuits: list[QuantumCircuit],
    *,
    shots: int,
    seed: int = 0,
    noise_model: NoiseModel | None = None,
) -> ExecutionResults:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer per unique circuit.")
    start = perf_counter()
    if not circuits:
        return ExecutionResults([], shots, 0, "aer-density-matrix", seed)
    backend = AerSimulator(method="density_matrix", noise_model=noise_model, max_parallel_threads=1)
    executable = transpile(
        circuits, backend, optimization_level=0, seed_transpiler=seed, num_processes=1
    )
    result = backend.run(executable, shots=shots, seed_simulator=seed).result()
    distributions = []
    for index in range(len(circuits)):
        counts = result.get_counts(index)
        distributions.append(
            {int(bits.replace(" ", ""), 2): count / shots for bits, count in counts.items()}
        )
    return ExecutionResults(
        distributions, shots, perf_counter() - start, "aer-density-matrix", seed
    )
