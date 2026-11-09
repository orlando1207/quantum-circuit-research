"""Compile manual gate/wire cuts through the Qiskit cutting add-on."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import SamplerResult
from qiskit.quantum_info import PauliList
from qiskit.result import QuasiDistribution
from qiskit_addon_cutting import (
    cut_gates,
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.instructions import CutWire
from qiskit_addon_cutting.qpd import SingleQubitQPDGate

from .circuits import CircuitProblem, fingerprint
from .execution import ExecutionResults, execute_exact, execute_shots
from .plans import CutPlan


@dataclass(frozen=True)
class CuttingCost:
    gate_cuts: int
    wire_cuts: int
    fragment_widths: dict[str, int]
    decomposition_terms: int
    generated_experiments: int
    unique_experiments: int
    gamma: float
    sampling_overhead: float
    observable_l1: float

    @property
    def bounded_estimator_factor(self) -> float:
        return self.sampling_overhead * self.observable_l1**2


@dataclass
class CompiledCut:
    problem: CircuitProblem
    plan: CutPlan
    fragments: dict[str, QuantumCircuit]
    subobservables: dict[str, PauliList]
    experiments: dict[str, list[QuantumCircuit]]
    unique_experiments: dict[str, list[QuantumCircuit]]
    experiment_indices: dict[str, list[int]]
    coefficients: list
    cost: CuttingCost

    def reconstruction_graph(self) -> nx.MultiGraph:
        """Fragment/factor graph retaining each local port of every QPD cut."""
        graph = nx.MultiGraph()
        for label, circuit in self.fragments.items():
            fragment = ("fragment", label)
            graph.add_node(fragment, kind="fragment", qubits=circuit.num_qubits)
            for item in circuit.data:
                if isinstance(item.operation, SingleQubitQPDGate):
                    cut_id = int(item.operation.label.rsplit("_", 1)[1])
                    factor = ("cut", cut_id)
                    graph.add_node(factor, kind="cut", gamma=float(item.operation.basis.kappa))
                    graph.add_edge(
                        fragment,
                        factor,
                        qubit=circuit.find_bit(item.qubits[0]).index,
                        decomposition_port=item.operation.qubit_id,
                    )
        return graph

    def reconstruct(self, results: dict[str, ExecutionResults]) -> float:
        self.plan.validate(self.problem)
        if set(results) != set(self.unique_experiments):
            raise ValueError("Execution results do not match the compiled fragments.")
        sampler_results = {}
        for label, execution in results.items():
            if len(execution.distributions) != len(self.unique_experiments[label]):
                raise ValueError("Execution result count does not match unique experiments.")
            sampler_results[label] = SamplerResult(
                quasi_dists=[
                    QuasiDistribution(execution.distributions[index])
                    for index in self.experiment_indices[label]
                ],
                metadata=[{} for _ in self.experiment_indices[label]],
            )
        terms = reconstruct_expectation_values(
            sampler_results, self.coefficients, self.subobservables
        )
        return float(np.dot(terms, self.problem.observable.coeffs.real))

    def run_exact(self, *, max_qubits: int = 12) -> CuttingResult:
        self.plan.validate(self.problem)
        execution = {
            label: execute_exact(circuits, max_qubits=max_qubits)
            for label, circuits in self.unique_experiments.items()
        }
        return CuttingResult(
            self.reconstruct(execution),
            execution,
            self.cost,
            self.problem.circuit_fingerprint,
            self.problem.query_fingerprint,
            self.plan,
        )

    def run_shots(self, *, shots: int, seed: int = 0, noise_model=None) -> CuttingResult:
        self.plan.validate(self.problem)
        execution = {
            label: execute_shots(circuits, shots=shots, seed=seed + i, noise_model=noise_model)
            for i, (label, circuits) in enumerate(self.unique_experiments.items())
        }
        return CuttingResult(
            self.reconstruct(execution),
            execution,
            self.cost,
            self.problem.circuit_fingerprint,
            self.problem.query_fingerprint,
            self.plan,
        )


@dataclass(frozen=True)
class CuttingResult:
    expectation: float
    execution: dict[str, ExecutionResults]
    cost: CuttingCost
    circuit_fingerprint: str
    query_fingerprint: str
    plan: CutPlan

    @property
    def total_shots(self) -> int:
        return sum(result.total_shots for result in self.execution.values())


def _deduplicate(circuits: list[QuantumCircuit]) -> tuple[list[QuantumCircuit], list[int]]:
    unique, indices, seen = [], [], {}
    for circuit in circuits:
        canonical = circuit.copy()
        canonical.name = "experiment"
        canonical.metadata = {}
        key = fingerprint(canonical)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(circuit)
        indices.append(seen[key])
    return unique, indices


def compile_cut(
    problem: CircuitProblem, plan: CutPlan, *, max_experiments: int = 10000
) -> CompiledCut:
    """Fully enumerate coefficients; finite-shot execution is a separate choice."""
    plan.validate(problem)
    if any(
        item.operation.num_qubits > 2 and item.operation.name != "barrier"
        for item in problem.circuit.data
    ):
        raise ValueError("Standard cutting requires one-/two-qubit gates; normalize larger gates.")
    marked = problem.circuit.copy_empty_like()
    marked.global_phase = problem.circuit.global_phase
    gate_indices = []
    for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
        for cut in plan.wire_cuts:
            if cut.before_operation == op_id:
                marked.append(CutWire(), [marked.qubits[cut.qubit]])
        if op_id in plan.gate_cuts:
            gate_indices.append(len(marked.data))
        marked.append(item.operation, item.qubits, item.clbits)
    # Idle output wires still carry observables (e.g. <X> on |0> is zero).
    active = {q for item in marked.data for q in item.qubits}
    for qubit in marked.qubits:
        if qubit not in active:
            marked.id(qubit)
    marked, _ = cut_gates(marked, gate_indices)
    expanded = cut_wires(marked)
    observables = expand_observables(problem.observable.paulis, marked, expanded)
    partitioned = partition_problem(expanded, observables=observables)
    fragments = {f"f{label}": circuit for label, circuit in partitioned.subcircuits.items()}
    subobservables = {f"f{label}": paulis for label, paulis in partitioned.subobservables.items()}
    widths = {label: circuit.num_qubits for label, circuit in fragments.items()}
    if plan.max_fragment_qubits is not None and any(
        width > plan.max_fragment_qubits for width in widths.values()
    ):
        raise ValueError(f"Materialized fragment widths {widths} exceed the device constraint.")
    possible_terms = prod(sum(c != 0 for c in basis.coeffs) for basis in partitioned.bases)
    # Each observable can add a measurement group per fragment. Bound before expansion.
    upper_bound = possible_terms * len(problem.observable) * len(fragments)
    if upper_bound > max_experiments:
        raise ValueError(
            f"Enumeration upper bound {upper_bound} exceeds max_experiments={max_experiments}."
        )
    experiments, coefficients = generate_cutting_experiments(
        fragments, subobservables, num_samples=np.inf
    )
    unique, indices = {}, {}
    for label, circuits in experiments.items():
        unique[label], indices[label] = _deduplicate(circuits)
    gamma = float(prod(basis.kappa for basis in partitioned.bases))
    cost = CuttingCost(
        len(plan.gate_cuts),
        len(plan.wire_cuts),
        widths,
        len(coefficients),
        sum(map(len, experiments.values())),
        sum(map(len, unique.values())),
        gamma,
        gamma**2,
        float(np.abs(problem.observable.coeffs).sum()),
    )
    return CompiledCut(
        problem, plan, fragments, subobservables, experiments, unique, indices, coefficients, cost
    )
