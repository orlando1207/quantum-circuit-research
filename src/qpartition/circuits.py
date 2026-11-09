"""Validated circuit/query snapshots and reproducible interchange."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


def fingerprint(circuit: QuantumCircuit) -> str:
    """Hash bound circuit semantics without QPY's random custom-gate UUIDs.

    Include wire/register order and metadata. Small unitary matrices distinguish
    custom definitions with identical names. QPY remains the storage format.
    """
    operations = []
    for item in circuit.data:
        op = item.operation
        matrix = None
        if op.name not in {"barrier", "measure", "reset"}:
            if op.num_qubits > 3:
                raise ValueError("Fingerprint supports unitary operations of at most three qubits.")
            value = Operator(op).data
            matrix = [value.real.tolist(), value.imag.tolist()]
        operations.append(
            {
                "name": op.name,
                "label": op.label,
                "qubits": [circuit.find_bit(q).index for q in item.qubits],
                "clbits": [circuit.find_bit(c).index for c in item.clbits],
                "matrix": matrix,
            }
        )
    data = {
        "name": circuit.name,
        "metadata": circuit.metadata,
        "global_phase": float(circuit.global_phase),
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "qregs": [(r.name, [circuit.find_bit(q).index for q in r]) for r in circuit.qregs],
        "cregs": [(r.name, [circuit.find_bit(c).index for c in r]) for r in circuit.cregs],
        "operations": operations,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def observable_fingerprint(observable: SparsePauliOp) -> str:
    data = [(p, float(c.real), float(c.imag)) for p, c in observable.to_list()]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def validate_unitary_circuit(circuit: QuantumCircuit) -> None:
    if circuit.num_qubits < 1:
        raise ValueError("A circuit must contain at least one qubit.")
    if circuit.parameters:
        raise ValueError("Bind all circuit parameters before creating a research problem.")
    if circuit.num_clbits:
        raise ValueError("Input circuits must have no classical bits or measurements.")
    for item in circuit.data:
        op = item.operation
        if op.name == "barrier":
            continue
        if op.name in {"measure", "reset", "initialize", "delay"} or hasattr(op, "blocks"):
            raise ValueError(f"Input operation {op.name!r} is outside the unitary circuit scope.")
        if op.num_qubits > 3:
            raise ValueError("Normalize operations wider than three qubits before validation.")
        try:
            matrix = np.asarray(op.to_matrix(), dtype=complex)
        except (AttributeError, TypeError, ValueError, QiskitError):
            try:
                matrix = Operator(op).data
            except Exception as exc:
                raise ValueError(f"Cannot validate operation {op.name!r} as a unitary.") from exc
        if not np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-10, rtol=0):
            raise ValueError(f"Operation {op.name!r} is not unitary.")


@dataclass(frozen=True)
class CircuitProblem:
    """A bound circuit on |0...0> and a Hermitian Pauli-sum query.

    Product-state preparations belong in the circuit. Qiskit's rightmost Pauli
    character acts on qubit 0. Operation IDs refer to this original snapshot.
    """

    circuit: QuantumCircuit
    observable: SparsePauliOp
    circuit_fingerprint: str
    query_fingerprint: str
    operation_ids: tuple[str, ...]

    @classmethod
    def create(cls, circuit: QuantumCircuit, observable: SparsePauliOp | str) -> CircuitProblem:
        validate_unitary_circuit(circuit)
        query = SparsePauliOp(observable) if isinstance(observable, str) else observable.copy()
        query = query.simplify(atol=0, rtol=0)
        if query.num_qubits != circuit.num_qubits:
            raise ValueError("Observable width must match circuit width.")
        if not np.all(np.isfinite(query.coeffs)) or np.any(np.abs(query.coeffs.imag) > 1e-12):
            raise ValueError("Observable coefficients must be finite and real (Hermitian query).")
        query.coeffs = query.coeffs.real.astype(complex)
        snapshot = circuit.copy()
        return cls(
            snapshot,
            query,
            fingerprint(snapshot),
            observable_fingerprint(query),
            tuple(f"op{i:06d}" for i in range(len(snapshot.data))),
        )

    def validate_snapshot(self) -> None:
        if fingerprint(self.circuit) != self.circuit_fingerprint:
            raise ValueError("Circuit snapshot changed; create a new problem and cut plan.")
        if observable_fingerprint(self.observable) != self.query_fingerprint:
            raise ValueError("Observable snapshot changed; create a new research problem.")

    def index(self, operation_id: str) -> int:
        try:
            return self.operation_ids.index(operation_id)
        except ValueError as exc:
            raise ValueError(f"Unknown operation ID: {operation_id}") from exc

    def exact_expectation(self, *, max_qubits: int = 12) -> float:
        self.validate_snapshot()
        if self.circuit.num_qubits > max_qubits:
            raise ValueError("Uncut statevector reference exceeds the configured qubit limit.")
        value = Statevector.from_instruction(self.circuit).expectation_value(self.observable)
        return float(np.real(value))


def save_problem(problem: CircuitProblem, directory: str | Path) -> None:
    """Save a circuit in QPY with an explicit query/provenance sidecar."""
    problem.validate_snapshot()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "circuit.qpy").open("wb") as stream:
        qpy.dump(problem.circuit, stream)
    data = {
        "schema_version": 1,
        "circuit_fingerprint": problem.circuit_fingerprint,
        "query_fingerprint": problem.query_fingerprint,
        "operation_ids": problem.operation_ids,
        "observable": [(p, float(c.real)) for p, c in problem.observable.to_list()],
    }
    (target / "problem.json").write_text(json.dumps(data, indent=2) + "\n")


def load_problem(directory: str | Path) -> CircuitProblem:
    target = Path(directory)
    data = json.loads((target / "problem.json").read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported problem schema version.")
    with (target / "circuit.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    if len(circuits) != 1:
        raise ValueError("A problem snapshot must contain exactly one circuit.")
    problem = CircuitProblem.create(circuits[0], SparsePauliOp.from_list(data["observable"]))
    if problem.circuit_fingerprint != data["circuit_fingerprint"]:
        raise ValueError("Circuit does not match its provenance sidecar.")
    if problem.query_fingerprint != data["query_fingerprint"]:
        raise ValueError("Observable does not match its provenance sidecar.")
    if list(problem.operation_ids) != data["operation_ids"]:
        raise ValueError("Operation IDs do not match the circuit snapshot.")
    return problem


def load_qasm2(path: str | Path, observable: SparsePauliOp | str) -> CircuitProblem:
    return CircuitProblem.create(qasm2.load(str(path)), observable)


def interaction_graph(problem: CircuitProblem) -> nx.Graph:
    problem.validate_snapshot()
    graph = nx.Graph()
    graph.add_nodes_from(range(problem.circuit.num_qubits))
    for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
        if item.operation.name == "barrier":
            continue
        qubits = [problem.circuit.find_bit(q).index for q in item.qubits]
        for i, a in enumerate(qubits):
            for b in qubits[i + 1 :]:
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, weight=0, operation_ids=[])
                graph[a][b]["weight"] += 1
                graph[a][b]["operation_ids"].append(op_id)
    return graph


def temporal_graph(problem: CircuitProblem) -> nx.DiGraph:
    """A directed port graph; retain all gates and ordered quantum wire edges."""
    problem.validate_snapshot()
    graph = nx.DiGraph(circuit_fingerprint=problem.circuit_fingerprint)
    previous = {}
    for q in range(problem.circuit.num_qubits):
        previous[q] = f"q{q}:input"
        graph.add_node(previous[q], kind="input", qubit=q)
    for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
        graph.add_node(op_id, kind="operation", name=item.operation.name)
        for local, qubit in enumerate(item.qubits):
            q = problem.circuit.find_bit(qubit).index
            start, end = f"{op_id}:in:{local}", f"{op_id}:out:{local}"
            graph.add_node(start, kind="port", qubit=q, direction="in")
            graph.add_node(end, kind="port", qubit=q, direction="out")
            graph.add_edge(previous[q], start, kind="wire", qubit=q, before_operation=op_id)
            graph.add_edge(start, op_id, kind="operation")
            graph.add_edge(op_id, end, kind="operation")
            previous[q] = end
    for q, node in previous.items():
        end = f"q{q}:output"
        graph.add_node(end, kind="output", qubit=q)
        graph.add_edge(node, end, kind="wire", qubit=q, before_operation=None)
    return graph
