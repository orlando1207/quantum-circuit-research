"""Cached, normalized local decompositions for the hybrid action grammar."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import log
from pathlib import Path

import numpy as np
from qiskit.circuit import Instruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli
from qiskit_addon_cutting.instructions import Move
from qiskit_addon_cutting.qpd import QPDBasis

from .excision import ExcisionDecomposition, decompose_excision


def _local_signed_map(matrix: np.ndarray, operations) -> np.ndarray:
    value = matrix.copy()
    for operation in operations:
        if operation.name == "qpd_measure":
            value = np.diag([value[0, 0], -value[1, 1]])
        elif operation.name == "reset":
            value = np.array([[np.trace(value), 0], [0, 0]], dtype=complex)
        else:
            unitary = Operator(operation).data
            value = unitary @ value @ unitary.conj().T
    return value


def _qpd_residual(basis: QPDBasis, target: np.ndarray, wire: bool = False) -> float:
    paulis = [Pauli(label).to_matrix() for label in ("I", "X", "Y", "Z")]
    residual = 0.0
    zero = np.array([[1, 0], [0, 0]], dtype=complex)
    if wire:
        for matrix in paulis:
            reconstructed = sum(
                coefficient
                * np.trace(_local_signed_map(matrix, maps[0]))
                * _local_signed_map(zero, maps[1])
                for coefficient, maps in zip(basis.coeffs, basis.maps)
            )
            residual = max(residual, float(np.max(np.abs(reconstructed - matrix))))
    else:
        for left in paulis:
            for right in paulis:
                matrix = np.kron(right, left)
                reconstructed = sum(
                    coefficient
                    * np.kron(_local_signed_map(right, maps[1]), _local_signed_map(left, maps[0]))
                    for coefficient, maps in zip(basis.coeffs, basis.maps)
                )
                residual = max(
                    residual,
                    float(np.max(np.abs(reconstructed - target @ matrix @ target.conj().T))),
                )
    return residual


@dataclass(frozen=True)
class ActionTemplate:
    template_id: str
    kind: str
    arity: int
    coefficients: tuple[float, ...]
    gamma: float
    basis: QPDBasis | None = None
    excision: ExcisionDecomposition | None = None
    reconstruction_residual: float | None = None

    @property
    def log_overhead(self) -> float:
        return 2 * log(self.gamma)


class TemplateRegistry:
    """One template per supported action; costs never depend on circuit query.

    Qiskit supplies gate/move QPDs. Excision is excluded by default; only the
    historical/refinement path can opt in to checked PTMs with tolerance zero.
    """

    def __init__(self, *, include_excision: bool = False) -> None:
        self.include_excision = include_excision
        self.entries: dict[str, ActionTemplate] = {}
        self.gate_cache: dict[str, tuple[str | None, str | None]] = {}
        basis = QPDBasis.from_instruction(Move())
        self.wire = self._qpd_template("wire:peng-v1", "wire", basis, 1, np.eye(2))

    def _qpd_template(
        self, key: str, kind: str, basis: QPDBasis, arity: int, target: np.ndarray
    ) -> ActionTemplate:
        coefficients = tuple(float(c) for c in basis.coeffs)
        gamma = float(sum(abs(c) for c in coefficients))
        if not np.isfinite(gamma) or gamma < 1 - 1e-10:
            raise ValueError("Invalid bounded-score template coefficient norm.")
        if any(op.num_qubits != 1 for maps in basis.maps for local in maps for op in local):
            raise ValueError("Hybrid compiler requires single-qubit local QPD instructions.")
        residual = _qpd_residual(basis, target, wire=kind == "wire")
        if residual > 1e-10:
            raise ValueError("QPD failed its full Pauli-basis channel check.")
        entry = ActionTemplate(
            key, kind, arity, coefficients, max(1.0, gamma), basis, reconstruction_residual=residual
        )
        self.entries[key] = entry
        return entry

    def for_gate(self, gate: Instruction) -> tuple[ActionTemplate | None, ActionTemplate | None]:
        if gate.num_qubits not in (1, 2):
            raise ValueError("Hybrid partitioning supports one-/two-qubit unitary gates.")
        matrix = Operator(gate).data
        digest = hashlib.sha256(
            gate.name.encode() + np.asarray(matrix, dtype=np.complex128).tobytes()
        ).hexdigest()
        if digest in self.gate_cache:
            cut_key, exc_key = self.gate_cache[digest]
            return self.entries.get(cut_key), self.entries.get(exc_key)
        cut = None
        if gate.num_qubits == 2:
            try:
                basis = QPDBasis.from_instruction(gate)
            except ValueError:
                basis = QPDBasis.from_instruction(UnitaryGate(matrix))
            if basis is not None:
                try:
                    cut = self._qpd_template(f"gate:{digest}", "gate", basis, 2, matrix)
                except ValueError:
                    # A custom instruction may reuse a standard gate's name.
                    basis = QPDBasis.from_instruction(UnitaryGate(matrix))
                    cut = self._qpd_template(f"gate:{digest}", "gate", basis, 2, matrix)
        if not self.include_excision:
            self.gate_cache[digest] = (cut.template_id if cut else None, None)
            return cut, None
        excision = decompose_excision(gate, max_qubits=2, tolerance=0)
        paulis = [Pauli(label).to_matrix() for label in excision.pauli_labels]
        residual = max(
            float(np.max(np.abs(excision.apply(p) - matrix @ p @ matrix.conj().T))) for p in paulis
        )
        if residual > 1e-10:
            raise ValueError("Excision failed its full Pauli-basis channel check.")
        coefficients = tuple(term.coefficient for term in excision.terms)
        gamma = float(sum(abs(c) for c in coefficients))
        entry = ActionTemplate(
            f"excision:{digest}",
            "excision",
            gate.num_qubits,
            coefficients,
            gamma,
            excision=excision,
            reconstruction_residual=residual,
        )
        self.entries[entry.template_id] = entry
        self.gate_cache[digest] = (cut.template_id if cut else None, entry.template_id)
        return cut, entry

    def save(self, path: str | Path) -> None:
        data = []
        for entry in self.entries.values():
            record = {
                "template_id": entry.template_id,
                "kind": entry.kind,
                "arity": entry.arity,
                "gamma": entry.gamma,
                "sampling_overhead": entry.gamma**2,
                "log_overhead": entry.log_overhead,
                "coefficients": entry.coefficients,
                "resource_model": "LO-no-ancillas-static-segments",
                "reconstruction_residual": entry.reconstruction_residual,
            }
            if entry.excision:
                record["pauli_branches"] = [
                    {
                        "input": t.input_pauli,
                        "output": t.output_pauli,
                        "eigenbits": t.eigenbits,
                        "coefficient": t.coefficient,
                    }
                    for t in entry.excision.terms
                ]
            if entry.basis:
                record["local_maps"] = [
                    [
                        [
                            {
                                "name": op.name,
                                "params": [repr(p) for p in op.params],
                                "matrix": (
                                    [
                                        Operator(op).data.real.tolist(),
                                        Operator(op).data.imag.tolist(),
                                    ]
                                    if op.name not in ("reset", "qpd_measure")
                                    else None
                                ),
                            }
                            for op in local
                        ]
                        for local in maps
                    ]
                    for maps in entry.basis.maps
                ]
            data.append(record)
        Path(path).write_text(json.dumps({"schema_version": 1, "templates": data}, indent=2) + "\n")
