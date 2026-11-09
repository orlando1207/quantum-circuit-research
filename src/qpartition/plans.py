"""Manual cut plans use original-snapshot operation IDs, never shifted indices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .circuits import CircuitProblem


@dataclass(frozen=True)
class WireCut:
    qubit: int
    before_operation: str


@dataclass(frozen=True)
class CutPlan:
    circuit_fingerprint: str
    gate_cuts: tuple[str, ...] = ()
    wire_cuts: tuple[WireCut, ...] = ()
    max_fragment_qubits: int | None = None

    @classmethod
    def create(
        cls,
        problem: CircuitProblem,
        *,
        gate_cuts: tuple[str, ...] = (),
        wire_cuts: tuple[WireCut, ...] = (),
        max_fragment_qubits: int | None = None,
    ) -> CutPlan:
        plan = cls(
            problem.circuit_fingerprint, tuple(gate_cuts), tuple(wire_cuts), max_fragment_qubits
        )
        plan.validate(problem)
        return plan

    def validate(self, problem: CircuitProblem) -> None:
        problem.validate_snapshot()
        if self.circuit_fingerprint != problem.circuit_fingerprint:
            raise ValueError("Cut plan belongs to a different circuit snapshot.")
        if self.max_fragment_qubits is not None and (
            isinstance(self.max_fragment_qubits, bool)
            or not isinstance(self.max_fragment_qubits, int)
            or self.max_fragment_qubits < 1
        ):
            raise ValueError("max_fragment_qubits must be a positive integer.")
        if len(set(self.gate_cuts)) != len(self.gate_cuts):
            raise ValueError("Duplicate gate cuts.")
        if len(set(self.wire_cuts)) != len(self.wire_cuts):
            raise ValueError("Duplicate wire cuts.")
        for op_id in self.gate_cuts:
            op = problem.circuit.data[problem.index(op_id)].operation
            if op.num_qubits != 2 or op.name == "barrier":
                raise ValueError("Standard gate cuts require a two-qubit unitary gate.")
        for cut in self.wire_cuts:
            if isinstance(cut.qubit, bool) or not isinstance(cut.qubit, int):
                raise ValueError("Wire-cut qubit must be an integer index.")
            if not 0 <= cut.qubit < problem.circuit.num_qubits:
                raise ValueError("Wire-cut qubit index is out of range.")
            item = problem.circuit.data[problem.index(cut.before_operation)]
            if problem.circuit.qubits[cut.qubit] not in item.qubits:
                raise ValueError("Wire cut must be placed before an operation on that wire.")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"schema_version": 1, **asdict(self)}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CutPlan:
        data = json.loads(Path(path).read_text())
        if data.pop("schema_version", None) != 1:
            raise ValueError("Unsupported cut-plan schema version.")
        data["gate_cuts"] = tuple(data["gate_cuts"])
        data["wire_cuts"] = tuple(WireCut(**cut) for cut in data["wire_cuts"])
        return cls(**data)
