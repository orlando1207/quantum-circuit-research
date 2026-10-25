"""Direct adapter to the installed Qiskit cut finder, with materialization checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isclose, log
from time import perf_counter

from qiskit_addon_cutting import cut_wires, expand_observables, partition_problem
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)


@dataclass(frozen=True)
class QiskitBaselineConfig:
    max_qubits: int
    seed: int = 0
    max_gamma: float = 1024
    max_backjumps: int | None = 10000
    gate_lo: bool = True
    wire_lo: bool = True


def run_qiskit_baseline(problem, config: QiskitBaselineConfig):
    """Call the genuine package; never substitute the project's SA/exact solver.

    No global QPD experiments are enumerated. Source operation IDs are restored
    by counting original instructions separately from inserted CutWire markers.
    """
    problem.validate_snapshot()
    if isinstance(config.max_qubits, bool) or not isinstance(config.max_qubits, int):
        raise ValueError("max_qubits must be a positive integer.")
    optimization = OptimizationParameters(
        seed=config.seed,
        max_gamma=config.max_gamma,
        max_backjumps=config.max_backjumps,
        gate_lo=config.gate_lo,
        wire_lo=config.wire_lo,
    )
    constraints = DeviceConstraints(config.max_qubits)
    started = perf_counter()
    marked, metadata = find_cuts(problem.circuit, optimization, constraints)
    optimizer_seconds = perf_counter() - started
    source_index = 0
    gates, wires = [], []
    for item in marked.data:
        if item.operation.name == "cut_wire":
            wires.append(
                {
                    "qubit": marked.find_bit(item.qubits[0]).index,
                    "before_operation_id": problem.operation_ids[source_index],
                }
            )
            continue
        if item.operation.name == "qpd_2q":
            gates.append(problem.operation_ids[source_index])
        source_index += 1
    if source_index != len(problem.circuit.data):
        raise RuntimeError("Cut finder changed source instruction alignment.")
    expanded = cut_wires(marked)
    observable = expand_observables(problem.observable.paulis, marked, expanded)
    partitioned = partition_problem(expanded, observables=observable)
    widths = tuple(c.num_qubits for c in partitioned.subcircuits.values())
    log_overhead = 2 * sum(log(b.kappa) for b in partitioned.bases)
    overhead = exp(log_overhead) if log_overhead < 709 else float("inf")
    if not widths or max(widths) > config.max_qubits:
        raise RuntimeError("SDK winner violates the materialized fragment-width constraint.")
    if sum(widths) != problem.circuit.num_qubits + len(wires):
        raise RuntimeError("SDK fragment registers violate static segment conservation.")
    if not isclose(overhead, metadata["sampling_overhead"], rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("SDK metadata and materialized QPD norms disagree.")
    return marked, {
        "method": "qiskit_addon_cutting.automated_cut_finding.find_cuts",
        "implementation": "installed official package, not a reimplementation",
        "config": asdict(config),
        "sampling_overhead": overhead,
        "log_sampling_overhead": log_overhead,
        "metadata_sampling_overhead": float(metadata["sampling_overhead"]),
        "minimum_reached": bool(metadata["minimum_reached"]),
        "widths": widths,
        "fragments": len(widths),
        "gate_cuts": gates,
        "wire_cuts": wires,
        "optimizer_seconds": optimizer_seconds,
        "total_seconds": perf_counter() - started,
        "physical_widths_checked": True,
        "physical_norm_checked": True,
        "source_fingerprint": problem.circuit_fingerprint,
    }
