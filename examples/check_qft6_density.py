"""Check actual QFT6 group recipes and causal anchors on entangled density inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator

from qpartition import GroupingConfig, HybridPlan, PartitionGraph, compile_hybrid, load_problem


def check(directory):
    folder = directory / "qft6"
    problem = load_problem(folder)
    graph = PartitionGraph(problem, grouping=GroupingConfig())
    plan = HybridPlan.load(folder / "enhanced/winner/best_plan.json")
    compiled = compile_hybrid(graph, plan)
    assert compiled.evaluation.action_counts["wire"] == 0
    library = graph.grouping
    selected = {library.by_id[s.group_id].anchor: s for s in compiled.evaluation.groups}
    covered = {
        op
        for s in selected.values()
        for op in (
            *library.by_id[s.group_id].operations,
            *library.by_id[s.group_id].absorbed_operations,
        )
    }
    n = problem.circuit.num_qubits
    dimension = 2**n
    indices = np.arange(dimension)
    rng = np.random.default_rng(159)
    target = Operator(problem.circuit).data
    checks = []
    for pure in (False, True):
        a = rng.normal(size=(dimension, 1 if pure else dimension)) + 1j * rng.normal(
            size=(dimension, 1 if pure else dimension)
        )
        density = a @ a.conj().T
        density /= np.trace(density)
        expected = target @ density @ target.conj().T
        current = density.copy()
        for op, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if op in selected:
                s = selected[op]
                candidate = library.by_id[s.group_id]
                basis = library.basis(s)
                sides = tuple(
                    tuple(
                        q for i, q in enumerate(candidate.qubits) if ((s.pattern >> i) & 1) == side
                    )
                    for side in (0, 1)
                )
                result = np.zeros_like(current)
                for weight, maps in zip(basis.coefficients, basis.maps, strict=True):
                    branch = current.copy()
                    for side in (0, 1):
                        for step in maps[side]:
                            qubits = [sides[side][q] for q in step.qubits]
                            if step.operation.name == "measure":
                                bits = (indices >> qubits[0]) & 1
                                branch *= bits[:, None] == bits[None, :]
                                if step.scored:
                                    branch *= 1 - 2 * bits[:, None]
                            else:
                                branch = (
                                    DensityMatrix(branch).evolve(step.operation, qargs=qubits).data
                                )
                    result += weight * branch
                current = result
            if op not in covered:
                current = (
                    DensityMatrix(current)
                    .evolve(
                        item.operation,
                        qargs=[problem.circuit.find_bit(q).index for q in item.qubits],
                    )
                    .data
                )
        residual = float(np.max(np.abs(current - expected)))
        assert residual < 2e-11, residual
        checks.append(
            {
                "input": "random_entangled_pure" if pure else "random_entangled_mixed",
                "max_density_residual": residual,
                "trace_real": float(np.trace(current).real),
            }
        )
    result = {
        "status": "passed",
        "source_fingerprint": problem.circuit_fingerprint,
        "physical_overhead": compiled.gamma**2,
        "checks": checks,
        "scope": "Actual signed group maps and their source anchors, "
        "with other source gates retained. "
        "Two dense six-qubit inputs, not a complete 4096-element channel basis and not "
        "enumeration of all cross-group experiment combinations.",
    }
    (directory / "qft6_density_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/grouping_benchmarks_v2"))
    check(parser.parse_args().directory)
