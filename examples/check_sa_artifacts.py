"""Cross-check saved no-excision winners using Qiskit's independent wire expansion."""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path

from qiskit_addon_cutting import cut_gates, cut_wires, expand_observables, partition_problem
from qiskit_addon_cutting.instructions import CutWire

from qpartition import HybridPlan, PartitionGraph, load_problem
from qpartition.legacy_graph import LegacyPartitionGraph


def check_artifacts(directory: Path) -> dict:
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    checks = {}
    for name, entry in report["instances"].items():
        folder = directory / name
        problem = load_problem(folder)
        plan = HybridPlan.load(folder / "best_plan.json")
        graph = (LegacyPartitionGraph if plan.graph_schema == "four-port-v1" else PartitionGraph)(
            problem
        )
        evaluation = plan.validate(graph)
        if evaluation.action_counts.get("excision", 0):
            checks[name] = {"status": "not-applicable-to-excision"}
            continue
        cuts = set()
        wire_cuts: dict[str, list[int]] = {}
        for variable, value in zip(graph.variables, plan.state):
            option = variable.options[value]
            if option is None:
                continue
            if option.kind == "gate":
                cuts.add(variable.operation_id)
            elif option.kind == "wire":
                wire_cuts.setdefault(variable.operation_id, []).append(variable.qubit)
        marked = problem.circuit.copy_empty_like()
        indices = []
        for op_id, item in zip(problem.operation_ids, problem.circuit.data):
            for qubit in wire_cuts.get(op_id, []):
                marked.append(CutWire(), [marked.qubits[qubit]])
            if op_id in cuts:
                indices.append(len(marked.data))
            marked.append(item.operation, item.qubits, item.clbits)
        marked, _ = cut_gates(marked, indices)
        expanded = cut_wires(marked)
        observable = expand_observables(problem.observable.paulis, marked, expanded)
        partitioned = partition_problem(expanded, observables=observable)
        widths = sorted(c.num_qubits for c in partitioned.subcircuits.values())
        gamma = float(prod(basis.kappa for basis in partitioned.bases))
        check = {
            "status": "passed",
            "qiskit_component_widths": widths,
            "graph_component_widths": sorted(evaluation.component_widths),
            "qiskit_sampling_overhead": gamma**2,
            "graph_sampling_overhead": evaluation.sampling_overhead,
            "scope": "Structural materialization and QPD norm; no global branch enumeration.",
        }
        assert widths == sorted(evaluation.component_widths), (name, check)
        assert abs(gamma**2 / evaluation.sampling_overhead - 1) < 1e-10, (name, check)
        checks[name] = check
        entry["independent_qiskit_check"] = check
        (folder / "qiskit_crosscheck.json").write_text(json.dumps(check, indent=2) + "\n")
        graph.registry.save(folder / "templates.json")
    (directory / "qiskit_crosschecks.json").write_text(json.dumps(checks, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks"))
    args = parser.parse_args()
    print(json.dumps(check_artifacts(args.directory), indent=2))
