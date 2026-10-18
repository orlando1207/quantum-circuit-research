"""Validate frozen inputs and independently materialize every fresh SA winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import exp, isclose, prod
from pathlib import Path

from qiskit_addon_cutting import cut_gates, cut_wires, partition_problem
from qiskit_addon_cutting.instructions import CutWire

from qpartition import HybridPlan, PartitionGraph, compile_hybrid, load_problem
from qpartition.paper_baseline import GateGroup, PaperTopology


def check(directory):
    report = json.loads((directory / "report.json").read_text())
    checks = {}
    for path, expected in report["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for key, entry in report["instances"].items():
        folder = directory / key
        problem = load_problem(folder)
        assert problem.circuit_fingerprint == entry["fingerprint"]
        # Current benchmark sources have no unused original wires. The SDK's
        # automatic partitioning otherwise discards fully idle qubits.
        used = {problem.circuit.find_bit(q).index for i in problem.circuit.data for q in i.qubits}
        assert len(used) == problem.circuit.num_qubits
        graph = PartitionGraph(problem)
        saved = json.loads((folder / "sa_fresh_winner.json").read_text())
        plan = HybridPlan.create(
            graph,
            tuple(saved["state"]),
            max_qubits=entry["max_qubits"],
            partition_mode="multiway",
        )
        plan.save(folder / "sa_fresh_plan.json")
        compiled = compile_hybrid(graph, plan)
        experiment = compiled.experiment((0,) * len(compiled.templates))
        assert tuple(c.num_qubits for c in experiment.circuits) == tuple(saved["widths"])
        assert isclose(compiled.gamma**2, saved["sampling_overhead"], rel_tol=1e-10)
        gate_ids, wire_ids = set(), {}
        for variable, value in zip(graph.variables, plan.state, strict=True):
            if not value:
                continue
            if variable.kind == "gate":
                gate_ids.add(variable.operation_id)
            else:
                wire_ids.setdefault(variable.operation_id, []).append(variable.qubit)
        marked = problem.circuit.copy_empty_like()
        indices = []
        for op, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            for q in wire_ids.get(op, ()):
                marked.append(CutWire(), [q])
            if op in gate_ids:
                indices.append(len(marked.data))
            marked.append(item.operation, item.qubits, item.clbits)
        marked, _ = cut_gates(marked, indices)
        partitioned = partition_problem(cut_wires(marked))
        widths = sorted(c.num_qubits for c in partitioned.subcircuits.values())
        norm = float(prod(b.kappa for b in partitioned.bases) ** 2)
        assert widths == sorted(saved["widths"])
        assert isclose(norm, saved["sampling_overhead"], rel_tol=1e-10)
        row = {
            "status": "passed",
            "source_fingerprint": problem.circuit_fingerprint,
            "sdk_widths": widths,
            "sdk_sampling_overhead": norm,
            "lazy_physical_branch_compiled": True,
            "global_experiment_family_enumerated": False,
        }
        manifest_path = folder / "own_group_manifest.json"
        no_gg_path = folder / "paper_capacity_winner.json"
        if manifest_path.exists() and no_gg_path.exists():
            manifest = json.loads(manifest_path.read_text())
            groups = tuple(GateGroup(**g) for g in manifest["groups"])
            grouped = PaperTopology(problem, groups)
            no_gg = json.loads(no_gg_path.read_text())
            candidate_cost = exp(2 * grouped.cost(no_gg["labels"]))
            assert candidate_cost <= no_gg["sampling_overhead"] * (1 + 1e-10)
            row["own_GG_reevaluated_no_GG_incumbent"] = {
                "ungrouped_overhead": no_gg["sampling_overhead"],
                "grouped_cost_same_labels": candidate_cost,
                "widths": no_gg["widths"],
                "scope": "Post hoc feasible candidate; not the grouped optimizer run "
                "or an author-manifest reproduction.",
            }
        checks[key] = row
        (folder / "sa_sdk_check.json").write_text(json.dumps(row, indent=2) + "\n")
    value = {
        "all_source_hashes_match": True,
        "cases_checked": len(checks),
        "checks": checks,
    }
    (directory / "independent_checks.json").write_text(json.dumps(value, indent=2) + "\n")
    print(f"Passed independent compilation/norm/width checks for {len(checks)} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/external_baselines_v1"))
    check(parser.parse_args().directory)
