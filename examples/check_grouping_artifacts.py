"""Independent SDK baseline norms plus grouped physical-allocation checks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from math import isclose, prod
from pathlib import Path

import numpy as np
from qiskit_addon_cutting import cut_gates, cut_wires, partition_problem
from qiskit_addon_cutting.instructions import CutWire

from qpartition import GroupingConfig, HybridPlan, PartitionGraph, compile_hybrid, load_problem


def executable_ast(source):
    """Ignore comments/docstrings, retaining all executable syntax and constants."""

    class StripDocstrings(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]
            return node

    return ast.dump(StripDocstrings().visit(ast.parse(source)), include_attributes=False)


def check_sources(report, directory):
    documentation_changes = []
    for path, expected in report["source_sha256"].items():
        current = Path(path)
        if hashlib.sha256(current.read_bytes()).hexdigest() == expected:
            continue
        snapshot = directory / "provenance/executed_source" / path
        assert snapshot.is_file(), f"Source changed without an execution snapshot: {path}"
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected, path
        assert executable_ast(current.read_text()) == executable_ast(snapshot.read_text()), path
        documentation_changes.append(path)
    return documentation_changes


def check(directory):
    report = json.loads((directory / "report.json").read_text())
    documentation_changes = check_sources(report, directory)
    results = {}
    for name, entry in report["instances"].items():
        folder = directory / name
        problem = load_problem(folder)
        assert problem.circuit_fingerprint == entry["fingerprint"]
        baseline = PartitionGraph(problem)
        grouped_graphs = {}
        results[name] = {}
        for method, row in entry.get("methods", entry.get("arms", {})).items():
            config = dict(entry["grouping_config"])
            config["families"] = tuple(config["families"])
            if "methods" in entry:
                config["group_moves"] = method != "cost_only"
            if method == "no_group":
                graph = baseline
            else:
                key = json.dumps(config, sort_keys=True)
                if key not in grouped_graphs:
                    grouped_graphs[key] = PartitionGraph(problem, grouping=GroupingConfig(**config))
                graph = grouped_graphs[key]
            plan = HybridPlan.load(folder / method / "winner/best_plan.json")
            evaluation = plan.validate(graph)
            compiled = compile_hybrid(graph, plan)
            assert isclose(compiled.gamma**2, row["reported_overhead"], rel_tol=1e-10)
            assert (
                sum(evaluation.block_widths)
                == problem.circuit.num_qubits + evaluation.action_counts["wire"]
            )
            marked = problem.circuit.copy_empty_like()
            wires, gates = {}, set()
            for v, bit in zip(graph.variables, plan.state, strict=True):
                if not bit:
                    continue
                if v.kind == "gate":
                    gates.add(v.operation_id)
                else:
                    wires.setdefault(v.operation_id, []).append(v.qubit)
            indices = []
            for op, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
                for q in wires.get(op, ()):
                    marked.append(CutWire(), [q])
                if op in gates:
                    indices.append(len(marked.data))
                marked.append(item.operation, item.qubits, item.clbits)
            marked, _ = cut_gates(marked, indices)
            sdk = partition_problem(cut_wires(marked))
            independent = baseline.evaluate(
                plan.state, max_qubits=plan.max_fragment_qubits, partition_mode=plan.partition_mode
            )
            sdk_gamma = float(prod(b.kappa for b in sdk.bases))
            assert isclose(sdk_gamma**2, independent.sampling_overhead, rel_tol=1e-10)
            assert sorted(c.num_qubits for c in sdk.subcircuits.values()) == sorted(
                independent.component_widths
            )
            assert evaluation.sampling_overhead <= independent.sampling_overhead * (1 + 1e-10)
            rng = np.random.default_rng(107)
            for _ in range(4):
                branches = tuple(int(rng.integers(len(t.coefficients))) for t in compiled.templates)
                physical = compiled.experiment(branches)
                assert tuple(c.num_qubits for c in physical.circuits) == evaluation.block_widths
                assert max(c.num_qubits for c in physical.circuits) <= plan.max_fragment_qubits
            covered = [
                op
                for g in evaluation.groups
                for op in (
                    *graph.grouping.by_id[g.group_id].operations,
                    *graph.grouping.by_id[g.group_id].absorbed_operations,
                )
            ]
            assert len(covered) == len(set(covered))
            results[name][method] = {
                "status": "passed",
                "sdk_independent_overhead_same_mask": sdk_gamma**2,
                "grouped_physical_overhead": compiled.gamma**2,
                "physical_widths": evaluation.block_widths,
                "groups": len(evaluation.groups),
                "all_branches_use_no_ancillas": True,
                "source_and_plan_fingerprint_checked": True,
                "no_overlapping_covered_operations": True,
                "scope": "SDK checks independent norm/connectivity; joint recipe and "
                "physical branch checks validate grouped cost and bundled registers.",
            }
    output = {
        "source_hashes_match": not documentation_changes,
        "executed_source_snapshots_and_executable_ast_match": True,
        "documentation_only_source_changes": documentation_changes,
        "winners_checked": sum(len(v) for v in results.values()),
        "checks": results,
    }
    (directory / "independent_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Passed grouped/SDK checks for {output['winners_checked']} winners")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/grouping_benchmarks_v1"))
    check(parser.parse_args().directory)
