"""Bounded wire-only feasibility references for cases with no heuristic solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qpartition import AnnealConfig, PartitionGraph, exact_search, load_problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--max-nodes", type=int, default=500000)
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    checks = {}
    for name in ("qft6", "qft12", "figure2", "grover4_control"):
        graph = PartitionGraph(load_problem(args.directory / name))
        limit = report["instances"][name]["max_qubits"]
        result = exact_search(
            graph, AnnealConfig(limit, allow_gate=False), max_nodes=args.max_nodes
        )
        checks[name] = {
            "complete": result.complete,
            "feasible": result.best is not None,
            "nodes": result.nodes_visited,
            "seconds": result.elapsed_seconds,
            "status": "proven-infeasible"
            if result.complete and result.best is None
            else "feasible-reference"
            if result.best is not None
            else "unresolved-node-limit",
            "scope": "Wire-only; contracted graph; two static blocks; original declared capacity.",
        }
        print(name, checks[name], flush=True)
    (args.directory / "wire_capacity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
