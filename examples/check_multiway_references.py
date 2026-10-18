"""Run cost-bounded exact references for selected saved multiway capacity cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qpartition import AnnealConfig, HybridPlan, PartitionGraph, exact_search, load_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/multiway_benchmarks_v3"))
    parser.add_argument("--max-nodes", type=int, default=2_000_000)
    parser.add_argument("--cases", nargs="+", default=["chain10_M6", "chain10_M5", "chain20_M10"])
    args = parser.parse_args()
    for name in args.cases:
        folder = args.directory / "capacity_sweep" / name
        graph = PartitionGraph(load_problem(folder))
        plan = HybridPlan.load(folder / "best_plan.json")
        config = AnnealConfig(
            plan.max_fragment_qubits,
            partition_mode=plan.partition_mode,
            max_components=plan.max_components,
        )
        result = exact_search(
            graph, config, incumbent=plan.validate(graph), max_nodes=args.max_nodes
        )
        record = {
            "complete": result.complete,
            "nodes_visited": result.nodes_visited,
            "states_evaluated": result.states_evaluated,
            "seconds": result.elapsed_seconds,
            "best_overhead": result.best.sampling_overhead,
            "best_state": result.best.state,
            "best_widths": result.best.block_widths,
            "best_actions": result.best.action_counts,
            "scope": "All available independent gate/wire cuts; multiway static width; "
            "same saved source circuit. Does not update the SA winner.",
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (folder / "exact_reference.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"{name}: cost={result.best.sampling_overhead:.6g}; complete={result.complete}; "
            f"nodes={result.nodes_visited:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
