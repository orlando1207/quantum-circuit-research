"""Run official Qiskit and explicit paper reconstructions on frozen identical inputs.

No result from this script is labeled an exact author-code reproduction. Manual
Table-I group manifests and several weighted-KL details are not published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from math import isclose, log
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import qasm2

from qpartition import AnnealConfig, PartitionGraph, load_problem, multi_start, save_problem
from qpartition.baseline_circuits import grover4_historical
from qpartition.paper_baseline import (
    GateGroup,
    PaperConfig,
    PaperTopology,
    run_paper_reconstruction,
)
from qpartition.qiskit_baseline import QiskitBaselineConfig, run_qiskit_baseline


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _one(job):
    method, problem, settings, groups = job
    if method == "qiskit":
        return run_qiskit_baseline(problem, QiskitBaselineConfig(**settings))[1]
    return run_paper_reconstruction(problem, PaperConfig(**settings), groups)[1]


def chain_manifest(problem):
    """Our deterministic disjoint adjacent-CX cascades, not the authors' catalog."""
    gates = [
        (op, item)
        for op, item in zip(problem.operation_ids, problem.circuit.data)
        if item.operation.num_qubits == 2
    ]
    result, i = [], 0
    while i < len(gates):
        possibilities = []
        for side, kind in ((0, "cascade_control"), (1, "cascade_target")):
            first = gates[i][1]
            same = first.qubits[side]
            others = set()
            end = i
            while end < len(gates):
                item = gates[end][1]
                if item.operation.name != "cx" or item.qubits[side] != same:
                    break
                other = item.qubits[1 - side]
                if other in others:
                    break
                others.add(other)
                end += 1
            possibilities.append((end - i, kind))
        length, kind = max(possibilities)
        if length < 2:
            i += 1
            continue
        result.append(
            GateGroup(
                f"own_cascade_{len(result)}",
                kind,
                tuple(op for op, _ in gates[i : i + length]),
                "Own deterministic maximal adjacent-CX cascades on the saved structural proxy; "
                "not authors' Table-I manual group definitions.",
            )
        )
        i += length
    return tuple(result)


def summarize(rows):
    feasible = [r for r in rows if r.get("status", "feasible") == "feasible"]
    if not feasible:
        return {"runs": len(rows), "feasible_runs": 0, "best_overhead": None}
    winner = min(feasible, key=lambda r: (r["sampling_overhead"], r["fragments"]))
    values = [r["sampling_overhead"] for r in feasible]
    result = {
        "runs": len(rows),
        "feasible_runs": len(feasible),
        "best_overhead": winner["sampling_overhead"],
        "median_overhead": float(np.median(values)),
        "best_seed": winner["config"]["seed"],
        "best_widths": winner["widths"],
        "best_gate_cuts": len(winner["gate_cuts"]),
        "best_wire_cuts": len(winner["wire_cuts"]),
        "best_fragments": winner["fragments"],
        "minimum_reached_runs": sum(r.get("minimum_reached", False) for r in rows),
    }
    if "stage_one_cut_gate_count" in winner:
        first_stage = min(
            feasible, key=lambda r: (r["stage_one_cut_gate_count"], r["config"]["seed"])
        )
        result["select_stage_one_by_count_then_refine"] = {
            "best_overhead": first_stage["sampling_overhead"],
            "seed": first_stage["config"]["seed"],
            "note": "Alternative restart interpretation; select stage-one winner by count first.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--output", type=Path, default=Path("validation/external_baselines_v1"))
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sa-steps", type=int, default=6000)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory.")
    if min(args.starts, args.workers, args.sa_steps) < 1:
        parser.error("Budgets must be positive.")
    args.output.mkdir(parents=True)
    frozen = json.loads((args.source / "report.json").read_text())
    cases = []
    for name, entry in frozen["instances"].items():
        cases.append(
            (name, load_problem(args.source / name), entry["max_qubits"], entry["provenance"])
        )
    grover, grover_provenance = grover4_historical()
    cases.append(("grover4_historical", grover, 3, grover_provenance))
    paper = json.loads(Path("benchmarks/table_i_baselines.json").read_text())
    report = {
        "schema_version": 1,
        "purpose": "External-method baselines under identical source circuits and capacities",
        "paper_source": paper["source"],
        "paper_pdf_sha256": hashlib.sha256(Path(paper["source"]).read_bytes()).hexdigest(),
        "paper_baselines": paper["rows"],
        "reproduction_status": "Official Qiskit package executed directly. Paper implementation "
        "is an explicit reconstruction; exact author reproduction "
        "unverified without code, environment, circuit and group manifests.",
        "arguments": {**vars(args), "source": str(args.source), "output": str(args.output)},
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("qiskit", "qiskit-addon-cutting", "networkx", "numpy")
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/qpartition").glob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "instances": {},
        "audits": {
            "eq3_shared_parallel_group": {
                "current_gamma": 7,
                "single_move_gamma": 3,
                "swapped_gamma": 7,
                "true_gain_log_gamma": 0,
                "printed_equation_gain": 2 * log(7 / 3),
                "interpretation": "Printed shared-group swap correction disagrees with "
                "the stated objective; authors' implementation unknown.",
            },
            "figure4_arithmetic_only": {
                "ungrouped_gamma": 3 * 3 * 4 * 4,
                "ungrouped_overhead": (3 * 3 * 4 * 4) ** 2,
                "grouped_gamma": 7 * 3 * 4,
                "grouped_overhead": (7 * 3 * 4) ** 2,
                "circuit_optimization_reproduced": False,
            },
        },
    }
    started = perf_counter()
    with ProcessPoolExecutor(args.workers) as executor:
        for name, problem, original_limit, provenance in cases:
            if args.only and name not in args.only:
                continue
            capacities = {
                original_limit,
                max(2, problem.circuit.num_qubits // 2),
                max(2, problem.circuit.num_qubits // 3),
            }
            if name == "chain10":
                capacities.add(6)
            for limit in sorted(capacities, reverse=True):
                key = f"{name}_M{limit}"
                folder = args.output / key
                folder.mkdir()
                save_problem(problem, folder)
                qasm2.dump(problem.circuit, folder / "source.qasm")
                entry = {
                    "benchmark": name,
                    "qubits": problem.circuit.num_qubits,
                    "two_qubit_gates": sum(
                        i.operation.num_qubits == 2 for i in problem.circuit.data
                    ),
                    "max_qubits": limit,
                    "paper_capacity": limit == original_limit,
                    "fingerprint": problem.circuit_fingerprint,
                    "provenance": provenance,
                    "methods": {},
                }
                methods = {
                    "qiskit": None,
                    "paper_balanced": PaperConfig(limit),
                    "paper_capacity": PaperConfig(limit, spatial_sizes="capacity"),
                }
                own_groups = chain_manifest(problem) if name.startswith("chain") else ()
                if own_groups:
                    methods["paper_GG_own_CX_manifest"] = PaperConfig(
                        limit, spatial_sizes="capacity"
                    )
                    write(
                        folder / "own_group_manifest.json",
                        {
                            "source_fingerprint": problem.circuit_fingerprint,
                            "groups": [asdict(g) for g in own_groups],
                            "author_manifest": False,
                        },
                    )
                for method, config in methods.items():
                    if method != "qiskit" and problem.circuit.num_qubits > 2 * limit:
                        entry["methods"][method] = {"status": "proven-infeasible-two-way"}
                        continue
                    groups = own_groups if method == "paper_GG_own_CX_manifest" else ()
                    settings = (
                        asdict(QiskitBaselineConfig(limit))
                        if method == "qiskit"
                        else asdict(config)
                    )
                    jobs = [
                        (
                            "qiskit" if method == "qiskit" else "paper",
                            problem,
                            {**settings, "seed": seed},
                            groups,
                        )
                        for seed in range(args.starts)
                    ]
                    rows = list(executor.map(_one, jobs))
                    write(folder / f"{method}_runs.json", rows)
                    summary = summarize(rows)
                    entry["methods"][method] = summary
                    winner = min(
                        (r for r in rows if r.get("status", "feasible") == "feasible"),
                        key=lambda r: r["sampling_overhead"],
                        default=None,
                    )
                    if winner:
                        write(folder / f"{method}_winner.json", winner)
                        if method != "qiskit":
                            graph = PaperTopology(problem, groups)
                            _, check = graph.materialize(winner["labels"])
                            assert max(check["sdk_component_widths"]) <= limit
                            assert sum(check["sdk_component_widths"]) == sum(winner["widths"])
                            if not groups:
                                assert isclose(
                                    check["sdk_individual_sampling_overhead"],
                                    winner["sampling_overhead"],
                                    rel_tol=1e-10,
                                )
                            write(folder / f"{method}_sdk_check.json", check)
                    print(
                        key,
                        method,
                        summary.get("best_overhead"),
                        summary.get("best_widths"),
                        flush=True,
                    )
                # A fixed-profile fresh SA comparator; no historical warm starts
                # or instance-specific tuning is imported into this comparison.
                graph = PartitionGraph(problem)
                config = AnnealConfig(limit, partition_mode="multiway", steps=args.sa_steps)
                runs = multi_start(
                    graph, config, seeds=tuple(range(args.starts)), workers=args.workers
                )
                rows = [r.to_dict() for r in runs]
                write(folder / "sa_fresh_runs.json", rows)
                best = min((r.best for r in runs if r.best), key=lambda e: e.log_overhead)
                entry["methods"]["sa_fresh"] = {
                    "best_overhead": best.sampling_overhead,
                    "best_widths": best.block_widths,
                    "best_fragments": len(best.block_widths),
                    "runs": len(runs),
                    "feasible_runs": sum(r.best is not None for r in runs),
                    "config": asdict(config),
                    "initialization": "fresh, no historical warm starts",
                }
                write(
                    folder / "sa_fresh_winner.json",
                    {
                        "state": best.state,
                        "widths": best.block_widths,
                        "sampling_overhead": best.sampling_overhead,
                        "action_counts": best.action_counts,
                    },
                )
                report["instances"][key] = entry
                report["wall_seconds"] = perf_counter() - started
                write(args.output / "report.json", report)
                print(key, "sa_fresh", best.sampling_overhead, best.block_widths, flush=True)
    print(f"Saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
