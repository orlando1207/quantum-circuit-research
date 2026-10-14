"""Multi-start/hyperparameter experiments with explicit Table-I provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from math import floor, log10
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from qiskit import qasm2, qpy

from qpartition import (
    AnnealConfig,
    HybridPlan,
    PartitionGraph,
    compile_hybrid,
    exact_search,
    multi_start,
    save_problem,
)
from qpartition.annealing import seed_states
from qpartition.benchmarks import initial_paper_suite


def summarize(runs):
    feasible = [run for run in runs if run.best is not None]
    overheads = [run.best.sampling_overhead for run in feasible]
    return {
        "runs": len(runs),
        "feasible_runs": len(feasible),
        "best_overhead": min(overheads) if overheads else None,
        "median_overhead": median(overheads) if overheads else None,
        "p90_overhead": float(np.quantile(overheads, 0.9)) if overheads else None,
        "median_seconds": median(run.elapsed_seconds for run in runs),
        "total_evaluations": sum(run.evaluations for run in runs),
        "improved_over_own_seed": sum(
            run.best.log_overhead < run.initial_log_overhead - 1e-10 for run in feasible
        ),
        "best_seed_overhead": min(np.exp(run.initial_log_overhead) for run in runs),
    }


def rounded_comparison(value, reported):
    if reported is None:
        return None
    half_step = 0.5 * 10 ** (floor(log10(reported)) - 1)
    low, high = reported - half_step, reported + half_step
    status = (
        "below-reported-rounding-interval"
        if value < low
        else "above-reported-rounding-interval"
        if value > high
        else "within-reported-rounding-interval"
    )
    return {
        "reported": reported,
        "rounding_interval": [low, high],
        "ratio_to_reported": value / reported,
        "status": status,
    }


def save_runs(path, runs):
    path.write_text(json.dumps([run.to_dict() for run in runs], indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/sa_benchmarks"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--tuning-starts", type=int, default=8)
    parser.add_argument("--final-starts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baselines = json.loads(Path("benchmarks/table_i_baselines.json").read_text())
    profiles = {
        "single_action": {"block_probability": 0.0},
        "mixed": {"block_probability": 0.3, "block_size": 4},
        "wide_blocks": {"block_probability": 0.6, "block_size": 8},
        "colder_start": {"initial_acceptance": 0.55},
        "fast_cooling": {"cooling_rate": 0.85},
        "slow_cooling": {"cooling_rate": 0.97},
        "no_excision_ablation": {"allow_excision": False},
    }
    report = {
        "schema_version": 1,
        "source_pdf_sha256": hashlib.sha256(Path(baselines["source"]).read_bytes()).hexdigest(),
        "constraints": "two static executable blocks; M=floor(3*n/4); no grouping",
        "comparison_scope": (
            "QFT count/spatial-cost matched reconstructions; chain structural proxies. "
            "No claim of beating original unpublished circuit artifacts."
        ),
        "selection_protocol": "Instance-specific tuning; final evaluation uses fresh seeds 1000+. "
        "No cross-circuit generalization claim. Initialization time is recorded separately; "
        "per-run evaluation counts include pilot probes and greedy polishing.",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("qiskit", "qiskit-aer", "qiskit-addon-cutting", "numpy", "networkx")
        },
        "arguments": {**vars(args), "output": str(args.output)},
        "profiles": profiles,
        "instances": {},
    }
    suite_start = perf_counter()
    for benchmark in initial_paper_suite():
        if args.only and benchmark.name not in args.only:
            continue
        print(f"Preparing {benchmark.name}", flush=True)
        folder = args.output / benchmark.name
        folder.mkdir(exist_ok=True)
        save_problem(benchmark.problem, folder)
        qasm2.dump(benchmark.problem.circuit, folder / "source.qasm")
        graph = PartitionGraph(benchmark.problem)
        graph.registry.save(folder / "templates.json")
        base = AnnealConfig(benchmark.max_qubits, steps=args.steps)
        init_start = perf_counter()
        pool = seed_states(graph, base)
        seed_evaluations = [graph.evaluate(state, max_qubits=base.max_qubits) for state in pool]
        initialization_seconds = perf_counter() - init_start
        (folder / "initialization.json").write_text(
            json.dumps(
                {
                    "seconds": initialization_seconds,
                    "states": pool,
                    "overheads": [evaluation.sampling_overhead for evaluation in seed_evaluations],
                },
                indent=2,
            )
            + "\n"
        )
        tuning = {}
        for name, changes in profiles.items():
            config = replace(base, **changes)
            started = perf_counter()
            runs = multi_start(
                graph,
                config,
                seeds=tuple(range(args.tuning_starts)),
                workers=args.workers,
                initial_states=pool,
            )
            summary = summarize(runs)
            summary["wall_seconds"] = perf_counter() - started
            summary["config"] = asdict(config)
            tuning[name] = summary
            save_runs(folder / f"tuning_{name}.json", runs)
            print(
                f"  {name}: best={summary['best_overhead']:.6g}, "
                f"median={summary['median_overhead']:.6g}",
                flush=True,
            )
        # Keep all three actions in the selected final configuration. The narrower
        # no-excision profile remains an explicit ablation, not a hidden replacement.
        selected = min(
            (name for name in profiles if name != "no_excision_ablation"),
            key=lambda name: (tuning[name]["best_overhead"], tuning[name]["median_overhead"], name),
        )
        config = replace(base, **profiles[selected])
        started = perf_counter()
        final_runs = multi_start(
            graph,
            config,
            seeds=tuple(range(1000, 1000 + args.final_starts)),
            workers=args.workers,
            initial_states=pool,
        )
        final_summary = summarize(final_runs)
        final_summary["wall_seconds"] = perf_counter() - started
        save_runs(folder / "final_runs.json", final_runs)
        best_run = min(
            (run for run in final_runs if run.best),
            key=lambda run: (run.best.log_overhead, len(run.best.block_widths)),
        )
        plan = HybridPlan.create(graph, best_run.best.state, max_qubits=base.max_qubits)
        plan.save(folder / "best_plan.json")
        compiled = compile_hybrid(graph, plan)
        rng = np.random.default_rng(331)
        widths_seen = set()
        for sample in range(16):
            branches = tuple(int(rng.integers(len(t.coefficients))) for t in compiled.templates)
            experiment = compiled.experiment(branches)
            widths_seen.add(tuple(c.num_qubits for c in experiment.circuits))
            if sample == 0:
                for i, circuit in enumerate(experiment.circuits):
                    with (folder / f"sample_branch_f{i}.qpy").open("wb") as stream:
                        qpy.dump(circuit, stream)
        validation = {
            "branch_configurations_checked": 16,
            "widths_seen": sorted(widths_seen),
            "all_widths_bounded": all(max(widths) <= base.max_qubits for widths in widths_seen),
            "static_allocation_is_branch_independent": True,
            "physical_gamma": compiled.gamma,
            "cost_matches_compiler": bool(
                abs(2 * np.log(compiled.gamma) - best_run.best.log_overhead) < 1e-10
            ),
            "scope": (
                "Compilation and static width check; "
                "no numerical/hardware execution of large instance."
            ),
        }
        (folder / "physical_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        exact = exact_search(graph, config, max_nodes=100000, incumbent=best_run.best)
        (folder / "exact_reference.json").write_text(
            json.dumps(
                {
                    "complete": exact.complete,
                    "nodes_visited": exact.nodes_visited,
                    "states_evaluated": exact.states_evaluated,
                    "seconds": exact.elapsed_seconds,
                    "best_overhead": exact.best.sampling_overhead if exact.best else None,
                    "scope": (
                        "Supported independent templates and two-static-block constraints only."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        paper = baselines["rows"][benchmark.paper_row]
        entry = {
            "provenance": benchmark.provenance,
            "fingerprint": benchmark.problem.circuit_fingerprint,
            "qubits": benchmark.problem.circuit.num_qubits,
            "max_qubits": benchmark.max_qubits,
            "initialization_seconds": initialization_seconds,
            "best_initial_overhead": min(e.sampling_overhead for e in seed_evaluations),
            "tuning": tuning,
            "selected_profile": selected,
            "final": final_summary,
            "best_seed": best_run.seed,
            "best_actions": best_run.best.action_counts,
            "best_component_widths": best_run.best.component_widths,
            "best_block_widths": best_run.best.block_widths,
            "comparison_without_groups": rounded_comparison(
                best_run.best.sampling_overhead, paper["paper_without_groups"]
            ),
            "comparison_with_groups": rounded_comparison(
                best_run.best.sampling_overhead, paper["paper_with_groups"]
            ),
            "exact_reference_complete": exact.complete,
            "exact_reference_overhead": exact.best.sampling_overhead if exact.best else None,
            "physical_validation": validation,
        }
        if benchmark.name.startswith("qft"):
            multi_config = replace(config, partition_mode="components")
            multi_runs = multi_start(
                graph, multi_config, seeds=tuple(range(2000, 2010)), workers=args.workers
            )
            entry["multiway_extension"] = {
                **summarize(multi_runs),
                "scope": "Relaxed fragment-count constraint; not a direct Table-I comparison.",
            }
            save_runs(folder / "multiway_runs.json", multi_runs)
        report["instances"][benchmark.name] = entry
        report["suite_wall_seconds"] = perf_counter() - suite_start
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"  final best-of-{args.final_starts}: {final_summary['best_overhead']:.6g}; "
            f"widths={best_run.best.block_widths}; exact_complete={exact.complete}",
            flush=True,
        )
    from check_sa_artifacts import check_artifacts

    check_artifacts(args.output)
    print(f"Saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
