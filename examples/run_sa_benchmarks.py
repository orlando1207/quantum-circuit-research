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
    refine_partition,
    save_problem,
)
from qpartition.annealing import seed_states
from qpartition.benchmarks import extended_suite


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
            run.initial_feasible and run.best.log_overhead < run.initial_log_overhead - 1e-10
            for run in feasible
        ),
        "best_seed_overhead": min(np.exp(run.initial_log_overhead) for run in runs),
        "best_before_polish": min(
            run.best_before_polish for run in feasible if run.best_before_polish is not None
        )
        if feasible
        else None,
        "polish_evaluations": sum(run.polish_evaluations for run in runs),
        "feasible_from_infeasible_seed": sum(not run.initial_feasible for run in feasible),
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
    parser.add_argument("--output", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--tuning-starts", type=int, default=6)
    parser.add_argument("--final-starts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ablation-starts", type=int, default=8)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if (args.output / "report.json").exists():
        raise ValueError("Use a fresh output directory; archived experiments are not overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    baselines = json.loads(Path("benchmarks/table_i_baselines.json").read_text())
    profiles = {
        "actions_uniform": {"search_space": "actions", "interval_polish": False},
        "actions_type_balanced": {
            "search_space": "actions",
            "interval_polish": False,
            "proposal_balance": "types",
        },
        "labels_single": {"block_probability": 0.0, "interval_polish": False},
        "labels_intervals": {"block_probability": 0.35, "block_size": 4},
        "labels_wide": {"block_probability": 0.6, "block_size": 8},
        "labels_cold": {"initial_acceptance": 0.55, "block_probability": 0.35},
    }
    report = {
        "schema_version": 2,
        "graph_schema": "temporal-incidence-v2",
        "core_actions": ["gate", "wire"],
        "source_pdf_sha256": hashlib.sha256(Path(baselines["source"]).read_bytes()).hexdigest(),
        "constraints": "two static executable blocks; M=floor(3*n/4); no grouping/excision",
        "comparison_scope": (
            "QFT/QAOA count/spatial-cost matched reconstructions; chain structural proxies; "
            "literal Fig.2 and own diagnostic controls. "
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
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("src/qpartition").glob("*.py"))
        },
        "instances": {},
    }
    suite_start = perf_counter()
    for benchmark in extended_suite():
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
        gate_init_start = perf_counter()
        gate_pool = seed_states(graph, replace(base, allow_wire=False))
        gate_initialization_seconds = perf_counter() - gate_init_start
        gate_seed = graph.evaluate(gate_pool[0], max_qubits=base.max_qubits)
        deterministic = refine_partition(graph, base, pool[0])
        (folder / "deterministic_refinement.json").write_text(
            json.dumps(
                {
                    "overhead": deterministic.best.sampling_overhead,
                    "evaluations": deterministic.evaluations,
                    "seconds": deterministic.elapsed_seconds,
                    "state": deterministic.best.state,
                    "actions": deterministic.best.action_counts,
                },
                indent=2,
            )
            + "\n"
        )
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
        selected = min(
            profiles,
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
        ablations = {}
        for mode, flags in {
            "gate_only": {"allow_wire": False},
            "wire_only": {"allow_gate": False},
        }.items():
            ablation_config = replace(base, **flags)
            runs = multi_start(
                graph,
                ablation_config,
                seeds=tuple(range(2000, 2000 + args.ablation_starts)),
                workers=args.workers,
            )
            ablations[mode] = summarize(runs)
            save_runs(folder / f"ablation_{mode}.json", runs)
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
        paper = baselines["rows"].get(benchmark.paper_row, {})
        entry = {
            "provenance": benchmark.provenance,
            "fingerprint": benchmark.problem.circuit_fingerprint,
            "qubits": benchmark.problem.circuit.num_qubits,
            "max_qubits": benchmark.max_qubits,
            "initialization_seconds": initialization_seconds,
            "best_initial_overhead": min(e.sampling_overhead for e in seed_evaluations),
            "gate_only_seed_overhead": gate_seed.sampling_overhead,
            "gate_seed_initialization_seconds": gate_initialization_seconds,
            "deterministic_refinement_overhead": deterministic.best.sampling_overhead,
            "deterministic_refinement_evaluations": deterministic.evaluations,
            "ablations": ablations,
            "graph_size": {
                "vertices": len(graph.nodes),
                "edges": len(graph.edges),
                "variables": len(graph.variables),
                "legacy_vertices": 2 * benchmark.problem.circuit.num_qubits
                + sum(
                    2 * item.operation.num_qubits
                    for item in benchmark.problem.circuit.data
                    if item.operation.name != "barrier"
                ),
            },
            "tuning": tuning,
            "selected_profile": selected,
            "final": final_summary,
            "best_seed": best_run.seed,
            "best_actions": best_run.best.action_counts,
            "best_component_widths": best_run.best.component_widths,
            "best_block_widths": best_run.best.block_widths,
            "comparison_without_groups": rounded_comparison(
                best_run.best.sampling_overhead, paper.get("paper_without_groups")
            ),
            "comparison_with_groups": rounded_comparison(
                best_run.best.sampling_overhead, paper.get("paper_with_groups")
            ),
            "exact_reference_complete": exact.complete,
            "exact_reference_overhead": exact.best.sampling_overhead if exact.best else None,
            "physical_validation": validation,
        }
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
