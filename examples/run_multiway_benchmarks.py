"""Reproducible multiway experiments on saved circuits and tighter width limits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from check_sa_artifacts import check_artifacts
from qiskit import qasm2
from run_sa_benchmarks import rounded_comparison, save_runs, summarize

from qpartition import (
    AnnealConfig,
    HybridPlan,
    PartitionGraph,
    compile_hybrid,
    exact_search,
    load_problem,
    multi_start,
    refine_partition,
    save_problem,
)
from qpartition.annealing import allowed_options, seed_states
from qpartition.multiway import bridge_seed_state


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate(graph, state, config):
    return graph.evaluate(
        state,
        max_qubits=config.max_qubits,
        partition_mode=config.partition_mode,
        max_components=config.max_components,
    )


def best_of(runs):
    return min(
        (r for r in runs if r.best),
        key=lambda r: (r.best.log_overhead, len(r.best.block_widths), r.best.state),
        default=None,
    )


def bridge_summary(graph, config):
    if config.partition_mode == "two_blocks":
        return None
    state = bridge_seed_state(
        graph, allowed_options(graph, config), max_qubits=config.max_qubits, include_gates=True
    )
    if state is None:
        return None
    evaluation = evaluate(graph, state, config)
    return {
        "state": state,
        "overhead": evaluation.sampling_overhead,
        "widths": evaluation.block_widths,
        "actions": evaluation.action_counts,
        "optimality_scope": "Exact only among partitions cutting available gate/wire bridges "
        "of the uncut graph.",
    }


def tune_capacity(graph, base, profiles, folder, args, prefix=""):
    init_start = perf_counter()
    pool = seed_states(graph, base)
    initialization_seconds = perf_counter() - init_start
    write_json(
        folder / f"{prefix}initialization.json",
        {
            "states": pool,
            "seconds": initialization_seconds,
            "bridge_seed": bridge_summary(graph, base),
        },
    )
    tuning = {}
    for name, parameters in profiles.items():
        config = replace(base, **parameters)
        runs = multi_start(
            graph,
            config,
            seeds=tuple(range(4000, 4000 + args.sweep_tuning_starts)),
            initial_states=pool,
            workers=args.workers,
        )
        save_runs(folder / f"{prefix}tuning_{name}.json", runs)
        tuning[name] = summarize(runs)
    selected = min(
        profiles, key=lambda p: (tuning[p]["best_overhead"], tuning[p]["median_overhead"], p)
    )
    config = replace(base, **profiles[selected])
    runs = multi_start(
        graph,
        config,
        seeds=tuple(range(3000, 3000 + args.sweep_starts)),
        initial_states=pool,
        workers=args.workers,
    )
    save_runs(folder / ("two_way_runs.json" if prefix else "final_runs.json"), runs)
    return config, runs, tuning, selected


def save_winner(graph, config, best, folder):
    plan = HybridPlan.create(
        graph,
        best.state,
        max_qubits=config.max_qubits,
        partition_mode=config.partition_mode,
        max_components=config.max_components,
    )
    plan.save(folder / "best_plan.json")
    save_problem(graph.problem, folder)
    qasm2.dump(graph.problem.circuit, folder / "source.qasm")
    compiled = compile_hybrid(graph, plan)
    rng = np.random.default_rng(331)
    widths_seen = set()
    for _ in range(8):
        branches = tuple(int(rng.integers(len(t.coefficients))) for t in compiled.templates)
        widths = tuple(c.num_qubits for c in compiled.experiment(branches).circuits)
        assert widths == best.block_widths and max(widths) <= config.max_qubits
        widths_seen.add(widths)
    assert np.isclose(2 * np.log(compiled.gamma), best.log_overhead, atol=1e-10, rtol=0)
    validation = {
        "branch_configurations_checked": 8,
        "widths_seen": sorted(widths_seen),
        "all_widths_bounded": True,
        "cost_matches_compiler": True,
        "static_segments": best.segment_count,
        "conservation": "n + number of wire cuts = total static registers",
        "scope": "Physical compilation and norms; no global numerical branch enumeration.",
    }
    write_json(folder / "physical_validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--output", type=Path, default=Path("validation/multiway_benchmarks_v3"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--tuning-starts", type=int, default=4)
    parser.add_argument("--final-starts", type=int, default=50)
    parser.add_argument("--ablation-starts", type=int, default=8)
    parser.add_argument("--sweep-starts", type=int, default=6)
    parser.add_argument("--sweep-tuning-starts", type=int, default=2)
    parser.add_argument("--sweep-steps", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory to preserve previous experiments.")
    if (
        min(
            args.steps,
            args.tuning_starts,
            args.final_starts,
            args.ablation_starts,
            args.sweep_starts,
            args.sweep_tuning_starts,
            args.sweep_steps,
            args.workers,
        )
        < 1
    ):
        parser.error("All experiment budgets must be positive.")
    source = json.loads((args.source / "report.json").read_text())
    baselines = json.loads(Path("benchmarks/table_i_baselines.json").read_text())
    args.output.mkdir(parents=True)
    profiles = {
        "local": {"block_probability": 0.0, "interval_polish": False},
        "intervals": {"block_probability": 0.45, "block_size": 4, "split_probability": 0.2},
        "wide": {"block_probability": 0.65, "block_size": 8, "split_probability": 0.35},
        "cold": {
            "initial_acceptance": 0.55,
            "block_probability": 0.45,
            "block_size": 6,
            "split_probability": 0.15,
        },
        "actions": {"search_space": "actions", "interval_polish": False},
    }
    report = {
        "schema_version": 3,
        "graph_schema": "temporal-incidence-v2",
        "core_actions": ["gate", "wire"],
        "constraints": "One executable fragment per connected component; width <= M; "
        "unbounded fragment count; static registers; no grouping/excision.",
        "selection_protocol": "Five profiles, separate tuning/final seeds, warm-start pool "
        "includes the historical two-way winner. Additional computation "
        "over v2; not an equal-total-budget comparison with the paper.",
        "comparison_scope": source["comparison_scope"],
        "arguments": {**vars(args), "source": str(args.source), "output": str(args.output)},
        "source_report_sha256": hashlib.sha256(
            (args.source / "report.json").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_pdf_sha256": hashlib.sha256(Path(baselines["source"]).read_bytes()).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/qpartition").glob("*.py"))
        },
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("qiskit", "qiskit-aer", "qiskit-addon-cutting", "numpy", "networkx")
        },
        "profiles": profiles,
        "instances": {},
        "paper_table_i": baselines["rows"],
    }
    started = perf_counter()
    for name, old_entry in source["instances"].items():
        if args.only and name not in args.only:
            continue
        print(f"Preparing {name}", flush=True)
        folder = args.output / name
        folder.mkdir()
        graph = PartitionGraph(load_problem(args.source / name))
        previous = HybridPlan.load(args.source / name / "best_plan.json")
        previous.validate(graph)
        base = AnnealConfig(
            previous.max_fragment_qubits, partition_mode="multiway", steps=args.steps
        )
        init_start = perf_counter()
        fresh = seed_states(graph, base)
        pool = tuple(dict.fromkeys((previous.state, *fresh)))
        seed_evaluations = [evaluate(graph, state, base) for state in pool]
        seed_best = min((e for e in seed_evaluations if e.feasible), key=lambda e: e.log_overhead)
        initialization_seconds = perf_counter() - init_start
        deterministic = refine_partition(graph, base, seed_best.state)
        write_json(
            folder / "initialization.json",
            {
                "seconds": initialization_seconds,
                "states": pool,
                "overheads": [e.sampling_overhead for e in seed_evaluations],
                "widths": [e.block_widths for e in seed_evaluations],
                "historical_two_way_overhead": old_entry["final"]["best_overhead"],
                "fresh_best_overhead": min(
                    evaluate(graph, s, base).sampling_overhead for s in fresh
                ),
                "bridge_seed": bridge_summary(graph, base),
            },
        )
        write_json(
            folder / "deterministic_refinement.json",
            {
                "overhead": deterministic.best.sampling_overhead,
                "state": deterministic.best.state,
                "widths": deterministic.best.block_widths,
                "actions": deterministic.best.action_counts,
                "evaluations": deterministic.evaluations,
                "seconds": deterministic.elapsed_seconds,
            },
        )
        # Every final trajectory can retain the deterministic incumbent; report
        # this initializer contribution explicitly instead of attributing it to SA.
        pool = tuple(dict.fromkeys((deterministic.best.state, *pool)))
        tuning = {}
        for profile, parameters in profiles.items():
            config = replace(base, **parameters)
            runs = multi_start(
                graph,
                config,
                seeds=tuple(range(args.tuning_starts)),
                initial_states=pool,
                workers=args.workers,
            )
            save_runs(folder / f"tuning_{profile}.json", runs)
            tuning[profile] = summarize(runs)
            print(f"  {profile}: {tuning[profile]['best_overhead']:.6g}", flush=True)
        selected = min(
            profiles, key=lambda p: (tuning[p]["best_overhead"], tuning[p]["median_overhead"], p)
        )
        config = replace(base, **profiles[selected])
        final_runs = multi_start(
            graph,
            config,
            seeds=tuple(range(1000, 1000 + args.final_starts)),
            initial_states=pool,
            workers=args.workers,
        )
        save_runs(folder / "final_runs.json", final_runs)
        winner = best_of(final_runs)
        best = winner.best
        assert best.log_overhead <= deterministic.best.log_overhead + 1e-10
        validation = save_winner(graph, config, best, folder)
        ablations = {}
        for mode, flags in {
            "gate_only": {"allow_wire": False},
            "wire_only": {"allow_gate": False},
        }.items():
            ablation_config = replace(config, **flags)
            runs = multi_start(
                graph,
                ablation_config,
                seeds=tuple(range(2000, 2000 + args.ablation_starts)),
                workers=args.workers,
            )
            save_runs(folder / f"ablation_{mode}.json", runs)
            ablations[mode] = summarize(runs)
        exact = exact_search(graph, config, max_nodes=100000, incumbent=best)
        write_json(
            folder / "exact_reference.json",
            {
                "complete": exact.complete,
                "nodes_visited": exact.nodes_visited,
                "states_evaluated": exact.states_evaluated,
                "seconds": exact.elapsed_seconds,
                "best_overhead": exact.best.sampling_overhead,
            },
        )
        paper = baselines["rows"].get(name, {})
        entry = {
            "provenance": old_entry["provenance"],
            "fingerprint": graph.problem.circuit_fingerprint,
            "qubits": graph.problem.circuit.num_qubits,
            "two_qubit_gates": len(graph.gate_variables),
            "max_qubits": config.max_qubits,
            "initialization_seconds": initialization_seconds,
            "historical_two_way_overhead": old_entry["final"]["best_overhead"],
            "historical_two_way_widths": old_entry["best_block_widths"],
            "best_initial_overhead": seed_best.sampling_overhead,
            "deterministic_refinement_overhead": deterministic.best.sampling_overhead,
            "tuning": tuning,
            "selected_profile": selected,
            "final": summarize(final_runs),
            "best_seed": winner.seed,
            "best_actions": best.action_counts,
            "best_block_widths": best.block_widths,
            "best_component_widths": best.component_widths,
            "best_fragments": len(best.block_widths),
            "fragment_shot_factor": len(best.block_widths) * best.sampling_overhead,
            "ablations": ablations,
            "physical_validation": validation,
            "comparison_without_groups": rounded_comparison(
                best.sampling_overhead, paper.get("paper_without_groups")
            ),
            "comparison_with_groups": rounded_comparison(
                best.sampling_overhead, paper.get("paper_with_groups")
            ),
            "exact_reference_complete": exact.complete,
        }
        report["instances"][name] = entry
        report["suite_wall_seconds"] = perf_counter() - started
        write_json(args.output / "report.json", report)
        print(
            f"  BEST {best.sampling_overhead:.6g}; widths={best.block_widths}; "
            f"cuts={best.action_counts}; selected={selected}",
            flush=True,
        )
    check_artifacts(args.output)

    sweep_root = args.output / "capacity_sweep"
    sweep_root.mkdir()
    sweep = {
        "schema_version": 3,
        "description": "Additional capacities; not Table-I comparisons. "
        "Two-way and multiway use matched fresh seeds/steps, without historical warm starts.",
        "instances": {},
    }
    for name, entry in report["instances"].items():
        graph = PartitionGraph(load_problem(args.source / name))
        n = graph.problem.circuit.num_qubits
        capacities = {max(2, n // 2), max(2, n // 3)} - {entry["max_qubits"]}
        if name == "chain10":
            capacities.add(6)
        for limit in sorted(capacities, reverse=True):
            key = f"{name}_M{limit}"
            folder = sweep_root / key
            folder.mkdir()
            base = AnnealConfig(limit, partition_mode="multiway", steps=args.sweep_steps)
            config, runs, tuning, selected = tune_capacity(graph, base, profiles, folder, args)
            winner = best_of(runs)
            assert winner is not None, key
            best = winner.best
            validation = save_winner(graph, config, best, folder)
            two = None
            two_tuning = None
            two_selected = None
            if n <= 2 * limit:
                _, two_runs, two_tuning, two_selected = tune_capacity(
                    graph,
                    replace(base, partition_mode="two_blocks"),
                    profiles,
                    folder,
                    args,
                    "two_way_",
                )
                two = summarize(two_runs)
            sweep["instances"][key] = {
                "source_benchmark": name,
                "qubits": n,
                "max_qubits": limit,
                "fingerprint": graph.problem.circuit_fingerprint,
                "final": summarize(runs),
                "best_actions": best.action_counts,
                "best_block_widths": best.block_widths,
                "best_fragments": len(best.block_widths),
                "two_way": two,
                "tuning": tuning,
                "selected_profile": selected,
                "two_way_tuning": two_tuning,
                "two_way_selected_profile": two_selected,
                "bridge_seed": bridge_summary(graph, config),
                "two_way_proven_infeasible": n > 2 * limit,
                "physical_validation": validation,
            }
            write_json(sweep_root / "report.json", sweep)
            binary_cost = two["best_overhead"] if two else "infeasible"
            print(
                f"Sweep {key}: {best.sampling_overhead:.6g}; widths={best.block_widths}; "
                f"two-way={binary_cost}",
                flush=True,
            )
    check_artifacts(sweep_root)
    report = json.loads((args.output / "report.json").read_text())
    report["suite_wall_seconds"] = perf_counter() - started
    report["capacity_sweep"] = str(sweep_root / "report.json")
    write_json(args.output / "report.json", report)
    print(f"Saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
