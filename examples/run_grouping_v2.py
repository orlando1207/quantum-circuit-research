"""Source-pinned factorial test of grouping heat baths and temperature calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from run_grouping_benchmarks import save_winner, write
from run_sa_benchmarks import summarize
from summarize_multiway_results import number, table

from qpartition import (
    AnnealConfig,
    GroupingConfig,
    HybridPlan,
    PartitionGraph,
    load_problem,
    multi_start,
    save_problem,
)
from qpartition.annealing import seed_states


def render(directory, report):
    rows, arm_rows = [], []
    for name, e in report["instances"].items():
        costs = [e["arms"][arm]["reported_overhead"] for arm in report["arm_definitions"]]
        rows.append(
            [
                name,
                e["max_qubits"],
                number(e["independent_v1"]),
                number(e["grouped_v1"]),
                number(e["post_v1_mask"]["overhead"]),
                *[number(c) for c in costs],
            ]
        )
        for arm, m in e["arms"].items():
            arm_rows.append(
                [
                    name,
                    arm,
                    number(m["median_overhead"]),
                    number(m["best_seed_overhead"]),
                    number(m["best_before_polish"]),
                    m["runs_at_pooled_best"],
                    m["total_evaluations"],
                    round(m["median_seconds"], 3),
                ]
            )
    text = [
        "# Grouping v2: recognition, executable cost compression, and coupled search\n",
        "Same frozen circuits and width limits as v1. Independent and grouped v1 columns "
        "are historical references, not reruns. The four v2 arms share a LUT and initial "
        "pool; they form a 2×2 factorial experiment over coupled heat baths and "
        "feasible-only temperature calibration.\n",
        table(
            [
                "Case",
                "M",
                "Independent v1",
                "Grouped v1",
                "V2 on v1 mask",
                *report["arm_definitions"],
            ],
            rows,
        ),
        "\nReported winners include the common pool incumbent. Raw run distributions and "
        "pre-refinement costs are separate below. Every trajectory has the same configured "
        "cache-miss evaluation ceiling (including pilot and refinement); actual counts and "
        "runtimes can differ. All arms use the same seeds and per-circuit final-start count.\n",
        table(
            [
                "Case",
                "Arm",
                "Median",
                "Best seed",
                "Best before refinement",
                "Runs ≤ pool best",
                "Evaluations",
                "Median s",
            ],
            arm_rows,
        ),
        "\nThe retained best of four arms is an experimental portfolio, not a single "
        "preselected method. Compare individual arms for causal attribution. Paper Table-I "
        "grouped costs are contextual targets on unpinned author inputs; Figure 4 remains "
        "unavailable.\n",
    ]
    (directory / "SUMMARY.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/grouping_benchmarks_v2"))
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--qft6-starts", type=int, default=50)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--evaluations", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing results; choose a fresh output directory.")
    if min(args.starts, args.qft6_starts, args.steps, args.evaluations, args.workers) < 1:
        raise ValueError("Budgets must be positive.")
    args.output.mkdir(parents=True)
    oldroot = Path("validation/grouping_benchmarks_v1")
    old = json.loads((oldroot / "report.json").read_text())
    arms = {
        "standard": {},
        "feasible_temperature": {"temperature_calibration": "feasible"},
        "coupled_heatbath": {"group_heatbath_probability": 0.4},
        "enhanced": {"group_heatbath_probability": 0.4, "temperature_calibration": "feasible"},
    }
    profiles = {
        "default": {},
        "colder": {"initial_acceptance": 0.5},
        "fast_cooling": {"cooling_rate": 0.90},
    }
    report = {
        "schema_version": 2,
        "reference": "reference/2608.05287v1.pdf",
        "arguments": {**vars(args), "output": str(args.output)},
        "arm_definitions": arms,
        "profiles": profiles,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/qpartition").glob("*.py"))
        },
        "paper_table_i": old["paper_table_i"],
        "instances": {},
    }
    start = perf_counter()
    for name, prior in old["instances"].items():
        if args.only and name not in args.only:
            continue
        folder = args.output / name
        folder.mkdir()
        problem = load_problem(oldroot / name)
        save_problem(problem, folder)
        t = perf_counter()
        graph = PartitionGraph(problem, grouping=GroupingConfig())
        prep = perf_counter() - t
        graph.grouping.save(folder / "groups.json")
        config = AnnealConfig(
            prior["max_qubits"],
            partition_mode="multiway",
            steps=args.steps,
            max_evaluations=args.evaluations,
        )
        oldmask = HybridPlan.load(oldroot / name / "grouped_multiway/winner/best_plan.json").state
        independent = HybridPlan.load(oldroot / name / "no_group/winner/best_plan.json").state
        pool = tuple(dict.fromkeys((oldmask, independent, *seed_states(graph, config))))
        write(folder / "shared_initial_masks.json", pool)

        def evaluate(s):
            return graph.evaluate(s, max_qubits=config.max_qubits, partition_mode="multiway")

        incumbent = min(
            (evaluate(s) for s in pool), key=lambda e: (not e.feasible, e.log_overhead, e.state)
        )
        post = evaluate(oldmask)
        entry = {
            "fingerprint": problem.circuit_fingerprint,
            "qubits": problem.circuit.num_qubits,
            "max_qubits": config.max_qubits,
            "grouping_config": asdict(graph.grouping.config),
            "preprocessing_seconds": prep,
            "candidates": len(graph.grouping.candidates),
            "independent_v1": prior["methods"]["no_group"]["reported_overhead"],
            "grouped_v1": prior["methods"]["grouped_multiway"]["reported_overhead"],
            "post_v1_mask": {"overhead": post.sampling_overhead, "widths": post.block_widths},
            "pool_best_overhead": incumbent.sampling_overhead,
            "arms": {},
        }
        starts = args.qft6_starts if name == "qft6" else args.starts
        print(
            name,
            "preprocess",
            round(prep, 2),
            "groups",
            entry["candidates"],
            "post",
            post.sampling_overhead,
            flush=True,
        )
        for arm, parameters in arms.items():
            path = folder / arm
            path.mkdir()
            tuning = {}
            base = replace(config, **parameters)
            for profile, changes in profiles.items():
                runs = multi_start(
                    graph,
                    replace(base, **changes),
                    seeds=(100, 101, 102),
                    workers=args.workers,
                    initial_states=pool,
                )
                write(path / f"tuning_{profile}.json", [r.to_dict() for r in runs])
                tuning[profile] = summarize(runs)
            selected = min(
                profiles,
                key=lambda p: (tuning[p]["median_overhead"], tuning[p]["best_overhead"], p),
            )
            finalconfig = replace(base, **profiles[selected])
            runs = multi_start(
                graph,
                finalconfig,
                seeds=tuple(range(2000, 2000 + starts)),
                workers=args.workers,
                initial_states=pool,
            )
            write(path / "final_runs.json", [r.to_dict() for r in runs])
            result = summarize(runs)
            trajectory = min(
                (r.best for r in runs if r.best), key=lambda e: (e.log_overhead, e.state)
            )
            winner = min((incumbent, trajectory), key=lambda e: (e.log_overhead, e.state))
            result.update(
                {
                    "reported_overhead": winner.sampling_overhead,
                    "widths": winner.block_widths,
                    "selected_profile": selected,
                    "tuning": tuning,
                    "final_config": asdict(finalconfig),
                    "retained_source": "pool" if winner is incumbent else "trajectory",
                    "runs_at_pooled_best": sum(
                        r.best is not None and r.best.log_overhead <= incumbent.log_overhead + 1e-10
                        for r in runs
                    ),
                    "heatbath_improvements": sum(r.heatbath_improvements for r in runs),
                    "heatbath_proposals": sum(r.heatbath_proposals for r in runs),
                    "budget_exhausted_runs": sum(r.budget_exhausted for r in runs),
                }
            )
            result["physical_check"] = save_winner(graph, finalconfig, winner, path / "winner")
            entry["arms"][arm] = result
            print(
                name,
                arm,
                "best",
                winner.sampling_overhead,
                "median",
                result["median_overhead"],
                "pool hits",
                result["runs_at_pooled_best"],
                "/",
                starts,
                flush=True,
            )
            report["instances"][name] = entry
            report["wall_seconds"] = perf_counter() - start
            write(args.output / "report.json", report)
        render(args.output, report)
    print("Saved", args.output / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
