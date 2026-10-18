"""Matched-budget, source-pinned grouping-aware SA experiments at M=floor(3N/4)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from run_sa_benchmarks import rounded_comparison, summarize
from summarize_multiway_results import number, table

from qpartition import (
    AnnealConfig,
    GroupingConfig,
    HybridPlan,
    PartitionGraph,
    compile_hybrid,
    load_problem,
    multi_start,
    save_problem,
)
from qpartition.annealing import seed_states


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def historical(name):
    if name == "grover4_historical":
        p = Path("validation/external_baselines_v1/grover4_historical_M3/sa_fresh_winner.json")
        value = json.loads(p.read_text())
        return tuple(value["state"]), value["sampling_overhead"], str(p)
    folder = Path("validation/multiway_benchmarks_v3") / name
    plan = HybridPlan.load(folder / "best_plan.json")
    report = json.loads(Path("validation/multiway_benchmarks_v3/report.json").read_text())
    return plan.state, report["instances"][name]["final"]["best_overhead"], str(folder)


def save_winner(graph, config, evaluation, folder):
    folder.mkdir()
    plan = HybridPlan.create(
        graph,
        evaluation.state,
        max_qubits=config.max_qubits,
        partition_mode=config.partition_mode,
        max_components=config.max_components,
    )
    plan.save(folder / "best_plan.json")
    compiled = compile_hybrid(graph, plan)
    rng = np.random.default_rng(845)
    checks = []
    for _ in range(4):
        branches = tuple(int(rng.integers(len(t.coefficients))) for t in compiled.templates)
        experiment = compiled.experiment(branches)
        widths = tuple(c.num_qubits for c in experiment.circuits)
        assert widths == evaluation.block_widths
        assert max(widths) <= config.max_qubits
        checks.append({"branches": branches, "widths": widths})
    assert np.isclose(compiled.gamma**2, evaluation.sampling_overhead, rtol=1e-10)
    assert (
        sum(evaluation.block_widths)
        == graph.problem.circuit.num_qubits + evaluation.action_counts["wire"]
    )
    check = {
        "status": "passed",
        "compiler_gamma": compiled.gamma,
        "compiled_overhead": compiled.gamma**2,
        "widths": evaluation.block_widths,
        "groups": [asdict(g) for g in evaluation.groups],
        "selected_group_selection_complete": evaluation.grouping_selection_complete,
        "branch_checks": checks,
        "joint_qpd_unitary_residuals": {
            g.group_id: getattr(graph.grouping.basis(g), "unitary_residual", None)
            for g in evaluation.groups
        }
        if graph.grouping is not None
        else {},
        "scope": "Executable local branch construction and coefficient norm; "
        "no global exponentially sized experiment family enumerated",
    }
    write(folder / "physical_check.json", check)
    return check


def render(directory, report):
    paper_rows, all_rows = [], []
    tied = sum(
        np.isclose(
            e["methods"]["grouped_multiway"]["reported_overhead"],
            e["methods"]["cost_only"]["reported_overhead"],
            rtol=1e-10,
        )
        for e in report["instances"].values()
    )
    for name, row in report["paper_table_i"].items():
        key = "grover4_historical" if name == "grover4" else name
        e = report["instances"].get(key)
        methods = e["methods"] if e else {}
        costs = [
            methods.get(m, {}).get("reported_overhead")
            for m in ("no_group", "cost_only", "grouped_two_blocks", "grouped_multiway")
        ]
        paper_rows.append(
            [
                name,
                row["max_qubits"],
                number(row["paper_without_groups"]),
                number(row["paper_with_groups"]),
                number(e["historical_overhead"]) if e else "—",
                *[number(c) for c in costs],
                methods.get("grouped_multiway", {}).get("winner_widths", "—"),
            ]
        )
    for name, e in report["instances"].items():
        for method, m in e["methods"].items():
            all_rows.append(
                {
                    "benchmark": name,
                    "qubits": e["qubits"],
                    "max_qubits": e["max_qubits"],
                    "method": method,
                    "historical_no_GG": e["historical_overhead"],
                    "trajectory_best": m["best_overhead"],
                    "retained_best": m["reported_overhead"],
                    "retained_source": m["retained_source"],
                    "post_only_overhead": e["post_only"]["sampling_overhead"],
                    "best_seed_overhead": m["best_seed_overhead"],
                    "best_before_polish": m["best_before_polish"],
                    "median": m["median_overhead"],
                    "fragments": len(m["winner_widths"]),
                    "widths": json.dumps(m["winner_widths"]),
                    "groups": len(m["winner_groups"]),
                    "selected_profile": m["selected_profile"],
                    "source_fingerprint": e["fingerprint"],
                }
            )
    text = [
        "# Grouping-aware SA: recorded comparison\n",
        "Primary reference: supplied arXiv:2608.05287v1, Table I. All costs are "
        "squared coefficient norms. Original manual group manifests and exact source "
        "circuits remain unavailable; these are identified reconstructions/proxies.\n",
        f"Each arm gets {report['arguments']['starts']} final starts and identical "
        f"{report['arguments']['steps']}-step trajectory budgets. Tuning starts, shared "
        "initial masks, historical warm starts, physical checks, and all raw outcomes "
        "are recorded. Runtime/evaluation counts are not matched across different pools.\n",
        table(
            [
                "Case",
                "M",
                "Paper no GG",
                "Paper GG",
                "Previous no GG",
                "Fresh no GG",
                "GG cost only",
                "GG two blocks",
                "GG multiway",
                "Multiway widths",
            ],
            paper_rows,
        ),
        "\nReported winners retain feasible historical/post-only candidates as well as "
        "trajectory winners. Raw trajectory costs are separate in all_results.csv. "
        "Two-block and multiway columns have different fragment-count constraints; "
        "every fragment obeys the same M. QFT12 GG has no reported paper target. "
        "Figure 4 is unavailable as a complete circuit and is not replaced by Figure 2.\n",
        "## All implemented benchmarks\n",
        table(
            [
                "Case",
                "M",
                "Previous",
                "No GG",
                "Post-only GG",
                "GG cost only",
                "GG two blocks",
                "GG multiway",
            ],
            [
                [
                    name,
                    e["max_qubits"],
                    number(e["historical_overhead"]),
                    number(e["methods"]["no_group"]["reported_overhead"]),
                    number(e["post_only"]["sampling_overhead"]),
                    *[
                        number(e["methods"][m]["reported_overhead"])
                        for m in ("cost_only", "grouped_two_blocks", "grouped_multiway")
                    ],
                ]
                for name, e in report["instances"].items()
            ],
        ),
        "\nAutomatic bounded groups include fusion, shared-control/target CX or CZ "
        "cascades, parallel gates, commuting rotation packets, and general Clifford "
        "packets, plus bounded known-local non-Clifford packets with explicitly "
        "computed achievable coefficient norms. Intervening local operations are "
        "checked or explicitly absorbed. "
        "Selected groups never double count source operations. Their joint local "
        "register bundles are counted in physical widths. Group selection is exact "
        "only when the recorded eligible-placement limit permits exhaustive selection; "
        "larger selectors use two deterministic greedy orders. No global group "
        "discovery or partition optimality claim is made.\n",
        "## Attribution and limits\n",
        "The CSV separates re-costing the historical mask, the best initial seed, "
        "the best annealing result before refinement, and the retained result. "
        "These are aggregate minima, not necessarily one trajectory.\n",
        table(
            ["Case", "Post-only", "Best GG seed", "GG before refinement", "GG final"],
            [
                [
                    name,
                    number(e["post_only"]["sampling_overhead"]),
                    *[
                        number(e["methods"]["grouped_multiway"][key])
                        for key in ("best_seed_overhead", "best_before_polish", "reported_overhead")
                    ],
                ]
                for name, e in report["instances"].items()
            ],
        ),
        f"\nGroup-aware moves and cost-only grouped multiway moves tie on "
        f"{tied}/{len(report['instances'])} retained best costs. These ties do not "
        "demonstrate a best-cost advantage for the group move family. The "
        "two-block arm also has a different proposal kernel, so it is not a "
        "controlled fragment-count-only ablation. Distinguish gains available "
        "through re-costing and initialization from gains during annealing and "
        "refinement using the stage columns. Source identity and missing manual "
        "group manifests prevent an exact paper reproduction claim.\n",
    ]
    (directory / "SUMMARY.md").write_text("\n".join(text))
    with (directory / "all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/grouping_benchmarks_v1"))
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--tuning-starts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous results; choose a fresh output directory.")
    if min(args.starts, args.tuning_starts, args.steps, args.workers) < 1:
        raise ValueError("Budgets must be positive.")
    args.output.mkdir(parents=True)
    old = json.loads(Path("validation/sa_benchmarks_v2/report.json").read_text())
    rows = [
        (name, Path("validation/sa_benchmarks_v2") / name, e["provenance"])
        for name, e in old["instances"].items()
    ]
    ext = Path("validation/external_baselines_v1")
    external = json.loads((ext / "report.json").read_text())
    rows.append(
        (
            "grover4_historical",
            ext / "grover4_historical_M3",
            external["instances"]["grover4_historical_M3"]["provenance"],
        )
    )
    report = {
        "schema_version": 1,
        "primary_reference": "reference/2608.05287v1.pdf",
        "paper_table_i": json.loads(Path("benchmarks/table_i_baselines.json").read_text())["rows"],
        "arguments": {**vars(args), "output": str(args.output)},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/qpartition").glob("*.py"))
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("qiskit", "qiskit-addon-cutting", "networkx", "numpy")
        },
        "instances": {},
    }
    profiles = {
        "intervals": {},
        "wider": {"block_probability": 0.6, "block_size": 6},
        "colder": {"initial_acceptance": 0.5},
    }
    started = perf_counter()
    for name, source, provenance in rows:
        if args.only and name not in args.only:
            continue
        folder = args.output / name
        folder.mkdir()
        problem = load_problem(source)
        save_problem(problem, folder)
        n = problem.circuit.num_qubits
        limit = 3 * n // 4
        baseline = PartitionGraph(problem)
        preprocessing_start = perf_counter()
        grouped = PartitionGraph(problem, grouping=GroupingConfig())
        cost_only = PartitionGraph(problem, grouping=GroupingConfig(group_moves=False))
        preprocess_seconds = perf_counter() - preprocessing_start
        grouped.grouping.save(folder / "groups.json")
        previous_state, previous_cost, previous_path = historical(name)
        base = AnnealConfig(limit, partition_mode="multiway", steps=args.steps)
        pool = tuple(
            dict.fromkeys(
                (previous_state, *seed_states(baseline, base), *seed_states(grouped, base))
            )
        )
        write(folder / "shared_initial_masks.json", pool)
        previous = baseline.evaluate(previous_state, max_qubits=limit, partition_mode="multiway")
        assert np.isclose(previous.sampling_overhead, previous_cost, rtol=1e-10)
        post_only = grouped.evaluate(previous_state, max_qubits=limit, partition_mode="multiway")
        assert post_only.feasible and post_only.sampling_overhead <= previous_cost * (1 + 1e-10)
        entry = {
            "qubits": n,
            "max_qubits": limit,
            "fingerprint": problem.circuit_fingerprint,
            "provenance": provenance,
            "historical_overhead": previous_cost,
            "historical_path": previous_path,
            "group_count": len(grouped.grouping.candidates),
            "preprocessing_seconds": preprocess_seconds,
            "grouping_config": asdict(grouped.grouping.config),
            "post_only": {
                "sampling_overhead": post_only.sampling_overhead,
                "widths": post_only.block_widths,
                "groups": [asdict(g) for g in post_only.groups],
            },
            "methods": {},
        }
        arms = {
            "no_group": (baseline, base),
            "cost_only": (cost_only, base),
            "grouped_two_blocks": (grouped, replace(base, partition_mode="two_blocks")),
            "grouped_multiway": (grouped, base),
        }
        print(
            name,
            "groups",
            entry["group_count"],
            "post-only",
            post_only.sampling_overhead,
            flush=True,
        )
        for method, (graph, config) in arms.items():
            method_folder = folder / method
            method_folder.mkdir()
            tuning = {}
            for profile, parameters in profiles.items():
                runs = multi_start(
                    graph,
                    replace(config, **parameters),
                    seeds=tuple(range(100, 100 + args.tuning_starts)),
                    workers=args.workers,
                    initial_states=pool,
                )
                write(method_folder / f"tuning_{profile}.json", [r.to_dict() for r in runs])
                tuning[profile] = summarize(runs)
            profile = min(
                profiles,
                key=lambda p: (tuning[p]["best_overhead"], tuning[p]["median_overhead"], p),
            )
            selected_config = replace(config, **profiles[profile])
            runs = multi_start(
                graph,
                selected_config,
                seeds=tuple(range(2000, 2000 + args.starts)),
                workers=args.workers,
                initial_states=pool,
            )
            write(method_folder / "final_runs.json", [r.to_dict() for r in runs])
            summary = summarize(runs)
            trajectory = min(
                (r.best for r in runs if r.best),
                key=lambda e: (e.log_overhead, len(e.block_widths), e.state),
            )
            retained = graph.evaluate(
                previous_state, max_qubits=limit, partition_mode=config.partition_mode
            )
            winner = min(
                (retained, trajectory),
                key=lambda e: (not e.feasible, e.log_overhead, len(e.block_widths), e.state),
            )
            summary.update(
                {
                    "reported_overhead": winner.sampling_overhead,
                    "retained_source": "historical/post-only"
                    if winner is retained
                    else "trajectory",
                    "selected_profile": profile,
                    "tuning": tuning,
                    "winner_widths": winner.block_widths,
                    "winner_groups": [asdict(g) for g in winner.groups],
                    "winner_actions": winner.action_counts,
                }
            )
            summary["physical_check"] = save_winner(
                graph, selected_config, winner, method_folder / "winner"
            )
            paper_key = "grover4" if name == "grover4_historical" else name
            target = report["paper_table_i"].get(paper_key)
            if target:
                summary["paper_GG_comparison"] = rounded_comparison(
                    winner.sampling_overhead, target["paper_with_groups"]
                )
            entry["methods"][method] = summary
            print(
                name,
                method,
                summary["reported_overhead"],
                winner.block_widths,
                "groups",
                len(winner.groups),
                flush=True,
            )
        report["instances"][name] = entry
        report["wall_seconds"] = perf_counter() - started
        write(args.output / "report.json", report)
        render(args.output, report)
    print(f"Saved {args.output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
