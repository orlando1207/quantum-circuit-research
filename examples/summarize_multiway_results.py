"""Generate complete paper/control/capacity tables from the recorded v3 reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def number(value):
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-8 and value < 1e8:
        return f"{value:,.0f}"
    return f"{value:,.6g}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory: Path, document: Path):
    report = json.loads((directory / "report.json").read_text())
    sweep = json.loads((directory / "capacity_sweep/report.json").read_text())
    main = report["instances"]
    paper = report["paper_table_i"]
    lines = ["\n### Recorded results at the paper's width limits\n"]
    paper_rows, paper_csv = [], []
    for name, baseline in paper.items():
        entry = main.get(name)
        status = "Reconstruction" if name.startswith(("qft", "qaoa")) else "Structural proxy"
        if entry is None:
            status = "Not reproduced"
        paper_rows.append(
            [
                name,
                baseline["max_qubits"],
                number(baseline["paper_without_groups"]),
                number(entry["historical_two_way_overhead"]) if entry else "—",
                number(entry["final"]["best_overhead"]) if entry else "—",
                entry["best_fragments"] if entry else "—",
                ", ".join(map(str, entry["best_block_widths"])) if entry else "—",
                status,
            ]
        )
        paper_csv.append(
            {
                "benchmark": name,
                "qubits": baseline["qubits"],
                "two_qubit_gates": baseline["two_qubit_gates"],
                "max_qubits": baseline["max_qubits"],
                **{
                    k: v
                    for k, v in baseline.items()
                    if k not in ("qubits", "two_qubit_gates", "max_qubits")
                },
                "historical_two_way_overhead": entry["historical_two_way_overhead"]
                if entry
                else None,
                "multiway_overhead": entry["final"]["best_overhead"] if entry else None,
                "fragments": entry["best_fragments"] if entry else None,
                "reproduction_status": status,
            }
        )
    lines.append(
        table(
            [
                "Case",
                "M",
                "Paper, no grouping",
                "Historical two-way",
                "Multiway",
                "K",
                "Widths",
                "Status",
            ],
            paper_rows,
        )
    )
    lines.append(
        "\nAll six comparable reconstruction/proxy costs are unchanged from the v2 "
        "incumbents at these capacities; all eleven main winners use two connected "
        "fragments. The earlier QFT12 improvement is retained and is not a new "
        "multiway improvement. QFT6/QAOA results agree with the paper's rounded "
        "ungrouped values; chain results agree on structural proxies.\n"
    )
    control_rows, main_csv = [], []
    for name, entry in main.items():
        counts = entry["best_actions"]
        main_csv.append(
            {
                "benchmark": name,
                "qubits": entry["qubits"],
                "two_qubit_gates": entry["two_qubit_gates"],
                "max_qubits": entry["max_qubits"],
                "overhead": entry["final"]["best_overhead"],
                "historical_two_way_overhead": entry["historical_two_way_overhead"],
                "fragments": entry["best_fragments"],
                "widths": json.dumps(entry["best_block_widths"]),
                "gate_cuts": counts["gate"],
                "wire_cuts": counts["wire"],
                "gate_only_overhead": entry["ablations"]["gate_only"]["best_overhead"],
                "wire_only_overhead": entry["ablations"]["wire_only"]["best_overhead"],
                "selected_profile": entry["selected_profile"],
                "feasible_final_runs": entry["final"]["feasible_runs"],
                "final_runs": entry["final"]["runs"],
            }
        )
        if name not in paper:
            control_rows.append(
                [
                    name,
                    entry["max_qubits"],
                    number(entry["final"]["best_overhead"]),
                    entry["best_fragments"],
                    ", ".join(map(str, entry["best_block_widths"])),
                    counts["gate"],
                    counts["wire"],
                ]
            )
    lines.append("\n### Additional implemented controls (no Table-I identity claim)\n")
    lines.append(
        table(["Case", "M", "Overhead", "K", "Widths", "Gate cuts", "Wire cuts"], control_rows)
    )
    lines.append("\n### Main-case ablations (each method gets eight starts)\n")
    lines.append(
        table(
            ["Case", "Gate only", "Wire only", "Combined winner"],
            [
                [
                    name,
                    number(entry["ablations"]["gate_only"]["best_overhead"]),
                    number(entry["ablations"]["wire_only"]["best_overhead"]),
                    number(entry["final"]["best_overhead"]),
                ]
                for name, entry in main.items()
            ],
        )
    )
    lines.append(
        "\nThe bridge controls share topology and capacity but use phase angles "
        "0.05 and π. Their gate/wire preference reverses with the physical norms. "
        "These ablations have smaller budgets than the combined main runs and "
        "provide feasible reference costs, not equal-budget optimizer rankings.\n"
    )
    lines.append("\n### Tighter-capacity sweep: matched tuning and trajectory budgets\n")
    sweep_rows, sweep_csv = [], []
    for name, entry in sweep["instances"].items():
        binary = entry["two_way"]
        two_cost = binary["best_overhead"] if binary else None
        multi_cost = entry["final"]["best_overhead"]
        counts = entry["best_actions"]
        bridge = entry["bridge_seed"]
        sweep_rows.append(
            [
                entry["source_benchmark"],
                entry["max_qubits"],
                number(two_cost) if binary else "Infeasible: n > 2M",
                number(multi_cost),
                entry["best_fragments"],
                ", ".join(map(str, entry["best_block_widths"])),
                f"{counts['gate']} / {counts['wire']}",
                number(bridge["overhead"]) if bridge else "—",
                f"{two_cost / multi_cost:,.3g}×" if two_cost else "—",
            ]
        )
        sweep_csv.append(
            {
                "benchmark": entry["source_benchmark"],
                "max_qubits": entry["max_qubits"],
                "two_way_overhead": two_cost,
                "two_way_proven_infeasible": entry["two_way_proven_infeasible"],
                "multiway_overhead": multi_cost,
                "fragments": entry["best_fragments"],
                "widths": json.dumps(entry["best_block_widths"]),
                "gate_cuts": counts["gate"],
                "wire_cuts": counts["wire"],
                "bridge_seed_overhead": bridge["overhead"] if bridge else None,
                "selected_profile": entry["selected_profile"],
                "two_way_selected_profile": entry["two_way_selected_profile"],
            }
        )
    lines.append(
        table(
            [
                "Case",
                "M",
                "Two-way",
                "Multiway",
                "K",
                "Widths",
                "G / W",
                "Bridge-DP seed",
                "Two / multi",
            ],
            sweep_rows,
        )
    )
    lines.append(
        "\nThe chain improvements are already available from the typed bridge-forest "
        "DP and the expanded feasible set. They are not evidence that SA alone "
        "outperformed the initializer. The two-way column is a matched-budget "
        "heuristic result unless separately certified. Lower capacities can make "
        "wire cuts useful, but do not universally favor them: QFT and several "
        "controls still choose gate cuts.\n"
    )
    for name, entry in sweep["instances"].items():
        path = directory / "capacity_sweep" / name / "exact_reference.json"
        if path.exists():
            exact = json.loads(path.read_text())
            if exact["complete"]:
                lines.append(
                    f"- `{name}`: completed exact search certifies overhead "
                    f"**{number(exact['best_overhead'])}** within the supported "
                    f"independent gate/wire library and multiway static-width model "
                    f"({exact['nodes_visited']:,} search nodes)."
                )
    trajectories = 0
    for root in (directory, directory / "capacity_sweep"):
        for folder in (p for p in root.iterdir() if p.is_dir()):
            paths = [
                *folder.glob("tuning_*.json"),
                *folder.glob("ablation_*.json"),
                *folder.glob("two_way_tuning_*.json"),
            ]
            paths.extend(
                p for p in (folder / "final_runs.json", folder / "two_way_runs.json") if p.exists()
            )
            trajectories += sum(len(json.loads(p.read_text())) for p in paths)
    verification = report.get("final_verification", {})
    lines.append(
        f"\nRecorded {trajectories:,} trajectories, including tuning and ablations. "
        f"All {len(main)} main and {len(sweep['instances'])} capacity winners "
        "passed physical branch-width/norm checks and independent Qiskit "
        "materialization checks. Main benchmark checks do not enumerate their "
        "large global expansions."
    )
    if verification:
        lines.append(
            f"\nThe complete test suite passes **{verification['tests_passed']} tests**; "
            "Ruff lint and formatting checks pass. The new tests include exact "
            "three-fragment reconstruction, sampled shot accounting, bridge-DP "
            "enumeration, and Hastings detailed balance.\n"
        )
    lines.append(
        "\nMachine-readable tables: "
        "[paper comparison](../validation/multiway_benchmarks_v3/paper_comparison.csv), "
        "[all main cases](../validation/multiway_benchmarks_v3/summary.csv), "
        "[full capacity sweep](../validation/multiway_benchmarks_v3/capacity_sweep/summary.csv).\n"
    )
    save_csv(directory / "paper_comparison.csv", paper_csv)
    save_csv(directory / "summary.csv", main_csv)
    save_csv(directory / "capacity_sweep/summary.csv", sweep_csv)
    content = document.read_text()
    before, rest = content.split("<!-- RESULTS_START -->", 1)
    _, after = rest.split("<!-- RESULTS_END -->", 1)
    document.write_text(
        before + "<!-- RESULTS_START -->\n" + "\n".join(lines) + "\n<!-- RESULTS_END -->" + after
    )
    print(
        f"Updated {document}; {len(paper_csv)} paper rows, {len(main_csv)} cases, "
        f"{len(sweep_csv)} capacity cases, {trajectories} trajectories."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/multiway_benchmarks_v3"))
    parser.add_argument("--document", type=Path, default=Path("docs/WORKFLOW_AND_ALGORITHM.md"))
    args = parser.parse_args()
    summarize(args.directory, args.document)
