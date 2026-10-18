"""Generate external-baseline tables without asserting original-artifact identity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_sa_benchmarks import rounded_comparison
from summarize_multiway_results import number, table


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory):
    report = json.loads((directory / "report.json").read_text())
    paper_rows, comparisons = [], []
    cases = report["instances"]
    for name, expected in report["paper_baselines"].items():
        source = "grover4_historical" if name == "grover4" else name
        entry = cases.get(f"{source}_M{expected['max_qubits']}")
        methods = entry["methods"] if entry else {}
        values = {
            method: methods.get(method, {}).get("best_overhead")
            for method in ("qiskit", "paper_balanced", "paper_capacity", "sa_fresh")
        }
        identity = (
            "Unavailable circuit"
            if entry is None
            else "Historical native-basis candidate"
            if name == "grover4"
            else "Structural proxy"
            if name.startswith("chain")
            else "Reconstructed, original files unverified"
        )
        row = {
            "benchmark": name,
            "max_qubits": expected["max_qubits"],
            "paper_qiskit": expected["qiskit"],
            "paper_no_GG": expected["paper_without_groups"],
            "paper_GG": expected["paper_with_groups"],
            **values,
            "GG_author_manifest_reproduced": False,
            "input_identity": identity,
            "qiskit_rounding_comparison": rounded_comparison(values["qiskit"], expected["qiskit"])
            if values["qiskit"] is not None
            else None,
            "no_GG_rounding_comparison": rounded_comparison(
                values["paper_capacity"], expected["paper_without_groups"]
            )
            if values["paper_capacity"] is not None
            else None,
        }
        comparisons.append(row)
        paper_rows.append(
            [
                name,
                expected["max_qubits"],
                number(expected["qiskit"]),
                number(values["qiskit"]),
                number(expected["paper_without_groups"]),
                number(values["paper_balanced"]),
                number(values["paper_capacity"]),
                number(values["sa_fresh"]),
            ]
        )
    lines = [
        "# Recorded external-baseline results\n",
        "Generated from `report.json`; all costs below are squared QPD coefficient norms. "
        "Official Qiskit is executed directly. KL columns are explicit reconstructions, "
        "not the authors' implementation. Input identity is unverified; chain inputs "
        "are structural proxies. Grover uses the new historical native-basis candidate.\n",
        f"Each available method uses {report['arguments']['starts']} fresh seeds. "
        f"SA uses {report['arguments']['sa_steps']} steps per seed without historical warm starts. "
        "Budgets are not matched in wall time or objective evaluations.\n",
        "## All eight Table-I rows\n",
        table(
            [
                "Case",
                "M",
                "Paper Qiskit",
                "Official Qiskit",
                "Paper no GG",
                "KL balanced",
                "KL capacity",
                "Fresh SA",
            ],
            paper_rows,
        ),
        "\nKL balanced fixes the initial qubit cardinality to floor(N/2). KL capacity "
        "samples cardinalities permitted by M. Both use the same physical width guard. "
        "Neither resolves the paper's unspecified weighted-KL mechanics.\n",
        "## Gate grouping: reported and measured scope\n",
        table(
            ["Case", "Paper GG", "Measured GG", "Status"],
            [
                [
                    row["benchmark"],
                    number(row["paper_GG"]),
                    number(
                        cases.get(f"{row['benchmark']}_M{row['max_qubits']}", {})
                        .get("methods", {})
                        .get("paper_GG_own_CX_manifest", {})
                        .get("best_overhead")
                    ),
                    "Own CX manifest on proxy; not author reproduction"
                    if row["benchmark"].startswith("chain")
                    else "Paper lists future work"
                    if row["paper_GG"] is None
                    else "Original circuit/group manifest missing",
                ]
                for row in comparisons
            ],
        ),
        "\nCanonical parallel/common-control/common-target CX costs are independently "
        "verified on small complete unitaries. These checks do not reproduce Table-I "
        "manual groups or compile a joint QPD. Figure 4 arithmetic gives 20,736 "
        "without GG and 7,056 with GG; the original block circuit remains unavailable.\n",
        "## Every tested circuit and capacity\n",
    ]
    rows, capacity_rows = [], []
    for key, entry in cases.items():
        for method, result in entry["methods"].items():
            rows.append(
                {
                    "case": key,
                    "benchmark": entry["benchmark"],
                    "qubits": entry["qubits"],
                    "two_qubit_gates": entry["two_qubit_gates"],
                    "max_qubits": entry["max_qubits"],
                    "paper_capacity": entry["paper_capacity"],
                    "method": method,
                    "status": result.get("status", "feasible"),
                    "best_overhead": result.get("best_overhead"),
                    "median_overhead": result.get("median_overhead"),
                    "fragments": result.get("best_fragments"),
                    "widths": json.dumps(result.get("best_widths")),
                    "gate_cuts": result.get("best_gate_cuts"),
                    "wire_cuts": result.get("best_wire_cuts"),
                    "runs": result.get("runs", 0),
                    "minimum_reached_runs": result.get("minimum_reached_runs"),
                    "source_fingerprint": entry["fingerprint"],
                }
            )
        methods = entry["methods"]
        qiskit, sa = methods["qiskit"], methods["sa_fresh"]
        kl = methods["paper_capacity"]
        capacity_rows.append(
            [
                entry["benchmark"],
                entry["max_qubits"],
                number(qiskit.get("best_overhead")),
                qiskit.get("best_fragments", "—"),
                qiskit.get("minimum_reached_runs", 0),
                number(kl.get("best_overhead"))
                if kl.get("best_overhead") is not None
                else kl.get("status", "No feasible run"),
                number(sa["best_overhead"]),
                sa["best_fragments"],
            ]
        )
    lines.append(
        table(
            ["Case", "M", "Qiskit", "K", "Certified runs", "KL capacity", "Fresh SA", "K"],
            capacity_rows,
        )
    )
    lines.append(
        "\n`minimum_reached` is the official SDK's search certificate within its "
        "supported search model. Zero certified runs means no optimality claim. "
        "Two-way infeasibility for N > 2M follows from static wire-segment accounting; "
        "it is not an optimizer failure. No depth constraint or qubit reuse is enabled.\n"
    )
    budget_runs = {}
    for path in sorted(directory.glob("qiskit*budget_sensitivity.json")):
        for run in json.loads(path.read_text())["runs"]:
            config = run["config"]
            key = (run["case"], config["max_gamma"], config["max_backjumps"])
            assert run["source_fingerprint"] == cases[run["case"]]["fingerprint"]
            budget_runs[key] = run
    if budget_runs:
        lines.extend(
            [
                "## Qiskit search-budget sensitivity (seed 0)\n",
                table(
                    ["Case", "Gamma bound", "Backjumps", "Overhead", "K", "Certified", "Seconds"],
                    [
                        [
                            case,
                            number(gamma),
                            jumps,
                            number(run["sampling_overhead"]),
                            run["fragments"],
                            run["minimum_reached"],
                            f"{run['optimizer_seconds']:.3f}",
                        ]
                        for (case, gamma, jumps), run in sorted(budget_runs.items())
                    ],
                ),
                "\nThese extra runs use the exact same circuit fingerprints. Stronger search "
                "can eliminate apparent advantages over default SDK settings. Search certification "
                "and cost must be reported together; these are not equal-time comparisons.\n",
            ]
        )
    checks_path = directory / "independent_checks.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text())
        lines.append(
            f"\nIndependent source-hash, physical compilation, SDK norm, and width checks "
            f"passed for all {checks['cases_checked']} fresh SA winners. One physical branch "
            "was compiled per winner; no global experiment family was enumerated.\n"
        )
        group_checks = [
            [
                case,
                number(row["own_GG_reevaluated_no_GG_incumbent"]["ungrouped_overhead"]),
                number(row["own_GG_reevaluated_no_GG_incumbent"]["grouped_cost_same_labels"]),
            ]
            for case, row in checks["checks"].items()
            if "own_GG_reevaluated_no_GG_incumbent" in row
        ]
        if group_checks:
            lines.extend(
                [
                    "## Own GG cost check on saved ungrouped KL incumbents\n",
                    table(
                        ["Case", "Ungrouped incumbent", "Grouped cost, same labels"], group_checks
                    ),
                    "\nThis post hoc re-evaluation retains a feasible ungrouped placement. It "
                    "is separate from the fresh grouped optimizer runs above. In particular, "
                    "a grouped run returning 81 while a retained candidate costs 16 reflects "
                    "initialization/search behavior, not an increase in the grouped cost of "
                    "that same placement. None uses the authors' manual manifests.\n",
                ]
            )
    (directory / "SUMMARY.md").write_text("\n".join(lines))
    (directory / "table_i_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    write_csv(directory / "all_results.csv", rows)
    write_csv(
        directory / "table_i_results.csv",
        [{k: v for k, v in row.items() if not k.endswith("comparison")} for row in comparisons],
    )
    print(f"Saved {directory / 'SUMMARY.md'} ({len(cases)} capacities, {len(rows)} method rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/external_baselines_v1"))
    summarize(parser.parse_args().directory)
