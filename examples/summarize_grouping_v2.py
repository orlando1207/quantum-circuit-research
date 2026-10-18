"""Summarize recorded grouping-v2 comparisons and paired quality measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_grouping_v2 import render
from summarize_multiway_results import number, table


def summarize(directory, document=None):
    report = json.loads((directory / "report.json").read_text())
    render(directory, report)
    rows = []
    for name, paper in report["paper_table_i"].items():
        entry = report["instances"].get("grover4_historical" if name == "grover4" else name)
        rows.append(
            [
                name,
                paper["max_qubits"],
                number(paper["paper_with_groups"]),
                number(entry["grouped_v1"]) if entry else "—",
                number(entry["arms"]["enhanced"]["reported_overhead"]) if entry else "—",
                number(min(m["reported_overhead"] for m in entry["arms"].values()))
                if entry
                else "—",
            ]
        )
    paper_table = table(
        ["Case", "M", "Paper GG", "Our grouped v1", "Enhanced v2", "Best four-arm portfolio"], rows
    )
    stats = {}
    rng = np.random.default_rng(20261018)
    for name, e in report["instances"].items():
        base = json.loads((directory / name / "standard/final_runs.json").read_text())
        enhanced = json.loads((directory / name / "enhanced/final_runs.json").read_text())
        assert [r["seed"] for r in base] == [r["seed"] for r in enhanced]
        assert all(
            a["initial_feasible"] == b["initial_feasible"]
            and abs(a["initial_log_overhead"] - b["initial_log_overhead"]) < 1e-10
            for a, b in zip(base, enhanced, strict=True)
        )
        x = np.array([r["best"]["sampling_overhead"] for r in base])
        y = np.array([r["best"]["sampling_overhead"] for r in enhanced])
        differences = np.log(x) - np.log(y)
        target = np.log(e["pool_best_overhead"])
        nonbest = np.array(
            [not r["initial_feasible"] or r["initial_log_overhead"] > target + 1e-10 for r in base]
        )
        means = np.mean(rng.choice(differences, (10000, len(differences)), replace=True), axis=1)
        stats[name] = {
            "paired_runs": len(x),
            "standard_median": float(np.median(x)),
            "enhanced_median": float(np.median(y)),
            "enhanced_lower_cost": int(sum(differences > 1e-10)),
            "standard_lower_cost": int(sum(differences < -1e-10)),
            "ties": int(sum(np.abs(differences) <= 1e-10)),
            "nonbest_initial_runs": int(sum(nonbest)),
            "standard_recovers_pool_best": int(sum(nonbest & (np.log(x) <= target + 1e-10))),
            "enhanced_recovers_pool_best": int(sum(nonbest & (np.log(y) <= target + 1e-10))),
            "mean_log_standard_over_enhanced": float(np.mean(differences)),
            "paired_bootstrap_95pct_mean_log_ratio": np.quantile(means, [0.025, 0.975]).tolist(),
            "same_selected_profile": e["arms"]["standard"]["selected_profile"]
            == e["arms"]["enhanced"]["selected_profile"],
        }
    (directory / "paired_quality.json").write_text(json.dumps(stats, indent=2) + "\n")
    csv_rows = []
    for name, entry in report["instances"].items():
        for arm, result in entry["arms"].items():
            csv_rows.append(
                {
                    "benchmark": name,
                    "arm": arm,
                    "max_qubits": entry["max_qubits"],
                    "independent_v1": entry["independent_v1"],
                    "grouped_v1": entry["grouped_v1"],
                    "v2_on_v1_mask": entry["post_v1_mask"]["overhead"],
                    "pool_best": entry["pool_best_overhead"],
                    "trajectory_best": result["best_overhead"],
                    "retained_best": result["reported_overhead"],
                    "before_refinement": result["best_before_polish"],
                    "median": result["median_overhead"],
                    "starts": result["runs"],
                    "runs_at_pool_best": result["runs_at_pooled_best"],
                    "evaluations": result["total_evaluations"],
                    "selected_profile": result["selected_profile"],
                    "widths": json.dumps(result["widths"]),
                    "source_fingerprint": entry["fingerprint"],
                }
            )
    with (directory / "all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    comparisons = table(
        [
            "Case",
            "Standard median",
            "Enhanced median",
            "Enhanced wins / losses / ties",
            "Goal recoveries: standard → enhanced",
        ],
        [
            [
                name,
                number(s["standard_median"]),
                number(s["enhanced_median"]),
                f"{s['enhanced_lower_cost']} / {s['standard_lower_cost']} / {s['ties']}",
                f"{s['standard_recovers_pool_best']}/{s['nonbest_initial_runs']} → "
                f"{s['enhanced_recovers_pool_best']}/{s['nonbest_initial_runs']}",
            ]
            for name, s in stats.items()
        ],
    )
    with (directory / "SUMMARY.md").open("a") as stream:
        stream.write("\n## Paper targets and historical reference\n\n" + paper_table + "\n")
        stream.write("\n## Paired final-run quality\n\n" + comparisons + "\n")
        stream.write(
            "\nMatched seeds and initial states allow paired descriptive comparisons. "
            "Profiles can differ because each arm tunes independently; in that case "
            "this compares the full tuned policies. Bootstrap intervals are exploratory, "
            "not corrected for multiple benchmarks or metrics. No global optimality "
            "or universal stochastic-search superiority is claimed. Goal recoveries "
            "exclude starts already feasible and at the shared pool's best cost.\n"
        )
        no_better = sum(
            all(
                m["best_overhead"] >= e["pool_best_overhead"] * (1 - 1e-10)
                for m in e["arms"].values()
            )
            for e in report["instances"].values()
        )
        stream.write(
            f"\nOn {no_better}/{len(report['instances'])} circuits, no arm beats the "
            "common pool incumbent. Improved costs over v1 must therefore be "
            "distinguished from improved recovery of a known goal by SA.\n"
        )
        if "qft6" in stats:
            low, high = stats["qft6"]["paired_bootstrap_95pct_mean_log_ratio"]
            stream.write(
                f"\nQFT6's paired bootstrap mean log cost-ratio interval is "
                f"[{low:.3f}, {high:.3f}]. Read the median/hit-rate changes as "
                "descriptive results; an interval containing zero does not establish "
                "overall cost superiority.\n"
            )
    count = sum(
        m["runs"] + sum(t["runs"] for t in m["tuning"].values())
        for e in report["instances"].values()
        for m in e["arms"].values()
    )
    if document:
        start, end = "<!-- GROUPING_V2_RESULTS_START -->", "<!-- GROUPING_V2_RESULTS_END -->"
        text = document.read_text()
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        block = (
            f"{start}\n\nRecorded **{count:,} trajectories**. The new candidate library and "
            "compressed recipes are shared by all four arms.\n\n"
            + paper_table
            + "\n\n"
            + comparisons
            + f"\n\n{end}"
        )
        document.write_text(before + block + after)
    print("Summarized", len(stats), "circuits;", count, "trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/grouping_benchmarks_v2"))
    parser.add_argument("--document", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.document)
