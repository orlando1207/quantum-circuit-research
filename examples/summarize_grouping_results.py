"""Regenerate grouping tables from recorded results without rerunning search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_grouping_benchmarks import render
from summarize_multiway_results import number, table


def summarize(directory, document=None):
    report = json.loads((directory / "report.json").read_text())
    render(directory, report)
    if document is None:
        return
    rows = []
    for name, paper in report["paper_table_i"].items():
        entry = report["instances"].get("grover4_historical" if name == "grover4" else name)
        grouped = entry["methods"]["grouped_multiway"] if entry else None
        rows.append(
            [
                name,
                paper["max_qubits"],
                number(paper["paper_without_groups"]),
                number(paper["paper_with_groups"]),
                number(entry["historical_overhead"]) if entry else "—",
                number(grouped["reported_overhead"]) if grouped else "—",
            ]
        )
    trajectories = sum(
        m["runs"] + sum(t["runs"] for t in m["tuning"].values())
        for entry in report["instances"].values()
        for m in entry["methods"].values()
    )
    start, end = "<!-- GROUPING_RESULTS_START -->", "<!-- GROUPING_RESULTS_END -->"
    result = "\n\n".join(
        [
            start,
            f"Recorded **{trajectories:,} trajectories** across "
            f"{len(report['instances'])} circuits, four arms, three tuning profiles, "
            f"and {report['arguments']['starts']} final starts per arm. "
            "Costs below are sampling overheads $\\Omega=\\Gamma^2$.",
            table(
                ["Case", "M", "Paper no GG", "Paper GG", "Previous no GG", "Grouped multiway"],
                rows,
            ),
            "The all-circuit table and stage-by-stage attribution are in the linked "
            "generated summary. Original authors' source identity and manual group "
            "manifests remain unverified. Figure 4 has no complete source circuit; "
            "the QFT12 grouped paper entry has no numeric target.",
            end,
        ]
    )
    text = document.read_text()
    if start not in text or end not in text:
        raise ValueError("Document needs GROUPING_RESULTS_START/END markers.")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    document.write_text(before + result + after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/grouping_benchmarks_v1"))
    parser.add_argument("--document", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.document)
