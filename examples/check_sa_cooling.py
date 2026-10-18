"""Paired held-out QFT12 cooling-rate sensitivity with a fixed initializer pool."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from run_sa_benchmarks import summarize

from qpartition import AnnealConfig, PartitionGraph, load_problem, multi_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    folder = args.directory / "qft12"
    graph = PartitionGraph(load_problem(folder))
    pool = tuple(
        tuple(s) for s in json.loads((folder / "initialization.json").read_text())["states"]
    )
    config = AnnealConfig(9, steps=6000, block_probability=0.35)
    report = {
        "scope": "Paired fresh seeds 3000+; QFT12 only; no generalization claim. "
        "Sensitivity analysis, not retrospective replacement of the primary final-50 experiment.",
        "profiles": {},
    }
    for rate in (0.85, 0.93, 0.97):
        runs = multi_start(
            graph,
            replace(config, cooling_rate=rate),
            seeds=tuple(range(3000, 3000 + args.starts)),
            initial_states=pool,
            workers=args.workers,
        )
        summary = summarize(runs)
        best = summary["best_overhead"]
        summary["best_hits"] = sum(run.best.sampling_overhead <= best * (1 + 1e-10) for run in runs)
        report["profiles"][str(rate)] = summary
        (folder / f"cooling_{rate}.json").write_text(
            json.dumps([run.to_dict() for run in runs], indent=2) + "\n"
        )
        print(rate, summary, flush=True)
    (args.directory / "cooling_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
