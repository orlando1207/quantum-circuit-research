"""Check how SDK search bounds affect selected external-baseline instances."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from qpartition import load_problem
from qpartition.qiskit_baseline import QiskitBaselineConfig, run_qiskit_baseline


def one(job):
    case, folder, config = job
    _, result = run_qiskit_baseline(load_problem(folder), config)
    return {"case": case, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/external_baselines_v1"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--filename", default="qiskit_budget_sensitivity.json")
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    output = args.directory / args.filename
    if output.exists():
        raise ValueError("Existing sensitivity results must be preserved.")
    cases = (("qft6_M3", 3), ("qft12_M9", 9), ("grover4_historical_M3", 3), ("chain20_M6", 6))
    configs = [(1024, 10000), (10000000, 10000), (10000000, 100000)]
    if args.extended:
        configs.append((10000000, 1000000))
    jobs = [
        (case, args.directory / case, QiskitBaselineConfig(limit, max_gamma=g, max_backjumps=b))
        for case, limit in cases
        if not args.only or case in args.only
        for g, b in configs
    ]
    if not all(folder.exists() for _, folder, _ in jobs):
        raise ValueError("Run the main baseline sweep first.")
    results = []
    with ProcessPoolExecutor(args.workers) as executor:
        for row in executor.map(one, jobs):
            results.append(row)
            output.write_text(
                json.dumps(
                    {
                        "purpose": "Search-budget sensitivity; original environment unverified",
                        "seed": 0,
                        "settings": [asdict(config) for _, _, config in jobs],
                        "runs": results,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                row["case"],
                row["config"],
                row["sampling_overhead"],
                row["minimum_reached"],
                row["optimizer_seconds"],
                flush=True,
            )


if __name__ == "__main__":
    main()
