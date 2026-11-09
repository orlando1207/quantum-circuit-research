"""Solve the excision LP once per distinct two-qubit gate class and cache it.

Scans saved CircuitProblem snapshots (QPY), collects the distinct KAK points of
their (fused) two-qubit gates, solves each canonical LP in parallel, and writes
  <cache>          pickle of ExcisionLPResult by KAK key (QPARTITION_EXCISION_CACHE)
  <cache>.csv      kak, gamma_cut, gamma_ex, lower-bound estimate, status, seconds

Usage:
  python examples/precompute_excision_costs.py --cases results/setting_a/cases \
      results/stage3_suite/cases --fuse-stage3 --cache results/excision_lp_cache.pkl --workers 6
"""

from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_setting_a_certification import _classify_kak, setting_a_problem  # noqa: E402

from qpartition import excision_lp  # noqa: E402
from qpartition.circuits import load_problem  # noqa: E402


def _solve_one(kak, out):
    """Child-process entry: solve one canonical LP and pickle (result, seconds)."""
    started = perf_counter()
    result = excision_lp.gate_excision_cost(excision_lp.canonical_unitary(*kak))
    with open(out, "wb") as handle:
        pickle.dump((result, perf_counter() - started), handle)


def _solve(kak, timeout):
    """Run `_solve_one` in its own process; a hang or crash only loses this gate."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.pkl"
        cmd = [sys.executable, __file__, "--solve-one", *map(repr, kak), "--out", str(out)]
        try:
            subprocess.run(cmd, timeout=timeout, check=False, capture_output=True)
        except subprocess.TimeoutExpired:
            return kak, None, timeout
        if not out.exists():
            return kak, None, None
        with open(out, "rb") as handle:
            result, seconds = pickle.load(handle)
        return kak, result, seconds


def collect_kak_points(directories, fuse_stage3):
    points = {}
    for directory in directories:
        for case_dir in sorted(Path(directory).iterdir()):
            if not (case_dir / "circuit.qpy").exists():
                continue
            problem = load_problem(case_dir)
            if fuse_stage3 and (case_dir / "stage3_meta.json").exists():
                problem = setting_a_problem(problem)
            for item in problem.circuit.data:
                if item.operation.num_qubits == 2:
                    U = np.asarray(Operator(item.operation).data, dtype=complex)
                    kak = excision_lp.kak_coordinates(U)
                    points.setdefault(excision_lp.kak_key(kak), kak)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-one", nargs=3, type=float, metavar=("A", "B", "C"))
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--cache", type=Path, default=Path("results/excision_lp_cache.pkl"))
    parser.add_argument("--fuse-stage3", action="store_true",
                        help="apply Setting A fusion to raw Stage 3 snapshots")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.solve_one:
        _solve_one(tuple(args.solve_one), args.out)
        return

    if args.cache.exists():
        excision_lp.load_cache(args.cache)
    points = collect_kak_points(args.cases, args.fuse_stage3)
    todo = [kak for key, kak in points.items() if key not in excision_lp._CACHE]
    print(f"{len(points)} distinct gate classes, {len(todo)} to solve", flush=True)
    seconds = {}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_solve, kak, args.timeout) for kak in todo]
        for done, future in enumerate(as_completed(futures), 1):
            kak, result, elapsed = future.result()
            if result is None:
                failed.append(kak)
                print(f"[{done}/{len(todo)}] {_classify_kak(kak)} FAILED/TIMEOUT "
                      "(registry falls back to the PTM recipe)", flush=True)
                continue
            key = excision_lp.kak_key(kak)
            excision_lp._CACHE[key] = result
            seconds[key] = elapsed
            excision_lp.save_cache(args.cache)
            print(f"[{done}/{len(todo)}] {_classify_kak(kak)} gamma_ex={result.gamma_ex:.4f} "
                  f"{result.status} ({elapsed:.0f}s)", flush=True)
    with open(f"{args.cache}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "gate_class", "gamma_cut", "gamma_ex",
                         "gamma_ex_lower_estimate", "status", "seconds"])
        for key in sorted(points):
            result = excision_lp._CACHE.get(key)
            if result is None:
                writer.writerow([*key, _classify_kak(key), excision_lp.gamma_cut_kak(*key),
                                 "", "", "FAILED", ""])
                continue
            writer.writerow([*key, _classify_kak(key), excision_lp.gamma_cut_kak(*key),
                             result.gamma_ex, result.lower_bound, result.status,
                             seconds.get(key, "")])
    print(f"wrote {args.cache} and {args.cache}.csv")


if __name__ == "__main__":
    main()
