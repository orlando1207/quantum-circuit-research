"""Sensitivity sweep: how would a CHEAPER excision recipe change the WG/WGE result?

This is an exploratory, hypothetical-cost study, not a physical-compilability
claim. `docs/MILP_VERIFICATION.md` certifies WG vs. WGE using each gate's
REAL, unassumed excision recipe norm (gamma_exc = 16 for CX). This script
instead ASSUMES a given gamma_exc (e.g. 5, well below any known recipe) for
every excision candidate, via the existing `TemplateRegistry.excision_gamma`
override, to answer: "if a cheaper excision construction existed, which
circuits/capacities would it help, and by how much?" Plans from this script
are NOT physically compilable (`compile_hybrid` raises "cost study only" for
any assumed gamma that doesn't match a gate's real recipe) -- this script
does not attempt to compile them; it only reports the MILP's own objective,
certified bounds, and a lightweight (non-physical) width/objective check.

IMPORTANT SOUNDNESS NOTE: docs/MILP_prompt.txt Section 8's K_safe derivation
(Lemma 2) requires gamma_exc(c) >= 4^(wire edges covered by c) for every
candidate -- i.e. gamma_exc >= 16 for a 2-wire-covering candidate, which the
real CX recipe satisfies exactly. An assumed gamma below 16 can VIOLATE that
assumption, which would make the usual compute_k_safe() bound UNSOUND (it
could under-count K and silently exclude the true optimum). This script
detects that case and falls back to a gamma-INDEPENDENT, always-valid bound
(assume, worst case, that every wire edge in the graph could be cut) --
larger/slower models, but never unsound. See `safe_k_for_gamma` below.

Usage (see also `python examples/run_milp_excision_cost_sweep.py --help`):

  python examples/run_milp_excision_cost_sweep.py \\
      --excision-gamma 5 --output results/milp-excision-gamma5.csv \\
      --workers 10 --threads 2 --solver auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from qpartition import (
    AnnealConfig,
    GroupingConfig,
    PartitionGraph,
    build_milp,
    cheap_feasible_upper_bound,
    compute_k_safe,
    excision_selection_from_solution,
    solve_milp,
    state_from_solution,
)
from qpartition.annealing import seed_states
from qpartition.benchmarks import chain_benchmark
from qpartition.large_benchmarks import large_benchmark
from qpartition.milp_benchmarks import (
    alternating_brickwork,
    qcnn,
    ripple_carry_adder,
    tree_tensor_network,
)

TOL = 1e-6
CSV_FIELDS = [
    "name",
    "family",
    "params",
    "n_qubits",
    "two_qubit_gates",
    "max_qubits",
    "k_safe",
    "k_safe_sound",
    "excision_gamma",
    "wg_status",
    "wg_proven_optimal",
    "wg_log10_overhead",
    "wg_bound_log10",
    "wg_gap_rel",
    "wg_runtime_seconds",
    "wge_status",
    "wge_proven_optimal",
    "wge_log10_overhead",
    "wge_bound_log10",
    "wge_gap_rel",
    "wge_runtime_seconds",
    "wge_excision_count",
    "omega_ratio_wg_over_wge",
    "classification",
    "delta_low_log10",
    "delta_high_log10",
    "objective_consistent",
]


def detect_solver(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import gurobipy as gp

        with gp.Env():
            pass
        return "gurobi"
    except Exception:
        return "highs"


def safe_k_for_gamma(graph, max_qubits, upper_bound_log_overhead, excision_gamma):
    """compute_k_safe when sound (gamma >= 16); a gamma-independent fallback otherwise."""
    if excision_gamma is None or excision_gamma >= 16:
        return compute_k_safe(graph, max_qubits, upper_bound_log_overhead), True
    n = graph.problem.circuit.num_qubits
    total_wires = sum(1 for v in graph.variables if v.kind == "wire")
    s = n + total_wires  # Trivial, always-valid bound: W can't exceed the graph's wire count.
    k_lemma1 = 2 * math.ceil(s / max_qubits) - 1
    k_min = math.ceil(n / max_qubits)
    return max(k_lemma1, k_min, 1), False


def upper_bound(graph, max_qubits, seed=991):
    cfg = AnnealConfig(max_qubits, partition_mode="multiway")
    pool = seed_states(graph, cfg, count=8, seed=seed)
    best = cheap_feasible_upper_bound(graph, max_qubits)
    for state in pool:
        evaluation = graph.evaluate(state, max_qubits=max_qubits, partition_mode="multiway")
        if evaluation.feasible:
            best = min(best, evaluation.log_overhead)
    return best


def build_benchmark(family: str, params: dict):
    if family == "separator":
        return large_benchmark(
            "excision_butterfly", params["n"], layers=params["layers"], seed=params["seed"]
        )
    if family == "brickwork":
        return alternating_brickwork(params["n"], layers=params["layers"], seed=params["seed"])
    if family == "adder":
        return ripple_carry_adder(params["bits"], seed=params["seed"])
    if family == "qcnn":
        return qcnn(params["n"], seed=params["seed"])
    if family == "ttn":
        return tree_tensor_network(params["n"], seed=params["seed"])
    if family == "chain":
        return chain_benchmark(params["n"])
    if family == "aqft":
        return large_benchmark("aqft", params["n"], layers=1, seed=params["seed"])
    if family == "trotter_grid":
        return large_benchmark("trotter_grid", params["n"], layers=3, seed=params["seed"])
    raise ValueError(f"Unknown family: {family}")


def default_grid():
    """Same families/sizes/capacities as docs/MILP_VERIFICATION.md's final grid,
    so results are directly comparable to the real-cost (gamma=16) study.
    """
    specs = []

    def add(name, family, params, capacities, time_limit=None):
        for capacity in capacities:
            spec = dict(name=f"{name}_M{capacity}", family=family, params=params, capacity=capacity)
            if time_limit is not None:
                spec["time_limit"] = time_limit
            specs.append(spec)

    for seed in (19, 23):
        for n in (10, 14, 18, 22, 26, 30):
            b = large_benchmark("excision_butterfly", n, layers=1, seed=seed)
            caps = sorted({b.max_qubits, math.ceil(n / 2)})
            add(f"separator_{n}_s{seed}", "separator", dict(n=n, layers=1, seed=seed), caps)

    for seed in (0, 1):
        for n in (8, 16, 24, 32):
            add(
                f"brickwork_{n}_s{seed}",
                "brickwork",
                dict(n=n, layers=4, seed=seed),
                (math.ceil(n / 2), math.ceil(n / 3)),
            )
            add(
                f"ttn_{n}_s{seed}",
                "ttn",
                dict(n=n, seed=seed),
                (math.ceil(n / 2), math.ceil(n / 3)),
            )

    for n in (8, 12, 16, 20):
        add(f"aqft_{n}", "aqft", dict(n=n, seed=19), (math.ceil(n / 2),), time_limit=900)
        add(
            f"trotter_grid_{n}",
            "trotter_grid",
            dict(n=n, seed=19),
            (math.ceil(n / 2),),
            time_limit=900,
        )

    for n in (8, 12, 16, 20, 24):
        add(f"chain_{n}", "chain", dict(n=n), (math.ceil(n / 2),))
    for n in (8, 10, 12):
        add(f"chain_tight_{n}", "chain", dict(n=n), (math.ceil(n / 3),), time_limit=1800)

    for seed in (0, 1):
        for n in (8, 12, 16):
            add(
                f"qcnn_{n}_s{seed}",
                "qcnn",
                dict(n=n, seed=seed),
                (math.ceil(n / 2), math.ceil(n / 3)),
                time_limit=1800,
            )

    for bits in (2, 3, 4, 5):
        n = 2 * bits + 2
        add(
            f"adder_{bits}b",
            "adder",
            dict(bits=bits, seed=0),
            (math.ceil(n / 2), math.ceil(n / 3)),
            time_limit=1800,
        )
    return specs


def classify(wg_obj, wg_bound, wg_opt, wge_obj, wge_bound, wge_opt, tol=TOL):
    delta_low = wg_bound - wge_obj if wg_bound is not None and wge_obj is not None else None
    delta_high = wg_obj - wge_bound if wg_obj is not None and wge_bound is not None else None
    if delta_low is not None and delta_low > tol:
        return "proven_improvement", delta_low, delta_high
    if (
        wg_opt
        and wge_opt
        and wg_obj is not None
        and wge_obj is not None
        and abs(wg_obj - wge_obj) <= tol
    ):
        return "proven_no_gain", delta_low, delta_high
    return "inconclusive", delta_low, delta_high


def solve_instance(task):
    name = task["name"]
    family = task["family"]
    params = task["params"]
    capacity = task["capacity"]
    time_limit = task["time_limit"]
    mip_gap_abs = task["mip_gap_abs"]
    mip_gap_rel = task["mip_gap_rel"]
    threads = task.get("threads")
    solver = task.get("solver", "highs")
    excision_gamma = task["excision_gamma"]
    out_dir = task["out_dir"]

    path = out_dir / "cases" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())

    started = perf_counter()
    benchmark = build_benchmark(family, params)
    n = benchmark.problem.circuit.num_qubits
    two_qubit_gates = sum(
        item.operation.num_qubits == 2 for item in benchmark.problem.circuit.data
    )
    graph = PartitionGraph(
        benchmark.problem,
        excision=True,
        excision_gamma=excision_gamma,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    ub = upper_bound(graph, capacity)
    k, k_sound = safe_k_for_gamma(graph, capacity, ub, excision_gamma)

    solver_kwargs = {} if threads is None else {"threads": threads}
    wg_model = build_milp(graph, max_qubits=capacity, K=k, excision=False)
    wge_model = build_milp(graph, max_qubits=capacity, K=k, excision=True)
    wg_sol = solve_milp(
        wg_model, solver=solver, time_limit=time_limit, mip_gap_abs=mip_gap_abs,
        mip_gap_rel=mip_gap_rel, **solver_kwargs,
    )
    wge_sol = solve_milp(
        wge_model, solver=solver, time_limit=time_limit, mip_gap_abs=mip_gap_abs,
        mip_gap_rel=mip_gap_rel, **solver_kwargs,
    )

    def lightweight_check(model, solution):
        """Width/objective sanity check without physical compilation (which
        would reject any assumed gamma that doesn't match a real recipe).
        """
        if solution.raw_x is None:
            return None
        state = state_from_solution(model, solution.raw_x)
        forced = excision_selection_from_solution(model, solution.raw_x) if model.excision else ()
        evaluation = graph.evaluate(
            state, max_qubits=capacity, partition_mode="multiway", forced_groups=forced
        )
        matches = np.isclose(evaluation.log_overhead, solution.objective, atol=1e-6, rtol=0)
        within_capacity = max(evaluation.block_widths) <= capacity
        return bool(matches and within_capacity), len(forced)

    wg_check = lightweight_check(wg_model, wg_sol)
    wge_check = lightweight_check(wge_model, wge_sol)
    excision_count = wge_check[1] if wge_check else None
    objective_consistent = (wg_check is None or wg_check[0]) and (
        wge_check is None or wge_check[0]
    )

    classification, delta_low, delta_high = classify(
        wg_sol.objective, wg_sol.bound, wg_sol.proven_optimal,
        wge_sol.objective, wge_sol.bound, wge_sol.proven_optimal,
    )
    ln10 = math.log(10)
    omega_ratio = None
    if wg_sol.objective is not None and wge_sol.objective is not None:
        omega_ratio = float(np.exp(wg_sol.objective - wge_sol.objective))

    record = dict(
        name=name,
        family=family,
        params=params,
        n_qubits=n,
        two_qubit_gates=two_qubit_gates,
        max_qubits=capacity,
        k_safe=k,
        k_safe_sound=k_sound,
        excision_gamma=excision_gamma,
        wg_status=wg_sol.status,
        wg_proven_optimal=wg_sol.proven_optimal,
        wg_log10_overhead=wg_sol.objective / ln10 if wg_sol.objective is not None else None,
        wg_bound_log10=wg_sol.bound / ln10 if wg_sol.bound is not None else None,
        wg_gap_rel=wg_sol.gap_rel,
        wg_runtime_seconds=wg_sol.runtime_seconds,
        wge_status=wge_sol.status,
        wge_proven_optimal=wge_sol.proven_optimal,
        wge_log10_overhead=wge_sol.objective / ln10 if wge_sol.objective is not None else None,
        wge_bound_log10=wge_sol.bound / ln10 if wge_sol.bound is not None else None,
        wge_gap_rel=wge_sol.gap_rel,
        wge_runtime_seconds=wge_sol.runtime_seconds,
        wge_excision_count=excision_count,
        omega_ratio_wg_over_wge=omega_ratio,
        classification=classification,
        delta_low_log10=delta_low / ln10 if delta_low is not None else None,
        delta_high_log10=delta_high / ln10 if delta_high is not None else None,
        objective_consistent=objective_consistent,
        elapsed_seconds=perf_counter() - started,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(
        f"{name} (M={capacity}, gamma={excision_gamma}): {classification}  "
        f"WG={wg_sol.status}  WGE={wge_sol.status}  ({record['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return record


def parallel(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            results.append(result)
            if len(results) % 5 == 0 or len(results) == len(futures):
                print(f"{function.__name__}: {len(results)}/{len(futures)}", flush=True)
    return results


def write_csv(records, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda r: r["name"]):
            row = dict(record)
            row["params"] = str(row["params"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excision-gamma", type=float, required=True,
        help="Assumed excision recipe norm for every candidate (e.g. 5). "
        "The real CX recipe is 16; values below that trigger the sound-K fallback.",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV output path.")
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Per-instance JSON checkpoints (resumable); default: alongside --output.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--threads", type=int, help="Solver threads per solve (default: cpus/workers)."
    )
    parser.add_argument(
        "--solver", choices=("auto", "gurobi", "highs"), default="auto",
        help="auto picks gurobi if gurobipy+license are available, else highs.",
    )
    parser.add_argument(
        "--time-limit", type=float, default=300.0,
        help="Per-model (WG and WGE each get this) time limit in seconds. Acts as a FLOOR: "
        "families with a higher built-in default (qcnn/adder/chain_tight/aqft/trotter_grid) "
        "use max(their default, this value), so raising it raises everyone.",
    )
    parser.add_argument("--mip-gap-abs", type=float, default=1e-6)
    parser.add_argument("--mip-gap-rel", type=float, default=0.0)
    parser.add_argument(
        "--families", type=str, default=None,
        help="Comma-separated family filter (default: all). "
        "Choices: separator,brickwork,ttn,qcnn,adder,chain,aqft,trotter_grid",
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="Skip solving; just write --output from whatever --work-dir checkpoints "
        "already exist (use to peek at progress, or after interrupting a run).",
    )
    parser.add_argument(
        "--retry-inconclusive", action="store_true",
        help="Before solving, delete the checkpoints of any already-completed instance "
        "classified 'inconclusive' so it gets re-solved (with --time-limit, typically "
        "raised) instead of being skipped as done. Proven instances are left alone.",
    )
    args = parser.parse_args()
    if args.excision_gamma < 1:
        parser.error("--excision-gamma must be >= 1.")

    work_dir = args.work_dir or args.output.parent / f"{args.output.stem}_cases"
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    specs = default_grid()
    if args.families:
        wanted = set(args.families.split(","))
        specs = [s for s in specs if s["family"] in wanted]
    if not specs:
        parser.error("No instances selected (check --families).")

    if args.collect:
        done = {spec["name"] for spec in specs} & {
            p.stem for p in (work_dir / "cases").glob("*.json")
        }
        records = [
            json.loads((work_dir / "cases" / f"{name}.json").read_text()) for name in done
        ]
        write_csv(records, args.output)
        print(f"Collected {len(records)}/{len(specs)} completed instances into {args.output}")
        return

    if args.retry_inconclusive:
        retried = 0
        for spec in specs:
            path = work_dir / "cases" / f"{spec['name']}.json"
            if not path.exists():
                continue
            if json.loads(path.read_text()).get("classification") == "inconclusive":
                path.unlink()
                retried += 1
        print(f"Cleared {retried} inconclusive checkpoint(s) for re-solving.", flush=True)

    threads = args.threads or max(1, (os.cpu_count() or args.workers) // args.workers)
    solver = detect_solver(args.solver)
    sound = args.excision_gamma >= 16
    k_note = "sound K_safe" if sound else "gamma < 16: using the trivial gamma-independent K bound"
    print(
        f"Solver: {solver} (threads/solve: {threads}); "
        f"excision_gamma={args.excision_gamma} ({k_note})",
        flush=True,
    )
    print(f"{len(specs)} instances -> {work_dir}", flush=True)

    jobs = [
        dict(
            name=spec["name"],
            family=spec["family"],
            params=spec["params"],
            capacity=spec["capacity"],
            # --time-limit is a FLOOR: some families (qcnn, adder, chain_tight,
            # aqft, trotter_grid) have a higher built-in default than the
            # overall default (300s) since they were harder in the real-cost
            # study; raising --time-limit raises everyone, including those.
            time_limit=max(spec.get("time_limit", args.time_limit), args.time_limit),
            mip_gap_abs=args.mip_gap_abs,
            mip_gap_rel=args.mip_gap_rel,
            threads=threads,
            solver=solver,
            excision_gamma=args.excision_gamma,
            out_dir=work_dir,
        )
        for spec in specs
    ]
    try:
        records = parallel(solve_instance, jobs, args.workers)
    except BaseException:
        # Preserve whatever already-completed instances' checkpoints exist so
        # `--collect` (or a plain rerun, which resumes via the same checkpoints)
        # can recover partial results after an interruption or a crash.
        print("Interrupted/failed -- completed instances are still on disk as JSON "
              "checkpoints; rerun the same command to resume, or pass --collect to "
              "export what finished so far.", flush=True)
        raise
    write_csv(records, args.output)
    print(f"Wrote {len(records)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
