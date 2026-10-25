"""Formal MILP verification of the WG vs. WGE partition optimum (docs/MILP_prompt.txt).

Builds the exact MILP models in `qpartition.milp` from the SAME excision-
enabled `PartitionGraph` for both WG (excision disabled in the model) and
WGE (excision enabled), solves each with a per-instance time budget, recompiles
and re-evaluates every available incumbent through the existing evaluator, and
classifies the instance per Section 10 (PROVEN IMPROVEMENT / PROVEN NO GAIN /
INCONCLUSIVE). Supports `--phase pilot|final|summarize` for staged, resumable
execution; reuse the same `--output` directory to resume, a fresh one for
changed sizes/time limits/source code.

Run with the research conda interpreter:
  python examples/run_milp_verification.py --phase pilot --output results/milp-verification-pilot
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from qpartition import (
    AnnealConfig,
    GroupingConfig,
    PartitionGraph,
    build_milp,
    cheap_feasible_upper_bound,
    compile_hybrid,
    compute_k_safe,
    excision_selection_from_solution,
    milp_plan,
    solve_milp,
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


def write_json(path, value):
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


# --- Instance set (Section 12): CX-only families across N ~ 8-32 qubits. ---


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


def pilot_definitions():
    """A small, cheap breadth-first sweep to find each family's solvable frontier.

    Deliberately narrow (not the full Section 12 grid): the pilot's only job is
    to locate, per family, the largest size/capacity that reaches proven
    optimality within a short time limit, so the final grid can be sized from
    real data instead of a guess.
    """
    specs = []

    def add(name, family, params, capacities):
        for capacity in capacities:
            specs.append(
                dict(name=f"{name}_M{capacity}", family=family, params=params, capacity=capacity)
            )

    for n in (10, 14):
        b = large_benchmark("excision_butterfly", n, layers=1, seed=19)
        add(
            f"separator_{n}",
            "separator",
            dict(n=n, layers=1, seed=19),
            sorted({b.max_qubits, math.ceil(n / 2)}),
        )
    for n in (8, 16):
        add(
            f"brickwork_{n}_L3",
            "brickwork",
            dict(n=n, layers=3, seed=0),
            (math.ceil(n / 2), math.ceil(n / 3)),
        )
    for bits in (2, 3, 4):
        n = 2 * bits + 2
        add(f"adder_{bits}b", "adder", dict(bits=bits, seed=0), (math.ceil(n / 2),))
    for n in (8, 16):
        add(f"qcnn_{n}", "qcnn", dict(n=n, seed=0), (math.ceil(n / 2), math.ceil(n / 3)))
        add(f"ttn_{n}", "ttn", dict(n=n, seed=0), (math.ceil(n / 2), math.ceil(n / 3)))
    for n in (8, 14):
        add(f"chain_{n}", "chain", dict(n=n), (math.ceil(n / 2), math.ceil(n / 3)))
    for n in (8, 14):
        add(f"aqft_{n}", "aqft", dict(n=n, seed=19), (math.ceil(n / 2),))
        add(f"trotter_grid_{n}", "trotter_grid", dict(n=n, seed=19), (math.ceil(n / 2),))
    return specs


def final_definitions():
    """The Section 12 grid, sized from the pilot's solvable frontier.

    Most families (separator, brickwork, ttn, chain-loose) solve in seconds
    and get a broad size/seed sweep at the default time budget. aqft/
    trotter_grid (structured "expected no gain" controls) and chain-tight get
    a longer budget since they are moderately harder. qcnn and the ripple-
    carry adder were the pilot's hardest families — a dedicated CPU thread
    per solve (see --threads) resolves most of that apparent difficulty (it
    was largely worker-contention under the pilot's shared cores, not the
    model itself), but they still get the longest budget and the smallest
    sizes to keep the whole grid inside a few hours.
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


def upper_bound(graph, max_qubits, seed=991):
    """A cheap, SA-seed-pool-informed valid WGE upper bound, tightening K_safe."""
    cfg = AnnealConfig(max_qubits, partition_mode="multiway")
    pool = seed_states(graph, cfg, count=8, seed=seed)
    best = cheap_feasible_upper_bound(graph, max_qubits)
    for state in pool:
        evaluation = graph.evaluate(state, max_qubits=max_qubits, partition_mode="multiway")
        if evaluation.feasible:
            best = min(best, evaluation.log_overhead)
    return best


def classify(wg, wge, tol=TOL):
    delta_low = (
        wg["bound"] - wge["objective"]
        if wg["bound"] is not None and wge["objective"] is not None
        else None
    )
    delta_high = (
        wg["objective"] - wge["bound"]
        if wg["objective"] is not None and wge["bound"] is not None
        else None
    )
    if delta_low is not None and delta_low > tol:
        label = "proven_improvement"
    elif (
        wg["proven_optimal"]
        and wge["proven_optimal"]
        and wg["objective"] is not None
        and wge["objective"] is not None
        and abs(wg["objective"] - wge["objective"]) <= tol
    ):
        label = "proven_no_gain"
    else:
        label = "inconclusive"
    return label, delta_low, delta_high


def solution_record(solution):
    return dict(
        status=solution.status,
        objective=solution.objective,
        bound=solution.bound,
        gap_abs=solution.gap_abs,
        gap_rel=solution.gap_rel,
        proven_optimal=solution.proven_optimal,
        runtime_seconds=solution.runtime_seconds,
        node_count=solution.node_count,
        solver=solution.solver,
        solver_options={k: v for k, v in solution.solver_options.items() if k != "disp"},
    )


def verify_and_describe(graph, model, solution, *, max_qubits):
    """Section 9 recompilation check; returns (ok, details) or (None, reason) if no incumbent."""
    if solution.raw_x is None:
        return None, "no_incumbent"
    try:
        plan = milp_plan(model, solution.raw_x)
        compiled = compile_hybrid(graph, plan)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the batch.
        return False, f"compile_error: {exc}"
    matches_objective = np.isclose(
        compiled.evaluation.log_overhead, solution.objective, atol=1e-6, rtol=0
    )
    widths_ok = max(compiled.evaluation.block_widths) <= max_qubits
    width_conservation = sum(compiled.evaluation.block_widths) == sum(
        compiled.evaluation.component_widths
    )
    excision_ids = excision_selection_from_solution(model, solution.raw_x) if model.excision else ()
    ok = bool(matches_objective and widths_ok and width_conservation)
    return ok, dict(
        recompiled_log_overhead=compiled.evaluation.log_overhead,
        matches_milp_objective=bool(matches_objective),
        widths=compiled.evaluation.block_widths,
        widths_within_capacity=bool(widths_ok),
        segment_conservation_ok=bool(width_conservation),
        excision_count=len(excision_ids),
        excision_group_ids=list(excision_ids),
        compiled_gamma=compiled.gamma,
    )


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
    out_dir = task["out_dir"]
    path = out_dir / "cases" / f"{name}.json"
    if path.exists():
        return name
    started = perf_counter()
    benchmark = build_benchmark(family, params)
    n = benchmark.problem.circuit.num_qubits
    two_qubit_gates = sum(
        item.operation.num_qubits == 2 for item in benchmark.problem.circuit.data
    )
    graph = PartitionGraph(
        benchmark.problem,
        excision=True,
        grouping=GroupingConfig(families=("excision",), exact_selection_limit=16),
    )
    ub = upper_bound(graph, capacity)
    k = compute_k_safe(graph, capacity, ub)

    solver_kwargs = {} if threads is None else {"threads": threads}
    wg_model = build_milp(graph, max_qubits=capacity, K=k, excision=False)
    wge_model = build_milp(graph, max_qubits=capacity, K=k, excision=True)
    wg_solution = solve_milp(
        wg_model,
        solver=solver,
        time_limit=time_limit,
        mip_gap_abs=mip_gap_abs,
        mip_gap_rel=mip_gap_rel,
        **solver_kwargs,
    )
    wge_solution = solve_milp(
        wge_model,
        solver=solver,
        time_limit=time_limit,
        mip_gap_abs=mip_gap_abs,
        mip_gap_rel=mip_gap_rel,
        **solver_kwargs,
    )

    wg_record = solution_record(wg_solution)
    wge_record = solution_record(wge_solution)
    label, delta_low, delta_high = classify(wg_record, wge_record)

    wg_ok, wg_detail = verify_and_describe(graph, wg_model, wg_solution, max_qubits=capacity)
    wge_ok, wge_detail = verify_and_describe(graph, wge_model, wge_solution, max_qubits=capacity)

    omega_ratio = None
    if wg_solution.objective is not None and wge_solution.objective is not None:
        omega_ratio = float(np.exp(wg_solution.objective - wge_solution.objective))

    record = dict(
        name=name,
        family=family,
        params=params,
        n_qubits=n,
        two_qubit_gates=two_qubit_gates,
        max_qubits=capacity,
        k_safe=k,
        upper_bound_used=ub,
        wg=wg_record,
        wge=wge_record,
        wg_verification=wg_detail,
        wge_verification=wge_detail,
        wg_verification_ok=wg_ok,
        wge_verification_ok=wge_ok,
        classification=label,
        delta_low=delta_low,
        delta_high=delta_high,
        omega_ratio_wg_over_wge=omega_ratio,
        build_seconds=dict(wg=wg_model.build_seconds, wge=wge_model.build_seconds),
        elapsed_seconds=perf_counter() - started,
    )
    write_json(path, record)
    print(
        f"{name} (M={capacity}): {label}  WG={wg_record['status']}  WGE={wge_record['status']}"
        f"  ({record['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return name


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


def summarize(out_dir: Path):
    cases_dir = out_dir / "cases"
    records = [json.loads(p.read_text()) for p in sorted(cases_dir.glob("*.json"))]
    by_family = {}
    for record in records:
        by_family.setdefault(record["family"], []).append(record)
    table = []
    for family, rows in sorted(by_family.items()):
        counts = {"proven_improvement": 0, "proven_no_gain": 0, "inconclusive": 0}
        ratios = []
        runtimes = []
        for row in rows:
            counts[row["classification"]] += 1
            if row["classification"] == "proven_improvement" and row["omega_ratio_wg_over_wge"]:
                ratios.append(row["omega_ratio_wg_over_wge"])
            runtimes.append(row["wg"]["runtime_seconds"] + row["wge"]["runtime_seconds"])
        table.append(
            dict(
                family=family,
                instances=len(rows),
                proven_improvement=counts["proven_improvement"],
                proven_no_gain=counts["proven_no_gain"],
                inconclusive=counts["inconclusive"],
                omega_ratio_min=min(ratios) if ratios else None,
                omega_ratio_max=max(ratios) if ratios else None,
                omega_ratio_median=float(np.median(ratios)) if ratios else None,
                mean_runtime_seconds=float(np.mean(runtimes)) if runtimes else None,
                max_runtime_seconds=float(np.max(runtimes)) if runtimes else None,
            )
        )
    write_json(out_dir / "summary.json", dict(families=table, total_instances=len(records)))
    write_json(out_dir / "report.json", records)
    for row in table:
        print(
            f"{row['family']:14s} n={row['instances']:3d}  "
            f"improve={row['proven_improvement']:3d}  no_gain={row['proven_no_gain']:3d}  "
            f"inconclusive={row['inconclusive']:3d}  "
            f"median_ratio={row['omega_ratio_median']}",
            flush=True,
        )


def write_metadata(out_dir: Path, args):
    path = out_dir / "metadata.json"
    source_hashes = {
        str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*Path("src/qpartition").glob("*.py"), Path(__file__)]
    }
    write_json(
        path,
        dict(
            source_hashes=source_hashes,
            packages={
                p: importlib.metadata.version(p)
                for p in ("qiskit", "numpy", "scipy", "networkx")
            },
            scipy_version=scipy.__version__,
            python=platform.python_version(),
            platform=platform.platform(),
            command=sys.argv,
            config=vars(args) | {"output": str(out_dir)},
            scope="Certified WG vs WGE optima; CX-only instances, real gamma_exc=16, "
            "no multi-gate grouping, no assumed excision costs.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/milp-verification-pilot"))
    parser.add_argument("--phase", choices=("pilot", "final", "summarize"), default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--threads", type=int, help="Solver threads per solve (default: cpus/workers)."
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "gurobi", "highs"),
        default="auto",
        help="auto picks gurobi if gurobipy+license are available, else highs.",
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-6)
    parser.add_argument("--mip-gap-rel", type=float, default=0.0)
    parser.add_argument(
        "--final-spec", type=Path, help="Optional JSON file overriding the built-in final grid."
    )
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "metadata.json").exists():
        write_metadata(out_dir, args)

    if args.phase == "pilot":
        specs = pilot_definitions()
    elif args.phase == "final":
        specs = json.loads(args.final_spec.read_text()) if args.final_spec else final_definitions()
    else:
        specs = []

    threads = args.threads or max(1, (os.cpu_count() or args.workers) // args.workers)
    solver = detect_solver(args.solver)
    print(f"Using solver backend: {solver} (threads per solve: {threads})", flush=True)
    if specs:
        jobs = [
            dict(
                name=spec["name"],
                family=spec["family"],
                params=spec["params"],
                capacity=spec["capacity"],
                time_limit=spec.get("time_limit", args.time_limit),
                mip_gap_abs=args.mip_gap_abs,
                mip_gap_rel=args.mip_gap_rel,
                threads=threads,
                solver=solver,
                out_dir=out_dir,
            )
            for spec in specs
        ]
        parallel(solve_instance, jobs, args.workers)

    summarize(out_dir)


if __name__ == "__main__":
    main()
