"""Setting A: certified WG vs WGE MILP comparison (docs/setting_a_prompt.txt).

Setting A is a DIFFERENT cost convention than the SWAP-peeled "fair
baseline" used by Part 1/Phase 2a (docs/second_stage_cont_prompt.txt): here
SWAP is priced as an ordinary gate (gate-cut cost 7, matching the paper's own
convention), and SWAP relabeling/peeling is disabled everywhere. This is a
pure re-orchestration, not a redesign:

- Gate-cut cost: the registry's OWN `templates.py`/`QPDBasis`-based cost,
  used as-is (it never had SWAP peeling baked in -- that was entirely this
  project's `preprocessing.peel_swaps`, which this script simply never
  calls). Fusion (`preprocessing.fuse_two_qubit_gates`) IS still applied.
- Excision cost: `TemplateRegistry.for_gate` already computes a REAL,
  gate-content-specific cost for every two-qubit gate via
  `excision.decompose_excision` (confirmed by direct inspection: no
  Clifford restriction, no gate-type branching) whenever `excision_gamma`
  is left at its default `None` -- i.e. the MILP's excision costs are
  already real per-gate values with zero code changes, not an assumed
  global constant. `excision_lp.gate_excision_cost` is run alongside (not
  instead) to report each gate's EXACT/CERTIFIED/UPPER_BOUND status, and as
  a cross-check that the two independent constructions agree (they are
  documented to coincide for every gate tested: `excision_lp.py`'s own
  docstring states its Pauli-pool construction reproduces `excision.py`'s).
- Method: MILP only (`qpartition.milp.build_milp`/`solve_milp`), no SA, no
  junction-count screening. `seed_states`+`cheap_feasible_upper_bound`
  (used only to tighten K_safe) are a deterministic spatial-bipartition
  heuristic, not simulated annealing -- same machinery
  `run_milp_verification.py` already uses for this purpose.

Two phases (checkpointed, like the project's other staged runners):
  prepare  -- generate every instance once, save as a QPY+json CircuitProblem
              snapshot under --output/cases/<name>/ (sequential; the family
              generators use closures that aren't picklable for workers).
  solve    -- load each saved snapshot and run the WG/WGE MILP pair
              (parallelizable; each task is a plain path + config dict).
  summarize -- gather all solved cases into the deliverable CSV + print a
              per-family and overall summary.

Usage:
  python examples/run_setting_a_certification.py --phase prepare --output results/setting_a
  python examples/run_setting_a_certification.py --phase solve --output results/setting_a \
      --workers 6 --threads 2 --time-limit 120 --solver auto
  python examples/run_setting_a_certification.py --phase summarize --output results/setting_a
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import pi
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_breadth_survey as breadth  # noqa: E402 -- reuse FAMILIES/_tight_capacities

from qpartition import (  # noqa: E402
    GroupingConfig,
    PartitionGraph,
    build_milp,
    compile_hybrid,
    compute_k_safe,
    milp_plan,
    solve_milp,
)
from qpartition.annealing import AnnealConfig, seed_states  # noqa: E402
from qpartition.circuits import CircuitProblem, load_problem, save_problem  # noqa: E402
from qpartition.cover_cuts import cover_cut_pool  # noqa: E402
from qpartition.excision_lp import gate_excision_cost  # noqa: E402
from qpartition.lattice_benchmarks import heisenberg_xy_lattice  # noqa: E402
from qpartition.milp import (  # noqa: E402
    cheap_feasible_upper_bound,
    excision_selection_from_solution,
)
from qpartition.preprocessing import fuse_two_qubit_gates, two_qubit_gate_count  # noqa: E402

TOL = 1e-6


def detect_solver(requested: str) -> str:
    """Resolve "auto" to Gurobi when available, else HiGHS -- same logic as
    `run_milp_verification.py`'s own `detect_solver` (`solve_milp` itself
    has no "auto" branch; that resolution is always done by the caller)."""
    if requested != "auto":
        return requested
    try:
        import gurobipy as gp

        with gp.Env():
            pass
        return "gurobi"
    except Exception:
        return "highs"


def setting_a_problem(raw_problem: CircuitProblem) -> CircuitProblem:
    """Mandatory Setting A preprocessing: fusion only, no SWAP relabel/peel."""
    fused = fuse_two_qubit_gates(raw_problem.circuit)
    return CircuitProblem.create(fused, raw_problem.observable)


def _lattice_sweep_tasks():
    """4 thetas x 3 lattices x 2 models x 2 sizes -- docs/setting_a_prompt.txt Section 3."""
    thetas = [pi / 16, pi / 8, 3 * pi / 16, pi / 4]
    tasks = []
    for model in ("xy", "heisenberg"):
        for topology in ("chain", "ladder", "2d_square"):
            for theta_index, theta in enumerate(thetas, start=1):
                for n in (8, 12):
                    def generator(n=n, model=model, topology=topology, theta=theta):
                        return heisenberg_xy_lattice(
                            topology, n, model=model, theta=theta, layers=1
                        ).problem

                    family = f"{model}_{topology}_theta{theta_index}of4"
                    tasks.append(
                        dict(
                            family=family,
                            source="self-generated (lattice physics)",
                            version="v1",
                            generator=generator,
                            n=n,
                            theta_over_pi=theta / pi,
                            is_negative_control=False,
                        )
                    )
    return tasks


def _safe_name(family: str, n_qubits: int, capacity: int) -> str:
    return f"{family}__n{n_qubits}__cap{capacity}".replace("/", "_")


def phase_prepare(output: Path) -> None:
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(family, source, version, params, raw_problem, is_negative_control):
        try:
            prepared = setting_a_problem(raw_problem)
        except Exception as exc:  # noqa: BLE001
            print(f"[{family} {params}] PREPROCESSING ERROR: {exc}")
            return
        n_qubits = prepared.circuit.num_qubits
        for capacity in breadth._tight_capacities(n_qubits):
            name = _safe_name(family, n_qubits, capacity)
            case_dir = cases_dir / name
            meta_path = case_dir / "setting_a_meta.json"
            if meta_path.exists():
                manifest.append(name)
                continue
            save_problem(prepared, case_dir)
            meta_path.write_text(
                json.dumps(
                    dict(
                        family=family,
                        source=source,
                        version=version,
                        params=params,
                        is_negative_control=is_negative_control,
                        n_qubits=n_qubits,
                        capacity=capacity,
                        raw_two_qubit_gates=two_qubit_gate_count(raw_problem.circuit),
                        fused_two_qubit_gates=two_qubit_gate_count(prepared.circuit),
                    ),
                    indent=2,
                )
            )
            manifest.append(name)
            print(f"[prepare] {name}")

    for family, source, version, generator, sizes, is_negative_control in breadth.FAMILIES:
        for n in sizes:
            try:
                raw_problem = generator(n)
            except Exception as exc:  # noqa: BLE001
                print(f"[{family} n={n}] GENERATION ERROR: {exc}")
                continue
            emit(family, source, version, f"n={n}", raw_problem, is_negative_control)

    for task in _lattice_sweep_tasks():
        try:
            raw_problem = task["generator"]()
        except Exception as exc:  # noqa: BLE001
            print(f"[{task['family']} n={task['n']}] GENERATION ERROR: {exc}")
            continue
        params = f"n={task['n']},theta_over_pi={task['theta_over_pi']:.4f}"
        emit(
            task["family"], task["source"], task["version"], params,
            raw_problem, task["is_negative_control"],
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nPrepared {len(manifest)} cases under {cases_dir}")


def upper_bound(graph, max_qubits, seed=991):
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
        solver=solution.solver,
    )


_KNOWN_KAK_POINTS = {
    (0.0, 0.0, 0.0): "identity",
    (pi / 4, 0.0, 0.0): "CX/CZ-equivalent",
    (pi / 4, pi / 4, 0.0): "iSWAP-equivalent",
    (pi / 4, pi / 4, pi / 4): "SWAP-equivalent",
}


def _classify_kak(kak: tuple[float, float, float], tol: float = 1e-3) -> str:
    """Name a well-known KAK point, since post-fusion gates are generic
    `UnitaryGate`s (fusion always re-wraps even an unfused single gate,
    losing its original name -- see `preprocessing.fuse_two_qubit_gates`)."""
    for point, label in _KNOWN_KAK_POINTS.items():
        if all(abs(a - b) < tol for a, b in zip(kak, point, strict=True)):
            return label
    a, b, c = kak
    return f"generic(a={a:.4f},b={b:.4f},c={c:.4f})"


def _gate_status_report(problem: CircuitProblem) -> dict:
    """Run the LP module on every distinct (by KAK point) two-qubit gate in
    the circuit, reporting gamma_ex and its EXACT/CERTIFIED/UPPER_BOUND
    status -- docs/setting_a_prompt.txt Section 1's reporting requirement."""

    seen = {}
    for item in problem.circuit.data:
        if item.operation.num_qubits != 2:
            continue
        U = np.asarray(Operator(item.operation).data, dtype=complex)
        result = gate_excision_cost(U)
        key = tuple(round(v, 6) for v in result.kak)
        seen[key] = dict(
            gate_class=_classify_kak(result.kak), gamma_ex=result.gamma_ex, status=result.status
        )
    return seen


def solve_case(task: dict) -> dict:
    name = task["name"]
    case_dir = Path(task["case_dir"])
    out_path = case_dir / "setting_a_result.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    meta = json.loads((case_dir / "setting_a_meta.json").read_text())
    capacity = meta["capacity"]
    time_limit = task["time_limit"]
    solver = detect_solver(task["solver"])
    threads = task.get("threads")
    solver_kwargs = {} if threads is None else {"threads": threads}

    started = perf_counter()
    try:
        problem = load_problem(case_dir)
        gate_statuses = _gate_status_report(problem)
        graph = PartitionGraph(
            problem, excision=True, grouping=GroupingConfig(families=("excision",))
        )
        ub = upper_bound(graph, capacity)
        k = compute_k_safe(graph, capacity, ub)
        cuts = (
            cover_cut_pool(graph, max_qubits=capacity, excision=True, time_budget=30.0)
            if task.get("cover_cuts") else []
        )

        wg_model = build_milp(graph, max_qubits=capacity, K=k, excision=False, cover_cuts=cuts)
        wge_model = build_milp(graph, max_qubits=capacity, K=k, excision=True, cover_cuts=cuts)
        wg_solution = solve_milp(
            wg_model, solver=solver, time_limit=time_limit, mip_gap_abs=1e-7,
            mip_gap_rel=0.0, **solver_kwargs,
        )
        wge_start = None
        if wg_solution.raw_x is not None:
            extra = wge_model.n_vars - wg_model.n_vars
            wge_start = np.concatenate([wg_solution.raw_x, np.zeros(extra)])
        wge_solution = solve_milp(
            wge_model, solver=solver, time_limit=time_limit, mip_gap_abs=1e-7,
            mip_gap_rel=0.0, mip_start=wge_start, **solver_kwargs,
        )

        wg_record = solution_record(wg_solution)
        wge_record = solution_record(wge_solution)
        label, delta_low, delta_high = classify(wg_record, wge_record)

        excisions = []
        if wge_solution.raw_x is not None:
            group_ids = excision_selection_from_solution(wge_model, wge_solution.raw_x)
            for gid in group_ids:
                candidate = graph.grouping.by_id[gid]
                index = problem.index(candidate.anchor)
                instruction = problem.circuit.data[index].operation
                U = np.asarray(Operator(instruction).data, dtype=complex)
                gate_class = _classify_kak(gate_excision_cost(U).kak)
                excisions.append(dict(op_id=candidate.anchor, gate_class=gate_class, index=index))

        verification = None
        if wge_solution.raw_x is not None:
            try:
                plan = milp_plan(wge_model, wge_solution.raw_x)
                compiled = compile_hybrid(graph, plan)
                verification = dict(
                    matches_objective=bool(
                        np.isclose(
                            compiled.evaluation.log_overhead, wge_solution.objective,
                            atol=1e-6, rtol=0,
                        )
                    ),
                    widths_within_capacity=bool(max(compiled.evaluation.block_widths) <= capacity),
                )
            except Exception as exc:  # noqa: BLE001
                verification = dict(error=str(exc))

        result = dict(
            name=name, **meta, wg=wg_record, wge=wge_record, classification=label,
            delta_low=delta_low, delta_high=delta_high,
            omega_ratio=(
                float(np.exp(wg_record["objective"] - wge_record["objective"]))
                if wg_record["objective"] is not None and wge_record["objective"] is not None
                else None
            ),
            excisions_selected=excisions, gate_statuses=list(gate_statuses.values()),
            cover_cuts=len(cuts), excision_recipe=graph.registry.excision_recipe,
            verification=verification, elapsed_seconds=perf_counter() - started, error=None,
        )
    except Exception as exc:  # noqa: BLE001
        result = dict(
            name=name, **meta, error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(), elapsed_seconds=perf_counter() - started,
        )
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def phase_solve(
    output: Path, *, workers: int, threads: int | None, time_limit: float, solver: str,
    cover_cuts: bool = False,
) -> None:
    cases_dir = output / "cases"
    names = sorted(p.name for p in cases_dir.iterdir() if p.is_dir())
    tasks = [
        dict(name=name, case_dir=str(cases_dir / name), time_limit=time_limit,
             solver=solver, threads=threads, cover_cuts=cover_cuts)
        for name in names
    ]
    print(f"Solving {len(tasks)} cases with {workers} worker(s)...")
    if workers <= 1:
        for task in tasks:
            result = solve_case(task)
            print(f"[{result['name']}] {result.get('classification', result.get('error'))}")
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_case, task): task["name"] for task in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                print(f"[{name}] {result.get('classification', result.get('error'))}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] WORKER ERROR: {exc}")


def phase_summarize(output: Path) -> None:
    cases_dir = output / "cases"
    rows = []
    for case_dir in sorted(cases_dir.iterdir()):
        result_path = case_dir / "setting_a_result.json"
        if not result_path.exists():
            continue
        rows.append(json.loads(result_path.read_text()))

    csv_path = output / "setting_a_certification.csv"
    import csv as csv_module

    fields = [
        "name", "family", "source", "version", "params", "is_negative_control",
        "n_qubits", "raw_two_qubit_gates", "fused_two_qubit_gates", "capacity",
        "wg_objective", "wg_bound", "wg_proven_optimal", "wg_runtime_seconds",
        "wge_objective", "wge_bound", "wge_proven_optimal", "wge_runtime_seconds",
        "classification", "omega_ratio", "excision_count", "error",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            wg = r.get("wg") or {}
            wge = r.get("wge") or {}
            writer.writerow(
                dict(
                    name=r["name"],
                    family=r.get("family"),
                    source=r.get("source"),
                    version=r.get("version"),
                    params=r.get("params"),
                    is_negative_control=r.get("is_negative_control"),
                    n_qubits=r.get("n_qubits"),
                    raw_two_qubit_gates=r.get("raw_two_qubit_gates"),
                    fused_two_qubit_gates=r.get("fused_two_qubit_gates"),
                    capacity=r.get("capacity"),
                    wg_objective=wg.get("objective"),
                    wg_bound=wg.get("bound"),
                    wg_proven_optimal=wg.get("proven_optimal"),
                    wg_runtime_seconds=wg.get("runtime_seconds"),
                    wge_objective=wge.get("objective"),
                    wge_bound=wge.get("bound"),
                    wge_proven_optimal=wge.get("proven_optimal"),
                    wge_runtime_seconds=wge.get("runtime_seconds"),
                    classification=r.get("classification"),
                    omega_ratio=r.get("omega_ratio"),
                    excision_count=len(r.get("excisions_selected") or []),
                    error=r.get("error"),
                )
            )
    print(f"Wrote {len(rows)} rows to {csv_path}")

    by_family = {}
    for r in rows:
        by_family.setdefault(r.get("family"), []).append(r)
    print("\nPer-family summary:")
    for family, family_rows in sorted(by_family.items()):
        counts = {"proven_improvement": 0, "proven_no_gain": 0, "inconclusive": 0, "error": 0}
        ratios = []
        for r in family_rows:
            if r.get("error"):
                counts["error"] += 1
                continue
            counts[r["classification"]] += 1
            if r["classification"] == "proven_improvement" and r.get("omega_ratio"):
                ratios.append(r["omega_ratio"])
        ratio_str = f" ratios={[round(x, 2) for x in ratios]}" if ratios else ""
        print(f"  {family}: {counts}{ratio_str}")

    improved = [r for r in rows if r.get("classification") == "proven_improvement"]
    print(f"\n{len(improved)}/{len(rows)} instances PROVEN IMPROVEMENT under Setting A.")
    for r in improved:
        print(f"  {r['name']}: omega_ratio={r.get('omega_ratio'):.3f} "
              f"excisions={r.get('excisions_selected')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["prepare", "solve", "summarize"])
    parser.add_argument("--output", type=Path, default=Path("results/setting_a"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--cover-cuts", action="store_true",
                        help="add capacity cover cuts (valid inequalities) to both models")
    args = parser.parse_args()

    if args.phase == "prepare":
        phase_prepare(args.output)
    elif args.phase == "solve":
        phase_solve(
            args.output, workers=args.workers, threads=args.threads,
            time_limit=args.time_limit, solver=args.solver, cover_cuts=args.cover_cuts,
        )
    else:
        phase_summarize(args.output)


if __name__ == "__main__":
    main()
