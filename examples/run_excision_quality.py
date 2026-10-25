"""Reproducible no-grouping tuning/holdout study, using the original SA kernel.

Run with the research conda interpreter. Worker processes, not alternative search
kernels, provide parallelism and interruptible external-baseline deadlines.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

from qpartition import (
    AnnealConfig,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    load_problem,
    save_problem,
)
from qpartition.annealing import seed_states
from qpartition.large_benchmarks import large_benchmark
from qpartition.paper_baseline import PaperConfig, run_paper_reconstruction
from qpartition.qiskit_baseline import QiskitBaselineConfig, run_qiskit_baseline

PROFILES = {
    "original": {},
    "feasible_temperature": {"temperature_calibration": "feasible"},
    "slow_cooling": {"temperature_calibration": "feasible", "cooling_rate": 0.98},
    "fast_cooling": {"temperature_calibration": "feasible", "cooling_rate": 0.88},
    "more_regions": {"temperature_calibration": "feasible", "block_probability": 0.7},
    "fewer_regions": {"temperature_calibration": "feasible", "block_probability": 0.1},
    "long_intervals": {
        "temperature_calibration": "feasible",
        "block_probability": 0.6,
        "block_size": 8,
    },
    "fewer_splits": {"temperature_calibration": "feasible", "split_probability": 0.08},
    "more_splits": {"temperature_calibration": "feasible", "split_probability": 0.5},
    "combined": {
        "temperature_calibration": "feasible",
        "cooling_rate": 0.95,
        "initial_acceptance": 0.6,
        "block_probability": 0.6,
        "block_size": 8,
        "split_probability": 0.12,
    },
}


def write_json(path, value):
    """Atomic checkpoints and strict JSON; logarithmic scores survive overflow."""

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


def config(capacity, budget, profile):
    return AnnealConfig(
        capacity,
        partition_mode="multiway",
        max_evaluations=budget,
        steps=3 * budget // 5,
        pilot_steps=48,
        **PROFILES[profile],
    )


def definitions():
    train = [
        (f"train_{f}", f, 64, m, 17, layers)
        for f, m, layers in (
            ("graphstate", 48, 1),
            ("aqft", 22, 1),
            ("trotter_grid", 22, 3),
            ("random_brickwork", 48, 6),
            ("excision_butterfly", 17, 3),
        )
    ]
    test = []
    for family, layers in (
        ("graphstate", 1),
        ("aqft", 1),
        ("trotter_grid", 3),
        ("random_brickwork", 6),
    ):
        for n in (50, 100, 150, 250):
            test.append((f"{family}_{n}_M{3 * n // 4}", family, n, 3 * n // 4, 53, layers))
    for family, layers in (("aqft", 1), ("random_brickwork", 6)):
        for n in (100, 250):
            test.append((f"{family}_{n}_M{n // 3}", family, n, n // 3, 53, layers))
    for n in (100, 250):
        m = 1 + math.ceil((n - 2) / 4)
        test.append((f"control_{n}_M{m}", "excision_butterfly", n, m, 53, 3))
    return train, test


def prepare_one(args):
    root, spec, starts = args
    name, family, n, capacity, seed, layers = spec
    directory = Path(root) / "cases" / name
    if (directory / "case.json").exists():
        return name
    t = perf_counter()
    benchmark = large_benchmark(family, n, layers=layers, seed=seed)
    graph = PartitionGraph(benchmark.problem)
    excision = PartitionGraph(benchmark.problem, excision=True)
    assert graph.nodes == excision.nodes and graph.edges == excision.edges
    assert excision.grouping.config.families == ("excision",)
    assert not excision.grouping.orbits and not excision.group_regions
    preprocessing = perf_counter() - t
    t = perf_counter()
    pool = seed_states(graph, config(capacity, 100, "original"), count=starts, seed=991)
    seeding = perf_counter() - t
    save_problem(benchmark.problem, directory / "input")
    write_json(directory / "pool.json", pool)
    write_json(
        directory / "case.json",
        dict(
            name=name,
            family=family,
            qubits=n,
            max_qubits=capacity,
            two_qubit_gates=len(graph.gate_variables),
            vertices=len(graph.nodes),
            variables=len(graph.variables),
            depth=benchmark.problem.circuit.depth(),
            source_fingerprint=benchmark.problem.circuit_fingerprint,
            provenance=benchmark.provenance,
            preprocessing_seconds=preprocessing,
            seed_pool_seconds=seeding,
            pool_size=len(pool),
            pool_policy="Original seed_states on wire/gate graph; identical pool for both methods. "
            "No known separators or baseline winners supplied.",
        ),
    )
    return name


def run_sa(task):
    root, name, method, profile, seed, budget, audit = task
    directory = Path(root) / "cases" / name
    phase = "final" if audit else "tuning"
    target = directory / phase / f"{method}_{profile}_{seed}.json"
    if target.exists():
        row = json.loads(target.read_text())
        expected = asdict(config(row["config"]["max_qubits"], budget, profile))
        if row["config"] != expected:
            raise ValueError("Checkpoint configuration differs; use a fresh output directory.")
        return row
    case = json.loads((directory / "case.json").read_text())
    problem = load_problem(directory / "input")
    graph = PartitionGraph(problem, excision=method == "wge")
    pool = json.loads((directory / "pool.json").read_text())
    initial = tuple(pool[seed % len(pool)])
    cfg = config(case["max_qubits"], budget, profile)
    result = anneal(graph, cfg, seed=1000 + seed, initial_state=initial)
    best = result.best
    row = dict(
        case=name,
        method=method,
        profile=profile,
        seed=1000 + seed,
        seed_pool_index=seed % len(pool),
        config=asdict(cfg),
        status="feasible" if best else "no-feasible-incumbent",
        initial_log_overhead=result.initial_log_overhead,
        initial_feasible=result.initial_feasible,
        initial_temperature=result.initial_temperature,
        final_temperature=result.final_temperature,
        seconds=result.elapsed_seconds,
        evaluations=result.evaluations,
        accepted=result.accepted_moves,
        changed=result.changed_proposals,
        proposals=result.proposals,
        polish_evaluations=result.polish_evaluations,
        before_polish=result.best_before_polish,
        budget_exhausted=result.budget_exhausted,
        proposed_action_changes=result.proposed_action_changes,
        accepted_action_changes=result.accepted_action_changes,
        trace=result.trace,
    )
    if best:
        row.update(
            log_sampling_overhead=best.log_overhead,
            sampling_overhead=best.sampling_overhead,
            widths=best.block_widths,
            fragments=len(best.block_widths),
            counts=best.action_counts,
            segments=best.segment_count,
            excision_wire_cuts=best.excision_wire_cuts,
            raw_wire_cuts=sum(
                v.kind == "wire" and bool(x)
                for v, x in zip(graph.variables, best.state, strict=True)
            ),
            realization_selection_complete=best.grouping_selection_complete,
        )
        assert max(best.block_widths) <= cfg.max_qubits
        assert best.segment_count == problem.circuit.num_qubits + row["raw_wire_cuts"]
        if audit:
            target.parent.mkdir(parents=True, exist_ok=True)
            plan = HybridPlan.create(
                graph, best.state, max_qubits=cfg.max_qubits, partition_mode="multiway"
            )
            compiled = compile_hybrid(graph, plan)
            recipe_log = 2 * sum(math.log(t.gamma) for t in compiled.templates)
            assert math.isclose(recipe_log, best.log_overhead, abs_tol=1e-8)
            branch = compiled.experiment((0,) * len(compiled.templates))
            assert tuple(c.num_qubits for c in branch.circuits) == best.block_widths
            row["physical_recipe_and_width_audit"] = True
            plan.save(target.with_suffix(".plan.json"))
    write_json(target, row)
    return row


def parallel(function, jobs, workers):
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
            if len(results) % 10 == 0 or len(results) == len(futures):
                print(f"{function.__name__}: {len(results)}/{len(futures)}", flush=True)
    return results


def choose_profile(root, names):
    """Equal case weights, median over starts; failure rate precedes cost."""
    rows = [
        json.loads(p.read_text())
        for name in names
        for p in (Path(root) / "cases" / name / "tuning").glob("*.json")
    ]
    cells = {}
    for name in names:
        for profile in PROFILES:
            subset = [r for r in rows if r["case"] == name and r["profile"] == profile]
            valid = [
                r["log_sampling_overhead"] / math.log(10)
                for r in subset
                if r["status"] == "feasible"
            ]
            cells[name, profile] = median(valid) if len(valid) == len(subset) and valid else None
    floors = {
        name: min(cells[name, p] for p in PROFILES if cells[name, p] is not None) for name in names
    }
    scores = []
    for profile in PROFILES:
        valid = [
            cells[name, profile] - floors[name]
            for name in names
            if cells[name, profile] is not None
        ]
        subset = [r for r in rows if r["profile"] == profile]
        scores.append(
            dict(
                profile=profile,
                changes=PROFILES[profile],
                failures=sum(r["status"] != "feasible" for r in subset),
                mean_median_log10_regret=sum(valid) / len(valid) if valid else float("inf"),
                median_seconds=median(r["seconds"] for r in subset),
                median_acceptance=median(r["accepted"] / max(1, r["changed"]) for r in subset),
                median_polish_evaluations=median(r["polish_evaluations"] for r in subset),
                cells={name: cells[name, profile] for name in names},
            )
        )
    scores.sort(key=lambda r: (r["failures"], r["mean_median_log10_regret"], r["profile"]))
    report = dict(
        selected=scores[0]["profile"],
        profiles=scores,
        selection="Failures, then mean across training cases of median-start "
        "log10 regret; profile name breaks exact ties. Holdouts not used.",
    )
    write_json(Path(root) / "tuning.json", report)
    return report["selected"]


def baseline_worker(root, name, method, starts):
    directory = Path(root) / "cases" / name
    case = json.loads((directory / "case.json").read_text())
    problem = load_problem(directory / "input")
    capacity = case["max_qubits"]
    target = directory / f"{method}.json"
    if method == "paper" and problem.circuit.num_qubits > 2 * capacity:
        write_json(target, dict(status="proven-infeasible-two-way", runs=[], completed=0))
        return
    runs, best = [], None
    for seed in range(starts):
        try:
            if method == "qiskit":
                _, row = run_qiskit_baseline(
                    problem,
                    QiskitBaselineConfig(
                        capacity,
                        seed=7000 + seed,
                        max_gamma=1e100,
                        max_backjumps=10000,
                    ),
                )
                row["status"] = "feasible"
            else:
                _, row = run_paper_reconstruction(
                    problem,
                    PaperConfig(
                        capacity,
                        seed=7000 + seed,
                        max_iter=10,
                        spatial_sizes="balanced",
                        capacity_model="physical",
                    ),
                    groups=(),
                )
                row.pop("groups", None)
                row.pop("grouped_status", None)
            runs.append(row)
            if row["status"] == "feasible" and (
                best is None or row["log_sampling_overhead"] < best["log_sampling_overhead"]
            ):
                best = row
        except Exception as exc:
            runs.append(dict(status="error", error=f"{type(exc).__name__}: {exc}"))
        write_json(
            target,
            dict(
                status="completed" if seed + 1 == starts else "running",
                completed=len(runs),
                requested=starts,
                best=best,
                runs=runs,
            ),
        )


def bounded_baseline(task):
    root, name, method, starts, timeout = task
    target = Path(root) / "cases" / name / f"{method}.json"
    if target.exists() and json.loads(target.read_text())["status"] != "running":
        return name, method
    t = perf_counter()
    with target.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--phase",
                "baseline-worker",
                "--case",
                name,
                "--method",
                method,
                "--baseline-starts",
                str(starts),
            ],
            stdout=log,
            stderr=log,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    row = json.loads(target.read_text()) if target.exists() else dict(completed=0, runs=[])
    if timed_out:
        row["status"] = "time-limit-with-incumbent" if row.get("best") else "timeout"
    elif process.returncode:
        row["status"] = "worker-error"
    row.update(
        deadline_seconds=timeout,
        wall_seconds=perf_counter() - t,
        process_returncode=process.returncode,
    )
    write_json(target, row)
    print(f"baseline: {name} {method} {row['status']} ({row['completed']})", flush=True)
    return name, method


def summarize(root, names):
    root = Path(root)
    tuning = json.loads((root / "tuning.json").read_text())
    report = []
    for name in names:
        directory = root / "cases" / name
        case = json.loads((directory / "case.json").read_text())
        methods = {}
        for method in ("wg", "wge"):
            rows = [
                json.loads(p.read_text())
                for p in (directory / "final").glob(f"{method}_{tuning['selected']}_*.json")
                if not p.name.endswith(".plan.json")
            ]
            valid = [r for r in rows if r["status"] == "feasible"]
            best = min(valid, key=lambda r: r["log_sampling_overhead"]) if valid else None
            methods[method] = dict(
                best=best,
                starts=len(rows),
                feasible=len(valid),
                median_log_sampling_overhead=median(r["log_sampling_overhead"] for r in valid)
                if valid
                else None,
                total_search_seconds=sum(r["seconds"] for r in rows),
                total_evaluations=sum(r["evaluations"] for r in rows),
            )
            if best:
                source = (
                    directory
                    / "final"
                    / (f"{method}_{best['profile']}_{best['seed'] - 1000}.plan.json")
                )
                destination = root / "winners" / f"{name}_{method}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        for method in ("qiskit", "paper"):
            path = directory / f"{method}.json"
            baseline = json.loads(path.read_text()) if path.exists() else dict(status="pending")
            methods[method] = {k: v for k, v in baseline.items() if k != "runs"}
            methods[method]["run_status_counts"] = {
                status: sum(r["status"] == status for r in baseline.get("runs", []))
                for status in {r["status"] for r in baseline.get("runs", [])}
            }
        report.append(case | dict(methods=methods))
    write_json(root / "report.json", report)
    lines = [
        "# Large-circuit excision quality study",
        "",
        "All values in the comparison table are **log10 sampling overhead** (squared "
        "coefficient norm). Physical PTM excision recipes; no assumed discounts. "
        "Only wire cuts, gate cuts, and individual two-qubit excisions are enabled.",
        "",
        f"Training selected `{tuning['selected']}`. Starts use the original generic seed "
        "pool, shared between WG and WGE. No hand-supplied separators or baseline winners. "
        "The synthetic control is deliberately favorable and is not an application workload.",
        "",
        "| Circuit | N | M | WG | WGE | Excisions | Fragments | Qiskit | Paper reconstruction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def score(row):
        return f"{row['log_sampling_overhead'] / math.log(10):.4f}" if row else "—"

    for r in report:
        m = r["methods"]
        wge = m["wge"]["best"]
        baselines = []
        for method in ("qiskit", "paper"):
            b = m[method]
            baselines.append(
                f"{score(b.get('best'))} ({b.get('completed', 0)} starts; {b['status']})"
            )
        lines.append(
            f"| {r['name']} | {r['qubits']} | {r['max_qubits']} | "
            f"{score(m['wg']['best'])} | {score(wge)} | "
            f"{wge['counts'].get('excision', 0) if wge else '—'} | "
            f"{wge['fragments'] if wge else '—'} | " + " | ".join(baselines) + " |"
        )
    lines += [
        "",
        "## Hyperparameter sweep",
        "",
        "| Profile | Mean median log10 regret | Median acceptance | Median polish evaluations |",
        "|---|---:|---:|---:|",
    ]
    for r in tuning["profiles"]:
        lines.append(
            f"| {r['profile']} | {r['mean_median_log10_regret']:.5f} | "
            f"{r['median_acceptance']:.3f} | {r['median_polish_evaluations']:.0f} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Qiskit is the installed official addon, using gate and wire cuts. The paper "
        "baseline is our two-stage KL reconstruction with physical static-width guards, "
        "not author code. Its two-way infeasibility under tight bounds is not a quality "
        "victory. Deadlines preserve only completed restarts; interrupted runs do not "
        "supply an incumbent. Errors/timeouts do not receive a numerical score.",
        "",
        "Each SA run has a fixed cache-miss budget, including pilots and polishing. "
        "Preprocessing and shared original seed construction are timed separately in "
        "`report.json`. Baseline time budgets and restart counts differ, so these are "
        "bounded quality comparisons, not equal-runtime rankings or optimality claims. "
        "Large plans are checked by physical recipe norms and a materialized branch; "
        "large quantum states and all QPD branches are not simulated.",
        "",
    ]
    analysis, extra = analyze(root, report)
    write_json(root / "analysis.json", analysis)
    lines += extra
    (root / "SUMMARY.md").write_text("\n".join(lines))


def analyze(root, report):
    """Separate physical repricing, co-optimization, and finite-budget search effects."""
    rows, baseline_counts = [], {}
    for case in report:
        methods = case["methods"]
        wg, wge = methods["wg"]["best"], methods["wge"]["best"]
        if not wg or not wge:
            continue
        directory = root / "cases" / case["name"]
        path = root / "winners" / f"{case['name']}_wg.json"
        graph = PartitionGraph(load_problem(directory / "input"), excision=True)
        post = graph.evaluate(
            HybridPlan.load(path).state, max_qubits=case["max_qubits"], partition_mode="multiway"
        )
        assert post.feasible and post.log_overhead <= wg["log_sampling_overhead"] + 1e-8
        initial = {}
        for method in ("wg", "wge"):
            starts = [
                json.loads(p.read_text())
                for p in (directory / "final").glob(
                    f"{method}_{methods[method]['best']['profile']}_*.json"
                )
                if not p.name.endswith(".plan.json")
            ]
            initial[method] = min(r["initial_log_overhead"] for r in starts)
        rows.append(
            dict(
                case=case["name"],
                control=case["family"] == "excision_butterfly",
                wg_log10=wg["log_sampling_overhead"] / math.log(10),
                wge_log10=wge["log_sampling_overhead"] / math.log(10),
                post_excision_log10=post.log_overhead / math.log(10),
                post_counts=post.action_counts,
                search_gain_log10={
                    m: (initial[m] - methods[m]["best"]["log_sampling_overhead"]) / math.log(10)
                    for m in ("wg", "wge")
                },
                best_initial_log10={m: initial[m] / math.log(10) for m in ("wg", "wge")},
                excisions=wge["counts"].get("excision", 0),
            )
        )
    natural = [r for r in rows if not r["control"]]
    stats = dict(
        completed_cases=len(rows),
        natural_wge_vs_wg={
            "better": sum(r["wge_log10"] < r["wg_log10"] - 1e-7 for r in natural),
            "equal": sum(abs(r["wge_log10"] - r["wg_log10"]) <= 1e-7 for r in natural),
            "worse": sum(r["wge_log10"] > r["wg_log10"] + 1e-7 for r in natural),
        },
        natural_winners_with_excision=sum(r["excisions"] > 0 for r in natural),
        co_optimization_beats_post_only=sum(
            r["wge_log10"] < r["post_excision_log10"] - 1e-7 for r in rows
        ),
    )
    for method in ("qiskit", "paper"):
        comparisons = []
        for case in report:
            ours, other = case["methods"]["wge"]["best"], case["methods"][method].get("best")
            if ours and other:
                comparisons.append(ours["log_sampling_overhead"] - other["log_sampling_overhead"])
        baseline_counts[method] = dict(
            comparable=len(comparisons),
            better=sum(x < -1e-7 for x in comparisons),
            equal=sum(abs(x) <= 1e-7 for x in comparisons),
            worse=sum(x > 1e-7 for x in comparisons),
        )
    extra = [
        "",
        "## Excision attribution",
        "",
        "`WG + post` reprices the WG winner with the same physical excision LUT and "
        "performs no additional search. It distinguishes an available decomposition "
        "saving from a benefit of changing the search trajectory. These diagnostics "
        "do not add starts to WGE in the main comparison.",
        "",
        "| Case | WG log10 | WG + post log10 | WGE search log10 | WGE excisions |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        extra.append(
            f"| {r['case']} | {r['wg_log10']:.4f} | "
            f"{r['post_excision_log10']:.4f} | {r['wge_log10']:.4f} | {r['excisions']} |"
        )
    s = stats["natural_wge_vs_wg"]
    extra += [
        "",
        f"Among completed ordinary cases, WGE is better/equal/worse than WG "
        f"on {s['better']}/{s['equal']}/{s['worse']} cases; "
        f"{stats['natural_winners_with_excision']} ordinary winners contain excision.",
        "",
        "A finite-budget WGE trajectory can lose to WG despite pointwise cheaper "
        "realizations: changed costs change acceptance and the visited states. "
        "Its comparison with the post-only column makes that distinction explicit.",
        "",
    ]
    confirmation = root / "confirmation.json"
    if confirmation.exists():
        extra += [
            "## Held-out profile confirmation",
            "",
            "Same five starts and 4000-evaluation limits on four 100-qubit cases. "
            "This diagnostic does not retune the selected profile or enter the main table.",
            "",
            "| Case | Original best / median log10 | Selected best / median log10 |",
            "|---|---:|---:|",
        ]
        selected = json.loads((root / "tuning.json").read_text())["selected"]
        for row in json.loads(confirmation.read_text()):
            a, b = row["original"], row[selected]

            def pair(cell):
                return (
                    f"{cell['best_log10']:.4f} / {cell['median_log10']:.4f}"
                    if cell["best_log10"] is not None
                    else "pending"
                )

            extra.append(f"| {row['case']} | {pair(a)} | {pair(b)} |")
    return dict(stats=stats, external_comparisons=baseline_counts, cases=rows), extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/excision_quality_v1"))
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "prepare",
            "tune",
            "final",
            "baselines",
            "summarize",
            "confirmation",
            "baseline-worker",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tuning-starts", type=int, default=3)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--tuning-evaluations", type=int, default=1600)
    parser.add_argument("--evaluations", type=int, default=4000)
    parser.add_argument("--baseline-starts", type=int, default=50)
    parser.add_argument("--baseline-timeout", type=float, default=60)
    parser.add_argument("--case")
    parser.add_argument("--method", choices=("paper", "qiskit"))
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    root = args.output.resolve()
    if args.phase == "baseline-worker":
        baseline_worker(root, args.case, args.method, args.baseline_starts)
        return
    train, test = definitions()
    if args.case:
        test = [s for s in test if s[0] == args.case]
        if not test:
            parser.error("Unknown test case")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "metadata.json").exists():
        hashes = {
            str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path("src/qpartition").glob("*.py"), Path(__file__)]
        }
        preserved = {
            f: Path("src/qpartition", f).read_bytes()
            == subprocess.run(
                ["git", "show", f"HEAD:src/qpartition/{f}"],
                capture_output=True,
                check=True,
            ).stdout
            for f in ("annealing.py", "label_moves.py", "multiway.py", "group_heatbath.py")
        }
        assert all(preserved.values())
        write_json(
            root / "metadata.json",
            dict(
                source_hashes=hashes,
                original_kernels_unchanged=preserved,
                packages={
                    p: importlib.metadata.version(p)
                    for p in ("qiskit", "qiskit-addon-cutting", "networkx", "numpy", "scipy")
                },
                python=platform.python_version(),
                platform=platform.platform(),
                command=sys.argv,
                config=vars(args) | {"output": str(root)},
                profiles=PROFILES,
                train=[s[0] for s in train],
                test=[s[0] for s in test],
                scope="WG versus WGE; no multi-gate grouping, no assumed excision costs.",
            ),
        )
    write_json(
        root / f"execution_{args.phase}.json",
        dict(
            config=vars(args) | {"output": str(root)},
            source_hashes={
                str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*Path("src/qpartition").glob("*.py"), Path(__file__)]
            },
        ),
    )
    if args.phase in ("all", "prepare"):
        parallel(prepare_one, [(root, s, args.starts) for s in train + test], args.workers)
    if args.phase in ("all", "tune"):
        parallel(
            run_sa,
            [
                (root, s[0], "wge", p, seed, args.tuning_evaluations, False)
                for s in train
                for p in PROFILES
                for seed in range(args.tuning_starts)
            ],
            args.workers,
        )
        print("Selected:", choose_profile(root, [s[0] for s in train]), flush=True)
    if args.phase in ("all", "final"):
        selected = json.loads((root / "tuning.json").read_text())["selected"]
        parallel(
            run_sa,
            [
                (root, s[0], method, selected, seed, args.evaluations, True)
                for s in test
                for method in ("wg", "wge")
                for seed in range(args.starts)
            ],
            args.workers,
        )
    if args.phase in ("all", "confirmation"):
        # This diagnostic never feeds profile selection or the main WG/WGE table.
        confirmation = [s for s in test if s[2] == 100 and s[3] == 75]
        parallel(
            run_sa,
            [
                (root, s[0], "wge", "original", seed, args.evaluations, True)
                for s in confirmation
                for seed in range(args.starts)
            ],
            args.workers,
        )
        rows = []
        selected = json.loads((root / "tuning.json").read_text())["selected"]
        for s in confirmation:
            result = dict(case=s[0])
            for profile in ("original", selected):
                data = [
                    json.loads(p.read_text())
                    for p in (root / "cases" / s[0] / "final").glob(f"wge_{profile}_*.json")
                    if not p.name.endswith(".plan.json")
                ]
                values = [
                    r["log_sampling_overhead"] / math.log(10)
                    for r in data
                    if r["status"] == "feasible"
                ]
                result[profile] = dict(
                    starts=len(data),
                    feasible=len(values),
                    best_log10=min(values) if values else None,
                    median_log10=median(values) if values else None,
                )
            rows.append(result)
        write_json(root / "confirmation.json", rows)
    if args.phase in ("all", "baselines"):
        tasks = [
            (root, s[0], method, args.baseline_starts, args.baseline_timeout)
            for s in test
            for method in ("qiskit", "paper")
        ]
        with ThreadPoolExecutor(max_workers=min(4, args.workers)) as executor:
            list(executor.map(bounded_baseline, tasks))
    if args.phase in ("all", "final", "baselines", "summarize", "confirmation"):
        summarize(root, [s[0] for s in test])


if __name__ == "__main__":
    main()
