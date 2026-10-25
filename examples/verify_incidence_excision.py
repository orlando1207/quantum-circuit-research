"""Matched-label excision checks using the original graph and SA kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import shutil
import subprocess
from dataclasses import asdict
from math import log
from pathlib import Path
from time import perf_counter

from qpartition import (
    AnnealConfig,
    HybridPlan,
    PartitionGraph,
    anneal,
    compile_hybrid,
    save_problem,
)
from qpartition.annealing import seed_states
from qpartition.benchmarks import qft_benchmark
from qpartition.large_benchmarks import large_benchmark
from qpartition.paper_baseline import PaperConfig, run_paper_reconstruction
from qpartition.qiskit_baseline import QiskitBaselineConfig, run_qiskit_baseline


def baseline_worker(channel, problem, capacity, method):
    try:
        if method == "qiskit":
            _, row = run_qiskit_baseline(problem, QiskitBaselineConfig(capacity, max_backjumps=200))
        else:
            _, row = run_paper_reconstruction(problem, PaperConfig(capacity, max_iter=2))
        channel.put(
            dict(
                status=row.get("status", "feasible"),
                log_sampling_overhead=row.get("log_sampling_overhead"),
                sampling_overhead=row.get("sampling_overhead"),
                widths=row.get("widths"),
            )
        )
    except Exception as exc:
        channel.put(dict(status="error", error=f"{type(exc).__name__}: {exc}"))


def baseline(problem, capacity, method, timeout):
    n = problem.circuit.num_qubits
    if method == "paper" and n > 2 * capacity:
        return dict(status="infeasible-two-way")
    vertices = 2 * n + 2 * sum(i.operation.num_qubits == 2 for i in problem.circuit.data)
    if method == "paper" and vertices > 1500:
        return dict(status="local-dense-reconstruction-resource-limit")
    ctx = mp.get_context("spawn")
    channel = ctx.Queue()
    process = ctx.Process(target=baseline_worker, args=(channel, problem, capacity, method))
    process.start()
    try:
        return channel.get(timeout=timeout)
    except queue.Empty:
        return dict(status="timeout", seconds=timeout)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/incidence-excision"))
    parser.add_argument("--evaluations", type=int, default=80)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory.")
    if args.starts < 1 or args.evaluations < 2:
        parser.error("Use positive starts and >=2 evaluations.")
    args.output.mkdir(parents=True, exist_ok=True)
    kernels = ["annealing.py", "label_moves.py", "multiway.py", "group_heatbath.py"]
    preserved = {
        f: Path("src/qpartition", f).read_bytes()
        == subprocess.run(
            ["git", "show", f"HEAD:src/qpartition/{f}"], capture_output=True, check=True
        ).stdout
        for f in kernels
    }
    assert all(preserved.values())
    source_hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("src/qpartition").glob("*.py")
    }
    for name in [*source_hashes, str(Path(__file__).resolve().relative_to(Path.cwd()))]:
        target = args.output / "executed_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    (args.output / "metadata.json").write_text(
        json.dumps(
            dict(
                unchanged_from_HEAD=preserved,
                source_hashes=source_hashes,
                config=vars(args) | {"output": str(args.output)},
                statement="Original label SA with matched initial edge states. "
                "Large-case starts are known witnesses, not new seed algorithms. "
                "Baselines have one bounded run; no grouping enabled.",
            ),
            indent=2,
        )
        + "\n"
    )
    suite = [large_benchmark("excision_butterfly", n) for n in (6, 18, 100, 1000)]
    suite += [qft_benchmark(n) for n in (6, 12)]
    suite += [large_benchmark("trotter_chain", n) for n in (100, 1000)]
    report = []
    for b in suite:
        t = perf_counter()
        a = PartitionGraph(b.problem)
        g = PartitionGraph(b.problem, excision=True)
        assert a.nodes == g.nodes and a.edges == g.edges and a.variables == g.variables
        cfg = AnnealConfig(
            b.max_qubits,
            partition_mode="multiway",
            steps=3 * args.evaluations,
            max_evaluations=args.evaluations,
            pilot_steps=16,
            cache_size=64,
        )
        save_problem(b.problem, args.output / b.name / "input")
        if b.problem.circuit.num_qubits <= 20:
            pool = seed_states(a, cfg, count=args.starts)
            start_kind = "unchanged independent seed_states pool, shared by both models"
        elif b.provenance["family"] == "excision_butterfly":
            state = [0] * len(a.variables)
            op = b.provenance["boundary_operation"]
            state[a.gate_variables[op]] = 1
            for wire in a.incoming_wires[op]:
                state[wire] = 1
            pool = (tuple(state),)
            start_kind = "known separator edge mask, identical for both models"
        else:
            labels = tuple(int(d["qubit"] >= b.max_qubits) for d in a.nodes)
            pool = (a.state_from_labels(labels),)
            start_kind = "contiguous spatial labels, identical for both models"
        record = dict(
            name=b.name,
            qubits=b.problem.circuit.num_qubits,
            max_qubits=b.max_qubits,
            depth=b.problem.circuit.depth(),
            vertices=len(a.nodes),
            variables=len(a.variables),
            preprocessing_seconds=perf_counter() - t,
            start_kind=start_kind,
            source_fingerprint=b.problem.circuit_fingerprint,
            anneal_config=asdict(cfg),
            methods={},
        )
        for name, graph in [("wire_gate", a), ("wire_gate_excision", g)]:
            rows = []
            best = None
            for seed in range(args.starts):
                run = anneal(graph, cfg, seed=seed, initial_state=pool[seed % len(pool)])
                ev = run.best
                assert ev is not None
                if best is None or ev.log_overhead < best.log_overhead:
                    best = ev
                rows.append(
                    dict(
                        seed=seed,
                        initial_log_overhead=run.initial_log_overhead,
                        log_overhead=ev.log_overhead,
                        counts=ev.action_counts,
                        widths=ev.block_widths,
                        evaluations=run.evaluations,
                        accepted=run.accepted_moves,
                        seconds=run.elapsed_seconds,
                    )
                )
            plan = HybridPlan.create(
                graph, best.state, max_qubits=b.max_qubits, partition_mode="multiway"
            )
            plan.save(args.output / f"{b.name}_{name}_plan.json")
            compiled = compile_hybrid(graph, plan)
            assert abs(2 * sum(log(t.gamma) for t in compiled.templates) - best.log_overhead) < 1e-9
            branch = compiled.experiment((0,) * len(compiled.templates))
            assert tuple(c.num_qubits for c in branch.circuits) == best.block_widths
            record["methods"][name] = dict(
                log_overhead=best.log_overhead,
                overhead=best.sampling_overhead,
                counts=best.action_counts,
                widths=best.block_widths,
                segments=best.segment_count,
                excision_wire_cuts=best.excision_wire_cuts,
                runs=rows,
            )
        for method in ("qiskit", "paper"):
            record["methods"][method] = baseline(b.problem, b.max_qubits, method, args.timeout)
        report.append(record)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            b.name,
            record["methods"]["wire_gate"]["overhead"],
            record["methods"]["wire_gate_excision"]["overhead"],
            flush=True,
        )
    lines = [
        "# Original incidence-graph excision validation",
        "",
        f"Matched starting states, {args.starts} label-SA starts per model. "
        "Only excision factors change. Large known-witness starts test integration, "
        "not search quality. Qiskit and reconstructed paper KL each receive one bounded run.",
        "",
        "| Circuit | N | M | Gate/wire | With excision | Qiskit | Paper reconstruction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def display(row):
        return (
            f"{row['sampling_overhead']:.6g}"
            if row.get("sampling_overhead") is not None
            else row["status"]
        )

    for r in report:
        methods = r["methods"]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    str(r["qubits"]),
                    str(r["max_qubits"]),
                    f"{methods['wire_gate']['overhead']:.6g}",
                    f"{methods['wire_gate_excision']['overhead']:.6g}",
                    display(methods["qiskit"]),
                    display(methods["paper"]),
                ]
            )
            + " |"
        )
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
