"""LNS vs Qiskit on the surveyed circuit families (fixed Setting A costs).

Instances: every case of the exact WG/WGE family survey (`results/family_survey/cases`),
plus the proven-improvement cases of the project's own lattice certification
(`results/setting_a_lp/cases`). The exact survey values are the reference.

Methods (single-threaded; T is the per-tier budget, every method gets 2T in total):
  qiskit    qiskit_addon_cutting.find_cuts. One task runs a max_backjumps ladder
            (1e4 = Qiskit's default, then 1e5, 1e6, unlimited) until 2T is used;
            "qiskit" reports the default rung, "qiskit_best" the best valid rung.
            Each plan is re-evaluated in this project's cost model and checked
            against Qiskit's own overhead. Rungs are checkpointed, so a rung
            killed for memory keeps the earlier ones.
  lns       the proposed pipeline: MILP-LNS on the WG model for T (-> "LNS-WG"),
            then MILP-LNS on the WGE model for T started from that solution
            (-> "LNS-WGE"; never worse than LNS-WG by construction).
  lns_wge_cold  ablation: MILP-LNS on the WGE model alone for 2T.

Phases: select | run | summarize. `summarize` writes per-instance and per-family CSVs.

  python examples/run_lns_family_benchmark.py --phase select --output results/lns_family
  QPARTITION_EXCISION_CACHE=results/excision_lp_cache.pkl \
      python examples/run_lns_family_benchmark.py --phase run --output results/lns_family
  python examples/run_lns_family_benchmark.py --phase summarize --output results/lns_family
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

METHODS = ("qiskit", "lns")
QISKIT_LADDER = (10_000, 100_000, 1_000_000, None)
TOL = 1e-6


def tier_of(g2: int) -> str:
    return "S" if g2 <= 32 else ("M" if g2 <= 100 else ("L" if g2 <= 300 else "X"))


# ---------------------------------------------------------------- selection


def _f(value):
    return None if value in (None, "None", "") else float(value)


def _reference(result: dict | None) -> dict:
    """Exact-MILP reference from a survey / certification result."""
    if not result or result.get("error"):
        return dict(ref_class="not_run", wg_exact=None, wg_lb=None, wg_proven=False,
                    wge_exact=None, wge_lb=None, wge_proven=False)
    wg, wge = result.get("wg") or {}, result.get("wge") or {}
    return dict(
        ref_class=result.get("classification", "inconclusive"),
        wg_exact=_f(wg.get("objective")), wg_lb=_f(wg.get("bound")),
        wg_proven=str(wg.get("proven_optimal")) == "True",
        wge_exact=_f(wge.get("objective")), wge_lb=_f(wge.get("bound")),
        wge_proven=str(wge.get("proven_optimal")) == "True",
    )


def select(output: Path, survey: Path, setting_a: Path) -> None:
    instances = []
    for case in sorted((survey / "cases").iterdir()):
        meta_path = case / "survey_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        res_path = case / "survey_result.json"
        result = json.loads(res_path.read_text()) if res_path.exists() else None
        instances.append(dict(
            name=case.name, case_dir=str(case), source_set="family_survey",
            family=meta["family"], category=meta["category"], topology=meta["topology"],
            params=meta.get("params", ""), n_qubits=meta["n_qubits"],
            g2=meta["fused_two_qubit_gates"], capacity=meta["capacity"],
            has_3param=meta.get("has_3param"), family_group=meta.get("family_group"),
            **_reference(result),
        ))
    setting_a_cases = sorted((setting_a / "cases").iterdir()) \
        if (setting_a / "cases").exists() else []
    for case in setting_a_cases:
        res_path = case / "setting_a_result.json"
        if not res_path.exists():
            continue
        result = json.loads(res_path.read_text())
        if result.get("classification") != "proven_improvement":
            continue
        instances.append(dict(
            name=f"setting_a__{case.name}", case_dir=str(case), source_set="setting_a_lp",
            family=f"own_{re.sub(r'_theta\d+of\d+$', '', result['family'])}"
            if not result["family"].startswith(
                "excision_butterfly") else "excision_butterfly",
            category="lattice-hamiltonian" if "butterfly" not in result["family"]
            else "synthetic-control",
            topology="planar-cyclic", params=result.get("params", ""),
            n_qubits=result["n_qubits"], g2=result["fused_two_qubit_gates"],
            capacity=result["capacity"], has_3param=None, **_reference(result),
        ))
    for inst in instances:
        inst["tier"] = tier_of(inst["g2"])
    # A family is an "excision family" if any of its instances is a proven improvement.
    improving = {i["family"] for i in instances if i["ref_class"] == "proven_improvement"}
    for inst in instances:
        if inst.get("family_group"):
            continue  # fixed by the instance generator (larger-circuit benchmark)
        inst["family_group"] = "excision-improves" if inst["family"] in improving \
            else "no-improvement"
    output.mkdir(parents=True, exist_ok=True)
    (output / "instances.json").write_text(json.dumps(instances, indent=2))
    count = defaultdict(int)
    for inst in instances:
        count[inst["family_group"], inst["tier"]] += 1
    print(f"{len(instances)} instances: {dict(sorted(count.items()))}")


# ---------------------------------------------------------------- one task


def _load(inst):
    from qpartition import GroupingConfig, PartitionGraph
    from qpartition.circuits import load_problem

    problem = load_problem(Path(inst["case_dir"]))
    graph = PartitionGraph(problem, excision=True, grouping=GroupingConfig(families=("excision",)))
    return problem, graph


def _verify(graph, plan, objective):
    from qpartition import compile_hybrid

    compiled = compile_hybrid(graph, plan)
    return dict(
        recompiled=float(compiled.evaluation.log_overhead),
        matches=bool(abs(compiled.evaluation.log_overhead - objective) < 1e-6),
        widths_ok=bool(max(compiled.evaluation.block_widths) <= plan.max_fragment_qubits),
        max_width=int(max(compiled.evaluation.block_widths)),
    )


def qiskit_state(problem, graph, marked):
    """Map Qiskit's marked circuit (qpd_2q gates, cut_wire markers) to a graph state."""
    circuit = problem.circuit
    state = [0] * len(graph.variables)
    source = 0
    pending = []  # (qubit, source index of the next original instruction)
    for item in marked.data:
        name = item.operation.name
        if name == "cut_wire":
            pending.append((marked.find_bit(item.qubits[0]).index, source))
            continue
        if name == "qpd_2q":
            state[graph.gate_variables[problem.operation_ids[source]]] = 1
        source += 1
    dropped = 0
    for q, start in pending:
        target = None
        for idx in range(start, len(circuit.data)):
            inst = circuit.data[idx]
            qs = [circuit.find_bit(x).index for x in inst.qubits]
            if q in qs and inst.operation.num_qubits == 2:
                target = problem.operation_ids[idx]
                break
        var = graph.wire_variables.get((target, q)) if target is not None else None
        if var is None:
            dropped += 1  # cut before the first or after the last 2q gate: no effect
        else:
            state[var] = 1
    return tuple(state), dropped


def run_qiskit(problem, graph, cap, budget, checkpoint=None):
    from qiskit_addon_cutting.automated_cut_finding import (
        DeviceConstraints,
        OptimizationParameters,
        find_cuts,
    )

    def alarm(_signum, _frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, alarm)
    t0 = perf_counter()
    rungs = []
    for backjumps in QISKIT_LADDER:
        remaining = budget - (perf_counter() - t0)
        if rungs and remaining < 1:
            break
        # Every rung, including Qiskit's default, shares the method's wall-clock budget.
        limit = remaining
        signal.setitimer(signal.ITIMER_REAL, max(limit, 1))
        start = perf_counter()
        rung = dict(max_backjumps=backjumps)
        try:
            marked, meta = find_cuts(
                problem.circuit,
                OptimizationParameters(seed=0, max_gamma=1e300, max_backjumps=backjumps),
                DeviceConstraints(cap),
            )
            signal.setitimer(signal.ITIMER_REAL, 0)
            state, dropped = qiskit_state(problem, graph, marked)
            ev = graph.evaluate(state, max_qubits=cap, partition_mode="multiway",
                                forced_groups=())
            q_log = math.log(meta["sampling_overhead"]) if meta["sampling_overhead"] > 0 \
                else None
            widths = list(ev.block_widths)
            rung.update(
                status="ok" if ev.feasible else "invalid_width",
                qiskit_log=q_log, our_log=float(ev.log_overhead),
                cost_match=bool(q_log is not None and abs(q_log - ev.log_overhead) < 1e-6),
                max_width=max(widths) if widths else None,
                minimum_reached=bool(meta["minimum_reached"]),
                gate_cuts=sum(1 for i in marked.data if i.operation.name == "qpd_2q"),
                wire_cuts=sum(1 for i in marked.data if i.operation.name == "cut_wire"),
                dropped_wire_cuts=dropped,
            )
        except TimeoutError:
            rung.update(status="timeout")
        except Exception as exc:  # noqa: BLE001
            signal.setitimer(signal.ITIMER_REAL, 0)
            rung.update(status=f"error: {type(exc).__name__}: {exc}"[:300])
        rung["seconds"] = perf_counter() - start
        rungs.append(rung)
        if checkpoint is not None:
            checkpoint(_qiskit_record(rungs, perf_counter() - t0))
        if rung.get("minimum_reached"):
            break  # Qiskit proved its own search optimal; deeper rungs cannot improve
    return _qiskit_record(rungs, perf_counter() - t0)


def _qiskit_record(rungs, seconds):
    default = rungs[0]
    valid = [r for r in rungs if r.get("status") == "ok"]
    best = min(valid, key=lambda r: r["our_log"]) if valid else None
    return dict(
        objective=default["our_log"] if default.get("status") == "ok" else None,
        status=default.get("status"), minimum_reached=default.get("minimum_reached"),
        best_objective=best["our_log"] if best else None,
        best_backjumps=best["max_backjumps"] if best else None,
        rungs=rungs, seconds=seconds,
    )


def _lns(graph, cap, excision, budget, lns_options, initial_state=None, seed=0):
    from qpartition.lns import LNSConfig, run_lns

    config = LNSConfig(cap, excision=excision, time_limit=budget, seed=seed,
                       initial_state=initial_state, **lns_options)
    res = run_lns(graph, config)
    verification = _verify(graph, res.plan, res.objective)
    return res, dict(
        objective=verification["recompiled"], solver_objective=res.objective,
        proven_optimal=res.proven_optimal, iterations=res.iterations,
        trajectory=res.trajectory, excisions=list(res.excisions),
        start_objective=res.start_objective, start_source=res.start_source,
        operator_stats=res.operator_stats, seconds=res.seconds, verification=verification,
    )


def run_method(inst, method, budget, lns_options, checkpoint=None):
    started = perf_counter()
    problem, graph = _load(inst)
    cap = inst["capacity"]
    record = dict(name=inst["name"], method=method, budget=budget,
                  n_vertices=len(graph.nodes), load_seconds=perf_counter() - started)
    base, _, seed_tag = method.partition(":s")
    seed = int(seed_tag or 0)
    if base == "qiskit":
        def save(partial):
            if checkpoint is not None:
                checkpoint(dict(record, **partial, partial=True))

        record.update(run_qiskit(problem, graph, cap, 2 * budget, save))
    elif base == "lns":
        wg_res, wg = _lns(graph, cap, False, budget, lns_options, seed=seed)
        record["wg"] = wg
        if checkpoint is not None:
            checkpoint(dict(record, partial=True))
        _, wge = _lns(graph, cap, True, budget, lns_options, initial_state=wg_res.state,
                      seed=seed)
        record["wge"] = wge
        record["seconds"] = wg["seconds"] + wge["seconds"]
    elif base == "lns_wge_cold":
        _, wge = _lns(graph, cap, True, 2 * budget, lns_options, seed=seed)
        record["wge"] = wge
        record["seconds"] = wge["seconds"]
    else:
        raise ValueError(method)
    return record


def task_main(task_json: str) -> None:
    task = json.loads(task_json)
    out = Path(task["out"])
    limit_gb = float(os.environ.get("RSS_LIMIT_GB", "3"))

    def watchdog():
        page = os.sysconf("SC_PAGE_SIZE")
        while True:
            with open("/proc/self/statm") as fh:
                rss = int(fh.read().split()[1]) * page / 1e9
            if rss > limit_gb:
                record = dict(name=task["inst"]["name"], method=task["method"])
                if out.exists():  # keep checkpointed partial results
                    try:
                        record = json.loads(out.read_text())
                    except ValueError:
                        pass
                record["error" if not record.get("partial") else "aborted"] = \
                    f"aborted: RSS > {limit_gb} GB"
                out.write_text(json.dumps(record, indent=2, default=float))
                os._exit(3)
            time.sleep(0.5)

    def checkpoint(partial):
        out.write_text(json.dumps(dict(partial, error=None), indent=2, default=float))

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        record = run_method(task["inst"], task["method"], task["budget"],
                            task.get("lns_options", {}), checkpoint)
        record["error"] = None
        import resource

        record["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception as exc:  # noqa: BLE001
        record = dict(name=task["inst"]["name"], method=task["method"],
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    out.write_text(json.dumps(record, indent=2, default=float))


# ---------------------------------------------------------------- run


def run(output: Path, runs_dir: str, workers: int, budgets: dict, methods, names=None,
        lns_options=None, limit=None) -> None:
    instances = json.loads((output / "instances.json").read_text())
    if names:
        wanted = set(names)
        instances = [i for i in instances if i["name"] in wanted]
    runs = output / runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    tasks = []
    order = {"S": 0, "M": 1, "L": 2, "X": 3}
    for inst in sorted(instances, key=lambda i: (order[i["tier"]], i["g2"], i["name"])):
        for method in methods:
            out = runs / f"{inst['name']}__{method}.json"
            if out.exists():
                continue
            tasks.append(dict(inst=inst, method=method, budget=budgets[inst["tier"]],
                              out=str(out), lns_options=lns_options or {}))
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} task(s), {workers} worker(s), runs -> {runs}", flush=True)

    def launch(task):
        cmd = [sys.executable, __file__, "--task", json.dumps(task)]
        try:
            subprocess.run(cmd, timeout=task["budget"] * 8 + 600, check=False,
                           capture_output=True)
        except subprocess.TimeoutExpired:
            Path(task["out"]).write_text(json.dumps(dict(
                name=task["inst"]["name"], method=task["method"], error="subprocess timeout")))
        return task

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            task = future.result()
            out = Path(task["out"])
            status = "missing"
            if out.exists():
                r = json.loads(out.read_text())
                status = r.get("error") or r.get("aborted") or ""
                if r.get("wg"):
                    status += f" wg={r['wg']['objective']:.3f} wge={r['wge']['objective']:.3f}" \
                        if r.get("wge") else f" wg={r['wg']['objective']:.3f}"
                elif "best_objective" in r:
                    status += f" qiskit={r.get('objective')} best={r.get('best_objective')}"
            print(f"[{done}/{len(tasks)}] {time.strftime('%H:%M:%S')} {task['inst']['name']} "
                  f"{task['method']}: {status}", flush=True)


# ---------------------------------------------------------------- summarize


def _ratio(a, b):
    """Overhead ratio e^(a-b) (how many times larger overhead a is than b)."""
    if a is None or b is None:
        return None
    return math.exp(min(a - b, 700))


def _fmt(x, digits=3):
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if x != 0 and (abs(x) >= 1e6 or abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return str(x)


def collect(output: Path, runs_dirs):
    """Latest record per (instance, method) over the given run directories (later wins)."""
    runs = {}
    for runs_dir in runs_dirs:
        for path in (output / runs_dir).glob("*.json"):
            r = json.loads(path.read_text())
            runs[r["name"], r["method"]] = r
    return runs


def summarize(output: Path, runs_dirs) -> None:
    instances = json.loads((output / "instances.json").read_text())
    runs = collect(output, runs_dirs)
    rows = []
    for inst in instances:
        q = runs.get((inst["name"], "qiskit"), {})
        lns = runs.get((inst["name"], "lns"), {})
        wg = {} if lns.get("error") else (lns.get("wg") or {})
        wge = {} if lns.get("error") else (lns.get("wge") or {})

        def obj(r):
            return r.get("objective")

        lns_wg, lns_wge = obj(wg), obj(wge)
        q_default = q.get("objective") if not q.get("error") else None
        q_best = q.get("best_objective") if not q.get("error") else None
        if q.get("error"):
            q_status = "error: " + q["error"][:40]
        elif not q:
            q_status = "not_run"
        else:
            q_status = q.get("status") or ""
        # Best WG-only value known from any method (exact MILP, LNS-WG, Qiskit).
        wg_candidates = [v for v in (inst["wg_exact"], lns_wg, q_best) if v is not None]
        best_wg = min(wg_candidates) if wg_candidates else None
        row = dict(
            instance=inst["name"], source_set=inst["source_set"], family=inst["family"],
            family_group=inst["family_group"], category=inst["category"],
            topology=inst["topology"], params=inst["params"], N=inst["n_qubits"],
            G2_fused=inst["g2"], M=inst["capacity"], tier=inst["tier"],
            survey_class=inst["ref_class"],
            exact_wg=inst["wg_exact"], exact_wg_proven=inst["wg_proven"],
            exact_wge=inst["wge_exact"], exact_wge_proven=inst["wge_proven"],
            qiskit_status=q_status + (" (" + q["aborted"] + ")" if q.get("aborted") else ""),
            qiskit=q_default,
            qiskit_minimum_reached=q.get("minimum_reached"),
            qiskit_best=q_best, qiskit_best_backjumps=q.get("best_backjumps"),
            qiskit_cost_check=all(r.get("cost_match", True) for r in q.get("rungs", [])
                                  if r.get("status") in ("ok", "invalid_width")) if q else None,
            lns_wg=lns_wg, lns_wg_proven=wg.get("proven_optimal"),
            lns_wge=lns_wge, lns_wge_proven=wge.get("proven_optimal"),
            lns_wge_excisions=len(wge.get("excisions") or []) if lns_wge is not None else None,
            lns_verified=all(r.get("verification", {}).get("matches", False)
                             and r.get("verification", {}).get("widths_ok", False)
                             for r in (wg, wge) if obj(r) is not None) or None,
            lns_wg_seconds=wg.get("seconds"), lns_wge_seconds=wge.get("seconds"),
            qiskit_seconds=q.get("seconds"),
            lns_peak_rss_gb=lns.get("peak_rss_gb"), qiskit_peak_rss_gb=q.get("peak_rss_gb"),
        )
        # Comparisons (log units; overhead ratio = e^difference).
        ref_q = q_best
        row["lns_wg_vs_qiskit"] = _compare(lns_wg, ref_q, q_status)
        row["lns_wge_vs_qiskit"] = _compare(lns_wge, ref_q, q_status)
        row["overhead_ratio_qiskit_over_lns_wge"] = _ratio(ref_q, lns_wge)
        row["lns_wg_gap_to_exact_wg"] = (lns_wg - inst["wg_exact"]) \
            if lns_wg is not None and inst["wg_exact"] is not None else None
        row["lns_wge_gap_to_exact_wge"] = (lns_wge - inst["wge_exact"]) \
            if lns_wge is not None and inst["wge_exact"] is not None else None
        row["best_wg_known"] = best_wg
        row["lns_wge_vs_best_wg"] = None if lns_wge is None or best_wg is None else (
            "better" if lns_wge < best_wg - TOL else ("tie" if lns_wge <= best_wg + TOL
                                                       else "worse"))
        row["overhead_ratio_best_wg_over_lns_wge"] = _ratio(best_wg, lns_wge)
        # A gain over the best WG solution counts as an excision gain only if the
        # LNS-WGE plan actually uses an excision (otherwise it is extra search).
        row["excision_gain_over_best_wg"] = bool(
            row["lns_wge_vs_best_wg"] == "better" and (row["lns_wge_excisions"] or 0) > 0)
        row["qiskit_deep_search_out_of_memory"] = bool(q.get("aborted"))
        rows.append(row)
    fields = list(rows[0])
    with (output / "lns_vs_qiskit_instances.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})
    family_rows = family_table(rows)
    with (output / "lns_vs_qiskit_families.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(family_rows[0]))
        writer.writeheader()
        for row in family_rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})
    group_rows = family_table(rows, key="family_group")
    with (output / "lns_vs_qiskit_groups.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(group_rows[0]))
        writer.writeheader()
        for row in group_rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})
    for row in group_rows:
        print({k: _fmt(v) for k, v in row.items()})


def _compare(ours, qiskit, q_status):
    if ours is None:
        return "lns_missing"
    if qiskit is None:
        return "qiskit_failed" if q_status not in ("not_run", "") else "qiskit_missing"
    if ours < qiskit - TOL:
        return "better"
    return "tie" if ours <= qiskit + TOL else "worse"


def family_table(rows, key="family"):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = []
    for name, rs in sorted(groups.items()):
        ran = [r for r in rs if r["lns_wg"] is not None or r["lns_wge"] is not None]
        ratios = [r["overhead_ratio_qiskit_over_lns_wge"] for r in rs
                  if r["overhead_ratio_qiskit_over_lns_wge"] is not None]
        excision_gain = [r["overhead_ratio_best_wg_over_lns_wge"] for r in rs
                         if r["excision_gain_over_best_wg"]]
        better_ratios = [r["overhead_ratio_qiskit_over_lns_wge"] for r in rs
                         if r["lns_wge_vs_qiskit"] == "better"]

        def count(field, value):
            return sum(1 for r in rs if r[field] == value)

        out.append({
            key: name,
            "family_group": rs[0]["family_group"] if key == "family" else name,
            "instances": len(rs),
            "lns_run": len(ran),
            "qiskit_valid": sum(1 for r in rs if r["qiskit_best"] is not None),
            "qiskit_invalid_or_failed": count("lns_wge_vs_qiskit", "qiskit_failed"),
            "lns_wg_better_than_qiskit": count("lns_wg_vs_qiskit", "better"),
            "lns_wg_tie_qiskit": count("lns_wg_vs_qiskit", "tie"),
            "lns_wg_worse_than_qiskit": count("lns_wg_vs_qiskit", "worse"),
            "lns_wge_better_than_qiskit": count("lns_wge_vs_qiskit", "better"),
            "lns_wge_tie_qiskit": count("lns_wge_vs_qiskit", "tie"),
            "lns_wge_worse_than_qiskit": count("lns_wge_vs_qiskit", "worse"),
            "median_overhead_ratio_qiskit_over_lns_wge": _median(ratios),
            "median_ratio_where_lns_better": _median(better_ratios),
            "max_overhead_ratio_qiskit_over_lns_wge": max(ratios) if ratios else None,
            "lns_wg_equals_exact_wg": sum(
                1 for r in rs if r["lns_wg_gap_to_exact_wg"] is not None
                and abs(r["lns_wg_gap_to_exact_wg"]) <= TOL),
            "lns_wg_better_than_exact_incumbent": sum(
                1 for r in rs if r["lns_wg_gap_to_exact_wg"] is not None
                and r["lns_wg_gap_to_exact_wg"] < -TOL),
            "lns_wg_worse_than_exact": sum(
                1 for r in rs if r["lns_wg_gap_to_exact_wg"] is not None
                and r["lns_wg_gap_to_exact_wg"] > TOL),
            "exact_wg_available": sum(1 for r in rs if r["exact_wg"] is not None),
            "lns_wge_better_than_best_wg": count("lns_wge_vs_best_wg", "better"),
            "lns_wge_tie_best_wg": count("lns_wge_vs_best_wg", "tie"),
            "lns_wge_worse_than_best_wg": count("lns_wge_vs_best_wg", "worse"),
            "excision_gain_over_best_wg": len(excision_gain),
            "max_excision_gain_ratio": max(excision_gain) if excision_gain else None,
            "qiskit_deep_search_out_of_memory": sum(
                1 for r in rs if r["qiskit_deep_search_out_of_memory"]),
        })
    return out


def _median(values):
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else math.sqrt(values[mid - 1] * values[mid])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["select", "run", "summarize"])
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path, default=Path("results/lns_family"))
    parser.add_argument("--survey", type=Path, default=Path("results/family_survey"))
    parser.add_argument("--setting-a", type=Path, default=Path("results/setting_a_lp"))
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--summary-runs", nargs="+", default=["runs"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-s", type=float, default=60.0)
    parser.add_argument("--budget-m", type=float, default=180.0)
    parser.add_argument("--budget-l", type=float, default=300.0)
    parser.add_argument("--budget-x", type=float, default=450.0)
    parser.add_argument("--lns-options", default="{}", help="JSON dict of LNSConfig overrides")
    args = parser.parse_args()
    if args.task:
        task_main(args.task)
    elif args.phase == "select":
        select(args.output, args.survey, args.setting_a)
    elif args.phase == "run":
        names = args.names
        if args.names_file:
            names = [n.strip() for n in args.names_file.read_text().split() if n.strip()]
        run(args.output, args.runs_dir, args.workers,
            dict(S=args.budget_s, M=args.budget_m, L=args.budget_l, X=args.budget_x),
            args.methods, names,
            json.loads(args.lns_options), args.limit)
    else:
        summarize(args.output, args.summary_runs)


if __name__ == "__main__":
    main()
