"""Exact WG vs WGE survey over external benchmark families (Setting A, fixed excision LP).

Families: MQT Bench (algorithm level), HamLib (2-local lattice and graph
Hamiltonians, Trotterized), Google ReCirq re-implementations (random circuit
sampling, QAOA, Fermi-Hubbard ladder), CutQC-style supremacy grids, QASMBench,
Qiskit Quantum Volume / IQP. See `src/qpartition/external_benchmarks.py`.

Phases:
  prepare     build, fuse (Setting A), filter (6 <= N <= 24, fused G2 <= 250), and
              freeze each circuit x capacity in {ceil(N/2), ceil(N/3)} under
              <output>/cases/<name>/ with structural features in survey_meta.json
  precompute  solve the excision LP for every new gate class
              (examples/precompute_excision_costs.py, shared cache)
  solve       exact WG and WGE MILPs with capacity cover cuts, each case in its own
              subprocess (RSS watchdog); improvements on circuits that contain
              3-parameter gates are re-solved with the 2(sum|u|)^2 - 1 gate-cut
              costs as a sensitivity check
  summarize   per-family / per-category tables (<output>/summary.md, survey.csv)

Usage:
  python examples/run_family_survey.py --phase prepare --output results/family_survey
  python examples/run_family_survey.py --phase precompute --output results/family_survey
  QPARTITION_EXCISION_CACHE=results/excision_lp_cache.pkl python examples/run_family_survey.py \
      --phase solve --output results/family_survey --workers 5 --threads 2 --time-limit 1200
  python examples/run_family_survey.py --phase summarize --output results/family_survey
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAMLIB = Path("data/raw/hamlib")
QASMBENCH = Path("data/raw/qasmbench")
MAX_N, MIN_N, MAX_G2 = 24, 6, 250


# ---------------------------------------------------------------- circuit sources


def _clean(circuit):
    """Drop final measurements and barriers; reject anything non-unitary that remains."""
    from qiskit import QuantumCircuit, transpile

    circuit = circuit.copy()
    circuit.remove_final_measurements(inplace=True)
    out = QuantumCircuit(circuit.num_qubits)
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        if name in {"measure", "reset", "if_else", "while_loop", "for_loop", "switch_case"}:
            raise ValueError(f"non-unitary operation {name!r} inside the circuit")
        out.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    if out.num_parameters:
        out = out.assign_parameters({p: 0.13 + 0.011 * i for i, p in enumerate(out.parameters)})
    if any(item.operation.num_qubits > 2 for item in out.data) or any(
        not hasattr(item.operation, "to_matrix") and item.operation.num_qubits == 2
        for item in out.data
    ):
        out = transpile(out, basis_gates=["cx", "u", "rzz", "cp"], optimization_level=0)
    return out


def _from_circuit(circuit, provenance):
    from qpartition.benchmarks import ResearchBenchmark
    from qpartition.circuits import CircuitProblem

    circuit = _clean(circuit)
    problem = CircuitProblem.create(circuit, "Z" * circuit.num_qubits)
    return ResearchBenchmark(name=provenance["family"], problem=problem, max_qubits=0,
                             paper_row=None, provenance=provenance)


MQT = [
    ("ae", "algorithm"), ("bv", "algorithm"), ("dj", "algorithm"), ("ghz", "state-prep"),
    ("graphstate", "state-prep"), ("wstate", "state-prep"), ("qaoa", "optimization"),
    ("qft", "algorithm"), ("qftentangled", "algorithm"), ("qnn", "variational"),
    ("qpeexact", "algorithm"), ("qpeinexact", "algorithm"), ("qwalk", "algorithm"),
    ("vqe_real_amp", "variational"), ("vqe_su2", "variational"),
    ("vqe_two_local", "variational"), ("cdkm_ripple_carry_adder", "arithmetic"),
    ("draper_qft_adder", "arithmetic"), ("full_adder", "arithmetic"),
    ("modular_adder", "arithmetic"), ("multiplier", "arithmetic"), ("hhl", "algorithm"),
    ("bmw_quark_cardinality", "variational"), ("bmw_quark_copula", "variational"),
    ("grover", "algorithm"), ("randomcircuit", "random"),
]


def mqt_specs():
    for name, category in MQT:
        for n in (8, 12, 16):
            def make(name=name, n=n):
                from mqt.bench import BenchmarkLevel, get_benchmark

                circuit = get_benchmark(name, BenchmarkLevel.ALG, n)
                return _from_circuit(circuit, dict(source="MQT Bench 2.3.0 (alg)",
                                                   family=f"mqt_{name}", n=n))

            yield dict(family=f"mqt_{name}", category=category, source="MQT Bench",
                       params=f"n={n}", make=make)


def _hamlib_keys(path, wanted_types, max_n, h_value):
    import h5py

    keys = []
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if not key.endswith(f"_h-{h_value}"):
                continue
            if not any(t in key for t in wanted_types):
                continue
            n = int(handle[key].attrs["nqubits"])
            if MIN_N <= n <= max_n:
                keys.append((key, n))
    return sorted(keys, key=lambda kn: (kn[1], kn[0]))


LATTICES = ("1D-grid-nonpbc", "2D-grid-nonpbc", "2D-grid-pbc", "2D-triag-nonpbc",
            "2D-triag-pbc", "2D-hex-nonpbc", "2D-hex-pbc", "3D-grid-nonpbc")


def hamlib_specs():
    from qpartition import external_benchmarks as E

    for model, file, h in (("heisenberg", "heis.hdf5", 0), ("tfim", "tfim.hdf5", 1)):
        path = HAMLIB / file
        chosen = {}
        for key, n in _hamlib_keys(path, LATTICES, MAX_N, h):
            lattice = next(t for t in LATTICES if t in key)
            if lattice.startswith("1D") and n not in (8, 12, 16):
                continue
            chosen.setdefault(lattice, []).append((key, n))
        for lattice, keys in chosen.items():
            for key, n in keys[:5]:
                for steps in (1, 2):
                    def make(path=path, key=key, steps=steps, model=model):
                        return E.hamlib_trotter(path, key, dt=0.2, steps=steps, model=model)

                    category = "1d-control" if lattice.startswith("1D") else "lattice-hamiltonian"
                    yield dict(family=f"hamlib_{model}_{lattice}", category=category,
                               source="HamLib", params=f"{key},r={steps}", make=make)
    for n in (8, 12, 16):
        key = f"reg-3_n-{n}_rinst-00"
        for steps in (1, 2):
            def make(key=key, steps=steps):
                return E.hamlib_trotter(HAMLIB / "graph-regular_reg-3.hdf5", key, dt=0.2,
                                        steps=steps, model="qmaxcut")

            yield dict(family="hamlib_qmaxcut_reg3", category="graph-hamiltonian",
                       source="HamLib", params=f"{key},r={steps}", make=make)
        for p in (1, 2):
            def make(key=key, p=p):
                return E.hamlib_qaoa(HAMLIB / "ham-graph-regular_reg-3.hdf5", key, p=p)

            yield dict(family="hamlib_maxcut_reg3_qaoa", category="optimization",
                       source="HamLib", params=f"{key},p={p}", make=make)


def recirq_specs():
    from qpartition import external_benchmarks as E

    for rows, cols in ((3, 3), (3, 4), (4, 4), (3, 5)):
        for cycles in (4, 8):
            yield dict(family="recirq_rcs_sycamore", category="random-grid",
                       source="Google ReCirq (re-impl.)", params=f"{rows}x{cols},c={cycles}",
                       make=lambda r=rows, c=cols, k=cycles: E.random_circuit_sampling(r, c, k))
        yield dict(family="cutqc_supremacy_cz", category="random-grid", source="CutQC (re-impl.)",
                   params=f"{rows}x{cols},d=8",
                   make=lambda r=rows, c=cols: E.supremacy_cz(r, c, 8))
    for kind, sizes in (("hardware_grid", (8, 12, 16)), ("3_regular", (8, 12, 16)),
                        ("sk", (8, 10, 12))):
        for n in sizes:
            for p in ((1, 2) if kind != "sk" else (1,)):
                yield dict(family=f"recirq_qaoa_{kind}", category="optimization",
                           source="Google ReCirq (re-impl.)", params=f"n={n},p={p}",
                           make=lambda k=kind, n=n, p=p: E.recirq_qaoa(k, n, p))
    for sites in (4, 6, 8):
        for steps in (1, 2):
            yield dict(family="recirq_fermi_hubbard_ladder", category="lattice-hamiltonian",
                       source="Google ReCirq (re-impl.)", params=f"L={sites},r={steps}",
                       make=lambda s=sites, r=steps: E.fermi_hubbard_ladder(s, r))


def library_specs():
    from qpartition import external_benchmarks as E

    for n in (6, 8, 10, 12):
        yield dict(family="qiskit_quantum_volume", category="random", source="Qiskit library",
                   params=f"n={n}", make=lambda n=n: E.quantum_volume_circuit(n))
    for n in (8, 12, 16):
        yield dict(family="qiskit_iqp", category="random", source="Qiskit library",
                   params=f"n={n}", make=lambda n=n: E.iqp_circuit(n))


QASMBENCH_CATEGORY = {
    "adder": "arithmetic", "bigadder": "arithmetic", "multiply": "arithmetic",
    "sat": "algorithm", "qf21": "algorithm", "qram": "algorithm", "qft": "algorithm",
    "qpe": "algorithm", "qaoa": "optimization", "ising": "1d-control", "dnn": "variational",
    "bv": "algorithm", "knn": "algorithm", "swap_test": "algorithm", "qec9xz": "qec",
    "hhl": "algorithm",
}


def qasmbench_specs():
    for path in sorted(QASMBENCH.glob("*.qasm")):
        stem = path.stem
        base = stem.rsplit("_n", 1)[0]

        def make(path=path, stem=stem):
            from qiskit import qasm2

            circuit = qasm2.load(str(path), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            return _from_circuit(circuit, dict(source="QASMBench", family=f"qasmbench_{stem}",
                                               file=path.name))

        yield dict(family=f"qasmbench_{base}", category=QASMBENCH_CATEGORY.get(base, "other"),
                   source="QASMBench", params=stem, make=make)


def lattice_specs():
    """This project's Heisenberg/XY Trotter circuits (lattice_benchmarks), larger sizes."""
    from qpartition.lattice_benchmarks import heisenberg_xy_lattice

    thetas = {"pi/8": math.pi / 8, "3pi/16": 3 * math.pi / 16, "pi/4": math.pi / 4}
    shapes = {"2d_square": (dict(lx=3, ly=4), dict(lx=4, ly=4)),
              "ladder": (dict(leg_length=6), dict(leg_length=8))}
    for model in ("heisenberg", "xy"):
        for topology, sizes in shapes.items():
            for shape in sizes:
                for label, theta in thetas.items():
                    small = shape.get("lx", 0) * shape.get("ly", 0) == 12 or shape.get(
                        "leg_length") == 6
                    for steps in ((1, 2) if small and label == "3pi/16" else (1,)):
                        def make(model=model, topology=topology, shape=shape, theta=theta,
                                 steps=steps):
                            return heisenberg_xy_lattice(topology, 0, model=model, theta=theta,
                                                         layers=steps, **shape)

                        dims = "x".join(str(v) for v in shape.values())
                        yield dict(family=f"own_{model}_{topology}",
                                   category="lattice-hamiltonian",
                                   source="self (lattice_benchmarks)",
                                   params=f"{dims},theta={label},r={steps}", make=make)


def all_specs():
    for gen in (mqt_specs, hamlib_specs, recirq_specs, library_specs, qasmbench_specs,
                lattice_specs):
        yield from gen()


# ---------------------------------------------------------------- features


def _kak_class(kak, tol=1e-6):
    a, b, c = (abs(x) for x in kak)
    nonzero = sum(x > tol for x in (a, b, c))
    if nonzero == 0:
        return "local"
    if abs(a - math.pi / 4) < tol and b < tol and c < tol:
        return "CX-class"
    if abs(a - math.pi / 4) < tol and abs(b - math.pi / 4) < tol:
        return "SWAP-class" if abs(c - math.pi / 4) < tol else (
            "iSWAP-class" if c < tol else "3-param")
    return {1: "1-param", 2: "2-param", 3: "3-param"}[nonzero]


def features(circuit):
    from qiskit.quantum_info import Operator

    from qpartition.excision_lp import kak_coordinates

    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    classes = Counter()
    for item in circuit.data:
        if item.operation.num_qubits == 2:
            a, b = (circuit.find_bit(q).index for q in item.qubits)
            graph.add_edge(a, b)
            kak = kak_coordinates(np.asarray(Operator(item.operation).data, dtype=complex))
            classes[_kak_class(kak)] += 1
    active = graph.subgraph([v for v in graph if graph.degree(v) > 0])
    components = nx.number_connected_components(active) if active.number_of_nodes() else 0
    cycle_rank = active.number_of_edges() - active.number_of_nodes() + components
    degrees = [d for _, d in active.degree()]
    if active.number_of_edges() == 0:
        topology = "empty"
    elif cycle_rank == 0:
        topology = "path" if max(degrees) <= 2 else "tree"
    elif max(degrees) <= 2:
        topology = "ring"
    elif nx.check_planarity(active)[0]:
        topology = "planar-cyclic"
    else:
        topology = "non-planar"
    return dict(
        interaction_edges=active.number_of_edges(), max_degree=max(degrees, default=0),
        mean_degree=float(np.mean(degrees)) if degrees else 0.0, cycle_rank=cycle_rank,
        topology=topology, gate_classes=dict(classes),
        has_3param=bool(classes.get("3-param", 0)),
        two_qubit_depth=circuit.depth(filter_function=lambda it: it.operation.num_qubits == 2),
    )


def _safe(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def prepare(output: Path) -> None:
    from qpartition.circuits import CircuitProblem, save_problem
    from qpartition.preprocessing import fuse_two_qubit_gates, two_qubit_gate_count

    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for spec in all_specs():
        tag = f"{spec['family']} {spec['params']}"
        try:
            bench = spec["make"]()
            raw = bench.problem.circuit
            fused = fuse_two_qubit_gates(raw)
            n, g2 = fused.num_qubits, two_qubit_gate_count(fused)
            if not MIN_N <= n <= MAX_N or g2 > MAX_G2 or g2 == 0:
                log.append(dict(spec=tag, status=f"filtered N={n} G2={g2}"))
                continue
            problem = CircuitProblem.create(fused, bench.problem.observable)
            feats = features(fused)
        except Exception as exc:  # noqa: BLE001
            log.append(dict(spec=tag, status=f"error: {type(exc).__name__}: {exc}"))
            continue
        for capacity in sorted({max(2, math.ceil(n / 2)), max(2, math.ceil(n / 3))}):
            name = _safe(f"{spec['family']}__{spec['params']}__cap{capacity}")
            case = cases_dir / name
            if (case / "survey_meta.json").exists():
                continue
            save_problem(problem, case)
            meta = dict(name=name, family=spec["family"], category=spec["category"],
                        source=spec["source"], params=spec["params"], n_qubits=n,
                        raw_two_qubit_gates=two_qubit_gate_count(raw),
                        fused_two_qubit_gates=g2, capacity=capacity,
                        provenance=bench.provenance, **feats)
            (case / "survey_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log.append(dict(spec=tag, status=f"ok N={n} G2={g2}"))
        print(f"[prepare] {tag}: N={n} G2={g2}", flush=True)
    (output / "prepare_log.json").write_text(json.dumps(log, indent=2))
    ok = sum(entry["status"].startswith("ok") for entry in log)
    print(f"{ok} circuits kept, {len(log) - ok} filtered or failed; "
          f"{len(list(cases_dir.iterdir()))} cases")


# ---------------------------------------------------------------- solving


def _formula_cost_model(model, graph):
    """Same MILP with gate-cut costs 2 ln(2(sum|u|)^2 - 1) (sensitivity, not compilable)."""
    from qiskit.quantum_info import Operator

    from qpartition.excision_lp import gamma_cut_kak, kak_coordinates

    problem = graph.problem
    new = copy.copy(model)
    new.c = model.c.copy()
    new.var_ub = model.var_ub.copy()
    w = {}
    for e, var in enumerate(graph.variables):
        if len(var.options) < 2:
            continue
        cost = var.options[1].log_overhead
        if var.kind == "gate":
            op = problem.circuit.data[problem.index(var.operation_id)].operation
            kak = kak_coordinates(np.asarray(Operator(op).data, dtype=complex))
            cost = min(cost, 2 * math.log(max(gamma_cut_kak(*kak), 1.0)))
        w[e] = cost
        new.c[model.x_col(e)] = cost
    if model.excision:
        for i, cand in enumerate(model.candidates):
            saving = sum(w[e] for e in cand.variables) - 2 * math.log(cand.gamma)
            new.c[model.y_col(i)] = -saving
            if saving <= 1e-12:
                new.var_ub[model.y_col(i)] = 0.0
    return new, sum(w[e] for e, v in enumerate(graph.variables)
                    if v.kind == "gate" and e in w)


def solve_case(task):
    from run_setting_a_certification import (
        _classify_kak,
        classify,
        detect_solver,
        solution_record,
        upper_bound,
    )

    from qpartition import (
        GroupingConfig,
        PartitionGraph,
        build_milp,
        compile_hybrid,
        compute_k_safe,
        milp_plan,
        solve_milp,
    )
    from qpartition.circuits import load_problem
    from qpartition.cover_cuts import cover_cut_pool
    from qpartition.milp import excision_selection_from_solution

    case = Path(task["case_dir"])
    meta = json.loads((case / "survey_meta.json").read_text())
    cap = meta["capacity"]
    solver = detect_solver("auto")
    kwargs = dict(solver=solver, time_limit=task["time_limit"], mip_gap_abs=1e-7,
                  mip_gap_rel=0.0, threads=task["threads"])
    started = perf_counter()
    problem = load_problem(case)
    graph = PartitionGraph(problem, excision=True, grouping=GroupingConfig(families=("excision",)))
    ub = upper_bound(graph, cap)
    previous = case / "survey_result.superseded.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        incumbents = [(old.get(key) or {}).get("objective") for key in ("wg", "wge")]
        ub = min([ub, *[v for v in incumbents if v is not None]])
    k = compute_k_safe(graph, cap, ub)
    cuts = cover_cut_pool(graph, max_qubits=cap, excision=True, time_budget=60.0)
    wg_model = build_milp(graph, max_qubits=cap, K=k, excision=False, cover_cuts=cuts)
    wge_model = build_milp(graph, max_qubits=cap, K=k, excision=True, cover_cuts=cuts)
    wg = solve_milp(wg_model, **kwargs)
    start = None
    if wg.raw_x is not None:
        start = np.concatenate([wg.raw_x, np.zeros(wge_model.n_vars - wg_model.n_vars)])
    wge = solve_milp(wge_model, mip_start=start, **kwargs)
    wg_r, wge_r = solution_record(wg), solution_record(wge)
    label, delta_low, delta_high = classify(wg_r, wge_r)
    excisions, verification = [], None
    if wge.raw_x is not None:
        from qiskit.quantum_info import Operator

        from qpartition.excision_lp import gate_excision_cost

        for gid in excision_selection_from_solution(wge_model, wge.raw_x):
            cand = graph.grouping.by_id[gid]
            op = problem.circuit.data[problem.index(cand.anchor)].operation
            kak = gate_excision_cost(np.asarray(Operator(op).data, dtype=complex))
            excisions.append(dict(op_id=cand.anchor, gate_class=_classify_kak(kak.kak),
                                  wires_absorbed=len(cand.excision_wires), gamma_ex=cand.gamma))
        try:
            compiled = compile_hybrid(graph, milp_plan(wge_model, wge.raw_x))
            verification = dict(
                recompiled=float(compiled.evaluation.log_overhead),
                matches=bool(abs(compiled.evaluation.log_overhead - wge.objective) < 1e-5),
                widths_ok=bool(max(compiled.evaluation.block_widths) <= cap),
            )
        except Exception as exc:  # noqa: BLE001
            verification = dict(error=f"{type(exc).__name__}: {exc}")
    sensitivity = None
    if task.get("sensitivity") and label == "proven_improvement" and meta.get("has_3param"):
        wg_f, all_cut = _formula_cost_model(wg_model, graph)
        wge_f, _ = _formula_cost_model(wge_model, graph)
        k_f = compute_k_safe(graph, cap, all_cut)
        if k_f > k:
            wg_m = build_milp(graph, max_qubits=cap, K=k_f, excision=False, cover_cuts=cuts)
            wge_m = build_milp(graph, max_qubits=cap, K=k_f, excision=True, cover_cuts=cuts)
            wg_f, _ = _formula_cost_model(wg_m, graph)
            wge_f, _ = _formula_cost_model(wge_m, graph)
        a, b = solve_milp(wg_f, **kwargs), solve_milp(wge_f, **kwargs)
        s_label, s_low, _ = classify(solution_record(a), solution_record(b))
        sensitivity = dict(wg=solution_record(a), wge=solution_record(b), classification=s_label,
                           omega_ratio=(math.exp(a.objective - b.objective)
                                        if a.objective is not None and b.objective is not None
                                        else None))
    return dict(
        name=meta["name"], K=k, cover_cuts=len(cuts), wg=wg_r, wge=wge_r, classification=label,
        delta_low=delta_low, delta_high=delta_high,
        omega_ratio=(math.exp(wg.objective - wge.objective)
                     if wg.objective is not None and wge.objective is not None else None),
        excisions=excisions, verification=verification, sensitivity=sensitivity,
        seconds=perf_counter() - started, error=None,
    )


def task_main(task_json):
    task = json.loads(task_json)
    out = Path(task["case_dir"]) / "survey_result.json"
    limit = float(os.environ.get("RSS_LIMIT_GB", "4"))

    def watchdog():
        page = os.sysconf("SC_PAGE_SIZE")
        while True:
            with open("/proc/self/statm") as fh:
                rss = int(fh.read().split()[1]) * page / 1e9
            if rss > limit:
                out.write_text(json.dumps(dict(error=f"aborted: RSS > {limit} GB")))
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        result = solve_case(task)
    except Exception as exc:  # noqa: BLE001
        result = dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    out.write_text(json.dumps(result, indent=2, default=str))


EXCISION_RELEVANT = ("own_", "hamlib_heisenberg", "hamlib_qmaxcut", "recirq_rcs_sycamore",
                     "qiskit_quantum_volume")


def plan_case(meta):
    """(priority, per-model time limit). P1: excision-relevant families (3-parameter or
    XY-type gates on cyclic interaction graphs); P2: the rest up to G2 100; P3: larger."""
    relevant = meta["family"].startswith(EXCISION_RELEVANT) or (
        meta.get("has_3param") and meta["topology"] not in ("path", "tree", "ring"))
    if relevant:
        return 1, 1500.0
    if meta["fused_two_qubit_gates"] <= 100:
        return 2, 600.0
    return 3, 300.0


def solve(output: Path, workers: int, threads: int, time_limit: float | None,
          deadline_hours: float | None = None, priorities=None) -> None:
    started = time.time()
    cases = []
    for case in sorted((output / "cases").iterdir()):
        if (case / "survey_result.json").exists() or not (case / "survey_meta.json").exists():
            continue
        meta = json.loads((case / "survey_meta.json").read_text())
        priority, limit = plan_case(meta)
        if priorities and priority not in priorities:
            continue
        size = meta["fused_two_qubit_gates"] * meta["n_qubits"]
        cases.append((priority, size, str(case), time_limit or limit))
    cases.sort()
    counts = Counter(c[0] for c in cases)
    print(f"{len(cases)} case(s) to solve {dict(sorted(counts.items()))}, {workers} worker(s) x "
          f"{threads} thread(s), deadline {deadline_hours} h", flush=True)
    (output / f"plan_p{''.join(map(str, priorities or [0]))}.json").write_text(json.dumps(
        [dict(priority=p, case=Path(c).name, time_limit=t) for p, _, c, t in cases], indent=2))

    def launch(item):
        priority, _, case_dir, limit = item
        if deadline_hours and time.time() - started > deadline_hours * 3600:
            return case_dir, "skipped (deadline)"
        task = dict(case_dir=case_dir, threads=threads, time_limit=limit)
        try:
            subprocess.run([sys.executable, __file__, "--task", json.dumps(task)],
                           timeout=3 * limit + 900, check=False, capture_output=True)
        except subprocess.TimeoutExpired:
            (Path(case_dir) / "survey_result.json").write_text(
                json.dumps(dict(error="subprocess timeout")))
        return case_dir, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, c) for c in cases]
        for done, future in enumerate(as_completed(futures), 1):
            case_dir, skipped = future.result()
            case_dir = Path(case_dir)
            if skipped:
                continue
            path = case_dir / "survey_result.json"
            r = json.loads(path.read_text()) if path.exists() else {"error": "missing"}
            status = r.get("error") or f"{r['classification']} ratio={r.get('omega_ratio')}"
            print(f"[{done}/{len(cases)}] {case_dir.name}: {status} "
                  f"({(r.get('seconds') or 0):.0f}s)", flush=True)


# ---------------------------------------------------------------- summary


def summarize(output: Path) -> None:
    rows, not_run = [], []
    for case in sorted((output / "cases").iterdir()):
        meta_path, result_path = case / "survey_meta.json", case / "survey_result.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if not result_path.exists():
            not_run.append(meta)
            continue
        r = json.loads(result_path.read_text())
        sens = r.get("sensitivity") or {}
        rows.append(dict(
            name=meta["name"], family=meta["family"], category=meta["category"],
            source=meta["source"], params=meta["params"], n=meta["n_qubits"],
            g2=meta["fused_two_qubit_gates"], capacity=meta["capacity"],
            topology=meta["topology"], cycle_rank=meta["cycle_rank"],
            max_degree=meta["max_degree"], has_3param=meta["has_3param"],
            gate_classes=json.dumps(meta["gate_classes"], sort_keys=True),
            classification=r.get("classification") or "error",
            error=r.get("error"),
            wg=(r.get("wg") or {}).get("objective"), wge=(r.get("wge") or {}).get("objective"),
            wg_bound=(r.get("wg") or {}).get("bound"),
            wge_bound=(r.get("wge") or {}).get("bound"),
            ratio=r.get("omega_ratio"), excisions=len(r.get("excisions") or []),
            verified=(r.get("verification") or {}).get("matches"),
            sensitivity=sens.get("classification"), sensitivity_ratio=sens.get("omega_ratio"),
            seconds=r.get("seconds"),
        ))
    with (output / "survey.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def block(key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        lines = [f"| {key} | instances | improvement | no gain | inconclusive | error | "
                 "max ratio | topologies |", "|---|---|---|---|---|---|---|---|"]
        for name, group in sorted(groups.items()):
            count = Counter(row["classification"] for row in group)
            ratios = [row["ratio"] for row in group
                      if row["classification"] == "proven_improvement" and row["ratio"]]
            topo = ",".join(sorted({row["topology"] for row in group}))
            lines.append(
                f"| {name} | {len(group)} | {count['proven_improvement']} | "
                f"{count['proven_no_gain']} | {count['inconclusive']} | {count['error']} | "
                f"{max(ratios):.2f} | {topo} |" if ratios else
                f"| {name} | {len(group)} | 0 | {count['proven_no_gain']} | "
                f"{count['inconclusive']} | {count['error']} | - | {topo} |")
        return lines

    lines = ["# Family survey: exact WG vs WGE\n",
             f"{len(rows)} cases solved; {len(not_run)} not run (stopped: beyond exact-MILP "
             f"reach, fused G2 {min((m['fused_two_qubit_gates'] for m in not_run), default=0)}-"
             f"{max((m['fused_two_qubit_gates'] for m in not_run), default=0)}).\n",
             "## By category\n", *block("category"), "",
             "## By family\n", *block("family"), "", "## By interaction topology\n",
             *block("topology"), ""]
    improved = [row for row in rows if row["classification"] == "proven_improvement"]
    lines += ["## Proven improvements\n",
              "| instance | N | G2 | M | topology | WG | WGE | ratio | excisions | verified | "
              "3-param sensitivity |", "|" + "---|" * 11]
    for row in sorted(improved, key=lambda r: -r["ratio"]):
        sens = (f"{row['sensitivity']} ({row['sensitivity_ratio']:.2f}x)"
                if row["sensitivity"] and row["sensitivity_ratio"] else row["sensitivity"] or "n/a")
        lines.append(f"| {row['name']} | {row['n']} | {row['g2']} | {row['capacity']} | "
                     f"{row['topology']} | {row['wg']:.3f} | {row['wge']:.3f} | "
                     f"{row['ratio']:.2f} | {row['excisions']} | {row['verified']} | {sens} |")
    text = "\n".join(lines)
    (output / "summary.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "precompute", "solve", "summarize"])
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path, default=Path("results/family_survey"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="override the per-priority time limits (s per model)")
    parser.add_argument("--priorities", type=int, nargs="+", default=None,
                        help="only solve cases with these priorities (1, 2, 3)")
    parser.add_argument("--deadline-hours", type=float, default=None,
                        help="stop launching new cases after this many hours")
    args = parser.parse_args()
    if args.task:
        task_main(args.task)
    elif args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "precompute":
        subprocess.run([sys.executable, str(Path(__file__).parent / "precompute_excision_costs.py"),
                        "--cases", str(args.output / "cases"), "--cache",
                        "results/excision_lp_cache.pkl", "--workers", str(args.workers)],
                       check=True)
    elif args.phase == "solve":
        solve(args.output, args.workers, args.threads, args.time_limit, args.deadline_hours,
              args.priorities)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
