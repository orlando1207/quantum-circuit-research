"""Larger circuits for the LNS vs Qiskit comparison (no exact MILP at this scale).

Phases:
  prepare     build, fuse (Setting A), filter (16 <= N <= 128, 30 <= fused G2 <= 800),
              save <output>/cases/<name>/{circuit.qpy, problem.json, survey_meta.json}
  precompute  excision LP for every new gate class (shared cache)

Then run the comparison with the LNS family runner:
  python examples/run_lns_family_benchmark.py --phase select --survey results/lns_large \
      --setting-a none --output results/lns_large
  python examples/run_lns_family_benchmark.py --phase run --output results/lns_large ...

Instance groups (`family_group` in the metadata):
  excision-relevant  circuits built from large-angle 3-parameter gates, the regime where the
                     family survey found excision gains (Heisenberg/XY at theta >= pi/8,
                     Sycamore RCS, Quantum Volume)
  control            circuits from families with no excision gain in the survey
                     (1-parameter gates, small-angle Trotter steps, algorithms, arithmetic)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_family_survey import HAMLIB, QASMBENCH, _from_circuit, _safe, features  # noqa: E402

MIN_N, MAX_N, MIN_G2, MAX_G2 = 16, 128, 30, 800
PI8, PI4 = math.pi / 8, math.pi / 4


def _hamlib_key(file, lattice, n, h):
    import h5py

    with h5py.File(HAMLIB / file, "r") as handle:
        for key in sorted(handle.keys()):
            if key.startswith(f"graph-{lattice}-qubitnodes") and key.endswith(f"_h-{h}") \
                    and int(handle[key].attrs["nqubits"]) == n:
                return key
    raise KeyError(f"{file}: no {lattice} key with n={n}")


def specs():
    from qpartition import external_benchmarks as E
    from qpartition.lattice_benchmarks import heisenberg_xy_lattice

    rel, ctl = "excision-relevant", "control"
    # HamLib Heisenberg at a large Trotter angle (excision regime) and at dt = 0.2 (control).
    hamlib_sizes = {"2D-grid-nonpbc": (25, 36, 49, 64), "2D-triag-nonpbc": (27, 36, 48),
                    "2D-hex-nonpbc": (48, 96), "3D-grid-nonpbc": (27, 64)}
    for lattice, sizes in hamlib_sizes.items():
        for n in sizes:
            for dt, label, group in ((PI8, "pi/8", rel), (0.2, "0.2", ctl)):
                if group == ctl and lattice != "2D-grid-nonpbc":
                    continue
                yield dict(
                    family=f"hamlib_heisenberg_{lattice}", group=group,
                    category="lattice-hamiltonian", source="HamLib",
                    params=f"n={n},dt={label},r=1",
                    make=lambda lat=lattice, n=n, dt=dt: E.hamlib_trotter(
                        HAMLIB / "heis.hdf5", _hamlib_key("heis.hdf5", lat, n, 0), dt=dt,
                        steps=1, model="heisenberg"))
    for n in (25, 36, 49, 64):
        yield dict(family="hamlib_tfim_2D-grid-nonpbc", group=ctl,
                   category="lattice-hamiltonian", source="HamLib", params=f"n={n},dt=0.2,r=1",
                   make=lambda n=n: E.hamlib_trotter(
                       HAMLIB / "tfim.hdf5", _hamlib_key("tfim.hdf5", "2D-grid-nonpbc", n, 1),
                       dt=0.2, steps=1, model="tfim"))
    # This project's Heisenberg / XY lattices at large angles.
    for model in ("heisenberg", "xy"):
        for theta, label in ((PI8, "pi/8"), (PI4, "pi/4")):
            for shape in (dict(lx=5, ly=5), dict(lx=6, ly=6), dict(lx=8, ly=8)):
                yield dict(family=f"own_{model}_2d_square", group=rel,
                           category="lattice-hamiltonian", source="self (lattice_benchmarks)",
                           params=f"{shape['lx']}x{shape['ly']},theta={label},r=1",
                           make=lambda m=model, t=theta, s=shape: heisenberg_xy_lattice(
                               "2d_square", 0, model=m, theta=t, layers=1, **s))
            for leg in (12, 20, 30):
                yield dict(family=f"own_{model}_ladder", group=rel,
                           category="lattice-hamiltonian", source="self (lattice_benchmarks)",
                           params=f"2x{leg},theta={label},r=1",
                           make=lambda m=model, t=theta, g=leg: heisenberg_xy_lattice(
                               "ladder", 0, model=m, theta=t, layers=1, leg_length=g))
    # Google ReCirq Sycamore random circuits and CutQC supremacy (CZ) on larger grids.
    for rows, cols in ((4, 5), (5, 5), (6, 6)):
        for cycles in (8, 12):
            yield dict(family="recirq_rcs_sycamore", group=rel, category="random-grid",
                       source="Google ReCirq (re-impl.)", params=f"{rows}x{cols},c={cycles}",
                       make=lambda r=rows, c=cols, k=cycles: E.random_circuit_sampling(r, c, k))
        yield dict(family="cutqc_supremacy_cz", group=ctl, category="random-grid",
                   source="CutQC (re-impl.)", params=f"{rows}x{cols},d=8",
                   make=lambda r=rows, c=cols: E.supremacy_cz(r, c, 8))
    for n in (16, 20):
        yield dict(family="qiskit_quantum_volume", group=rel, category="random",
                   source="Qiskit library", params=f"n={n}",
                   make=lambda n=n: E.quantum_volume_circuit(n))
    # ReCirq QAOA and Fermi-Hubbard.
    for n in (24, 32, 48):
        for p in (1, 2):
            yield dict(family="recirq_qaoa_3_regular", group=ctl, category="optimization",
                       source="Google ReCirq (re-impl.)", params=f"n={n},p={p}",
                       make=lambda n=n, p=p: E.recirq_qaoa("3_regular", n, p))
    for sites in (12, 16):
        yield dict(family="recirq_fermi_hubbard_ladder", group=ctl,
                   category="lattice-hamiltonian", source="Google ReCirq (re-impl.)",
                   params=f"L={sites},r=2",
                   make=lambda s=sites: E.fermi_hubbard_ladder(s, 2))
    # MQT Bench algorithms at 24 and 32 qubits.
    mqt = (("qft", "algorithm"), ("qftentangled", "algorithm"), ("qpeexact", "algorithm"),
           ("ae", "algorithm"), ("qaoa", "optimization"), ("vqe_su2", "variational"),
           ("vqe_real_amp", "variational"), ("ghz", "state-prep"),
           ("graphstate", "state-prep"), ("cdkm_ripple_carry_adder", "arithmetic"),
           ("draper_qft_adder", "arithmetic"))
    for name, category in mqt:
        for n in (24, 32):
            def make(name=name, n=n):
                from mqt.bench import BenchmarkLevel, get_benchmark

                circuit = get_benchmark(name, BenchmarkLevel.ALG, n)
                return _from_circuit(circuit, dict(source="MQT Bench 2.3.0 (alg)",
                                                   family=f"mqt_{name}", n=n))

            yield dict(family=f"mqt_{name}", group=ctl, category=category,
                       source="MQT Bench", params=f"n={n}", make=make)
    for stem, category in (("knn_n25", "algorithm"), ("swap_test_n25", "algorithm")):
        def make(stem=stem):
            from qiskit import qasm2

            circuit = qasm2.load(str(QASMBENCH / f"{stem}.qasm"),
                                 custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            return _from_circuit(circuit, dict(source="QASMBench", family=f"qasmbench_{stem}"))

        yield dict(family=f"qasmbench_{stem.rsplit('_n', 1)[0]}", group=ctl, category=category,
                   source="QASMBench", params=stem, make=make)


def capacities(n):
    return sorted({math.ceil(n / 2), math.ceil(n / 3)})


def prepare(output: Path) -> None:
    from qpartition.circuits import CircuitProblem, save_problem
    from qpartition.preprocessing import fuse_two_qubit_gates, two_qubit_gate_count

    cases = output / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    log = []
    for spec in specs():
        tag = f"{spec['family']} {spec['params']}"
        try:
            bench = spec["make"]()
            raw = bench.problem.circuit
            fused = fuse_two_qubit_gates(raw)
            n, g2 = fused.num_qubits, two_qubit_gate_count(fused)
            if not (MIN_N <= n <= MAX_N and MIN_G2 <= g2 <= MAX_G2):
                log.append(dict(spec=tag, status=f"filtered N={n} G2={g2}"))
                print(f"[prepare] {tag}: filtered N={n} G2={g2}", flush=True)
                continue
            problem = CircuitProblem.create(fused, bench.problem.observable)
            feats = features(fused)
        except Exception as exc:  # noqa: BLE001
            log.append(dict(spec=tag, status=f"error: {type(exc).__name__}: {exc}"))
            print(f"[prepare] {tag}: error {exc}", flush=True)
            continue
        for capacity in capacities(n):
            name = _safe(f"{spec['family']}__{spec['params']}__cap{capacity}")
            case = cases / name
            if (case / "survey_meta.json").exists():
                continue
            save_problem(problem, case)
            meta = dict(name=name, family=spec["family"], family_group=spec["group"],
                        category=spec["category"], source=spec["source"],
                        params=spec["params"], n_qubits=n,
                        raw_two_qubit_gates=two_qubit_gate_count(raw),
                        fused_two_qubit_gates=g2, capacity=capacity,
                        provenance=bench.provenance, **feats)
            (case / "survey_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log.append(dict(spec=tag, status=f"ok N={n} G2={g2}"))
        print(f"[prepare] {tag}: N={n} G2={g2}", flush=True)
    (output / "prepare_log.json").write_text(json.dumps(log, indent=2))
    ok = sum(e["status"].startswith("ok") for e in log)
    print(f"{ok} circuits kept, {len(log) - ok} filtered/failed, "
          f"{len(list(cases.iterdir()))} cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "precompute"], required=True)
    parser.add_argument("--output", type=Path, default=Path("results/lns_large"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    else:
        subprocess.run([sys.executable, str(Path(__file__).parent / "precompute_excision_costs.py"),
                        "--cases", str(args.output / "cases"), "--cache",
                        "results/excision_lp_cache.pkl", "--workers", str(args.workers),
                        "--timeout", str(args.timeout)], check=True)


if __name__ == "__main__":
    main()
