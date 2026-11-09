"""Phase 2a breadth-first screening survey (docs/second_stage_cont_prompt.txt
Part 2). Builds a family registry spanning external sources (MQT Bench,
Qiskit's own circuit library, Stim-pattern, Qiskit Nature) and this project's
own self-generated structural/physics families, runs each through the
mandatory preprocessing + screening pipeline (`qpartition.screening`) at
2 sizes x up to 2 tight capacities per family, and writes a coverage table
(family, source, version, N, fused G2, capacity, WG cost, junction count,
predicted gain, status) -- the Phase 2a deliverable. Phase 2b (certification
to proven optimality via the existing MILP) is a separate follow-up step for
any family this flags with junction_count > 0, plus >=3 negative controls.

Usage: python examples/run_breadth_survey.py --output results/breadth_survey.csv
"""

from __future__ import annotations

import argparse
import csv
import traceback
from dataclasses import asdict
from pathlib import Path

import mqt.bench as mb
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2, MCXVChain, PhaseEstimation, QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from qpartition.circuits import CircuitProblem
from qpartition.large_benchmarks import large_benchmark
from qpartition.lattice_benchmarks import heisenberg_xy_lattice, repetition_code_syndrome_extraction
from qpartition.milp_benchmarks import qcnn, ripple_carry_adder, tree_tensor_network
from qpartition.screening import screen_problem

CSV_FIELDS = [
    "family",
    "source",
    "version",
    "params",
    "is_negative_control",
    "n_qubits",
    "fused_two_qubit_gates",
    "max_qubits",
    "wg_log_overhead",
    "wg_feasible",
    "junction_count",
    "predicted_gain",
    "predicted_gain_is_lower_bound_only",
    "status",
    "note",
]


def _to_project_circuit(circuit: QuantumCircuit, *, seed: int = 0) -> QuantumCircuit:
    """Strip measurements/resets/barriers/classical bits, bind any leftover
    free parameters, and flatten to a cx+u basis (no coupling map, so no
    routing SWAPs are introduced -- the circuit stays logical/unrouted)."""
    stripped = QuantumCircuit(circuit.num_qubits)
    for item in circuit.data:
        if item.operation.name in {"measure", "reset", "barrier"}:
            continue
        stripped.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    if stripped.num_parameters:
        binds = {p: 0.13 + 0.011 * i for i, p in enumerate(stripped.parameters)}
        stripped = stripped.assign_parameters(binds)
    return transpile(stripped, basis_gates=["cx", "u"], optimization_level=0, seed_transpiler=seed)


def _mqt_bench_problem(name: str, size: int, *, seed: int = 0) -> CircuitProblem:
    circuit = mb.get_benchmark(name, mb.BenchmarkLevel.ALG, circuit_size=size)
    flat = _to_project_circuit(circuit, seed=seed)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


def _efficient_su2_problem(n: int) -> CircuitProblem:
    circuit = EfficientSU2(n, reps=2, entanglement="linear")
    flat = _to_project_circuit(circuit)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


def _qaoa_ansatz_problem(n: int) -> CircuitProblem:
    import networkx as nx

    graph = nx.random_regular_graph(3, n, seed=7)
    terms = [("".join("Z" if i in (a, b) else "I" for i in reversed(range(n))), 1.0)
             for a, b in graph.edges()]
    cost = SparsePauliOp.from_list(terms)
    circuit = QAOAAnsatz(cost, reps=2)
    flat = _to_project_circuit(circuit)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


def _phase_estimation_problem(n: int) -> CircuitProblem:
    from qiskit.circuit.library import PhaseGate

    num_eval = max(2, n // 2)
    unitary = QuantumCircuit(n - num_eval)
    unitary.append(PhaseGate(0.41), [0])
    circuit = PhaseEstimation(num_eval, unitary)
    flat = _to_project_circuit(circuit)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


def _mcx_vchain_oracle_problem(n: int) -> CircuitProblem:
    # MCXVChain(controls) needs 2*controls-1 qubits total (controls + target
    # + controls-2 V-chain ancillas); size the oracle to exactly fill n.
    controls = max(2, (n + 1) // 2)
    total = 2 * controls - 1
    circuit = QuantumCircuit(total)
    circuit.h(range(controls + 1))
    circuit.append(MCXVChain(controls), range(total))
    flat = _to_project_circuit(circuit)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


def _uccsd_problem(_n: int) -> CircuitProblem:
    from qiskit_nature.second_q.circuit.library import UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    ansatz = UCCSD(2, (1, 1), JordanWignerMapper())
    circuit = ansatz.decompose(reps=4)
    flat = _to_project_circuit(circuit)
    return CircuitProblem.create(flat, "Z" * flat.num_qubits)


# Each entry: (name, source, version, generator(size) -> CircuitProblem | ResearchBenchmark,
#              sizes, is_negative_control)
def _lb(family, **kwargs):
    return lambda n: large_benchmark(family, n, **kwargs)


FAMILIES = [
    ("ghz", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("ghz", n), [8, 14], False),
    ("wstate", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("wstate", n), [8, 14], False),
    ("qft", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("qft", n), [8, 14], True),
    ("qftentangled", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("qftentangled", n),
     [8, 14], False),
    ("bv", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("bv", n), [8, 14], True),
    ("dj", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("dj", n), [8, 14], True),
    ("qpeexact", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("qpeexact", n), [8, 12], False),
    ("qaoa_mqt", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("qaoa", n), [8, 12], False),
    ("vqe_two_local", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("vqe_two_local", n),
     [8, 12], False),
    ("graphstate_mqt", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("graphstate", n),
     [8, 14], True),
    ("randomcircuit", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("randomcircuit", n),
     [8, 12], True),
    ("modular_adder", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("modular_adder", n),
     [8, 12], False),
    ("hhl", "MQT Bench", "2.3.0", lambda n: _mqt_bench_problem("hhl", n), [6, 8], False),
    ("efficient_su2", "Qiskit circuit library", "2.x", _efficient_su2_problem, [8, 14], False),
    ("qaoa_ansatz", "Qiskit circuit library", "2.x", _qaoa_ansatz_problem, [8, 12], False),
    ("phase_estimation", "Qiskit circuit library", "2.x", _phase_estimation_problem,
     [8, 12], False),
    ("mcx_vchain_oracle", "Qiskit circuit library", "2.x", _mcx_vchain_oracle_problem,
     [8, 12], False),
    ("uccsd", "Qiskit Nature", "0.x", _uccsd_problem, [8], False),
    ("ripple_carry_adder", "self-generated (CDKM/Qiskit library)", "v1",
     lambda n: ripple_carry_adder(n).problem, [4, 6], False),
    ("qcnn", "self-generated", "v1", lambda n: qcnn(n).problem, [8, 16], False),
    ("tree_tensor_network", "self-generated", "v1", lambda n: tree_tensor_network(n).problem,
     [8, 16], False),
    ("excision_butterfly", "self-generated", "v1", lambda n: _lb("excision_butterfly")(n).problem,
     [6, 10], False),
    ("trotter_chain", "self-generated", "v1", lambda n: _lb("trotter_chain")(n).problem,
     [8, 14], True),
    ("trotter_grid", "self-generated", "v1", lambda n: _lb("trotter_grid")(n).problem,
     [8, 14], False),
    ("aqft", "self-generated", "v1", lambda n: _lb("aqft")(n).problem, [8, 14], True),
    ("random_brickwork", "self-generated", "v1", lambda n: _lb("random_brickwork")(n).problem,
     [8, 14], True),
    ("heisenberg_2d_square", "self-generated (lattice physics)", "v1",
     lambda n: heisenberg_xy_lattice("2d_square", n, model="heisenberg").problem, [9, 16], False),
    ("heisenberg_ladder", "self-generated (lattice physics)", "v1",
     lambda n: heisenberg_xy_lattice("ladder", n, model="heisenberg").problem, [8, 12], False),
    ("xy_heavy_hex", "self-generated (lattice physics)", "v1",
     lambda n: heisenberg_xy_lattice("heavy_hex", n, model="xy").problem, [8, 16], False),
    ("repetition_code", "self-generated (Stim connectivity pattern)", "v1",
     lambda n: repetition_code_syndrome_extraction(n, 2).problem, [4, 6], False),
]


def _tight_capacities(n: int) -> list[int]:
    import math

    return sorted({max(2, math.ceil(n / 3)), max(2, math.ceil(n / 2))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/breadth_survey.csv"))
    parser.add_argument("--max-evaluations", type=int, default=1200)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    rows = []
    selected = FAMILIES if args.families is None else [
        f for f in FAMILIES if f[0] in args.families
    ]
    for family, source, version, generator, sizes, is_negative_control in selected:
        for n in sizes:
            try:
                problem = generator(n)
            except Exception as exc:  # noqa: BLE001 -- one family's failure must not stop the sweep
                rows.append(dict(
                    family=family, source=source, version=version, params=f"n={n}",
                    is_negative_control=is_negative_control, n_qubits=n,
                    fused_two_qubit_gates=0, max_qubits=0, wg_log_overhead=float("nan"),
                    wg_feasible=False, junction_count=0, predicted_gain=1.0,
                    predicted_gain_is_lower_bound_only=False, status="error",
                    note=f"generation: {exc}",
                ))
                print(f"[{family} n={n}] GENERATION ERROR: {exc}")
                traceback.print_exc()
                continue
            for capacity in _tight_capacities(problem.circuit.num_qubits):
                result = screen_problem(
                    problem, max_qubits=capacity, family=family, source=source,
                    version=version, params=f"n={n},capacity={capacity}",
                    max_evaluations=args.max_evaluations, seed=0,
                )
                row = asdict(result)
                row["is_negative_control"] = is_negative_control
                rows.append(row)
                print(
                    f"[{family} n={n} cap={capacity}] status={result.status} "
                    f"G2={result.fused_two_qubit_gates} junctions={result.junction_count} "
                    f"predicted_gain={result.predicted_gain:.3f}"
                    + ("" if result.status == "screened" else f" note={result.note}")
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"\nWrote {len(rows)} rows to {args.output}")
    screened = [r for r in rows if r.get("status") == "screened"]
    with_junctions = [r for r in screened if r.get("junction_count", 0) > 0]
    print(f"{len(screened)}/{len(rows)} screened successfully; "
          f"{len(with_junctions)} instance(s) show at least one excision junction.")


if __name__ == "__main__":
    main()
