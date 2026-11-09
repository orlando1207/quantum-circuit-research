"""Pinned-generator reconstructions; do not claim unpublished artifact identity."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import pi

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit.synthesis.qft import synth_qft_full

from .circuits import CircuitProblem


@dataclass(frozen=True)
class ResearchBenchmark:
    name: str
    problem: CircuitProblem
    max_qubits: int
    paper_row: str | None
    provenance: dict


def qft_benchmark(n: int) -> ResearchBenchmark:
    circuit = synth_qft_full(n, do_swaps=True)
    circuit.name = f"qft_{n}_algorithmic"
    two_qubit_count = sum(item.operation.num_qubits == 2 for item in circuit.data)
    assert two_qubit_count == n * (n - 1) // 2 + n // 2
    return ResearchBenchmark(
        f"qft{n}",
        CircuitProblem.create(circuit, "Z" * n),
        3 * n // 4,
        f"qft{n}",
        {
            "kind": "count-and-spatial-cost-matched-reconstruction",
            "generator": "qiskit.synthesis.qft.synth_qft_full(do_swaps=True)",
            "mqt_source": "https://github.com/munich-quantum-toolkit/bench/blob/"
            "2cd3f3b08d5d8866595948fbce955e701e86a5df/"
            "src/mqt/bench/benchmarks/qft.py",
            "two_qubit_gates": two_qubit_count,
            "comparison_caveat": "Paper does not pin circuit files or MQT/Qiskit versions; "
            "matching counts and rounded spatial costs does not prove identity.",
        },
    )


def chain_benchmark(n: int) -> ResearchBenchmark:
    """Four-qubit blocks overlap on one wire, as Fig. 8b; block contents are a proxy.

    Full blocks contain two directed CX layers on all six pairs (12 gates).
    A trailing two-qubit block contains one CX. This matches 36/73 gates for
    n=10/20, while deliberately retaining the unspecified-block caveat.
    """
    if n < 4:
        raise ValueError("Chain benchmark requires at least four qubits.")
    circuit = QuantumCircuit(n, name=f"chain_{n}_structural_proxy")
    circuit.h(range(n))
    windows = []
    for first in range(0, n - 1, 3):
        qubits = tuple(range(first, min(first + 4, n)))
        windows.append(qubits)
        pairs = tuple(combinations(qubits, 2))
        for a, b in pairs:
            circuit.cx(a, b)
        if len(qubits) == 4:
            for a, b in pairs:
                circuit.cx(b, a)
    return ResearchBenchmark(
        f"chain{n}",
        CircuitProblem.create(circuit, "Z" * n),
        3 * n // 4,
        f"chain{n}",
        {
            "kind": "structural-proxy",
            "generator": "overlapping-four-qubit-CX-blocks-v1",
            "windows": windows,
            "two_qubit_gates": sum(item.operation.num_qubits == 2 for item in circuit.data),
            "comparison_caveat": "Fig. 8b specifies block overlap but not block gates; "
            "this is not the authors' exact chain circuit.",
        },
    )


def initial_paper_suite() -> tuple[ResearchBenchmark, ...]:
    return (qft_benchmark(6), qft_benchmark(12), chain_benchmark(10), chain_benchmark(20))


def qaoa_benchmark(n: int) -> ResearchBenchmark:
    """Bound p=2 cycle-MaxCut reconstruction; optimized angles are ours, not supplied."""
    if n not in (8, 15):
        raise ValueError("The pinned QAOA reconstructions are defined for 8/15 qubits.")
    edges = sorted(nx.random_regular_graph(2, n, seed=111).edges())
    gammas = (-4.4310439746041945, -2.496868463247365)
    betas = (-0.32236282932132954, -0.9260703445189131)
    circuit = QuantumCircuit(n, name=f"qaoa_{n}_bound_reconstruction")
    circuit.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for a, b in edges:
            circuit.rzz(gamma, a, b)
        circuit.rx(2 * beta, range(n))
    return ResearchBenchmark(
        f"qaoa{n}",
        CircuitProblem.create(circuit, "Z" * n),
        3 * n // 4,
        f"qaoa{n}",
        {
            "kind": "count-and-spatial-cost-matched-reconstruction",
            "generator": "degree-2-MaxCut-p2-bound-v2",
            "edges": edges,
            "gammas": gammas,
            "betas": betas,
            "source": "https://github.com/munich-quantum-toolkit/bench/blob/"
            "9ac1ecf7b0fad995d8e04d327bb80b792873201e/src/mqt/bench/benchmarks/qaoa.py",
            "graph_source": "historical MQT helper: random_regular_graph(2,n,seed=111)",
            "angle_source": "Own exact eight-qubit MaxCut SLSQP run, maxiter=25, ftol=1e-8; "
            "initial_point=default_rng(111).uniform(-pi,pi,4), order=(gamma1,gamma2,beta1,beta2). "
            "Expectation=6.666666666652331; 133 evaluations. Frozen angles reused on 15-cycle.",
            "two_qubit_gates": 2 * n,
            "comparison_caveat": "Paper angles/artifacts are unavailable. Counts and spatial "
            "cost match do not establish the same circuit; no original-artifact win claim.",
        },
    )


def figure2_benchmark() -> ResearchBenchmark:
    circuit = QuantumCircuit(4, name="paper_figure2_literal")
    circuit.h(range(4))
    for a, b, kind in (
        (0, 1, "cz"),
        (2, 3, "cz"),
        (1, 2, "cz"),
        (0, 1, "cx"),
        (2, 3, "cx"),
        (0, 1, "cz"),
        (2, 3, "cz"),
        (1, 2, "cz"),
        (1, 2, "cz"),
    ):
        getattr(circuit, kind)(a, b)
    return ResearchBenchmark(
        "figure2",
        CircuitProblem.create(circuit, "XXXX"),
        3,
        None,
        {
            "kind": "literal-figure-topology",
            "two_qubit_gates": 9,
            "source": "https://arxiv.org/html/2608.05287v1/3_example_combinedcut.svg",
            "reference_gate_only_overhead": 729,
            "reference_mixed_overhead": 144,
            "comparison_caveat": "Figure example, not a Table-I row. Added local H preparations. "
            "Static two-block capacity 3; the wire-only cartoon is not assumed feasible.",
        },
    )


def bridge_benchmark(angle: float, name: str) -> ResearchBenchmark:
    circuit = QuantumCircuit(8, name=name)
    circuit.h(range(8))
    for layer in range(2):
        for a, b in combinations(range(3), 2):
            circuit.cz(a, b)
        for q in range(3):
            circuit.cp(angle, 3, q)
        if layer == 0:
            for q in range(4):
                circuit.ry(0.17 + 0.03 * q, q)
    for layer in range(2):
        for a, b in combinations(range(3, 8), 2):
            circuit.cz(a, b)
        if layer == 0:
            for q in range(3, 8):
                circuit.ry(0.11 + 0.02 * q, q)
    return ResearchBenchmark(
        name,
        CircuitProblem.create(circuit, "X" * 8),
        6,
        None,
        {
            "kind": "synthetic-cost-crossover-control",
            "phase_angle": angle,
            "generator": "noncommuting-overlapping-4/5-qubit-blocks-v3",
            "local_rotations": "RY(.17+.03*q) on 0..3 and RY(.11+.02*q) on 3..7, "
            "between repeated entangling layers, preventing strong-phase cancellation.",
            "two_qubit_gates": 32,
            "comparison_caveat": "Own diagnostic circuit; no paper quality comparison.",
        },
    )


def random_benchmark() -> ResearchBenchmark:
    rng = np.random.default_rng(71)
    circuit = QuantumCircuit(8, name="random8_seed71")
    circuit.h(range(8))
    for _ in range(4):
        permutation = rng.permutation(8)
        for q in range(8):
            circuit.ry(float(rng.uniform(-pi, pi)), q)
        for a, b in zip(permutation[::2], permutation[1::2]):
            circuit.cx(int(a), int(b))
    return ResearchBenchmark(
        "random8",
        CircuitProblem.create(circuit, "Z" * 8),
        6,
        None,
        {
            "kind": "synthetic-random-control",
            "seed": 71,
            "layers": 4,
            "two_qubit_gates": 16,
            "comparison_caveat": "Own diagnostic; no Table-I baseline.",
        },
    )


def grover_control() -> ResearchBenchmark:
    oracle = QuantumCircuit(4)
    oracle.mcx([0, 1, 2], 3)
    operator = grover_operator(oracle, reflection_qubits=[0, 1, 2])
    circuit = QuantumCircuit(4)
    circuit.x(3)
    circuit.h(range(4))
    circuit.compose(operator.power(2), inplace=True)
    circuit = transpile(circuit, basis_gates=["u", "cx"], optimization_level=2, seed_transpiler=7)
    circuit.name = "grover4_phase_kickback_control"
    count = sum(item.operation.num_qubits == 2 for item in circuit.data)
    assert count == 38
    return ResearchBenchmark(
        "grover4_control",
        CircuitProblem.create(circuit, "ZZZZ"),
        3,
        None,
        {
            "kind": "related-Grover-synthesis-control",
            "two_qubit_gates": count,
            "generator": "MCX phase-kickback oracle, two iterations, u/cx opt2 seed7",
            "comparison_caveat": "38-gate count matches Table I but the oracle flag preparation "
            "and decomposition differ from historical MQT. No direct Table-I comparison.",
        },
    )


def extended_suite() -> tuple[ResearchBenchmark, ...]:
    return (
        *initial_paper_suite(),
        qaoa_benchmark(8),
        qaoa_benchmark(15),
        figure2_benchmark(),
        bridge_benchmark(0.05, "bridge_weak"),
        bridge_benchmark(pi, "bridge_strong"),
        random_benchmark(),
        grover_control(),
    )
