"""Pinned-generator reconstructions; do not claim unpublished artifact identity."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from qiskit import QuantumCircuit
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
