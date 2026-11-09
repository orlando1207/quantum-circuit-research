"""Deterministic scalable circuits and a deliberately favorable excision control."""

from __future__ import annotations

from math import pi

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import GraphStateGate

from .benchmarks import ResearchBenchmark
from .circuits import CircuitProblem


def large_benchmark(family: str, n: int, *, layers: int = 3, seed: int = 19):
    if isinstance(n, bool) or not isinstance(n, int) or n < 6 or layers < 1:
        raise ValueError("Use at least six qubits and a positive layer count.")
    circuit = QuantumCircuit(n, name=f"{family}_{n}_L{layers}_s{seed}")
    circuit.h(range(n))
    provenance = dict(kind="synthetic-scalable-control", family=family, seed=seed, layers=layers)
    if family == "excision_butterfly":
        # Two past modules meet at one CX; the two output wires then enter
        # distinct future modules. Each module has its own private qubits.
        private = list(range(2, n))
        modules = [private[i * (n - 2) // 4 : (i + 1) * (n - 2) // 4] for i in range(4)]
        modules = [[i % 2, *qs] for i, qs in enumerate(modules)]

        def module(qs):
            for layer in range(layers):
                for q in qs:
                    circuit.ry(0.13 + 0.07 * layer, q)
                # Repeated star interactions give a controlled temporal-module
                # separator; this is a diagnostic, not a representative workload.
                for q in qs[1:]:
                    circuit.cx(qs[0], q)

        module(modules[0])
        module(modules[1])
        boundary = len(circuit.data)
        circuit.cx(0, 1)
        module(modules[2])
        module(modules[3])
        capacity = max(map(len, modules))
        provenance.update(
            modules=modules,
            boundary_operation=f"op{boundary:06d}",
            known_excision_overhead=256,
            diagnostic="Deliberately constructed four-module separator; no generic win claim",
        )
    elif family in {"trotter_chain", "trotter_grid", "sparse_expander"}:
        if family == "trotter_chain":
            edges = [(q, q + 1) for q in range(n - 1)]
        elif family == "trotter_grid":
            width = max(2, int(n**0.5))
            edges = [(q, q + 1) for q in range(n - 1) if (q + 1) % width]
            edges += [(q, q + width) for q in range(n - width)]
        else:
            degree = 3 if n % 2 == 0 else 4
            edges = sorted(nx.random_regular_graph(degree, n, seed=seed).edges())
        for layer in range(layers):
            for a, b in edges:
                circuit.rzz(0.37, a, b)
            circuit.rx(0.23 + 0.02 * layer, range(n))
        capacity = 3 * n // 4
        provenance.update(
            edges=edges,
            angles=[0.37, 0.23],
            interpretation="Bound Hamiltonian-simulation workload, no parameter optimization",
        )
    elif family == "graphstate":
        # Algorithmic-level construction in MQT Bench v2.3.0, with final
        # measurements omitted to match this project's unitary input contract.
        graph = nx.random_regular_graph(3, n, seed=seed)
        circuit = QuantumCircuit(n, name=f"graphstate_{n}_d3_s{seed}")
        circuit.append(GraphStateGate(nx.to_numpy_array(graph)), range(n))
        circuit = circuit.decompose(gates_to_decompose="graph_state")
        capacity = 3 * n // 4
        provenance.update(
            kind="attributed-generator-adaptation",
            degree=3,
            layers=None,
            source="https://github.com/munich-quantum-toolkit/bench/blob/v2.3.0/"
            "src/mqt/bench/benchmarks/graphstate.py",
            changes="Degree three; remove final measurements. Not a paper Table-I artifact.",
        )
    elif family == "random_brickwork":
        import numpy as np

        rng = np.random.default_rng(seed)
        for layer in range(layers):
            circuit.ry(0.19 + 0.03 * layer, range(n))
            order = rng.permutation(n)
            for i in range(0, n - 1, 2):
                circuit.cx(int(order[i]), int(order[i + 1]))
        capacity = 3 * n // 4
        provenance["interpretation"] = "Fixed-angle random-matching CX layers; own stress test"
    elif family == "aqft":
        # Bounded interaction range makes 1000-qubit generation linear in n.
        for q in reversed(range(n)):
            circuit.h(q)
            for delta in range(1, min(6, q + 1)):
                circuit.cp(pi / 2**delta, q, q - delta)
        capacity = 3 * n // 4
        provenance.update(
            cutoff=5, approximation="Omitted CP gates beyond distance five; no final swaps"
        )
    else:
        raise ValueError(f"Unknown scalable benchmark family: {family}")
    provenance["two_qubit_gates"] = sum(i.operation.num_qubits == 2 for i in circuit.data)
    provenance["source_identity"] = "Frozen generator; not an original paper Table-I circuit"
    return ResearchBenchmark(
        circuit.name, CircuitProblem.create(circuit, "Z" * n), capacity, None, provenance
    )
