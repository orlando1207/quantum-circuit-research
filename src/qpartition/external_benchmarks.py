"""Circuit families from external benchmark sources, in this project's input form.

Every generator returns a `ResearchBenchmark` (bound 1-/2-qubit unitary circuit,
"Z...Z" observable, provenance dict). Sources:

- HamLib (Sawaya et al., portal.nersc.gov/cfs/m888/dcamps/hamlib): 2-local
  Hamiltonians read from the published HDF5 files and Trotterized here with a
  bond-grouped Lie-Trotter step: one exact exp(-i dt H_bond) per bond (in
  HamLib's term order) followed by exp(-i dt h_q) for every single-qubit term.
- Google ReCirq, re-implemented in Qiskit (Cirq is not a dependency):
  random circuit sampling with Sycamore fSim(pi/2, pi/6) gates and the
  ABCDCDAB grid pattern; QAOA on hardware-grid, 3-regular and SK graphs; and
  the Fermi-Hubbard ladder (LineLayout).
- CutQC-style supremacy grids (CZ + sqrt(X)/sqrt(Y)/T).
- Qiskit circuit library: Quantum Volume and IQP.

MQT Bench and QASMBench circuits are loaded by the survey runner.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, SXGate, TGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from .benchmarks import ResearchBenchmark
from .circuits import CircuitProblem

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _bench(circuit: QuantumCircuit, provenance: dict) -> ResearchBenchmark:
    """`max_qubits` is not a property of these circuits; the survey chooses capacities."""
    problem = CircuitProblem.create(circuit, "Z" * circuit.num_qubits)
    name = provenance.get("family") or provenance.get("model") or "external"
    return ResearchBenchmark(name=name, problem=problem, max_qubits=0, paper_row=None,
                             provenance=provenance)


# ---------------------------------------------------------------- HamLib


_TERM = re.compile(r"\(?([-+0-9.eE]+)(?:[+-][0-9.eE]+j\))?\)?\s*\[([^\]]*)\]")


def parse_openfermion(text: str) -> list[tuple[float, tuple[tuple[str, int], ...]]]:
    """Parse HamLib's OpenFermion QubitOperator text (real coefficients, signs kept)."""
    terms = []
    for coefficient, ops in _TERM.findall(text):
        factors = tuple((op[0], int(op[1:])) for op in ops.split())
        terms.append((float(coefficient), factors))
    return terms


def hamlib_terms(hdf5_path: str | Path, key: str):
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        raw = handle[key][()]
        n = int(handle[key].attrs["nqubits"])
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    return parse_openfermion(text), n


def trotter_circuit(terms, n: int, *, dt: float, steps: int, initial: str = "plus"):
    """Bond-grouped Lie-Trotter circuit for a <=2-local Hamiltonian."""
    bonds: dict[tuple[int, int], np.ndarray] = {}
    fields: dict[int, np.ndarray] = {}
    for coefficient, factors in terms:
        if not factors:
            continue
        if len(factors) > 2:
            raise ValueError("Only 2-local Hamiltonians are supported.")
        if len(factors) == 1:
            (op, q), = factors
            fields[q] = fields.get(q, 0) + coefficient * _PAULI[op]
            continue
        (op_a, a), (op_b, b) = sorted(factors, key=lambda f: f[1])
        # Qiskit little-endian: the higher qubit index is the left kron factor.
        h = coefficient * np.kron(_PAULI[op_b], _PAULI[op_a])
        bonds[(a, b)] = bonds.get((a, b), 0) + h
    circuit = QuantumCircuit(n)
    if initial == "plus":
        circuit.h(range(n))
    for _ in range(steps):
        for (a, b), h in bonds.items():
            circuit.append(UnitaryGate(expm(-1j * dt * h), check_input=False), [a, b])
        for q, h in sorted(fields.items()):
            circuit.append(UnitaryGate(expm(-1j * dt * h), check_input=False), [q])
    return circuit


def hamlib_trotter(hdf5_path, key, *, dt: float, steps: int, model: str):
    terms, n = hamlib_terms(hdf5_path, key)
    circuit = trotter_circuit(terms, n, dt=dt, steps=steps)
    return _bench(circuit, dict(
        source="HamLib", model=model, key=key, file=Path(hdf5_path).name, dt=dt, steps=steps,
        construction="bond-grouped Lie-Trotter, |+>^n initial state",
    ))


def hamlib_qaoa(hdf5_path, key, *, p: int, gamma: float = 0.4, beta: float = 0.3):
    """QAOA for a HamLib MaxCut (ZZ) cost Hamiltonian: H^n, then p x (cost, RX mixer)."""
    terms, n = hamlib_terms(hdf5_path, key)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(p):
        g = gamma * (layer + 1) / p
        for coefficient, factors in terms:
            if len(factors) == 2:
                (_, a), (_, b) = factors
                circuit.rzz(2 * g * coefficient, a, b)
        circuit.rx(2 * beta * (p - layer) / p, range(n))
    return _bench(circuit, dict(source="HamLib", model="maxcut-qaoa", key=key, p=p,
                                gamma=gamma, beta=beta))


# ---------------------------------------------------------------- ReCirq


def fsim(theta: float, phi: float) -> UnitaryGate:
    c, s = np.cos(theta), np.sin(theta)
    m = np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0],
                  [0, 0, 0, np.exp(-1j * phi)]], dtype=complex)
    return UnitaryGate(m, label=f"fsim({theta:.3f},{phi:.3f})")


SYCAMORE = (np.pi / 2, np.pi / 6)
_SQRT_W = np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]], dtype=complex) / np.sqrt(2)


def _grid_layer(rows, cols, col_offset, vertical, stagger=True):
    """Cirq GridInteractionLayer membership (cirq/experiments/random_quantum_circuit_generation)."""
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if vertical:
                if r + 1 < rows and (c % 2, (r - col_offset) % 2) in {(0, 0), (1, int(stagger))}:
                    pairs.append(((r, c), (r + 1, c)))
            elif c + 1 < cols and (r % 2, (c - col_offset) % 2) in {(0, 0), (1, int(stagger))}:
                pairs.append(((r, c), (r, c + 1)))
    return pairs


# GRID_STAGGERED_PATTERN = ABCDCDAB.
_PATTERN = [(0, True), (1, True), (1, False), (0, False), (1, False), (0, False), (0, True),
            (1, True)]


def random_circuit_sampling(rows: int, cols: int, cycles: int, *, seed: int = 0,
                            two_qubit: str = "sycamore", single_set: str = "sycamore"):
    """ReCirq RCS: random {sqrt X, sqrt Y, sqrt W} layer, then a patterned 2q layer.

    `single_set="supremacy"` draws from {sqrt X, sqrt Y, T} instead (CutQC supremacy)."""
    rng = np.random.default_rng(seed)
    n = rows * cols

    def q(rc):
        return rc[0] * cols + rc[1]

    third = TGate().to_matrix() if single_set == "supremacy" else _SQRT_W
    singles = [SXGate().to_matrix(), expm(-1j * np.pi / 4 * _PAULI["Y"]), third]
    previous = [None] * n
    circuit = QuantumCircuit(n)

    def single_layer():
        for qubit in range(n):
            choices = [i for i in range(3) if i != previous[qubit]]
            k = int(rng.choice(choices))
            previous[qubit] = k
            circuit.append(UnitaryGate(singles[k], check_input=False), [qubit])

    for cycle in range(cycles):
        single_layer()
        offset, vertical = _PATTERN[cycle % len(_PATTERN)]
        for a, b in _grid_layer(rows, cols, offset, vertical):
            if two_qubit == "sycamore":
                circuit.append(fsim(*SYCAMORE), [q(a), q(b)])
            else:
                circuit.append(CZGate(), [q(a), q(b)])
    single_layer()
    return _bench(circuit, dict(
        source="Google ReCirq (re-implemented)", family="random_circuit_sampling",
        grid=[rows, cols], cycles=cycles, seed=seed, two_qubit=two_qubit,
        reference="recirq/random_circuit_sampling; cirq random_rotations_between_grid_"
        "interaction_layers_circuit, GRID_STAGGERED_PATTERN",
    ))


SYCAMORE23 = """
----------
----------
----------
---A------
--ABC-----
-ABCDE----
--CDEFG---
---EFGHI--
----GHI---
-----I----
"""


def sycamore23_graph() -> nx.Graph:
    qubits = set()
    for r, line in enumerate(SYCAMORE23.strip("\n").splitlines()):
        for c, ch in enumerate(line):
            if ch != "-":
                qubits.add((r, c))
    graph = nx.Graph()
    graph.add_nodes_from(qubits)
    for r, c in qubits:
        for nb in ((r + 1, c), (r, c + 1)):
            if nb in qubits:
                graph.add_edge((r, c), nb)
    return graph


def hardware_grid_problem(n: int, seed: int = 52) -> nx.Graph:
    """ReCirq hardware-grid problem: n qubits grown by BFS from (6, 3), random +-1 weights."""
    device = sycamore23_graph()
    order, seen, queue = [], {(6, 3)}, deque([(6, 3)])
    while queue and len(order) < n:
        v = queue.popleft()
        order.append(v)
        for u in sorted(device.neighbors(v)):
            if u not in seen:
                seen.add(u)
                queue.append(u)
    sub = device.subgraph(order[:n])
    index = {v: i for i, v in enumerate(sorted(sub.nodes))}
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for a, b in sorted(sub.edges):
        graph.add_edge(index[a], index[b], weight=float(rng.choice([-1, 1])))
    return graph


def qaoa_circuit(graph: nx.Graph, p: int, *, gamma: float = 0.4, beta: float = 0.3,
                 provenance=None):
    n = graph.number_of_nodes()
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(p):
        g = gamma * (layer + 1) / p
        for a, b, data in graph.edges(data=True):
            circuit.rzz(2 * g * data.get("weight", 1.0), a, b)
        circuit.rx(2 * beta * (p - layer) / p, range(n))
    return _bench(circuit, dict(provenance or {}, p=p, gamma=gamma, beta=beta))


def recirq_qaoa(kind: str, n: int, p: int, *, seed: int = 0):
    if kind == "hardware_grid":
        graph = hardware_grid_problem(n)
    elif kind == "3_regular":
        graph = nx.random_regular_graph(3, n, seed=seed)
        rng = np.random.default_rng(seed)
        for a, b in graph.edges:
            graph[a][b]["weight"] = float(rng.choice([-1, 1]))
    elif kind == "sk":
        graph = nx.complete_graph(n)
        rng = np.random.default_rng(seed)
        for a, b in graph.edges:
            graph[a][b]["weight"] = float(rng.choice([-1, 1]))
    else:
        raise ValueError(kind)
    return qaoa_circuit(graph, p, provenance=dict(
        source="Google ReCirq (re-implemented)", family=f"qaoa_{kind}", n=n, seed=seed,
        reference="recirq/qaoa/problems.py",
    ))


def fermi_hubbard_ladder(sites: int, steps: int, *, J: float = 1.0, U: float = 4.0,
                         dt: float = 0.3):
    """ReCirq Fermi-Hubbard LineLayout: spin-up chain 0..L-1 and spin-down chain L..2L-1
    on a 2 x L ladder. Step: FSim(-J dt, 0) hopping on even then odd chain bonds,
    then CPhase(-U dt) on rungs. Initial state: alternating occupation (product)."""
    n = 2 * sites
    circuit = QuantumCircuit(n)
    for i in range(sites):
        if i % 2 == 0:
            circuit.x(i)
        else:
            circuit.x(sites + i)
    for _ in range(steps):
        for parity in (0, 1):
            for chain in (0, sites):
                for i in range(parity, sites - 1, 2):
                    circuit.append(fsim(-J * dt, 0.0), [chain + i, chain + i + 1])
        for i in range(sites):
            circuit.cp(-U * dt, i, sites + i)
    return _bench(circuit, dict(
        source="Google ReCirq (re-implemented)", family="fermi_hubbard_ladder", sites=sites,
        steps=steps, J=J, U=U, dt=dt,
        reference="recirq/fermi_hubbard/layouts.py LineLayout, circuits.py",
        deviation="product initial state instead of the Givens-rotation preparation",
    ))


# ---------------------------------------------------------------- other


def supremacy_cz(rows: int, cols: int, depth: int, *, seed: int = 0):
    """CutQC-style supremacy grid: CZ on the ABCDCDAB pattern, random sqrt X / sqrt Y / T."""
    bench = random_circuit_sampling(rows, cols, depth, seed=seed, two_qubit="cz",
                                    single_set="supremacy")
    return _bench(bench.problem.circuit, dict(
        bench.provenance, family="supremacy_cz",
        reference="CutQC helper_functions/benchmarks.py (supremacy); CZ on the ABCDCDAB pattern",
    ))


def quantum_volume_circuit(n: int, *, seed: int = 0):
    from qiskit.circuit.library import QuantumVolume

    circuit = QuantumVolume(n, depth=n, seed=seed).decompose()
    return _bench(circuit, dict(source="Qiskit circuit library", family="quantum_volume",
                                n=n, seed=seed))


def iqp_circuit(n: int, *, seed: int = 0):
    from qiskit.circuit.library import IQP

    rng = np.random.default_rng(seed)
    upper = np.triu(rng.integers(0, 8, size=(n, n)))
    matrix = upper + upper.T - np.diag(np.diag(upper))
    circuit = IQP(matrix).decompose()
    return _bench(circuit, dict(source="Qiskit circuit library", family="iqp", n=n, seed=seed))


def observable_for(n: int) -> SparsePauliOp:
    return SparsePauliOp("Z" * n)
