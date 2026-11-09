"""CX-only benchmark families for the MILP verification study (docs/MILP_prompt.txt).

Every two-qubit gate in these generators is CX (or, where noted, its local-
Clifford-equivalent CZ), whose physical excision recipe norm is exactly 16 in
the existing registry (`templates.py`; confirmed by
`docs/WORKFLOW_AND_ALGORITHM.md` Section 5.2's gate-norm table). This lets the
study use "16" directly as each candidate's real, re-verifiable gamma_exc
rather than an unverifiable assumed cost (`TemplateRegistry.excision_gamma`),
so every MILP solution stays physically recompilable through the existing
`compile_hybrid` pipeline. Single-qubit rotations are free of charge and do
not affect partitioning cost; they are included only to avoid degenerate
all-identity wires and to keep circuit fingerprints distinct per seed.

Families here are deliberately small (N in roughly 8-32 qubits) and built for
temporal/spatial structure that can make excision locally worthwhile, unlike
the 50-250 qubit frozen workloads in `large_benchmarks.py`. The separator
family (`large_benchmarks.large_benchmark("excision_butterfly", ...)`) and
chain/modular family (`benchmarks.chain_benchmark`) are reused directly by the
runner rather than duplicated here.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder

from .benchmarks import ResearchBenchmark
from .circuits import CircuitProblem


def _angle(seed: int, index: int, base: float = 0.11, step: float = 0.037) -> float:
    return base + step * ((seed * 97 + index) % 23)


def alternating_brickwork(n: int, *, layers: int, seed: int = 0) -> ResearchBenchmark:
    """Deterministic 1D brickwork: alternating even/odd-offset CX layers.

    Unlike `large_benchmarks.large_benchmark("random_brickwork", ...)`, the
    pairing pattern is fixed (not a random matching per layer), giving a
    reproducible, well-understood nearest-neighbor structure; `seed` only
    perturbs the free single-qubit rotation angles between layers.
    """
    if n < 4:
        raise ValueError("Brickwork needs at least four qubits.")
    if layers < 2 or layers > 6:
        raise ValueError("This family targets 2-6 shallow layers (Section 12).")
    circuit = QuantumCircuit(n, name=f"brickwork_alt_{n}_L{layers}_s{seed}")
    for q in range(n):
        circuit.ry(_angle(seed, q), q)
    for layer in range(layers):
        offset = layer % 2
        for q in range(offset, n - 1, 2):
            circuit.cx(q, q + 1)
        for q in range(n):
            circuit.ry(_angle(seed, q + 1000 * (layer + 1)), q)
    two_qubit_gates = sum(item.operation.num_qubits == 2 for item in circuit.data)
    return ResearchBenchmark(
        circuit.name,
        CircuitProblem.create(circuit, "Z" * n),
        max(2, n // 2),
        None,
        {
            "kind": "milp-verification-instance",
            "family": "brickwork_alternating",
            "layers": layers,
            "seed": seed,
            "two_qubit_gates": two_qubit_gates,
            "interpretation": "Deterministic alternating even/odd CX brickwork, own generator.",
        },
    )


def ripple_carry_adder(bits: int, *, seed: int = 0) -> ResearchBenchmark:
    """Cuccaro-style ripple-carry adder (Qiskit's CDKM construction), decomposed to CX + u."""
    if bits < 2:
        raise ValueError("The adder needs at least two bits per register.")
    raw = CDKMRippleCarryAdder(bits)
    circuit = transpile(raw, basis_gates=["cx", "u"], optimization_level=0, seed_transpiler=seed)
    circuit.name = f"adder_{bits}b_s{seed}"
    n = circuit.num_qubits
    two_qubit_gates = sum(item.operation.num_qubits == 2 for item in circuit.data)
    other = {item.operation.name for item in circuit.data} - {"cx", "u"}
    if other:
        raise RuntimeError(f"Adder decomposition left non-CX/u operations: {other}")
    return ResearchBenchmark(
        circuit.name,
        CircuitProblem.create(circuit, "Z" * n),
        max(2, n // 2),
        None,
        {
            "kind": "milp-verification-instance",
            "family": "ripple_carry_adder",
            "bits": bits,
            "seed": seed,
            "two_qubit_gates": two_qubit_gates,
            "generator": "qiskit.circuit.library.CDKMRippleCarryAdder, transpiled to cx+u",
        },
    )


def qcnn(n: int, *, seed: int = 0) -> ResearchBenchmark:
    """A unitary QCNN-style ansatz: CX convolution layers plus CX pooling halvings.

    Mirrors the structure of Cong-Choi-Lukin QCNN circuits (convolution over
    nearest-neighbor active pairs, then pooling that halves the active set)
    without measurement: pooling is a CX from each pooled qubit into its
    partner, which keeps the circuit unitary and keeps all qubits physically
    present. Both convolution and pooling use CX (gamma_exc = 16).
    """
    if n < 4:
        raise ValueError("QCNN needs at least four qubits.")
    circuit = QuantumCircuit(n, name=f"qcnn_{n}_s{seed}")
    active = list(range(n))
    for q in active:
        circuit.ry(_angle(seed, q), q)
    level = 0
    while len(active) > 1:
        for a, b in zip(active, active[1:]):
            circuit.cx(a, b)
            circuit.ry(_angle(seed, a + 1000 * (level + 1)), a)
            circuit.ry(_angle(seed, b + 1000 * (level + 1)), b)
        pooled = []
        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]
            circuit.cx(b, a)  # Pool b into a; a survives to the next level.
            pooled.append(a)
        if len(active) % 2:
            pooled.append(active[-1])
        active = pooled
        level += 1
    two_qubit_gates = sum(item.operation.num_qubits == 2 for item in circuit.data)
    return ResearchBenchmark(
        circuit.name,
        CircuitProblem.create(circuit, "Z" * n),
        max(2, n // 2),
        None,
        {
            "kind": "milp-verification-instance",
            "family": "qcnn",
            "seed": seed,
            "two_qubit_gates": two_qubit_gates,
            "interpretation": "Unitary CX-only convolution+pooling ansatz; own generator, "
            "not a trained/paper QCNN instance.",
        },
    )


def tree_tensor_network(n_leaves: int, *, seed: int = 0) -> ResearchBenchmark:
    """A binary-tree-of-CX circuit: the canonical unitary tree-tensor-network ansatz."""
    if n_leaves < 4:
        raise ValueError("A tree-tensor-network needs at least four leaves.")
    circuit = QuantumCircuit(n_leaves, name=f"ttn_{n_leaves}_s{seed}")
    active = list(range(n_leaves))
    for q in active:
        circuit.ry(_angle(seed, q), q)
    level = 0
    while len(active) > 1:
        survivors = []
        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]
            circuit.cx(b, a)
            circuit.ry(_angle(seed, a + 1000 * (level + 1)), a)
            survivors.append(a)
        if len(active) % 2:
            survivors.append(active[-1])
        active = survivors
        level += 1
    two_qubit_gates = sum(item.operation.num_qubits == 2 for item in circuit.data)
    return ResearchBenchmark(
        circuit.name,
        CircuitProblem.create(circuit, "Z" * n_leaves),
        max(2, n_leaves // 2),
        None,
        {
            "kind": "milp-verification-instance",
            "family": "tree_tensor_network",
            "seed": seed,
            "two_qubit_gates": two_qubit_gates,
            "interpretation": "Binary-tree CX merge ansatz, own generator.",
        },
    )


def two_qubit_gate_names(problem: CircuitProblem) -> set[str]:
    return {
        item.operation.name for item in problem.circuit.data if item.operation.num_qubits == 2
    }
