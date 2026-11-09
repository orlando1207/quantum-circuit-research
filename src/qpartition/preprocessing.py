"""Mandatory fair-baseline preprocessing for externally sourced benchmark circuits
(docs/second_stage_prompt.txt Task 2, "Mandatory preprocessing"; SWAP relabeling
added per docs/second_stage_cont_prompt.txt Part 1 Action #2).

External/transpiled circuits routinely decompose one logical two-qubit
interaction into several consecutive native two-qubit gates on the same pair
(e.g. an RZZ realized as CX-RZ-CX). Partitioning each such gate independently
inflates WG for no physical reason (the prompt's own example: 3 independent
CX cuts cost Gamma=27, vs. ~5.85 for the fused SU(4)) and would make any
WG-vs-WGE comparison unfair. `fuse_two_qubit_gates` collapses every maximal
run of consecutive two-qubit gates on the same pair -- absorbing any
intervening single-qubit gates on either qubit -- into one explicit SU(4)
`UnitaryGate`, using Qiskit's own circuit composition (not hand-rolled tensor
algebra) to get the qubit-ordering conventions right by construction.

A run on pair (a, b) ends as soon as a gate touches a or b together with a
THIRD qubit (or the circuit ends); single-qubit gates on qubits not currently
in an open run, and gates with 3+ qubits, pass through unchanged (any 3+
qubit gate closes every run it touches, conservatively, since it cannot be
absorbed into a two-qubit block).

`peel_swaps` implements SWAP relabeling as an actual circuit-level transform
(not merely the cost-formula `swap_peeled_gamma_cut` in `excision_lp.py`,
which only affects baseline numbers used for comparison). A logical SWAP is
free: for any two-qubit U, U = SWAP . V with V = SWAP . U, so applying V on
the current physical wires and then recording that the two logical qubits'
data have exchanged physical position reproduces U's effect exactly, with no
extra gate. This pass walks the ORIGINAL (unrouted, logical) circuit once,
keeps a running logical-to-physical permutation (init: identity), and for
each two-qubit gate picks whichever of U or V=SWAP.U is cheaper to cut
(`excision_lp.gamma_cut_gate`) -- peeling (using V and swapping the
permutation) only when it is strictly cheaper. The final observable is then
remapped through the accumulated permutation so that expectation values are
preserved exactly (see `tests/test_preprocessing.py` for a direct
statevector-level check of this invariant). This must run before fusion and
before KAK/LP costing, per the mandatory preprocessing order; it must never
be applied to an already-hardware-routed circuit (those SWAPs are routing
artifacts, not logical structure, and peeling them would misattribute
routing overhead as algorithmic gain).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, PauliList, SparsePauliOp

from qpartition.excision_lp import gamma_cut_gate

_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def fuse_two_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    owner: dict[int, dict] = {}

    def close(qubit: int) -> None:
        cell = owner.get(qubit)
        if cell is not None:
            del owner[cell["order"][0]]
            del owner[cell["order"][1]]

    entries: list[tuple[str, object]] = []  # ("op", (op, qubits)) or ("cell", cell)

    for item in circuit.data:
        op = item.operation
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier" or len(qubits) >= 3:
            for q in qubits:
                close(q)
            entries.append(("op", (op, qubits)))
            continue
        if len(qubits) == 1:
            (q,) = qubits
            cell = owner.get(q)
            if cell is None:
                entries.append(("op", (op, qubits)))
            else:
                cell["circuit"].append(op, [cell["order"].index(q)])
            continue
        a, b = qubits
        cell_a, cell_b = owner.get(a), owner.get(b)
        if cell_a is not None and cell_a is cell_b:
            cell_a["circuit"].append(op, [cell_a["order"].index(a), cell_a["order"].index(b)])
        else:
            close(a)
            close(b)
            cell = {"order": (a, b), "circuit": QuantumCircuit(2)}
            cell["circuit"].append(op, [0, 1])
            owner[a] = cell
            owner[b] = cell
            entries.append(("cell", cell))

    fused = circuit.copy_empty_like()
    for kind, payload in entries:
        if kind == "op":
            op, qubits = payload
            fused.append(op, qubits)
        else:
            cell = payload
            matrix = Operator(cell["circuit"]).data
            fused.append(UnitaryGate(matrix, label="fused"), list(cell["order"]))
    return fused


def two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return sum(item.operation.num_qubits == 2 for item in circuit.data)


def _remap_observable(
    observable: SparsePauliOp, phys_of_logical: list[int]
) -> SparsePauliOp:
    """Permute each Pauli term's per-qubit operators from logical to physical
    position. `PauliList.z`/`.x` columns are indexed directly by qubit number
    (no string-direction reversal needed), so this is a pure column
    permutation; term phases are untouched since no operator changes, only
    which qubit it sits on."""
    paulis = observable.paulis
    new_z = np.zeros_like(paulis.z)
    new_x = np.zeros_like(paulis.x)
    for logical_q, physical_q in enumerate(phys_of_logical):
        new_z[:, physical_q] = paulis.z[:, logical_q]
        new_x[:, physical_q] = paulis.x[:, logical_q]
    new_paulis = PauliList.from_symplectic(new_z, new_x, paulis.phase)
    return SparsePauliOp(new_paulis, coeffs=observable.coeffs)


def peel_swaps(
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None = None,
    *,
    tol: float = 1e-9,
) -> tuple[QuantumCircuit, SparsePauliOp | None]:
    """Logical-level SWAP-relabeling pass (docs/second_stage_cont_prompt.txt
    Part 1 Action #2). See module docstring for the math; this is the
    circuit-level transform, not just the cost-formula baseline.

    `circuit` must be the original, UNROUTED logical circuit -- hardware
    routing SWAPs from a prior transpile are artifacts, not logical
    structure, and must not be present here. Single- and 3+-qubit operations
    (and barriers) pass through at the gates' current physical position with
    no effect on the running permutation; only two-qubit gates are ever
    candidates for peeling.

    Returns `(new_circuit, new_observable)`; `new_observable` is `None` iff
    `observable` was `None`. The transform is exact: evaluating `observable`
    against `circuit` and `new_observable` against `new_circuit` (both
    starting from the same initial state) gives identical expectation
    values -- this is checked directly, at the statevector level, in
    `tests/test_preprocessing.py`.
    """
    n = circuit.num_qubits
    phys_of_logical = list(range(n))
    new_circuit = circuit.copy_empty_like()
    for item in circuit.data:
        op = item.operation
        logical_qubits = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier" or len(logical_qubits) != 2:
            physical_qubits = [phys_of_logical[q] for q in logical_qubits]
            new_circuit.append(op, physical_qubits)
            continue
        a, b = logical_qubits
        pa, pb = phys_of_logical[a], phys_of_logical[b]
        U = np.asarray(Operator(op).data, dtype=complex)
        V = _SWAP @ U
        if gamma_cut_gate(V) < gamma_cut_gate(U) - tol:
            new_circuit.append(UnitaryGate(V, label=f"{op.name}*swap"), [pa, pb])
            phys_of_logical[a], phys_of_logical[b] = pb, pa
        else:
            new_circuit.append(op, [pa, pb])
    new_observable = None if observable is None else _remap_observable(observable, phys_of_logical)
    return new_circuit, new_observable
