"""Ancilla-free joint QPDs for Pauli packets and CX/CZ parity cascades.

Local signed maps are projective Pauli-parity instruments or signed mixtures
of Pauli rotations. No inter-fragment communication or uncounted qubits occur.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import cos, pi, sin

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli


@dataclass(frozen=True)
class LocalStep:
    operation: object
    qubits: tuple[int, ...]
    scored: bool = False


@dataclass(frozen=True)
class JointBasis:
    coefficients: tuple[float, ...]
    maps: tuple[tuple[tuple[LocalStep, ...], tuple[LocalStep, ...]], ...]
    gamma: float
    unitary_residual: float


@lru_cache(maxsize=65536)
def multiply_paulis(left, right):
    p = Pauli(left).dot(Pauli(right))
    return (-1j) ** int(p.phase), Pauli((p.z, p.x)).to_label()


def embedded_pauli(axes, qubits):
    return "".join(axes.get(q, "I") for q in reversed(qubits))


def native_lcu(operation, operands, support):
    """Return local post-unitaries and Pauli LCU, up to an irrelevant global phase."""
    name = operation.name
    a, b = operands
    posts = []
    if name == "swap":
        return posts, [(0.5, embedded_pauli({a: axis, b: axis}, support)) for axis in "IXYZ"]
    if name == "cx":
        axes, theta = {a: "Z", b: "X"}, pi / 4
        for q, axis in ((a, "Z"), (b, "X")):
            p = Pauli(axis).to_matrix()
            posts.append((q, cos(pi / 4) * np.eye(2) - 1j * sin(pi / 4) * p))
    elif name in {"cp", "cz"}:
        angle = pi if name == "cz" else float(operation.params[0])
        theta, axes = angle / 4, {a: "Z", b: "Z"}
        for q in operands:
            posts.append((q, np.diag([np.exp(-1j * theta), np.exp(1j * theta)])))
    elif name in {"rxx", "ryy", "rzz"}:
        theta = -float(operation.params[0]) / 2
        axes = {a: name[1].upper(), b: name[1].upper()}
    else:
        raise ValueError(f"No ancilla-free packet rule for {name}.")
    return posts, [
        (cos(theta), "I" * len(support)),
        (1j * sin(theta), embedded_pauli(axes, support)),
    ]


def pauli_instrument(label):
    """Lueders measurement of a whole Pauli string, preserving parity coherence."""
    n = len(label)
    active = [q for q in range(n) if label[-q - 1] != "I"]
    if not active:
        return ()
    anchor = active[-1]
    basis = QuantumCircuit(n)
    for q in active:
        axis = label[-q - 1]
        if axis == "Y":
            basis.sdg(q)
        if axis in "XY":
            basis.h(q)
    for q in active[:-1]:
        basis.cx(q, anchor)
    steps = [
        LocalStep(i.operation, tuple(basis.find_bit(q).index for q in i.qubits)) for i in basis.data
    ]
    steps.append(LocalStep(Measure(), (anchor,), True))
    for i in basis.inverse().data:
        steps.append(LocalStep(i.operation, tuple(basis.find_bit(q).index for q in i.qubits)))
    return tuple(steps)


def local_transition(left, right, imaginary):
    """Real/imaginary transition maps for left rho right^dagger, phase removed."""
    phase, relative = multiply_paulis(left, right)
    # Pauli strings are Hermitian: right = phase * left * relative.
    n = len(left)
    lmatrix = Pauli(left).to_matrix()
    if not imaginary:
        maps = ((1.0, pauli_instrument(relative)),)
    else:
        h = Pauli(relative).to_matrix()
        maps = tuple(
            (
                sign / 2,
                (
                    LocalStep(
                        UnitaryGate((np.eye(2**n) + sign * 1j * h) / np.sqrt(2)), tuple(range(n))
                    ),
                ),
            )
            for sign in (1, -1)
        )
    append = (LocalStep(UnitaryGate(lmatrix), tuple(range(n))),)
    return phase.conjugate(), tuple((weight, steps + append) for weight, steps in maps)


def _transition_signature(left, right, imaginary):
    phase, relative = multiply_paulis(left, right)
    if not imaginary:
        key = (
            ("u", left)
            if set(relative) == {"I"}
            else ("r", relative, min(left, multiply_paulis(left, relative)[1]))
        )
        return phase.conjugate(), ((1.0, key),)
    if set(relative) == {"I"}:
        return phase.conjugate(), ()  # The imaginary transition of identity is zero.
    phase2, other = multiply_paulis(left, relative)
    recipes = []
    for sign in (1, -1):
        words = sorted(((left, 1), (other, sign * 1j * phase2)))
        ratio = words[1][1] / words[0][1]
        key = ("rot", words[0][0], words[1][0], round(ratio.real), round(ratio.imag))
        recipes.append((sign / 2, key))
    return phase.conjugate(), tuple(recipes)


def coalesce_paulis(terms):
    combined = defaultdict(complex)
    for coefficient, label in terms:
        combined[label] += coefficient
    return tuple((c, p) for p, c in sorted(combined.items()) if abs(c) > 1e-14)


def joint_pauli_recipes(terms, support, sides):
    """Collect identical executable signed maps, without materializing matrices.

    L R_P and (L P) R_P are the same signed instrument. Rotations are identified
    by their two Pauli words up to global phase. This also cancels I_identity=0.
    No abstract Schmidt norm or optimal-instrument cost is substituted.
    """
    terms = coalesce_paulis(terms)
    labels = [
        ["".join(p[-support.index(q) - 1] for q in reversed(side)) for _, p in terms]
        for side in sides
    ]
    weights = defaultdict(float)
    for i, (c, _) in enumerate(terms):
        weights[(("u", labels[0][i]), ("u", labels[1][i]))] += abs(c) ** 2
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            transitions = [
                [_transition_signature(labels[s][i], labels[s][j], imag) for imag in (False, True)]
                for s in (0, 1)
            ]
            z = terms[i][0] * terms[j][0].conjugate()
            z *= transitions[0][0][0] * transitions[1][0][0]
            for ai, bi, coefficient in (
                (0, 0, 2 * z.real),
                (1, 1, -2 * z.real),
                (0, 1, -2 * z.imag),
                (1, 0, -2 * z.imag),
            ):
                if abs(coefficient) < 1e-14:
                    continue
                for (wa, a), (wb, b) in product(transitions[0][ai][1], transitions[1][bi][1]):
                    weights[a, b] += coefficient * wa * wb
    return tuple((c, a, b) for (a, b), c in sorted(weights.items()) if abs(c) > 1e-14)


def _signature_steps(key):
    if key[0] == "u":
        label = key[1]
        return (LocalStep(UnitaryGate(Pauli(label).to_matrix()), tuple(range(len(label)))),)
    if key[0] == "r":
        return pauli_instrument(key[1]) + _signature_steps(("u", key[2]))
    _, left, right, real, imag = key
    matrix = (Pauli(left).to_matrix() + complex(real, imag) * Pauli(right).to_matrix()) / np.sqrt(2)
    return (LocalStep(UnitaryGate(matrix), tuple(range(len(left)))),)


def joint_pauli_basis(terms, posts, support, sides, target):
    """Construct the compressed physical QPD for arbitrary coefficient phases.

    Duplicate LCU words and equivalent signed maps are collected. The returned
    norm can be strictly below the raw bound sum|c|^2 + 4 sum(|Re z|+|Im z|).
    """
    terms = coalesce_paulis(terms)
    local_posts = []
    for side in sides:
        qc = QuantumCircuit(len(side))
        for qubits, matrix in posts:
            if all(q in side for q in qubits):
                qc.unitary(matrix, [side.index(q) for q in qubits])
            elif set(qubits).intersection(side):
                raise ValueError("A purported local group post-unitary crosses its parties.")
        local_posts.append(
            tuple(
                LocalStep(i.operation, tuple(qc.find_bit(q).index for q in i.qubits))
                for i in qc.data
            )
        )
    weights, maps = [], []

    for weight, a, b in joint_pauli_recipes(terms, support, sides):
        weights.append(weight)
        maps.append((_signature_steps(a) + local_posts[0], _signature_steps(b) + local_posts[1]))

    # Check the complete composite unitary, including local post factors.
    reconstructed = sum(c * Pauli(p).to_matrix() for c, p in terms)
    post_circuit = QuantumCircuit(len(support))
    for qubits, matrix in posts:
        post_circuit.unitary(matrix, [support.index(q) for q in qubits])
    reconstructed = Operator(post_circuit).data @ reconstructed
    overlap = np.vdot(target, reconstructed)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    residual = float(np.max(np.abs(reconstructed - phase * target)))
    if residual > 1e-10:
        raise ValueError(f"Group LCU does not match its source unitary: {residual}")
    return JointBasis(tuple(weights), tuple(maps), float(sum(abs(w) for w in weights)), residual)


def general_joint_basis(coefficients, left, right, target=None, *, cost_only=False):
    """Executable upper bound for known local interleavings, without POVM ancillas.

    Expand U_i^dagger U_j in physical Pauli transitions. This may cost more than
    the abstract optimal-instrument bound, so only actual norm improvements are
    admitted into the LUT. An operator-Schmidt lower bound is never substituted.
    """
    local_bases = []
    for dimension in (left[0].shape[0], right[0].shape[0]):
        n = dimension.bit_length() - 1
        local_bases.append(
            tuple(("".join(p), Pauli("".join(p)).to_matrix()) for p in product("IXYZ", repeat=n))
        )
    weights, maps = [], []
    gamma = float(sum(abs(c) ** 2 for c in coefficients))

    def local_map(unitary, label, imaginary):
        n = len(label)
        after = (LocalStep(UnitaryGate(unitary), tuple(range(n))),)
        if not imaginary:
            return ((1.0, pauli_instrument(label) + after),)
        h = Pauli(label).to_matrix()
        return tuple(
            (
                sign / 2,
                (
                    LocalStep(
                        UnitaryGate((np.eye(2**n) + sign * 1j * h) / np.sqrt(2)), tuple(range(n))
                    ),
                )
                + after,
            )
            for sign in (1, -1)
        )

    if not cost_only:
        for c, a, b in zip(coefficients, left, right, strict=True):
            if abs(c) ** 2 > 1e-14:
                weights.append(float(abs(c) ** 2))
                maps.append(
                    tuple(
                        (LocalStep(UnitaryGate(u), tuple(range(u.shape[0].bit_length() - 1))),)
                        for u in (a, b)
                    )
                )
    for i in range(len(coefficients)):
        for j in range(i + 1, len(coefficients)):
            expansions = []
            for k, unitaries in enumerate((left, right)):
                relative = unitaries[i].conj().T @ unitaries[j]
                expansion = [
                    (np.trace(p @ relative) / len(relative), label) for label, p in local_bases[k]
                ]
                expansions.append([(c, label) for c, label in expansion if abs(c) > 1e-13])
            for (ca, a), (cb, b) in product(*expansions):
                z = coefficients[i] * coefficients[j].conjugate() * ca.conjugate() * cb.conjugate()
                gamma += 4 * (abs(z.real) + abs(z.imag))
                if cost_only:
                    continue
                for ai, bi, coefficient in (
                    (False, False, 2 * z.real),
                    (True, True, -2 * z.real),
                    (False, True, -2 * z.imag),
                    (True, False, -2 * z.imag),
                ):
                    if abs(coefficient) < 1e-14:
                        continue
                    for (wa, ma), (wb, mb) in product(
                        local_map(left[i], a, ai), local_map(right[i], b, bi)
                    ):
                        weights.append(float(coefficient * wa * wb))
                        maps.append((ma, mb))
    if cost_only:
        return gamma
    reconstructed = sum(
        c * np.kron(b, a) for c, a, b in zip(coefficients, left, right, strict=True)
    )
    phase = np.vdot(target, reconstructed)
    phase = phase / abs(phase) if abs(phase) else 1
    residual = float(np.max(np.abs(reconstructed - phase * target)))
    if residual > 1e-10:
        raise ValueError("General packet LCU differs from its original source block.")
    actual_gamma = float(sum(abs(w) for w in weights))
    if not np.isclose(actual_gamma, gamma, atol=1e-10, rtol=1e-10):
        raise ValueError("General packet physical coefficients do not match its cost LUT.")
    return JointBasis(tuple(weights), tuple(maps), actual_gamma, residual)
