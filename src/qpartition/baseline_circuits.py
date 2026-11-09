"""Source-pinned benchmark reconstructions for the external-method comparison."""

from __future__ import annotations

import warnings
from math import pi

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator, MCPhaseGate
from qiskit.quantum_info import Operator

from .circuits import CircuitProblem

MQT_GROVER_SOURCE = (
    "https://github.com/munich-quantum-toolkit/bench/blob/"
    "9ac1ecf7b0fad995d8e04d327bb80b792873201e/src/mqt/bench/benchmarks/grover.py"
)
GRAY_CODE_SOURCE = (
    "https://github.com/Qiskit/qiskit/blob/0.45.3/qiskit/circuit/library/standard_gates/p.py"
)


def historical_three_control_phase():
    """Mathematical Gray-code synthesis used by historical Qiskit MCPhaseGate.

    Independently expressed using integer Gray words: update control parity,
    then apply a parity-signed phase pi/4 to the target. Keeping CP primitive
    matters for the circuit partition problem; this is not an all-CX lowering.
    """
    circuit = QuantumCircuit(4, name="historical_mcp_pi_gray")
    previous = 0
    for index in range(1, 8):
        word = index ^ (index >> 1)
        anchor = 3 - word.bit_length()
        if previous:
            changed = 3 - (word ^ previous).bit_length()
            if changed != anchor:
                circuit.cx(changed, anchor)
            else:
                for qubit in range(anchor + 1, 3):
                    if word & (1 << (2 - qubit)):
                        circuit.cx(qubit, anchor)
        sign = 1 if word.bit_count() % 2 else -1
        circuit.cp(sign * pi / 4, anchor, 3)
        previous = word
    return circuit


def grover4_historical():
    """Reconstruct pinned MQT defaults with a checked historical native basis.

    The paper does not identify a generator commit. Matching gate count and
    rounded cut costs is evidence of compatibility, not original-file identity.
    """
    register = QuantumRegister(3, "q")
    flag = AncillaRegister(1, "flag")
    oracle = QuantumCircuit(register, flag)
    oracle.mcp(pi, register, flag)
    # The historical class preserves AncillaRegister reflection semantics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        operator = GroverOperator(oracle, mcx_mode="noancilla")
    source = QuantumCircuit(register, flag)
    source.h(register)
    source.x(flag)
    source.compose(operator.power(2), inplace=True)
    intermediate = source.decompose(reps=2)
    gray = historical_three_control_phase()
    circuit = intermediate.copy_empty_like()
    for item in intermediate.data:
        if item.operation.name == "mcphase":
            circuit.compose(gray, qubits=item.qubits, inplace=True)
        elif item.operation.num_qubits > 2:
            circuit.compose(item.operation.definition, qubits=item.qubits, inplace=True)
        else:
            circuit.append(item.operation, item.qubits)
    circuit.name = "grover4_historical_mqt_native_reconstruction"
    phase_residual = float(np.max(np.abs(Operator(gray).data - Operator(MCPhaseGate(pi, 3)).data)))
    residual = float(np.max(np.abs(Operator(circuit).data - Operator(source).data)))
    if max(residual, phase_residual) > 1e-10:
        raise RuntimeError("Historical decomposition failed its complete-unitary check.")
    if circuit.count_ops().get("cx") != 24 or circuit.count_ops().get("cp") != 14:
        raise RuntimeError("Pinned historical native-gate reconstruction changed.")
    provenance = {
        "kind": "pinned-source-and-native-decomposition-reconstruction",
        "generator_source": MQT_GROVER_SOURCE,
        "phase_decomposition_source": GRAY_CODE_SOURCE,
        "iterations": 2,
        "data_qubits": 3,
        "flag_qubits": 1,
        "native_two_qubit_gates": {"cx": 24, "cp": 14},
        "phase_unitary_residual": phase_residual,
        "complete_unitary_residual": residual,
        "artifact_identity": "Paper's generator commit, exact QPY/QASM, and environment unpinned.",
        "scope": "MQT source defaults; historical Gray-code CP lowering; no outcome fitting.",
    }
    return CircuitProblem.create(circuit, "ZZZZ"), provenance
