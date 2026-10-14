import pytest
from qiskit import QuantumCircuit
from qiskit_addon_cutting.utils.simulation import ExactSampler
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

from qpartition.execution import exact_distribution, execute_shots


def test_mid_measurement_reset_and_register_order_against_independent_sampler():
    circuit = QuantumCircuit(2, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 2)
    circuit.reset(0)
    circuit.h(1)
    circuit.measure(1, 0)
    circuit.measure(0, 1)
    actual = exact_distribution(circuit)
    oracle = ExactSampler().run([circuit]).result().quasi_dists[0]
    assert actual == pytest.approx(dict(oracle), abs=1e-12)
    assert actual == pytest.approx({0: 0.25, 1: 0.25, 4: 0.25, 5: 0.25})


def test_reset_is_trace_preserving_on_entangled_input():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.measure([0, 1], [0, 1])
    assert exact_distribution(circuit) == pytest.approx({0: 0.5, 2: 0.5})


def test_measurement_overwrite_retains_only_latest_bit():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.reset(0)
    circuit.x(0)
    circuit.measure(0, 0)
    assert exact_distribution(circuit) == pytest.approx({1: 1})


def test_execution_limits_and_invalid_shots():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure_all()
    with pytest.raises(ValueError, match="qubit limit"):
        exact_distribution(circuit, max_qubits=1)
    with pytest.raises(ValueError, match="branch limit"):
        exact_distribution(circuit, max_branches=1)
    with pytest.raises(ValueError, match="positive integer"):
        execute_shots([circuit], shots=0)


def test_shot_backend_applies_requested_noise_model():
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(amplitude_damping_error(0.25), ["x"])
    result = execute_shots([circuit], shots=4000, seed=17, noise_model=noise)
    assert result.total_shots == 4000
    assert result.distributions[0][1] == pytest.approx(0.75, abs=0.04)
