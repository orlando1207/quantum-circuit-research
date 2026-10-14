"""Verify file-backed circuits and save snapshots, physical experiments, and results."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import sin, sqrt
from pathlib import Path

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp

from qpartition import (
    CutPlan,
    WireCut,
    bootstrap_uncertainty,
    compile_cut,
    compile_excision,
    load_problem,
    save_problem,
)
from qpartition.circuits import load_qasm2

EXACT_TOLERANCE = 1e-10
STATISTICAL_SCREEN_SIGMAS = 6
CIRCUIT_DIRECTORY = Path(__file__).resolve().parent / "circuits"


@dataclass(frozen=True)
class VerificationCase:
    name: str
    filename: str
    observable: tuple[tuple[str, float], ...]
    analytic_expectation: float
    gamma: float
    gate_indices: tuple[int, ...] = ()
    wire_locations: tuple[tuple[int, int], ...] = ()  # Qubit, original instruction index.
    excision_indices: tuple[int, ...] = ()
    max_fragment_qubits: int | None = None


def verification_cases() -> tuple[VerificationCase, ...]:
    ghz_query = (("XXX", 0.3), ("ZIZ", 0.2), ("YXY", -0.1))
    return (
        VerificationCase(
            "bell_xx_gate",
            "bell.qasm",
            (("XX", 1),),
            1,
            3,
            gate_indices=(1,),
            max_fragment_qubits=1,
        ),
        VerificationCase(
            "bell_yy_gate",
            "bell.qasm",
            (("YY", 1),),
            -1,
            3,
            gate_indices=(1,),
            max_fragment_qubits=1,
        ),
        VerificationCase(
            "bell_iz_gate",
            "bell.qasm",
            (("IZ", 1),),
            0,
            3,
            gate_indices=(1,),
            max_fragment_qubits=1,
        ),
        VerificationCase(
            "ghz_wire",
            "ghz.qasm",
            ghz_query,
            0.6,
            4,
            wire_locations=((1, 2),),
            max_fragment_qubits=2,
        ),
        VerificationCase(
            "ghz_mixed",
            "ghz.qasm",
            ghz_query,
            0.6,
            12,
            gate_indices=(1,),
            wire_locations=((1, 2),),
            max_fragment_qubits=2,
        ),
        VerificationCase(
            "rzz_gate",
            "rzz.qasm",
            (("ZY", 1),),
            sin(0.3),
            1 + 2 * sin(0.3),
            gate_indices=(2,),
            max_fragment_qubits=1,
        ),
        VerificationCase(
            "bell_t_excision",
            "bell_t.qasm",
            (("XY", 1),),
            1 / sqrt(2),
            2 + 2 * sqrt(2),
            excision_indices=(2,),
        ),
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _check_exact(actual: float, expected: float, description: str) -> None:
    if abs(actual - expected) > EXACT_TOLERANCE:
        raise AssertionError(f"{description}: {actual} != {expected}")


def _shot_diagnostic(estimate: float, expected: float, standard_error: float) -> dict:
    # Normalize NumPy scalars before producing portable JSON data.
    estimate, expected, standard_error = map(float, (estimate, expected, standard_error))
    error = abs(estimate - expected)
    return {
        "estimate": estimate,
        "absolute_error": error,
        "standard_error": standard_error,
        "error_in_standard_errors": error / standard_error if standard_error > 0 else None,
        "within_statistical_screen": (
            error <= STATISTICAL_SCREEN_SIGMAS * standard_error + EXACT_TOLERANCE
        ),
    }


def verify_case(
    case: VerificationCase,
    output: Path,
    *,
    shots: int = 2000,
    seed: int = 42,
    bootstrap_replicates: int = 200,
) -> dict:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 2:
        raise ValueError("Verification needs at least two shots per executed circuit.")
    directory = output / case.name
    directory.mkdir(parents=True, exist_ok=True)
    source = CIRCUIT_DIRECTORY / case.filename
    imported = load_qasm2(source, SparsePauliOp.from_list(case.observable))
    _check_exact(imported.exact_expectation(), case.analytic_expectation, "QASM import")
    save_problem(imported, directory)
    problem = load_problem(directory)
    reference = problem.exact_expectation()
    _check_exact(reference, case.analytic_expectation, "QPY/query round trip")
    (directory / "circuit.txt").write_text(str(problem.circuit.draw(output="text")) + "\n")
    _write_json(
        directory / "operations.json",
        [
            {
                "id": op_id,
                "name": item.operation.name,
                "qubits": [problem.circuit.find_bit(q).index for q in item.qubits],
            }
            for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True)
        ],
    )

    if case.excision_indices:
        operation_ids = tuple(problem.operation_ids[i] for i in case.excision_indices)
        _write_json(
            directory / "excision.json",
            {
                "schema_version": 1,
                "circuit_fingerprint": problem.circuit_fingerprint,
                "operation_ids": operation_ids,
            },
        )
        selection = json.loads((directory / "excision.json").read_text())
        if selection["circuit_fingerprint"] != problem.circuit_fingerprint:
            raise AssertionError("Excision selection refers to another circuit.")
        compiled = compile_excision(problem, tuple(selection["operation_ids"]))
        exact = compiled.run_exact()
        sampled = compiled.run_shots(shots=shots, seed=seed)
        diagnostic = _shot_diagnostic(sampled.expectation, reference, sampled.standard_error)
        importance = compiled.run_sampled_terms(samples=sampled.total_shots, seed=seed)
        importance_diagnostic = _shot_diagnostic(
            importance.expectation, reference, importance.standard_error
        )
        experiments = {"excision": [item.circuit for item in compiled.experiments]}
        mapping = {
            "scored_experiments": [
                {"coefficient": item.coefficient, "score_mask": item.score_mask}
                for item in compiled.experiments
            ],
        }
        cost = {
            "gamma": compiled.gamma,
            "sampling_overhead": compiled.gamma**2,
            "algebraic_terms": [item.algebraic_terms for item in compiled.decompositions],
            "physical_terms": len(compiled.experiments),
            "executed_circuits": sampled.executed_circuits,
            "execution_qubits": problem.circuit.num_qubits,
        }
        additional = {
            "sampled_terms": {
                **importance_diagnostic,
                "total_shots": importance.total_shots,
                "executed_circuits": importance.executed_circuits,
            },
        }
    else:
        plan = CutPlan.create(
            problem,
            gate_cuts=tuple(problem.operation_ids[i] for i in case.gate_indices),
            wire_cuts=tuple(
                WireCut(qubit, problem.operation_ids[index]) for qubit, index in case.wire_locations
            ),
            max_fragment_qubits=case.max_fragment_qubits,
        )
        plan.save(directory / "plan.json")
        restored_plan = CutPlan.load(directory / "plan.json")
        restored_plan.validate(problem)
        compiled = compile_cut(problem, restored_plan)
        exact = compiled.run_exact()
        sampled = compiled.run_shots(shots=shots, seed=seed)
        uncertainty = bootstrap_uncertainty(
            compiled, sampled, replicates=bootstrap_replicates, seed=seed
        )
        diagnostic = {
            **_shot_diagnostic(sampled.expectation, reference, uncertainty.standard_error),
            "bootstrap": asdict(uncertainty),
        }
        experiments = compiled.unique_experiments
        mapping = {
            "experiment_indices": compiled.experiment_indices,
            "coefficients": [
                {"coefficient": float(coefficient), "weight_type": weight_type.name}
                for coefficient, weight_type in compiled.coefficients
            ],
            "fragment_observables": {
                label: values.to_labels() for label, values in compiled.subobservables.items()
            },
        }
        cost = {
            **asdict(compiled.cost),
            "bounded_estimator_factor": compiled.cost.bounded_estimator_factor,
        }
        additional = {}

    _check_exact(exact.expectation, reference, "Reconstruction vs uncut simulation")
    _check_exact(exact.expectation, case.analytic_expectation, "Reconstruction vs analytic value")
    _check_exact(cost["gamma"], case.gamma, "Physical decomposition norm")
    experiment_directory = directory / "experiments"
    experiment_directory.mkdir(exist_ok=True)
    for label, circuits in experiments.items():
        with (experiment_directory / f"{label}.qpy").open("wb") as stream:
            qpy.dump(circuits, stream)
    _write_json(directory / "experiment_mapping.json", mapping)
    result = {
        "case": case.name,
        "source_qasm": case.filename,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "circuit_fingerprint": problem.circuit_fingerprint,
        "query_fingerprint": problem.query_fingerprint,
        "observable": [
            (label, float(weight.real)) for label, weight in problem.observable.to_list()
        ],
        "analytic_expectation": case.analytic_expectation,
        "uncut_expectation": reference,
        "exact_reconstruction": exact.expectation,
        "exact_error": abs(exact.expectation - case.analytic_expectation),
        "shot_result": {**diagnostic, "total_shots": sampled.total_shots},
        "cost": cost,
        "exact_checks_passed": True,
        **additional,
    }
    _write_json(directory / "result.json", result)
    return result


def run_verification(output: Path, *, shots: int = 2000, seed: int = 42) -> dict:
    records = []
    for index, case in enumerate(verification_cases()):
        record = verify_case(case, output, shots=shots, seed=seed + 1000 * index)
        records.append(record)
        print(
            f"{case.name:18} analytic={record['analytic_expectation']: .6f} "
            f"exact_error={record['exact_error']:.2e} "
            f"sampled={record['shot_result']['estimate']: .6f} "
            f"SE={record['shot_result']['standard_error']:.6f} "
            f"shots={record['shot_result']['total_shots']}"
        )
    report = {
        "schema_version": 1,
        "seed": seed,
        "case_seed_stride": 1000,
        "shots_per_enumerated_circuit": shots,
        "exact_tolerance": EXACT_TOLERANCE,
        "statistical_screen_sigmas": STATISTICAL_SCREEN_SIGMAS,
        "versions": {
            name: version(name)
            for name in (
                "quantum-partition-research",
                "qiskit",
                "qiskit-aer",
                "qiskit-addon-cutting",
            )
        },
        "all_exact_checks_passed": all(record["exact_checks_passed"] for record in records),
        "all_statistical_screens_passed": all(
            record["shot_result"]["within_statistical_screen"]
            and record.get("sampled_terms", {}).get("within_statistical_screen", True)
            for record in records
        ),
        "results": records,
    }
    _write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/simple_verification"))
    args = parser.parse_args()
    report = run_verification(args.output, shots=args.shots, seed=args.seed)
    print(f"Saved snapshots, physical circuits, and report to {args.output}")
    if not report["all_statistical_screens_passed"]:
        raise SystemExit("A finite-shot statistical screen failed; inspect report.json.")
