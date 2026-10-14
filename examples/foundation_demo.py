"""Run reproducible gate, wire, mixed-cut, and gate-excision validation cases."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qpartition import (
    CircuitProblem,
    CutPlan,
    WireCut,
    bootstrap_uncertainty,
    compile_cut,
    compile_excision,
)


def run_demo(shots: int = 2000, seed: int = 42) -> dict:
    records = []
    gate = QuantumCircuit(2)
    gate.h(0)
    gate.ry(0.3, 1)
    gate.cx(0, 1)
    gate.rz(0.2, 0)
    gate_problem = CircuitProblem.create(gate, SparsePauliOp.from_list([("XX", 0.7), ("ZZ", 0.3)]))
    chain = QuantumCircuit(3)
    chain.h(0)
    chain.cx(0, 1)
    chain.ry(0.41, 1)
    chain.cx(1, 2)
    chain.rz(0.27, 2)
    chain_problem = CircuitProblem.create(
        chain, SparsePauliOp.from_list([("XXX", 0.4), ("ZIZ", 0.5), ("YXY", -0.2)])
    )
    cases = [
        (
            "gate",
            gate_problem,
            CutPlan.create(
                gate_problem, gate_cuts=(gate_problem.operation_ids[2],), max_fragment_qubits=1
            ),
        ),
        (
            "wire",
            chain_problem,
            CutPlan.create(
                chain_problem,
                wire_cuts=(WireCut(1, chain_problem.operation_ids[3]),),
                max_fragment_qubits=2,
            ),
        ),
        (
            "mixed",
            chain_problem,
            CutPlan.create(
                chain_problem,
                gate_cuts=(chain_problem.operation_ids[1],),
                wire_cuts=(WireCut(1, chain_problem.operation_ids[3]),),
                max_fragment_qubits=2,
            ),
        ),
    ]
    for name, problem, plan in cases:
        compiled = compile_cut(problem, plan)
        exact = compiled.run_exact()
        sampled = compiled.run_shots(shots=shots, seed=seed)
        uncertainty = bootstrap_uncertainty(compiled, sampled, replicates=200, seed=seed)
        reference = problem.exact_expectation()
        if abs(exact.expectation - reference) > 1e-10:
            raise RuntimeError(f"Exact reconstruction failed for {name}.")
        records.append(
            {
                "case": name,
                "reference": reference,
                "exact_reconstruction": exact.expectation,
                "exact_error": abs(exact.expectation - reference),
                "shot_estimate": sampled.expectation,
                "shot_error": abs(sampled.expectation - reference),
                "uncertainty": asdict(uncertainty),
                "total_shots": sampled.total_shots,
                "execution_seconds": sum(x.elapsed_seconds for x in sampled.execution.values()),
                "cost": {
                    **asdict(compiled.cost),
                    "bounded_estimator_factor": compiled.cost.bounded_estimator_factor,
                },
            }
        )
    excision = QuantumCircuit(2)
    excision.h(0)
    excision.cx(0, 1)
    excision.t(0)
    excision.ry(0.22, 1)
    problem = CircuitProblem.create(excision, "XY")
    compiled = compile_excision(problem, (problem.operation_ids[2],))
    exact = compiled.run_exact()
    enumerated = compiled.run_shots(shots=shots, seed=seed)
    sampled = compiled.run_sampled_terms(samples=shots * len(compiled.experiments), seed=seed)
    reference = problem.exact_expectation()
    if abs(exact.expectation - reference) > 1e-10:
        raise RuntimeError("Exact excision reconstruction failed.")
    records.append(
        {
            "case": "T-excision",
            "reference": reference,
            "exact_reconstruction": exact.expectation,
            "exact_error": abs(exact.expectation - reference),
            "algebraic_terms": compiled.decompositions[0].algebraic_terms,
            "physical_terms": len(compiled.experiments),
            "discarded_ptm_norm": compiled.decompositions[0].discarded_ptm_norm,
            "gamma": compiled.gamma,
            "sampling_overhead": compiled.gamma**2,
            "enumerated": asdict(enumerated),
            "sampled_terms": asdict(sampled),
        }
    )
    return {
        "schema_version": 1,
        "seed": seed,
        "shots_per_enumerated_unique_circuit": shots,
        "versions": {
            name: version(name)
            for name in [
                "quantum-partition-research",
                "qiskit",
                "qiskit-aer",
                "qiskit-addon-cutting",
                "numpy",
                "scipy",
                "networkx",
            ]
        },
        "results": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_demo(args.shots, args.seed)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
