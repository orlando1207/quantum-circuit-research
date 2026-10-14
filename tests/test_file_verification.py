"""Exercise the file-to-experiment workflow, including exported result mappings."""

import json

import pytest
from qiskit import qpy

from examples.verify_simple_circuits import verification_cases, verify_case
from qpartition import CutPlan, compile_cut, load_problem
from qpartition.execution import execute_exact


@pytest.mark.parametrize("case", verification_cases(), ids=lambda case: case.name)
def test_qasm_snapshot_and_saved_experiments_reconstruct(case, tmp_path):
    record = verify_case(case, tmp_path, shots=300, seed=29, bootstrap_replicates=20)
    assert record["exact_error"] < 1e-10
    assert record["cost"]["sampling_overhead"] == pytest.approx(case.gamma**2)
    directory = tmp_path / case.name
    saved_result = json.loads((directory / "result.json").read_text())
    assert saved_result["shot_result"]["estimate"] == record["shot_result"]["estimate"]
    problem = load_problem(directory)
    mapping = json.loads((directory / "experiment_mapping.json").read_text())

    if case.excision_indices:
        with (directory / "experiments" / "excision.qpy").open("rb") as stream:
            circuits = qpy.load(stream)
        execution = execute_exact(circuits)
        # Reconstruct exclusively from exported circuits and classical score metadata.
        estimate = 0.0
        for term, probabilities in zip(
            mapping["scored_experiments"], execution.distributions, strict=True
        ):
            mean = sum(
                probability * (-1 if (outcome & term["score_mask"]).bit_count() % 2 else 1)
                for outcome, probability in probabilities.items()
            )
            estimate += term["coefficient"] * mean
        assert estimate == pytest.approx(case.analytic_expectation, abs=1e-10)
        assert record["shot_result"]["total_shots"] == 300 * len(circuits)
        assert record["sampled_terms"]["total_shots"] == record["shot_result"]["total_shots"]
    else:
        plan = CutPlan.load(directory / "plan.json")
        compiled = compile_cut(problem, plan)
        assert mapping["experiment_indices"] == compiled.experiment_indices
        execution = {}
        for label in compiled.unique_experiments:
            with (directory / "experiments" / f"{label}.qpy").open("rb") as stream:
                circuits = qpy.load(stream)
            execution[label] = execute_exact(circuits)
        assert compiled.reconstruct(execution) == pytest.approx(
            case.analytic_expectation, abs=1e-10
        )
        assert record["shot_result"]["total_shots"] == 300 * compiled.cost.unique_experiments
