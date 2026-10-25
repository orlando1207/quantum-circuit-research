"""Independently recompile study winners and materialize no-GG KL cuts in Qiskit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from qpartition import HybridPlan, PartitionGraph, compile_hybrid, load_problem
from qpartition.large_benchmarks import large_benchmark
from qpartition.paper_baseline import PaperTopology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/excision_quality_v1"))
    args = parser.parse_args()
    root = args.directory
    metadata = json.loads((root / "metadata.json").read_text())
    report = json.loads((root / "report.json").read_text())
    selected = json.loads((root / "tuning.json").read_text())["selected"]
    checks = []
    for case in report:
        name = case["name"]
        directory = root / "cases" / name
        provenance = case["provenance"]
        regenerated = large_benchmark(
            case["family"],
            case["qubits"],
            seed=provenance["seed"],
            layers=provenance.get("layers") or 1,
        )
        source = (
            load_problem(directory / "input")
            if (directory / "input").exists()
            else regenerated.problem
        )
        assert source.circuit_fingerprint == case["source_fingerprint"]
        assert regenerated.problem.circuit_fingerprint == source.circuit_fingerprint
        row = dict(case=name, input_regeneration=True, methods={})
        graphs = {}
        for method in ("wg", "wge"):
            result = case["methods"][method]
            assert result["starts"] == metadata["config"]["starts"]
            assert result["feasible"] == result["starts"]
            winner = result["best"]
            graph = PartitionGraph(source, excision=method == "wge")
            graphs[method] = graph
            if method == "wge":
                assert graph.grouping.config.families == ("excision",)
            target = root / "winners" / f"{name}_{method}.json"
            compiled = compile_hybrid(graph, HybridPlan.load(target))
            norm = 2 * sum(math.log(t.gamma) for t in compiled.templates)
            assert math.isclose(norm, winner["log_sampling_overhead"], abs_tol=1e-8)
            experiment = compiled.experiment((0,) * len(compiled.templates))
            assert [c.num_qubits for c in experiment.circuits] == winner["widths"]
            assert max(winner["widths"]) <= case["max_qubits"]
            assert sum(winner["widths"]) == case["qubits"] + winner["raw_wire_cuts"]
            row["methods"][method] = dict(physical_norm=True, physical_widths=True)
        baseline = case["methods"]["paper"].get("best")
        if baseline:
            topology = PaperTopology(source, groups=())
            _, materialized = topology.materialize(baseline["labels"])
            assert math.isclose(
                materialized["sdk_individual_log_sampling_overhead"],
                baseline["log_sampling_overhead"],
                abs_tol=1e-8,
            )
            assert max(materialized["sdk_component_widths"]) <= case["max_qubits"]
            assert sum(materialized["sdk_component_widths"]) == (
                case["qubits"] + len(baseline["wire_cuts"])
            )
            row["methods"]["paper"] = dict(sdk_norm=True, sdk_widths=True)
        for method in ("wg", "wge"):
            for path in (directory / "final").glob(f"{method}_{selected}_*.json"):
                if path.name.endswith(".plan.json"):
                    continue
                run = json.loads(path.read_text())
                assert run["physical_recipe_and_width_audit"]
                assert run["evaluations"] <= run["config"]["max_evaluations"]
                assert run["seed_pool_index"] == (run["seed"] - 1000) % case["pool_size"]
                pool = json.loads((directory / "pool.json").read_text())
                initial = graphs[method].evaluate(
                    tuple(pool[run["seed_pool_index"]]),
                    max_qubits=case["max_qubits"],
                    partition_mode="multiway",
                )
                assert math.isclose(initial.log_overhead, run["initial_log_overhead"], abs_tol=1e-8)
        checks.append(row)
        print("Audited", name, flush=True)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *Path("src/qpartition").glob("*.py"),
            Path("examples/run_excision_quality.py"),
            Path(__file__),
        ]
    }
    # The original preparation manifest is retained. Record later orchestration
    # and overflow-safe audit changes explicitly rather than overwrite its hashes.
    original_hashes = {
        (f"examples/{Path(p).name}" if Path(p).is_absolute() else p): old
        for p, old in metadata["source_hashes"].items()
    }
    changed = [p for p, old in original_hashes.items() if hashes[p] != old]
    assert set(changed) <= {
        "src/qpartition/paper_baseline.py",
        "examples/run_excision_quality.py",
    }
    for kernel in ("annealing.py", "label_moves.py", "multiway.py", "group_heatbath.py"):
        assert f"src/qpartition/{kernel}" not in changed
    output = dict(
        cases=len(checks),
        checks=checks,
        audited_source_hashes=hashes,
        changed_since_preparation=changed,
        source_change_notes="The runner gained an output-directory fix, a shorter table heading, "
        "and cancellation of queued tasks on failure before successful final runs. "
        "Paper materialization gained overflow-safe log norms; baseline search is unchanged. "
        "Any later report/confirmation orchestration edits are also reflected in these hashes.",
    )
    (root / "audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
