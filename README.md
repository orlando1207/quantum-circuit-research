# Quantum circuit partitioning research

Partition a fixed quantum circuit using wire and gate cuts, execute smaller
fragment families, and reconstruct observables classically. The optimizer now
supports multiway partitions, multi-start simulated annealing, temporal region
moves, and capacity-aware graph initialization. Grouping and excision are future
refinement techniques; manual excision remains available.

The maintained specification and complete benchmark comparison are in
[Workflow and algorithm](docs/WORKFLOW_AND_ALGORITHM.md). It replaces six
superseded implementation/result notes. Original research plans and literature
surveys remain background material.

The [external-baseline reproduction](docs/BASELINE_REPRODUCTION.md) documents
the directly usable official Qiskit optimizer, an explicitly labeled
reconstruction of the paper's two-stage KL method, grouped-cost checks, and
the limits preventing an exact authors' reproduction. Its
[recorded results](validation/external_baselines_v1/SUMMARY.md) compare the
same frozen inputs under the paper's width limits and tighter capacities.

## Environment and checks

```bash
conda activate quantum-partition
python -m pytest -q
ruff check src tests examples
```

The project uses an isolated Python 3.12 conda environment. To reproduce it:

```bash
conda env create -f environment.yml
conda activate quantum-partition
pip install -e . --no-deps
```

`requirements.lock.txt` pins Python packages; `conda-linux-64.lock.txt` records
the tested platform. No IBM account or quantum hardware is required.

## Prepare a circuit

Inputs contain bound one-/two-qubit unitary gates and start in the all-zero
state; put other preparations in the circuit. Decompose wider gates first.
Original measurements, resets, classical registers, and adaptive control are
outside the optimizer input format. Queries are Hermitian `SparsePauliOp` sums;
Qiskit's rightmost Pauli character acts on qubit 0.

Use `save_problem(problem, directory)` and `load_problem(directory)` for the
canonical `circuit.qpy` + `problem.json` snapshot. Saved plans reference this
snapshot and reject incompatible circuits or template libraries. OpenQASM is
an editable exchange format. The [circuit guide](examples/circuits/README.md)
explains import and verification with Bell, GHZ, RZZ, and T examples.

## Optimize and execute

```python
from qiskit import QuantumCircuit
from qpartition import (
    AnnealConfig, CircuitProblem, HybridPlan, PartitionGraph,
    compile_hybrid, multi_start,
)

circuit = QuantumCircuit(4)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(2, 3)
graph = PartitionGraph(CircuitProblem.create(circuit, "XXXX"))
config = AnnealConfig(max_qubits=2, partition_mode="multiway", steps=3000)
runs = multi_start(graph, config, seeds=tuple(range(10)), workers=1)
best = min((r.best for r in runs if r.best), key=lambda e: e.log_overhead)
plan = HybridPlan.create(
    graph, best.state, max_qubits=config.max_qubits,
    partition_mode=config.partition_mode,
)
compiled = compile_hybrid(graph, plan)
print(best.block_widths, best.action_counts, best.sampling_overhead)
print(compiled.run_exact().expectation)
print(compiled.run_sampled(trials=1000, seed=42))
```

Use a main guard in scripts with `workers > 1`. Save plans with `plan.save`.
Exact reconstruction is a small-instance correctness oracle with explicit
branch and qubit guards; larger plans use lazy sampled execution.

`multiway` executes each connected component separately. Every fragment must
fit `max_qubits`; each wire cut adds one static segment. `max_components` can
limit connected components. Disconnected components are not bundled in this
mode. `objective="overhead"` minimizes the independent QPD overhead;
`objective="fragment_shots"` minimizes its product with the executed fragment
count for the independent-trial model.

`partition_mode="two_blocks"` retains the previous exact two-bin packing
behavior and remains the constructor default for compatibility. Select
`multiway` explicitly for the new workflow. `components` is its compatibility
alias. Automatic depth constraints, register reuse, noise optimization, and
runtime communication are not implemented.

## Benchmarks and artifacts

```bash
python examples/run_multiway_benchmarks.py --workers 4 --output results/multiway-rerun
python examples/check_sa_artifacts.py --directory results/multiway-rerun
python examples/check_sa_artifacts.py --directory results/multiway-rerun/capacity_sweep
```

The runner preserves previous results and requires a fresh output directory.
It uses the frozen source circuits in `validation/sa_benchmarks_v2`, five tuning
profiles, 50 fresh final starts per main case, gate-only/wire-only ablations,
and an additional capacity sweep. Main runs explicitly warm-start from the
historical two-way winners; the tighter-capacity comparison uses fresh pools
and matched tuning/trajectory budgets for both modes.

Current results are in
[`validation/multiway_benchmarks_v3`](validation/multiway_benchmarks_v3), including
plans, QPY circuits, raw trajectories, physical compilation checks, and independent
Qiskit width/norm checks. The [maintained document](docs/WORKFLOW_AND_ALGORITHM.md)
lists all eight original Table-I baselines, all implemented benchmark results,
missing-reproduction cases, and the capacity sweep. Matching counts or rounded
costs does not establish identity with the paper's unpinned circuit artifacts.

The earlier benchmark PNGs remain in
[`validation/sa_benchmarks_v2/partition_png`](validation/sa_benchmarks_v2/partition_png).
Regenerable template dumps and local pilot experiments are ignored by Git.

To rerun the external comparison in the same conda environment:

```bash
python examples/run_external_baselines.py --output results/external-rerun --starts 50 --workers 4 --sa-steps 2000
python examples/summarize_external_baselines.py --directory results/external-rerun
python examples/check_qiskit_search_budget.py --directory results/external-rerun
```

The SDK is called directly; it is not replaced with our optimizer. The paper
reconstruction records its missing-detail assumptions, and grouped diagnostics
use explicit manifests. A new source-derived historical Grover candidate has
24 CX + 14 CP gates; the old 38-CX control remains a separate circuit.

## Foundation APIs

Manual standard cuts use `CutPlan`, `WireCut`, and `compile_cut`; queries can be
reconstructed exactly or with Aer shots and `bootstrap_uncertainty`. Manual
`compile_excision` supports bounded small gates using validated physical
measure/prepare expansions. Automatic core search accepts only gate/wire cuts.

Current plans use `temporal-incidence-v2`. The explicit legacy graph/compiler
and `migrate_legacy_plan` handle compatible historical plans; core SA does not
silently accept old excision states.

```bash
python examples/foundation_demo.py --shots 2000 --seed 42 --output results/foundation.json
python examples/verify_simple_circuits.py --shots 2000 --seed 42 --output results/simple_verification
```

The [foundation validation](validation/README.md) and circuit guide describe the
saved exact/shot verification cases. Full-distribution reconstruction, gate-group
discovery, and IBM hardware adapters remain later milestones.
