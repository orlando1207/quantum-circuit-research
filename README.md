# Quantum circuit partitioning research

Partition fixed quantum circuits using wire cutting, gate cutting, and
individual two-qubit gate excision. The optimizer keeps the contracted
temporal incidence graph and label-based multiway simulated annealing, with
structural seeding, temporal region moves, Hastings correction, multistart,
and deterministic polishing. The current research and experiments disable
gate grouping.

The maintained specification is
[Workflow and algorithm](docs/WORKFLOW_AND_ALGORITHM.md). It explains physical
sampling costs, excision realization patterns, resource accounting, accepted
moves, and the experimental protocol. The
[large-circuit study](validation/excision_quality_v1/SUMMARY.md) compares
50–250-qubit cases against official Qiskit and an explicit reconstruction of
arXiv:2608.05287 without GG, with a separate hyperparameter sweep.

## Environment

```bash
conda activate quantum-partition
python -m pytest -q
ruff check src tests examples
```

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate quantum-partition
pip install -e . --no-deps
```

The project uses Python 3.12 in the isolated `quantum-partition` conda
environment. Package/platform pins are in `requirements.lock.txt` and
`conda-linux-64.lock.txt`. No IBM account or quantum device is needed for the
current partition-quality experiments.

## Circuit inputs

Use bound one-/two-qubit unitary gates starting from the all-zero state; put
other preparations in the circuit. Decompose wider gates first. Original
measurements, resets, classical registers, and adaptive control are outside
the optimizer input contract. Observables are Hermitian `SparsePauliOp` sums;
Qiskit's rightmost Pauli character acts on qubit zero.

`save_problem(problem, directory)` writes the canonical `circuit.qpy` and
`problem.json` snapshot. `load_problem(directory)` restores and validates it.
OpenQASM is an editable interchange format. See the
[circuit guide](examples/circuits/README.md) for small verification inputs.

## Optimize and reconstruct

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
problem = CircuitProblem.create(circuit, "XXXX")
graph = PartitionGraph(problem, excision=True)
config = AnnealConfig(
    max_qubits=2, partition_mode="multiway", steps=2400,
    max_evaluations=4000, pilot_steps=48,
    temperature_calibration="feasible", cooling_rate=0.98,
)
runs = multi_start(graph, config, seeds=tuple(range(5)), workers=1)
best = min((r.best for r in runs if r.best), key=lambda e: e.log_overhead)
plan = HybridPlan.create(graph, best.state, max_qubits=2, partition_mode="multiway")
compiled = compile_hybrid(graph, plan)
print(best.block_widths, best.action_counts, best.log_overhead)
print(compiled.run_exact().expectation)
print(compiled.run_sampled(trials=1000, seed=42))
```

`PartitionGraph(problem)` supplies the wire/gate ablation. Adding
`excision=True` enables physical measure-and-prepare alternatives for certain
single-gate/adjacent-wire cut patterns, with no new graph or search encoding.
For CX the available excision recipe has norm 16 and overhead 256; other gate
recipes can cost more. `excision_gamma` records an explicit cost assumption,
and a mismatching recipe cannot be compiled at that assumed price. The
quality study uses actual recipe norms throughout.

Each retained-wire segment needs one static register. In `multiway` mode each
connected component executes separately and must fit `max_qubits`. Segment
conservation is N plus the number of structural wire cuts, including those
absorbed by excision. `two_blocks` remains the compatibility default; select
`multiway` explicitly. Automatic depth constraints, scheduling of register
reuse, and hardware noise optimization are future work.

Large reconstruction uses lazy sampled branches. Exact reconstruction is a
small-instance correctness oracle with explicit branch/qubit guards. Put
multiprocess calls under a main guard when `workers > 1`.

## Quality experiments

```bash
python examples/run_excision_quality.py --output results/excision-quality-rerun --workers 6
python examples/check_excision_quality.py --directory results/excision-quality-rerun
```

This runs ten profiles on separate 64-qubit tuning cases, then five matched
starts with and without excision on 22 held-out cases. It includes graph
states, explicitly approximate QFT, fixed-angle Hamiltonian simulation,
random brickwork, and labeled synthetic controls. The original generic seed
pool is shared between methods; no known separators or baseline winners are
supplied. Baselines request 50 restarts with explicit per-case deadlines;
completed counts and failures remain visible.

The runner supports `--phase prepare`, `tune`, `final`, `baselines`,
`confirmation`, and `summarize` for staged execution. Use the same configuration to resume, or a
fresh directory for changed settings. Results retain source identities,
per-start diagnostics, widths, recipe audits, budgets, and timing. Frozen
inputs and regenerable plan dumps stay local and are ignored by Git.

The earlier [integration checks](validation/incidence_excision_v3/SUMMARY.md)
include known-witness 100/1000-qubit controls. Those validate the excision
integration; they are distinct from the new optimizer-quality experiment.
Historical benchmark artifacts remain available, but the linked workflow and
large-circuit study describe the current active methodology.

## Foundation APIs

Manual standard cuts use `CutPlan`, `WireCut`, and `compile_cut`. Manual
`compile_excision` supports small gates through physical measure/prepare
expansions. `HybridPlan` and `compile_hybrid` handle the automatic core's
wire/gate/excision plans and validate source/recipe fingerprints.

```bash
python examples/foundation_demo.py --shots 2000 --seed 42 --output results/foundation.json
python examples/verify_simple_circuits.py --shots 2000 --seed 42 --output results/simple_verification
```

The [foundation validation](validation/README.md) records saved small exact
and shot-based checks. Full-distribution reconstruction and IBM hardware
adapters remain later milestones.
