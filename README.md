# Quantum circuit partitioning research

This repository provides the first executable foundation for quantum circuit
partitioning: manual wire/gate cuts, gate excision, automatic hybrid partitioning
with multi-start simulated annealing, local execution, classical reconstruction,
and resource accounting. The literature and design rationale are
in [RESEARCH_FOUNDATION.md](RESEARCH_FOUNDATION.md).

## Environment and verification

Development uses the isolated conda environment `quantum-partition`, Python 3.12.
Activate the environment created for this workspace:

```bash
conda activate quantum-partition
python -m pytest
ruff check src tests examples
python examples/foundation_demo.py --shots 2000 --seed 42 --output results/foundation.json
```

To reproduce the environment in another checkout:

```bash
conda env create -f environment.yml
conda activate quantum-partition
pip install -e . --no-deps
```

`requirements.lock.txt` pins the tested Python dependencies. The conda package
specification in `conda-linux-64.lock.txt` records the development platform's
Python and system packages. The project dependency ranges are in `pyproject.toml`.
No IBM account or quantum-device access is required for these examples.

The current suite passes 86 tests. The original saved
[validation report](validation/README.md) includes four demonstration cases whose
exact reconstruction errors are below 2e-15, with finite-shot estimates and
resource counts in JSON.

For editable Bell, GHZ, RZZ, and T-gate examples, see the
[circuit preparation and verification guide](examples/circuits/README.md).
It explains supported input formats and the saved experiment files. Run its
file-to-experiment workflow with:

```bash
python examples/verify_simple_circuits.py --shots 2000 --seed 42 --output results/simple_verification
```

## Circuit and query conventions

Input circuits start in `|0...0>`. Include product-state preparations as gates.
Parameters must be bound, input circuits must contain no classical registers,
measurements, resets, or control flow, and individual gates are limited to three
qubits for bounded unitary validation. The standard cutting adapter requires
one-/two-qubit gates; small multiqubit gates can use the excision adapter.

Queries are Hermitian `SparsePauliOp` sums with real coefficients. Qiskit's
rightmost Pauli character acts on qubit 0: `IZ` measures Z on qubit 0.

`CircuitProblem.create` copies the circuit and query. Its operation IDs refer to
that original snapshot. Plans remain valid while cuts insert new instructions;
mutating the snapshot is detected and requires a new problem. IDs are not a
cross-transpiler identity scheme. Fingerprints include ordered operations and
their small unitary matrices; they do not hash QPY's random custom-gate IDs.
Use `save_problem` / `load_problem` for QPY plus
JSON query/provenance snapshots, and `CutPlan.save` / `CutPlan.load` for plans.

`interaction_graph(problem)` exposes qubit interactions. `temporal_graph(problem)`
retains operations, directed wire edges, and operation ports, including ordering
and single-qubit gates. Neither function chooses cuts or runs an optimizer.
`compiled.reconstruction_graph()` exposes fragments and QPD factors, retaining
both local ports of each cut, including cuts whose ports stay in one fragment.

## Manual gate cutting

```python
from qiskit import QuantumCircuit
from qpartition import CircuitProblem, CutPlan, compile_cut, bootstrap_uncertainty

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
problem = CircuitProblem.create(circuit, "XX")
plan = CutPlan.create(
    problem,
    gate_cuts=(problem.operation_ids[1],),
    max_fragment_qubits=1,
)
compiled = compile_cut(problem, plan)
exact = compiled.run_exact()
sampled = compiled.run_shots(shots=2000, seed=42)
uncertainty = bootstrap_uncertainty(compiled, sampled, seed=42)

print(problem.exact_expectation(), exact.expectation, sampled.expectation)
print(compiled.cost, sampled.total_shots, uncertainty)
```

The adapter uses the Qiskit cutting add-on's physical decompositions and
reconstruction. It infers connected fragments after the selected cuts and checks
the qubits of the materialized fragments against the width constraint.

For wire cuts, select the qubit and an original operation on that wire immediately
after the cut:

```python
from qpartition import WireCut

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
problem = CircuitProblem.create(circuit, "XXX")
plan = CutPlan.create(
    problem,
    wire_cuts=(WireCut(1, problem.operation_ids[2]),),
    max_fragment_qubits=2,
)
compiled = compile_cut(problem, plan)
print(compiled.run_exact().expectation)
```

A plan can contain both wire and gate cuts. Repeated cuts on the same original
wire are supported. Fragment widths include the new wire segments introduced by
cuts. A cut can incur overhead without reducing width; width validation is
performed on the actual resulting fragments.

Standard cuts currently **enumerate the QPD coefficients**. Exact coefficient
enumeration and exact circuit evaluation are separate choices: `run_shots` still
has measurement noise. Identical experiments are shared within each fragment,
with a mapping back to all reconstruction terms. Execution data are not shared
across fragments, preserving independence of fragment-product estimates.

## Gate excision

```python
from qpartition import compile_excision

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.t(0)
problem = CircuitProblem.create(circuit, "XY")
compiled = compile_excision(problem, (problem.operation_ids[2],))

print(compiled.decompositions[0].algebraic_terms)  # 6 for T
print(len(compiled.experiments))  # 12 physical terms
print(compiled.gamma)  # 2 + 2*sqrt(2), for this construction
print(compiled.run_exact())
print(compiled.run_shots(shots=2000, seed=42))
print(compiled.run_sampled_terms(samples=24000, seed=42))
```

The sparse algebra uses
`R[Q,P] = Tr(Q U P U†)/2**n` and coefficients `R[Q,P]/2**n` in the
measure/prepare expansion. Physical experiments expand output Paulis into signed
product eigenstates, measure the input Pauli factors, reset and prepare the
excised wires, and retain correlations with untouched qubits. Removing a gate
can simplify the remaining operations without disconnecting the circuit.

`decompose_excision(gate)` exposes the transfer matrix, physical terms, algebraic
term count, and discarded numerical PTM norm. It supports blocks of at most three
qubits by default; both compilers enforce explicit enumeration limits. Numerical
coefficients below the declared tolerance are removed and their residual norm is
reported. This physical construction is not claimed to be the optimal QPD.

Multiple excisions are supported. The manual adapters retain separate compilation
paths. The hybrid compiler below combines wire cuts, gate cuts, and excisions in
one physical reconstruction. Joint gate-group decompositions remain a later extension.

## Automatic hybrid partitioning

`PartitionGraph` implements the typed two-/four-port model for fixed one-/two-qubit
unitary gates. Gate actions are categorical: retain, gate cut (when supported),
or whole-gate excision. Each temporal wire boundary can be retained or cut.
The cached template library validates each signed channel decomposition on a
complete Pauli basis and stores its physical coefficient norm.

```python
from qpartition import (
    AnnealConfig,
    CircuitProblem,
    HybridPlan,
    PartitionGraph,
    compile_hybrid,
    multi_start,
)
from qiskit import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
graph = PartitionGraph(CircuitProblem.create(circuit, "XX"))
config = AnnealConfig(max_qubits=1, steps=1000)
runs = multi_start(graph, config, seeds=tuple(range(10)), workers=1)
best = min((run.best for run in runs if run.best), key=lambda result: result.log_overhead)
plan = HybridPlan.create(graph, best.state, max_qubits=config.max_qubits)
compiled = compile_hybrid(graph, plan)
print(best.block_widths, best.sampling_overhead)  # (1, 1), approximately 9
print(compiled.run_exact().expectation)  # approximately 1
print(compiled.run_sampled(trials=1000, seed=42))
```

Use an `if __name__ == "__main__":` guard in scripts when setting `workers > 1`.
Store a `CircuitProblem` using the existing QPY + JSON format; save the selected
state using `HybridPlan.save`, and export the LUT using `graph.registry.save`.
The hybrid optimizer does not accept wider gates: explicitly decompose them into
one-/two-qubit gates before creating the graph.

The default objective is `log(overhead)` plus a penalty for exceeding the qubit
limit. Width counts **static wire segments**, including input/output boundaries:
wire cutting adds one segment, and excision adds the gate's arity. Connected
components are packed exactly into at most two executable blocks; each block's
width is the sum of its allocated segments. There is no qubit reuse or ancilla
discount. `partition_mode="components"` executes each component separately and
supports more than two fragments. Optional `objective="fragment_shots"` minimizes
`log(overhead) + log(number of blocks)` under the independent fragment-trial model.

SA uses calibrated temperatures, geometric cooling, single-action changes,
fixed local neighborhood redraws, a feasible incumbent archive, and final
coordinate descent. Initial states include spatial gate-cut partitions and
single-wire cuts. Multi-start runs cycle through this pool. Returned evaluation
counts include pilot probes and polishing; initialization is a separate expense.
`exact_search` provides a bounded small-instance reference and explicitly reports
whether its search completed. SA itself does not certify optimality.

The hybrid compiler generates physical branches lazily. `run_exact` enumerates
all branches with an explicit size guard; `run_sampled` importance-samples joint
decomposition/observable terms. `trials` is the number of reconstruction samples,
and `physical_shots` counts all executed fragment shots. Neither path needs IBM
hardware. The optimized coefficient norm is a worst-case sampling factor, not a
prediction of observable-specific variance.

The first Table-I experiment uses seven configurations, 50 fresh final starts
per circuit, and four worker processes:

```bash
python examples/run_sa_benchmarks.py --steps 6000 --tuning-starts 8 --final-starts 50 --workers 4
python examples/check_sa_artifacts.py
```

See [benchmark provenance](benchmarks/README.md) and
[results and limitations](SA_BENCHMARK_RESULTS.md). Raw seeds, traces, circuits,
templates, plans, and Qiskit cross-checks are saved in `validation/sa_benchmarks`.
The [implemented algorithm and mathematics](docs/HYBRID_SA_ALGORITHM.md) explain
the actual state space, evaluator, acceptance rule, physical reconstruction,
and QFT12 cut replacement. Regenerable LUT dumps are ignored by Git.

## Execution, statistics, and cost interpretation

`run_exact` enumerates statevector branches for measurements and resets, retaining
the joint classical outcome distribution. It has a default 12-qubit limit per
executed circuit and a 65,536-branch limit. This is a small-circuit correctness
oracle. It does not access a large uncut circuit when evaluating fragments.

`run_shots` uses Aer density-matrix simulation, with a seed and an optional Aer
`NoiseModel`. Transpilation preserves classical bit indices. All probabilities
and expectation estimates remain raw; there is no clipping or renormalization of
the reconstructed answer. Exact execution takes no physical shots.

For an executable signed decomposition, `gamma = sum(abs(coefficients))`, and
`gamma**2` is its worst-case bounded-estimator sampling factor. Standard cuts
report the product of the instantiated QPD basis norms: a CX cut gives 9, a wire
cut gives 16, and a mixed CX/wire plan gives 144. The Pauli-sum coefficient norm
is reported separately. These factors are not the actual circuit or shot counts
of a fixed-shot enumerated experiment.

`shots` means shots **per unique executed circuit** for enumerated runs.
`total_shots` counts all fragment executions. Costs also distinguish decomposition
terms, generated circuits, unique circuits, and materialized fragment widths.
Excision term sampling takes a global `samples` budget and batches repeated term
draws; each draw contributes one signed score shot.

The standard-cut uncertainty helper bootstraps each unique circuit's **joint
bitstrings**, keeping covariance induced by experiment reuse and multiple
observables. Results retain the circuit, query, and plan provenance, which is
checked before computing uncertainty. It estimates measurement sampling
uncertainty. Percentile bootstrap
intervals are approximate; sparse counts and deterministic observed outcomes can
make them unreliable. They do not cover device bias. Excision reports an analytic
sample standard error for independent enumerated +/-1 scores, or a combined
term/measurement standard error for importance-sampled terms.

The demo writes exact-reference errors, shot estimates, uncertainty, shot budgets,
costs, timing for standard execution, and dependency versions to JSON. Tests
cover channel identities on arbitrary matrices, entangled environments, exact
branch probabilities against the add-on's independent sampler, mixed/repeated
cuts, and finite-shot resource accounting.

Current scope is observable reconstruction and manual or automatic hybrid plans.
Gate-group discovery, joint decompositions, full-distribution
reconstruction, efficient Clifford execution, and IBM submission are subsequent
milestones. The current compilers already produce physical measurement circuits
and explicit result mappings for future execution adapters.
