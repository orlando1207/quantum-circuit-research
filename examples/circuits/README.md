# Preparing circuits and verifying the current operations

The four editable source circuits here use **OpenQASM 2.0**. For research records,
save the imported circuit using **QPY plus the JSON sidecar** produced by
`save_problem`. The observable and cut selection are part of the experiment and
must accompany the circuit.

From the repository root, run:

```bash
conda activate quantum-partition
python examples/verify_simple_circuits.py --shots 2000 --seed 42 --output results/simple_verification
```

The checked-in [report](../../validation/simple_circuits/report.json) records a
run with those settings. Each case checks QASM import, QPY/query round-trip,
saved cut-plan loading where applicable, the decomposition norm, and exact
reconstruction against both an analytic value and uncut statevector simulation.
It then executes physical circuits with finite shots and estimates uncertainty.

| Case | Operation | Observable | Analytic value | Sampled estimate | Estimated standard error |
|---|---|---|---:|---:|---:|
| Bell | CX gate cut | XX | 1 | 1.000001 | 0.000693 |
| Bell | CX gate cut | YY | -1 | -0.999785 | 0.000775 |
| Bell | CX gate cut | IZ | 0 | -0.002233 | 0.017232 |
| GHZ | Wire cut | 0.3 XXX + 0.2 ZIZ - 0.1 YXY | 0.6 | 0.599521 | 0.000520 |
| GHZ | CX gate + wire cuts | Same weighted query | 0.6 | 0.600035 | 0.000520 |
| RZZ(0.3) on ++ | RZZ gate cut | ZY | sin(0.3) | 0.295387 | 0.000606 |
| Bell followed by T on q0 | T excision | XY | 1/sqrt(2) | 0.692147 | 0.029581 |

All exact errors are below 1e-15 in this run. The excision case also uses
importance sampling: estimate 0.754039 with standard error 0.030786 and 24,000
total shots. The enumerated excision run separately uses 24,000 shots.

Bell has `<XX>=<ZZ>=1`, `<YY>=-1`, and zero single-qubit Z expectation. GHZ has
`<XXX>=<ZIZ>=1` and `<YXY>=-1`, giving the weighted value 0.6. The other two
values follow from the RZZ rotation angle and T phase pi/4. These analytic
checks supplement the simulator reference.

The six-standard-error screen in the runner is a loose statistical diagnostic,
not a proof of unbiasedness or guaranteed confidence coverage. All estimates in
this run, including importance-sampled excision, are within 1.53 estimated
standard errors of their targets. Signed reconstruction estimates can lie
slightly outside [-1, 1], as the Bell XX example shows; they remain unclipped.
These are observable checks on small ideal simulated circuits. Hardware error
and optimal partitioning are separate questions.

## Input formats

| Format | Current use | Loading |
|---|---|---|
| OpenQASM **2.0**, `.qasm` | Editable/importable unitary circuit; supply its observable separately | `load_qasm2(path, observable)` |
| `circuit.qpy` + `problem.json` directory | Canonical circuit, observable, operation IDs, and fingerprints | `load_problem(directory)` |
| `plan.json` | Selected gate/wire cuts and width constraint for that saved circuit | `CutPlan.load(path)` then `plan.validate(problem)` |
| Qiskit `QuantumCircuit` | Build a circuit directly in Python before saving | `CircuitProblem.create(circuit, observable)` |

The current wrapper has no OpenQASM 3 import function. A `.qasm` extension alone
does not specify the version: use the `OPENQASM 2.0;` header for these examples.
QASM import creates a new snapshot and its operation IDs; QPY plus the sidecars
preserves the saved research snapshot. JSON alone is not a circuit format here.

An input circuit must:

- Start from the assumed all-zero state; express preparations through gates.
- Have bound numerical parameters and at least one qubit.
- Have no classical registers, measurements, resets, initialization instructions,
  delays, or control flow. The compilers add measurements and resets themselves.
- Use one-/two-qubit gates for the standard cutting path. Individual gates up to
  three qubits are accepted by circuit validation and the small-gate excision path.

Give an observable of the same width, with real coefficients. The rightmost
Pauli letter acts on qubit 0: `IZ` is Z on q0, while `ZI` is Z on q1. Full
output-distribution reconstruction is not part of the current interface.

For example, [bell.qasm](bell.qasm) contains:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
```

[rzz.qasm](rzz.qasm) explicitly declares its two-qubit gate because RZZ is not
defined in standard `qelib1.inc`. Its gate invocation is one top-level operation
in the imported circuit, even though its definition contains CX/RZ/CX.

## Load, select a cut, save, and execute

```python
from qpartition import CutPlan, compile_cut, load_problem, save_problem
from qpartition.circuits import load_qasm2

problem = load_qasm2("examples/circuits/bell.qasm", "XX")
for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
    print(op_id, item.operation.name)
# op000000 h
# op000001 cx

plan = CutPlan.create(
    problem,
    gate_cuts=(problem.operation_ids[1],),
    max_fragment_qubits=1,
)
save_problem(problem, "results/my_bell")
plan.save("results/my_bell/plan.json")

restored = load_problem("results/my_bell")
restored_plan = CutPlan.load("results/my_bell/plan.json")
compiled = compile_cut(restored, restored_plan)
print(restored.exact_expectation())  # approximately 1
print(compiled.run_exact().expectation)  # approximately 1
print(compiled.run_shots(shots=2000, seed=42).expectation)
```

To use a Pauli sum, pass `SparsePauliOp.from_list([("XXX", 0.3), ...])` instead
of the single Pauli string. For a wire cut, select `WireCut(qubit,
problem.operation_ids[index])`, placing the cut immediately before that original
instruction on the selected wire. Gate excision uses `compile_excision(problem,
(operation_id,))` separately.

Inspect operation IDs before selecting them. Editing or transpiling a circuit
changes the snapshot: create a new problem and selection instead of reusing its
old plan. Do not hand-edit fingerprints in `problem.json` or `plan.json`.

To add another case to the verification runner, add a `VerificationCase` in
`verification_cases()`: specify its QASM filename, observable, analytic target,
decomposition norm, and gate/wire/excision locations. Indices in those definitions
refer to the original imported instructions and are converted to snapshot IDs
before compilation. Cases without an analytic target can still be evaluated
through the core API against a small uncut simulator reference.

## Saved verification artifacts

Each case directory contains:

```text
circuit.qpy                 Original unitary input circuit
problem.json                Observable, operation IDs, and fingerprints
circuit.txt                 Readable original circuit
operations.json             Operation IDs, names, and qubits
plan.json                   Gate/wire selection; excision uses excision.json
experiments/*.qpy           Physical measurement/reset circuits
experiment_mapping.json     Reconstruction indices or excision coefficients/masks
result.json                 Exact checks, shot estimate, uncertainty, and costs
```

The enclosing `report.json` summarizes all cases. Load physical experiment files
using `qiskit.qpy.load`; they contain measurement/reset operations, so they are
not valid unitary `CircuitProblem` inputs. The file-workflow tests replay these
exported physical circuits and verify reconstruction from their saved mappings.

`--shots` is the number of shots per unique enumerated circuit, rather than the
total budget. Read each result's `total_shots` for the actual number used. The
importance-sampled excision mode uses a separate total budget matching its
enumerated run. Theoretical norm-squared factors are 9 for CX, 16 for a wire
cut, 144 for this mixed plan, approximately 2.53141 for RZZ(0.3), and 23.31371
for the current T excision construction. These are not physical circuit counts
or a claim that this excision construction is optimal.
