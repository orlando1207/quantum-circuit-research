# Foundation validation

Validated on 2026-10-11 in the conda environment `quantum-partition`, Python
3.12.15. The complete suite passed: **51 tests**. Ruff lint and formatting checks
passed, and `pip check` found no broken requirements.

The saved [foundation.json](foundation.json) was produced with:

```bash
python examples/foundation_demo.py --shots 2000 --seed 42 --output validation/foundation.json
```

| Case | Fragment widths | Decomposition norm squared | Unique circuits | Enumerated shots | Exact error |
|---|---|---:|---:|---:|---:|
| CX gate cut | 1, 1 | 9 | 20 | 40,000 | 5.6e-16 |
| Wire cut | 2, 2 | 16 | 30 | 60,000 | 2.2e-16 |
| Mixed CX/wire cuts | 1, 1, 2 | 144 | 53 | 106,000 | 1.6e-15 |
| T excision, product eigenstate construction | Full 2-qubit circuit | 23.3137 | 12 | 24,000 | 4.4e-16 |

The excision demonstration additionally spends 24,000 shots on importance-sampled
physical terms. Its reported uncertainty includes both term selection and
measurement noise. Standard cuts enumerate their coefficients and bootstrap
joint outcome distributions to estimate measurement uncertainty.

These are correctness examples with manually selected plans, rather than
optimizer benchmarks or optimal decompositions. The wire/mixed query has
coefficient norm 1.1, so its expectation slightly above 1 is valid. Execution
timings in the JSON are diagnostic and depend on the host and simulator startup.

The tests also verify T/Toffoli sparse Pauli-transfer counts, channel action on
arbitrary matrices, excision in an entangled environment, repeated wire cuts,
cuts that retain both ports in one fragment, finite-shot variance calibration,
and trace-preserving reset behavior against an independent sampler.

All execution was local. No quantum device was used.

The subsequent [simple circuit verification](../examples/circuits/README.md)
adds seven analytic checks using four OpenQASM 2 source circuits. Its
[report](simple_circuits/report.json) and per-case snapshots, selections,
physical experiment circuits, and reconstruction metadata are saved in
`simple_circuits/`. It also verifies the complete import/save/reload workflow.
With the seven file-workflow tests included, the complete suite passes **58 tests**.
