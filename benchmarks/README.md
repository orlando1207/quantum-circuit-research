# Benchmark sources

The maintained [workflow, algorithm, and result tables](../docs/WORKFLOW_AND_ALGORITHM.md)
contain the complete comparison and provenance discussion.

`table_i_baselines.json` transcribes all eight rows of Table I from the supplied
[paper](../reference/2608.05287v1.pdf), including gate-only MIP, wire-only CutQC,
Qiskit, and the proposed method with and without gate grouping. The width limit
is `floor(3*n/4)` and overheads are rounded to two significant figures. Missing
entries are `null`, not zero.

Current multiway experiments reuse the exact frozen QPY/query snapshots in
`validation/sa_benchmarks_v2`; they do not regenerate circuits. Results are in
`validation/multiway_benchmarks_v3`, with tighter-width experiments under
`capacity_sweep/`. Main runs include historical two-way warm starts; the width
sweep uses fresh pools and matched tuning/final budgets for both modes.

QFT and fixed-angle QAOA are count/spatial-cost-matched reconstructions. Chain
circuits are structural proxies. The 38-CX Grover control is not an identified
reproduction of the paper's Grover row. Figure 4 lacks complete block definitions
and is not replaced with Figure 2. Those distinctions are preserved in every
comparison. The paper's grouped results use a different action library.

The later [external-baseline study](../docs/BASELINE_REPRODUCTION.md) adds a
historical MQT/Gray-code Grover candidate with 24 CX + 14 CP gates and a full
unitary check. It is saved as `grover4_historical`, separately from the old
control. Original artifact identity remains unverified. The study directly
executes the official Qiskit package and measures two explicit KL
reconstructions, with and without our own canonical CX group manifests where
applicable. Full authors' GG results require their original manual manifests.

The circuit generators and full provenance are in `src/qpartition/benchmarks.py`
and the saved v2 report. Previous raw experiment directories are retained as
reproducibility evidence, rather than maintained algorithm documentation.
