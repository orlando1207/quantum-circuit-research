# Table-I benchmark provenance

The baseline is Table I of the supplied
[paper](../reference/2608.05287v1.pdf), also available from
[arXiv](https://arxiv.org/html/2608.05287v1).
`table_i_baselines.json` transcribes all eight rows, including the columns with
and without gate grouping. Missing values are `null`, not zero. The qubit limit
is `floor(3*n/4)` and the paper retains the best of 50 runs. Its overhead values
are rounded to two significant figures; the runner compares rounding intervals.

The executable initial suite is in `qpartition.benchmarks`. Saved QPY + query JSON
snapshots and supplementary OpenQASM 2 files are in
[`validation/sa_benchmarks`](../validation/sa_benchmarks).

| Instance | Construction | Two-qubit gates | Width limit | Reproduction status |
|---|---|---:|---:|---|
| QFT6 | Full QFT, controlled phases and final swaps | 18 | 4 | Counts and spatial-cut cost match |
| QFT12 | Full QFT, controlled phases and final swaps | 72 | 9 | Counts and spatial-cut cost match |
| Chain10 | Overlapping four-qubit blocks | 36 | 7 | Structural proxy |
| Chain20 | Overlapping four-qubit blocks | 73 | 15 | Structural proxy |

QFT uses Qiskit 2.5.2 `synth_qft_full(do_swaps=True)`, without terminal
measurements. The algorithmic construction agrees with
[MQT Bench's QFT generator at commit 2cd3f3b](https://github.com/munich-quantum-toolkit/bench/blob/2cd3f3b08d5d8866595948fbce955e701e86a5df/src/mqt/bench/benchmarks/qft.py).
We include each final SWAP as one two-qubit gate, matching the paper's 18/72
counts. The exact spatial gate-cut seed costs are 2,521.7503178 and 796,655.9190897,
which round to the paper's 2.5e3 and 8.0e5. These two checks provide a useful
comparison, but the paper does not supply pinned circuit files/compiler versions
in the provided reference. They do not establish artifact identity.

For the chains, Figure 8b specifies four-qubit blocks sharing one wire, but does
not specify each block's gates. Our full block contains two directed CX layers
over all six pairs; a final two-qubit block contains one CX. This reproduces both
the overlap pattern and the 36/73 counts. The block contents are our construction,
so a matching cost is only a structural comparison. Input H gates are included
for simulation queries and do not change the partition width or cut cost.

The remaining rows are retained as targets rather than reported as reproduced:

- **QAOA8/15:** the historical
  [MQT generator at commit 9ac1ecf7](https://github.com/munich-quantum-toolkit/bench/blob/9ac1ecf7b0fad995d8e04d327bb80b792873201e/src/mqt/bench/benchmarks/qaoa.py)
  optimizes circuit angles classically. The exact optimized angles and benchmark
  artifacts are needed for a cost comparison; inventing new angles changes the
  gate QPD norms.
- **Grover4:** reconstruction with the available Qiskit decomposition gives a
  different two-qubit count (52 rather than 38). A pinned decomposition is needed.
- **Figure 4:** the diagram contains blocks whose complete unitary definitions
  are unavailable in the supplied reference.

All experiments optimize the product of the instantiated physical QPD norms
squared. They use independent decompositions, static allocation, no communication,
and no grouped templates. Grouped paper results remain visible as a stronger
comparison with a different action library. Runtime is not compared with the
paper: implementations, initialization budgets, and hardware differ.
