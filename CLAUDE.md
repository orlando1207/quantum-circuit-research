# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for partitioning general quantum circuits under a qubit capacity M, to
minimize the sampling overhead of reconstructing an observable. It combines three
mechanisms:

- wire cuts;
- gate cuts (QPD);
- excision of individual two-qubit gates (measure-and-prepare).

The study compares **WG** (wire + gate cuts) with **WGE** (wire + gate cuts + single-gate
excision).

- **Core algorithm:** MILP-based large neighbourhood search (MILP-LNS,
  `src/qpartition/lns.py`). It repeatedly repairs small regions of the exact WG/WGE MILP.
- **Exact model:** `src/qpartition/milp.py`. It certifies results on small instances.
- **Baseline:** Qiskit's `qiskit_addon_cutting.find_cuts`.

Label-move simulated annealing (`annealing.py`) is no longer the core. It is still used for
seeding. Multi-gate grouping (`grouping.py`, `group_qpd.py`, `group_heatbath.py`) is
implemented but disabled. Only its single-gate `excision` family is active.

No IBM account or quantum device is used. All execution is local simulation.

This `main` branch holds only the core workflow:

| Branch | Contents |
|---|---|
| `main` | Core workflow only. Tests, debug/diagnostic scripts, plots and the long design docs are left out. |
| `grouping-integration` | The full development history: tests, diagnostics, the SA-era studies, and `validation/`. Check it out when you need any of those. |

## Environment

```bash
conda create -n quantum-partition python=3.12.15 pip
conda activate quantum-partition
pip install -r requirements.txt     # complete pinned freeze of the working env
pip install -e . --no-deps
```

or `conda env create -f environment.yml && conda activate quantum-partition && pip install -e . --no-deps`.

**Gurobi.** `solver="auto"` picks Gurobi when it is importable and licensed. It is
10-50x faster than HiGHS (`scipy.optimize.milp`) on this model. Without a license,
`gurobipy` is limited to small models, so put a license on the workstation (WLS or a
file license). HiGHS is the dependency-free fallback.

**Lint.** `ruff check src examples` (line length 100, rules E, F, I).

## Data that is NOT in git (upload separately)

| Path | Contents |
|---|---|
| `data/raw/hamlib/` | HamLib HDF5 files: `heis.hdf5`, `tfim.hdf5`, `maxcut`, etc. (`HAMLIB` in `examples/run_family_survey.py`) |
| `data/raw/qasmbench/` | QASMBench `.qasm` files (`QASMBENCH` in the same file) |
| `results/excision_lp_cache.pkl` | Excision-LP cost cache keyed by canonical KAK coordinates. Pass it with `QPARTITION_EXCISION_CACHE=results/excision_lp_cache.pkl`. It can be regenerated with `examples/precompute_excision_costs.py`, but that is slow. Copy it over if possible. |
| `results/` | Gitignored scratch output: case snapshots (`circuit.qpy` + `problem.json`), runs, CSVs |

MQT Bench, Qiskit library and ReCirq-style generators need no data files. They are built
in `src/qpartition/external_benchmarks.py` and `lattice_benchmarks.py`.

## Fixed experimental setting ("Setting A")

- **Preprocessing:** fuse consecutive two-qubit gates on the same pair
  (`preprocessing.fuse_two_qubit_gates`). No grouping.
- **Gate-cut cost:** Qiskit's QPD (Mitarai-Fujii) cost, as-is, so a SWAP costs 7.
- **Wire-cut cost:** gamma = 4.
- **Excision cost:** the excision LP (`excision_lp.gate_excision_cost`, `excision_recipe="lp"`).
- **QPD residual tolerance:** 1e-4 for gates, 1e-10 for wires (`templates.GATE_QPD_TOLERANCE`).
  The looser gate tolerance lets near-identity fused gates, such as tiny-angle CP gates in
  a QFT, pass the channel check.
- **Objective:** the natural log of the sampling overhead. A difference d means a ratio e^d.
- **Capacities:** M = ceil(N/2) and ceil(N/3).

## Source map (`src/qpartition/`)

Core workflow, in reading order:

| File | Role |
|---|---|
| `circuits.py` | `CircuitProblem`: input validation, `save_problem`/`load_problem` (QPY + JSON) |
| `preprocessing.py` | Same-pair gate fusion, two-qubit gate count, SWAP peeling |
| `partition_graph.py` | Contracted temporal incidence graph, `evaluate(state, forced_groups=...)`, fragment widths |
| `templates.py`, `excision.py`, `excision_lp.py` | Cut and excision recipes and costs; excision-LP cache (`QPARTITION_EXCISION_CACHE`) |
| `grouping.py` | Realization selector. Only the single-gate `excision` family is enabled. |
| `milp.py` | Exact WG/WGE MILP: `build_milp`, `compute_k_safe`, `solve_milp` (HiGHS/Gurobi), `milp_plan` (solution -> `HybridPlan`) |
| `cover_cuts.py` | Capacity cover cuts (valid inequalities on the cut variables x), separated on the LP relaxation |
| `lns.py` | **Core algorithm.** `run_lns(graph, LNSConfig) -> LNSResult` |
| `hybrid.py` | `HybridPlan` validation and physical compilation (`compile_hybrid`); used to verify every plan |
| `qiskit_baseline.py` | Adapter for Qiskit `find_cuts` plus independent checks |
| `external_benchmarks.py`, `lattice_benchmarks.py`, `large_benchmarks.py`, `milp_benchmarks.py` | Benchmark circuit generators (HamLib, MQT, ReCirq, QV, Heisenberg/XY lattices, ...) |

Supporting modules (`annealing.py`, `multiway.py`, `label_moves.py`, `cutting.py`, `plans.py`,
`migration.py`, `legacy_graph.py`, and others) are imported by the package or by the
seeding code. Keep them.

## MILP-LNS (`lns.py`)

The state is the cut assignment x. Fragments, widths and the best excision selection are
derived from it.

1. **Cut pool.** Build a cover-cut pool on the LP relaxation, within a time cap.
2. **Seeds.** Collect candidate starting states and keep the best:
   - all gates cut;
   - qubit-partition seeds (`qubit_partition_seeds`): greedy and swap refinement of a
     qubit grouping into groups of at most M, using a cost-weighted interaction matrix;
   - the spatial and agglomerative seeds of `annealing.seed_states`, with a deadline;
   - an optional `initial_state` passed in `LNSConfig`.
3. **Warm start and certification.**
   - Run a short full-MILP warm start from the best seed.
   - Then run a certify step: the full model with K_safe for
     `min(certify_seconds, T/4)`, which proves optimality on small instances.
4. **Main loop.** Repeat until the time limit T:
   - **Destroy.** Pick an operator, choosing with adaptive weights:

     | Operator | Region it frees |
     |---|---|
     | `ball` | BFS ball around a vertex |
     | `cut_ball` | ball around a cut edge |
     | `fragments` | a fragment plus its neighbours |
     | `excision` | ball around a gate with a large excision saving |
     | `time_window` | consecutive two-qubit-gate layers |
     | `qubits` | the whole wires of 1-3 qubits, so qubits can move between fragments |

   - **Repair.** Solve the MILP with every edge outside the region fixed, using the
     incumbent as the MIP start. A repair is therefore never worse than the incumbent.
   - **Adapt.** Accept results that are no worse. Adapt region sizes and operator weights.

**Pipeline used in experiments:** run LNS-WG for T, then LNS-WGE for T, seeded from the WG
state (`initial_state`). By construction, LNS-WGE is never worse than LNS-WG.

## Experiment runners (`examples/`)

| Script | Purpose |
|---|---|
| `run_family_survey.py` | Exact WG vs WGE MILP survey over about 64 external families (N <= 24). Builds case dirs under `results/family_survey/cases`. |
| `run_setting_a_certification.py` | Exact certification for the project's lattice suite. `setting_a_problem` is reused by the other runners. |
| `run_breadth_survey.py` | Family definitions reused by the certification runner |
| `run_milp_verification.py` | Original exact-MILP verification study |
| `precompute_excision_costs.py` | Fills the excision-LP cache for every gate class in a set of cases |
| `run_lns_family_benchmark.py` | **Main comparison:** LNS-WG and LNS-WGE vs Qiskit |
| `run_large_benchmark.py` | Builds the large-circuit suite (N 16-128, fused G2 30-800), then precomputes excision costs |

**`run_lns_family_benchmark.py` details.**

- **Phases:** `select | run | summarize`.
- **Budgets:** per tier, T = 60 / 180 / 300 / 450 s for fused G2 <= 32 / <= 100 /
  <= 300 / > 300. Each method gets 2T, single-threaded.
- **Qiskit:** runs a max_backjumps ladder of 1e4 (Qiskit's default), 1e5, 1e6 and
  unlimited, checkpointed per rung.
- **Memory:** `RSS_LIMIT_GB` sets an RSS cap shared by both methods.
- **Output:** CSVs `lns_vs_qiskit_{instances,families,groups}.csv`.
- **Resuming:** rerun on the same output directory. Finished tasks are skipped.

Large-circuit flow:

```bash
python examples/run_large_benchmark.py --phase prepare --output results/lns_large
QPARTITION_EXCISION_CACHE=results/excision_lp_cache.pkl \
  python examples/run_large_benchmark.py --phase precompute --output results/lns_large
python examples/run_lns_family_benchmark.py --phase select --survey results/lns_large \
  --setting-a none --output results/lns_large
RSS_LIMIT_GB=2.0 QPARTITION_EXCISION_CACHE=results/excision_lp_cache.pkl \
  python examples/run_lns_family_benchmark.py --phase run --output results/lns_large \
  --methods lns qiskit --workers 6
python examples/run_lns_family_benchmark.py --phase summarize --output results/lns_large
```

## Status (2026-11-10)

**1. Exact family survey.** 384 of 468 cases were solved.

- Seven proven improvements from excision (six distinct): Quantum Volume, Sycamore RCS,
  the HamLib 3-cube, plus the project's own Heisenberg/XY lattices at theta >= pi/8.
- Every gain needs three things: a graph with cycles, 3-parameter gates, and a tight M.
- The HamLib Heisenberg results differ from our own because of the Trotter angle.
  HamLib's dt = 0.2 is too small for excision to pay off.

**2. LNS vs Qiskit on the survey families.**

- LNS was never worse than Qiskit, except for one instance that has since been fixed.
- Excision-improves group: LNS-WG matches exact WG on 103 of 106 instances.

**3. Large circuits** (168 instances, 2 GB RSS cap for both methods). 325 of 336 tasks
are done.

- LNS-WG/WGE vs Qiskit, as better / tie / worse:

  | group | better | tie | worse |
  |---|---|---|---|
  | excision-relevant | 62 | 8 | 0 |
  | control | 50 | 24 | 0 |

- On 16 instances Qiskit's default search went over the 2 GB cap and returned no plan.
- Excision beat the best WG plan on 5 instances: Quantum Volume n20 (up to 2303x) and
  the HamLib 3D grid n27.
- Still missing: LNS on MQT n32 ae, qft, qftentangled and qpeexact; Qiskit on 3 of those.

## Working conventions

- Commit only when the user asks. Put the attribution line given in the session at the
  end of each commit message.
- `results/` and `data/` are gitignored scratch areas. Never commit QPY/QASM/HDF5/pickle
  or archive files.
- When results are superseded, move them under `results/superseded-<name>-<date>/`
  instead of deleting them.
- Guard multiprocess code with `if __name__ == "__main__":`.
- Cap solver threads relative to the number of workers. HiGHS and Gurobi otherwise use
  every core per solve, and the oversubscription looks like hanging solves.
- **Running heavy jobs:**
  - Run them detached (`setsid nohup ... &`), with an RSS cap.
  - Stop them by PID or process group.
  - Never use `pkill -f` or `pgrep -f`: the pattern can match the shell running the
    command and kill it.
- `PartitionGraph.evaluate(..., forced_groups=())` forces zero excisions. `None` means
  "search excisions". Use `()` when re-scoring WG or Qiskit plans on an excision-enabled
  graph.
