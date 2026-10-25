# MILP verification of WG vs. WGE: results

Produced 2026-10-25 with the Gurobi 13.0.3 backend (academic WLS license),
`MIPGap=0`, `MIPGapAbs=1e-6`, `IntFeasTol=FeasibilityTol=1e-9`. Full
methodology: [`docs/MILP_VERIFICATION.md`](../../docs/MILP_VERIFICATION.md).
Spec: [`docs/MILP_prompt.txt`](../../docs/MILP_prompt.txt). Reproduce with:

```bash
python examples/run_milp_verification.py --phase final \
  --output results/milp-verification-v1 --workers 10 --threads 2 --solver auto
```

92 instances, all CX-only (real, unassumed gamma_exc = 16 for every
excision). Every instance's winning plan was independently recompiled through
the existing evaluator (`compile_hybrid`); all 92 passed (register counts
match MILP widths, recompiled log-overhead matches the MILP objective within
1e-6, and every selected excision's covered edges are confirmed cut with the
correct recipe norm).

## Per-family results

| Family | Instances | Proven improvement | Proven no gain | Inconclusive | Omega ratio (WG/WGE) | Mean / max runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| separator | 24 | 12 | 12 | 0 | 9.000 (all 12) | 1.6 / 8.1 |
| brickwork | 16 | 0 | 16 | 0 | — | 26.9 / 139.5 |
| ttn | 16 | 0 | 16 | 0 | — | 1.8 / 12.0 |
| qcnn | 12 | 0 | 12 | 0 | — | 405.4 / 1949.2 |
| chain | 8 | 0 | 7 | 1 | — | 466.7 / 3600.3 |
| adder | 8 | 0 | 4 | 4 | — | 1916.6 / 3601.4 |
| aqft | 4 | 0 | 4 | 0 | — | 40.8 / 123.1 |
| trotter_grid | 4 | 0 | 4 | 0 | — | 207.6 / 691.0 |
| **Total** | **92** | **12** | **75** | **5** | | |

## The one family that excises: separator, and only at tight capacity

All 12 PROVEN IMPROVEMENT instances are the separator family (CX-butterfly
construction: two private-qubit CX modules feed one central CX, which feeds
two more modules) at its tight, module-width capacity. The ratio is **exactly
9** (Omega_WG / Omega_WGE = 2304/256 for every size N = 10, 14, 18, 22, 26,
30, two seeds each) — the improvement factor is **scale-invariant**: adding
more modules does not erode or amplify the benefit, because the gain comes
from one fixed local pattern (a CX gate plus its two adjacent cut wires,
independent cost 3*4*4=48 vs. the excision's 16, a 48²/16²=9× reduction in
sampling overhead), not from anything that scales with N.

At the same N with a looser capacity (ceil(N/2) instead of the module width),
every separator instance is PROVEN NO GAIN (ratio exactly 1): once there is
room to avoid cutting the central gate's neighborhood at all, the optimizer
does, and excision has nothing to discount. **Capacity tightness, not circuit
size, is what turns excision on for this family.**

Reference instance B — `large_benchmarks.large_benchmark("excision_butterfly",
6)` at its defaults (layers=3, seed=19), the same circuit as the existing
`excision_butterfly_6_L3_s19` entry in
[`validation/incidence_excision_v3/SUMMARY.md`](../incidence_excision_v3/SUMMARY.md)
— is in this set. That entry reported WG=2304, WGE=256 from best-of-3 SA
starts: a heuristic upper bound, never previously proven optimal. This study
independently **certifies** those are the true global optima
(`tests/test_milp.py::test_reference_instance_b_separator_matches_known_validated_result`).

## Everywhere else: proven no gain

Brickwork, tree-tensor-network, QCNN, AQFT, and Trotter-grid are PROVEN NO
GAIN at every tested size and capacity (brickwork/ttn up to N=32; AQFT/Trotter
up to N=20). These families lack the separator's specific local motif (a
retained-cost-concentrated gate bracketed by two independently-cut wire
boundaries); the optimizer's unconstrained search never finds a cut pattern
excision could discount. 7 of 8 chain instances are also proven no gain; the
one exception (`chain_tight_8_M3`, the tightest tested chain capacity) did not
reach proven optimality in time (see below) rather than showing a gain.

## Inconclusive instances (certified bounds, not dropped)

5 instances — four Cuccaro ripple-carry adders at their tightest tested
capacities and one tight-capacity chain — did not reach proven optimality
within the 1800-second-per-model budget. Per Section 10 of the spec, these
are reported with their certified bounds rather than discarded:

| Instance | N | M | WG status (gap) | WGE status (gap) | Delta* range (log units) |
|---|---:|---:|---|---|---|
| adder_2b_M2 | 6 | 2 | time limit (63.7%) | time limit (73.3%) | [0, 33.92] |
| adder_3b_M3 | 8 | 3 | time limit (40.1%) | time limit (62.3%) | [0, 15.18] |
| adder_4b_M4 | 10 | 4 | **optimal** | time limit (42.9%) | [0, 7.13] |
| adder_5b_M4 | 12 | 4 | time limit (59.3%) | time limit (66.7%) | [0, 14.79] |
| chain_tight_8_M3 | 8 | 3 | time limit (22.3%) | time limit (31.8%) | [0, 8.39] |

(Delta* lower bound is clipped to 0 per the spec's sanity property — WGE can
never be worse than WG — even where the raw certified bounds alone would not
yet rule out a negative value.) The ripple-carry adder's long sequential CX
chain with heavy structural symmetry makes this MILP formulation's branch-and-
bound converge slowly at its tightest capacities; this is a property of the
instance class under this exact formulation, not evidence for or against
excision. None of these instances' *raw MILP objectives* are used anywhere
else in this summary's counts beyond this table.

## Reproduction and verification

- `tests/test_milp.py` (13 tests): both named reference instances, 5 random
  small circuits cross-checked against brute-force enumeration, width/
  excision/K_safe checks, and a regression test guarding the exact bug found
  and fixed during this study (below).
- `tests/test_milp_gurobi.py`: Gurobi and HiGHS certify bit-identical optima
  on reference instance A.
- All 172 tests in the full suite (158 pre-existing + 14 new) pass.

### One bug found and fixed during this study

The WG and WGE models must share one `PartitionGraph` (the spec's explicit
"same graph" requirement). Converting a WG solution back into a `HybridPlan`
initially passed `forced_groups=None` for the WG case, which told the shared
graph's evaluator to fall back to its own unrestricted excision selector —
silently discounting the plan whenever the WG-optimal cut pattern happened to
also satisfy some excision candidate (exactly the separator family's
defining property). This never affected any instance's *classification* (that
was always computed from the raw MILP objectives, not from recompilation),
but it broke the Section 9 recompilation check for 12 separator instances.
Fixed in `src/qpartition/milp.py` (`forced_groups=()` forces zero selections,
distinct from `None`'s "run the normal selector"); covered by
`tests/test_milp.py::test_wg_recompilation_does_not_silently_apply_an_eligible_excision`.
All 92 instances now pass recompilation verification.
