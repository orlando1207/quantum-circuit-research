# Workflow and algorithm: multiway partitioning with two-qubit excision

Updated 2026-10-21. This document specifies the current research method and its
implemented limits. The active operation set is **wire cutting, gate cutting,
and individual two-qubit gate excision**. Multi-gate grouping is disabled in
both the workflow below and the associated experiments.

The implementation keeps the contracted temporal incidence graph, binary edge
cut decisions induced by vertex labels, and the existing multiway simulated
annealing engine. Excision is an alternative physical realization of certain
local cut patterns. It introduces no vertices, independent state variables, or
separate search kernel.

## 1. Research question and scope

Given a fixed, bound circuit on N input qubits and a maximum fragment width M,
find a set of independently executed fragments and local decompositions that
minimize the sampling overhead for reconstructing an observable. The number
of fragments can vary. The immediate research question is:

> When does replacing a two-qubit gate and neighboring cut boundaries by a
> single measure-and-prepare decomposition improve partition quality beyond
> optimizing independent wire and gate cuts?

The key comparison is the matched **WG versus WGE ablation**: identical source
circuits, width limits, seed pools, search configurations, and evaluation
budgets; only the availability of excision realizations changes. Comparison
against official Qiskit and the reconstructed paper method provides additional
context, but does not isolate the effect of excision by itself.

Inputs contain one- and two-qubit unitary gates with all parameters bound.
Decompose larger gates before constructing the graph. Single-qubit operations
remain in the source circuit, including state preparation. Inputs do not
contain measurements, resets, classical registers, or adaptive control.
Generated physical fragments can contain measurements and resets.

The current hard resource constraint is static fragment width. Depth, noise,
connectivity, total execution latency, and mid-execution inter-device
communication are not optimized. There is no requirement for VQE, QML, a
parameter-training loop, or access to a quantum device for this study.

## 2. Input, identity, and preprocessing

The canonical input snapshot consists of `circuit.qpy` and `problem.json`,
written by `save_problem` and loaded by `load_problem`. The latter stores the
observable and circuit/query identities. OpenQASM is an interchange format;
QPY is the reproducibility format. Stable IDs such as `op000123` identify
instructions in the original source snapshot.

Preprocessing performs the following steps:

1. Validate the bound unitary circuit and Pauli-sum observable.
2. Build the contracted temporal incidence graph and source-to-vertex maps.
3. Cache independent wire/gate decompositions by actual gate semantics,
   including parameter values, rather than by gate name alone.
4. Construct the checked physical two-qubit excision recipe for each distinct
   gate and enumerate its allowed bindings to nearby wire boundaries.
5. Cache only excision realizations with strictly positive potential savings.
6. Build the original label-move region pools when starting a trajectory.
7. Generate the original structural seed pool; record its time separately.

The source circuit, all single-qubit instructions, operation order, and
input/output mappings remain authoritative. Their identities and the recipe
catalogue fingerprint are checked when saving and compiling a plan. A graph
alone is not a complete execution representation.

## 3. Contracted temporal incidence graph

Let G = (V, E_g union E_w). For every two-qubit source gate g acting on operands
a and b, create exactly two vertices, v(g,a) and v(g,b), and one gate edge
between them. On each original qubit, connect consecutive incidence vertices
in time with wire edges. A qubit without any two-qubit gate receives one
local-only vertex so that its register is still counted.

```text
Source circuit:                 Contracted graph:
q0: ----g--------h----           v(g,0) -------- v(h,0)
        |        |                 |               |
q1: ----g----k---|----           v(g,1) -- v(k,1)    |
             |   |                         |       |
q2: --------k---h----                  v(k,2) -- v(h,2)

Vertical edges: two-qubit gate couplings.
Horizontal edges: temporal continuation on a source qubit.
Single-qubit chains: retained in the source IR, contracted in this graph.
```

Input/output boundaries alias the first/last incidence on each qubit. There
are no independently cuttable exterior input/output edges in this contracted
graph. The canonical ordinary wire boundary is immediately before the next
two-qubit gate; intervening one-qubit operations remain mapped through the
source representation.

For G2 two-qubit gates and A active qubits, the graph has G2 gate edges and
2 G2 - A wire edges. It has 2 G2 incidence vertices plus one vertex for each
inactive qubit. Graph size therefore depends on gate count as well as qubit
count: 250 qubits with a few layers and 250-qubit full QFT have very different
optimization costs.

A gate edge can be cut only if the registry supplies its supported independent
decomposition. Disabled cuts constrain the endpoint vertices to the same
label; the move builder contracts these mandatory connections into units.

## 4. Label state, components, and physical widths

Each vertex has an integer label l_v. Every edge e = (u,v) induces the binary
cut decision:

$$
x_e = \mathbf{1}[l_u \ne l_v].
$$

The graph still has only retain/cut edge options. There is no third edge value
for excision. Labels encode boundaries; they are not an allocation of one
physical qubit per vertex and do not determine width by vertex cardinality.

For a proposed cut state, the evaluator performs two union-find computations:

- Unite endpoints of every retained wire edge to obtain **static wire
  segments**. A cut wire starts a new segment of the same source qubit.
- Unite endpoints of every retained wire or gate edge to obtain connected
  components. Each component executes as one fragment in `multiway` mode.

A fragment's width is the number of distinct static segments in it. Counting
source-qubit names, labels, or incidence vertices would give the wrong answer.
Two disconnected components with the same label still execute separately.
The number of occupied label colors can therefore differ from the fragment
count K.

Let W_raw be the number of cut wire edges. The central resource identity is:

$$
\sum_{i=1}^{K} w_i = N + W_{\mathrm{raw}},
\qquad
\max_i w_i \le M.
$$

There is no arbitrary packing of unrelated components in `multiway` mode.
The optional compatibility mode `two_blocks` instead packs connected
components into at most two static registers using exact subset-sum packing.
That mode is impossible when N > 2M; it is not used for our main SA study.

## 5. Sampling objective and excision realization LUT

### 5.1 Independent cuts

For a decomposition into physical local maps with signed coefficients a_j,
write gamma = sum_j |a_j|. The sampling overhead reported by this project is
Omega = gamma squared. For independent factors:

$$
\Gamma = \prod_t \gamma_t,
\qquad
\Omega = \Gamma^2,
\qquad
L = \log\Omega = 2\sum_t\log\gamma_t.
$$

Search and large-instance comparisons use L to avoid numerical overflow.
Tables explicitly identify either Omega or log10(Omega); neither is confused
with the unsquared norm Gamma.

The installed Qiskit decompositions provide gate-specific norms. Examples are
gamma = 3 for CX/CZ and gamma = 7 for SWAP. The independent local wire-cut
norm is 4, so one wire cut contributes overhead 16. There is no extra width
surcharge hidden inside these norms; width is handled by the separate resource
constraint.

### 5.2 Physical two-qubit excision recipe

For a two-qubit unitary U, let P and Q range over the 16 Pauli products and set
D = 4. The implemented transfer matrix is

$$
R_{Q,P} = \frac{1}{D}\operatorname{Tr}(QUPU^\dagger),
\qquad
\mathcal U(\rho) = \frac{1}{D}
  \sum_{Q,P} R_{Q,P}\operatorname{Tr}(P\rho)Q.
$$

Each output Pauli product Q is expanded into its signed product eigenstate
projectors. Each branch measures product Pauli P on the input and prepares
one output product eigenstate, with coefficient R(Q,P)/D times the output
sign. Identity factors use both computational eigenstates with positive sign.
For this specific construction:

$$
\gamma_{\mathrm{exc}} = \sum_{Q,P}|R_{Q,P}|.
$$

The catalogue checks the channel action on the full Pauli operator basis,
using zero PTM truncation tolerance. This establishes correctness for states
entangled with other circuit qubits, rather than checking only the all-zero
input.

For CX this physical recipe has gamma_exc = 16 and overhead 256. This is
**the cost of an available recipe, not an established optimal excision cost**.
Other two-qubit gates may have larger recipe norms; the benchmark study uses
their actual norms. It does not assign a universal overhead 256 to all gates.
The API can record an explicit assumed norm through `excision_gamma`, but a
recipe with a different actual norm cannot be compiled at that assumed price.
Such assumptions are excluded from the current comparison.

The actual recipes used in the workload families illustrate why a universal
constant would change the comparison:

| Gate | Independent gate-cut norm | Physical excision norm | Physical excision overhead |
|---|---:|---:|---:|
| CX or CZ | 3 | 16 | 256 |
| RZZ(0.37) | 1.723231 | 18.351542 | 336.779102 |
| CP(pi/2) | 2.414214 | 28 | 784 |
| CP(pi/32) | 1.098135 | 17.176206 | 295.022042 |

These are computed from the current registry, not asserted optimal excision
norms. A small-angle gate cut can be much cheaper than its excision, even
though excision may still discount a sufficiently costly neighboring cut
pattern.

### 5.3 Excision as a cut-pattern realization

For each operand of g, a realization chooses one of:

| Binding | Input measurement register | Output preparation register | Required wire cut |
|---|---|---|---|
| Local | Incidence of g | Same incidence | None |
| Incoming | Previous incidence | Incidence of g | Incoming wire edge |
| Outgoing | Incidence of g | Next incidence | Outgoing wire edge |

Unavailable incoming/outgoing edges are omitted. These registers are bindings
to existing vertices; they are not new input/output ports in the graph.
There are at most nine operand-binding pairs, each considered with the gate
edge cut or retained.

For a candidate c, record its source gate, physical recipe, measurement and
preparation bindings, covered cut variables B_c, required retained gate edge
if applicable, and potential saving:

$$
s_c = \sum_{e\in B_c} 2\log\gamma_e
      - 2\log\gamma_{\mathrm{exc},c}.
$$

A cut-gate realization covers that gate edge and its chosen wire edges. A
retained-gate realization covers only its chosen wire edges and requires the
gate edge to stay retained. Only candidates with s_c > 0 are stored. Runtime
eligibility requires every covered edge to be cut and every guard to hold.

Example: a cut CX plus one cut boundary on each operand would independently
cost

$$
(3\cdot4\cdot4)^2 = 2304.
$$

The excision recipe implements that local separation at overhead 256, a
ninefold reduction **at the same cut state and widths**. Comparing 256 only
with a single CX cut's cost 9 would miss the neighboring wire factors that
excision replaces.

Multiple candidates cannot reuse a source gate or claim the same cut factor
twice. The existing realization selector constructs deterministic greedy
incumbents and uses exact subset selection when there are at most ten eligible
candidates. Above that limit, selection is heuristic and its completion flag
is recorded. Thus the priced objective is an available physical construction,
not a guaranteed minimum over all decompositions.

For a compatible selected set S, the objective is

$$
L_{\mathrm{WGE}}(l)
 = L_{\mathrm{WG}}(l) - \sum_{c\in S(l)}s_c.
$$

This objective is no longer a sum of independent edge weights. Each excision
is a conditional local cost factor coupling an existing gate edge to nearby
wire edges. The graph topology and label state are unchanged.

Excision candidates do not introduce proposal regions or change the underlying
connected components. Already-cut wire boundaries provide the extra segments.
If W_ind is the number of independently implemented wire cuts and W_exc the
number of wire boundaries covered by selected excisions, then

$$
W_{\mathrm{raw}}=W_{\mathrm{ind}}+W_{\mathrm{exc}},
\qquad
\sum_i w_i=N+W_{\mathrm{ind}}+W_{\mathrm{exc}}.
$$

An excision may cover zero, one, or two wire boundaries in this representation.
Do not add two extra registers for every excision on top of the already
counted boundaries. Local measurement followed by reset/preparation can use
the same static register.

This is a restricted, physically executable integration: at most one adjacent
boundary per operand is absorbed, endpoint aliases remain, and graph
components are retained conservatively when an excised gate edge is uncut.
We do not claim that it represents every possible four-terminal excision
partition or optimizes register scheduling/reuse.

## 6. Multiway SA objective and moves

### 6.1 Energy and feasible incumbents

For fragment widths w_i, use the squared overflow penalty

$$
P(l)=\sum_i\max(0,w_i-M)^2,
\qquad
E(l)=L(l)+\lambda P(l).
$$

If a maximum component count K_max is explicitly supplied, add
max(0,K-K_max) squared to P. The current experiments impose no component-count
limit and no soft term for K: they minimize sampling overhead alone. Adding
log K would instead approximate total fragment shots under one particular
execution model; it is not silently included here.

Let U_bound be the sum of the largest independent log cost for every enabled
cut variable, and s_scale the larger of one and the median positive single-cut
log cost. The default penalty is lambda = U_bound + s_scale. For multiplier
at least one, every width violation has higher energy than any feasible
independent-cut construction. Search can still visit infeasible states at
positive temperature, while the returned best solution must be feasible.

The implementation uses an LRU evaluation cache and an optional hard
cache-miss budget. That budget includes initial evaluations, temperature
pilots, main proposals, and polishing. Seed construction is outside it and
must be reported separately.

### 6.2 Units and region pools

A unit is a set of vertices linked by mandatory retained edges. With all
ordinary cuts allowed, units are individual vertices. Define expand(R) as
adding every mandatory unit touched by R.

On each qubit with temporal incidence sequence [v0, v1, ..., vt], the original
move builder creates:

1. Every consecutive interval of lengths two through `block_size`.
2. Every nonempty proper prefix and suffix for refinement.
3. In multiway mode, all those intervals/prefixes/suffixes, each entire wire,
   and both endpoints of every gate edge form the region-proposal pool.
4. Mandatory-unit expansion is applied and duplicate regions are removed.
   Units are also retained for single-unit proposals and polishing.

`block_size` bounds the short-interval construction. It does **not** bound the
size of all multiway regions: a prefix, suffix, or entire wire can be larger.
For example, on [v0,v1,v2,v3] with block_size = 2, short intervals are
[v0,v1], [v1,v2], and [v2,v3], but [v0,v1,v2] and the full wire are still
available in the multiway region pool.

The pool is fixed for a given graph, allowed-cut policy, and block_size. A
proposal selects the region pool with probability `block_probability`, or
the unit pool otherwise, then chooses uniformly within the selected pool.
When the region pool is empty it falls back to units. These are the existing
pools, with no additional excision-specific pool.

### 6.3 Color-swap proposal and Hastings correction

Keep a fixed palette of max(2, |V|) colors throughout a trajectory. After
choosing R, select a vertex uniformly from R and take its current color a.
Select target color b from a mixture:

- An occupied color other than a, uniformly, with total probability 1-p_split.
- An unused palette color, uniformly, with total probability p_split.

If only one category is available, it receives probability one. Swap a and b
at all vertices in R; other colors remain unchanged. This operation is its
own inverse. It can move a unit, shift a temporal boundary, split a component,
join components, or exchange two colors inside a larger region. Widths and
physical realizations are recomputed after each changed proposal.

Let n_a(R) and n_b(R) count colors a and b in R, and t_l(b|a) be the target
probability under state l. Both source-color orientations can generate the
same swap, so the conditional pair probability is

$$
q_R(l\to l')=
\frac{n_a(R)t_l(b\mid a)+n_b(R)t_l(a\mid b)}{|R|}.
$$

Recompute this expression in the reverse state and accept with

$$
A(l\to l')=\min\left\{1,
 \exp\left[-\frac{E(l')-E(l)}{T}\right]
 \frac{q_R(l'\to l)}{q_R(l\to l')}\right\}.
$$

The region distribution is state-independent and each conditioned proposal
uses its own correct reverse ratio. Labels are **not canonicalized inside
the chain**; doing so would change these probabilities. Downhill proposals
are not unconditionally accepted when the Hastings ratio is unfavorable.
The code compares log acceptance to avoid numerical underflow.

There is no separate excision move. A color swap changes gate/wire boundaries;
that pattern may make an excision eligible, ineligible, or replace it with
another realization. The resulting priced objective enters this same
acceptance rule. Gate/wire changes in the trajectory counters are structural
edge changes; final action counts additionally distinguish selected excisions.

The shared engine retains its optional heat-bath machinery. Current excision
candidates define no special heat-bath orbits, so this study makes zero
heat-bath proposals and does not tune the inactive setting.

### 6.4 Temperature calibration and cooling

Before the chain, propose `pilot_steps` moves from the starting state without
advancing it. Collect positive energy changes. `temperature_calibration="all"`
uses all uphill pilots; `"feasible"` uses only pilots whose start and endpoint
are both feasible. The latter prevents large overflow penalties from setting
the initial temperature when useful feasible moves have much smaller costs.

$$
T_0=\frac{\operatorname{median}(\Delta E_{\mathrm{positive}})}
{-\log(p_0)}.
$$

If there are no eligible positive samples, use s_scale in the numerator.
p_0 is `initial_acceptance`, not the measured acceptance rate of the full
Hastings-corrected chain. At level j, T_j = max(1e-12, T0 * cooling_rate^j).
The implemented schedule advances using the larger of step progress and
cache-miss budget progress over `temperature_levels` (default 120).

### 6.5 Best solution and polishing

Every evaluated feasible proposal can update the best incumbent, even if the
Markov-chain acceptance test rejects it. Pilots can also find incumbents.
Consequently, `accepted_moves` does not count every improvement remembered by
the optimizer. Equal-cost solutions are ranked by fragment count and then
cut-state tuple.

After the main loop, deterministic polishing starts from the best feasible
solution, enumerating useful color swaps over the original polishing pool.
It includes occupied targets and one representative unused color. It retains
only feasible rank improvements, including the stated tie breaks, repeats if
an improvement is found, and consumes
the same remaining evaluation budget. Once the budget is exhausted polishing
cannot perform more evaluations. We reserve budget by using fewer main-loop
steps than the maximum cache misses rather than assuming polishing is free.

### 6.6 Structural seeds and multistart

The original `seed_states` combines spatial assignments, contiguous capacity
boundaries, depth-first interaction-graph orderings, one-wire proposals,
all-gate/all-wire cut starts, capacity-respecting agglomeration, and a dynamic
program restricted to available graph bridges. Bridge optimization is exact
within that bridge family only. Agglomeration recomputes physical widths
while restoring edges; it is an initialization heuristic, not the SA kernel.

For large N the spatial candidate set uses balanced random assignments and
structural orderings. Small cases can enumerate spatial bipartitions. The
pool ranks feasible seeds by cost while preserving different wire/gate and
fragment-count signatures.

Each multistart trajectory has a fixed seed and one explicit initial cut
state from that pool. Our experiment builds this pool on the WG graph and
shares it with WGE. The two graphs are checked for identical vertices and
edges. The original SA, region, seeding, and heat-bath modules are checked
against the existing committed kernels.
No known excision separator or external-baseline winner is inserted into
these quality-study pools.

```python
from qpartition import AnnealConfig, PartitionGraph, anneal
from qpartition.annealing import seed_states

wg = PartitionGraph(problem)
wge = PartitionGraph(problem, excision=True)  # only individual 2Q excision
cfg = AnnealConfig(
    max_qubits=M, partition_mode="multiway",
    steps=2400, max_evaluations=4000, pilot_steps=48,
    temperature_calibration="feasible",
)
pool = seed_states(wg, cfg, count=5, seed=991)
runs = [anneal(wge, cfg, seed=1000+i, initial_state=pool[i % len(pool)])
        for i in range(5)]
best = min((r.best for r in runs if r.best), key=lambda e: e.log_overhead)
```

This is a minimal usage example. The selected profile keeps these settings
and uses `cooling_rate=0.98`;
other active defaults remain `block_probability=0.3`, `block_size=4`,
`split_probability=0.25`, `initial_acceptance=0.8`, and 120 temperature levels.
This is an explicit experiment configuration, not a global-default change.

## 7. Postprocessing, execution, and validation

Create a `HybridPlan` from the winning cut state and compile it with
`compile_hybrid`. The compiler removes the independent decompositions covered
by an excision and inserts its actual measure-and-prepare recipe at the
original source gate. It binds input measurements and output preparations to
the existing vertices stored in the LUT. It preserves intervening one-qubit
operations, including redirecting them onto the prepared register when an
outgoing wire boundary is absorbed.

The plan contains the source identity, cut decisions, capacity/mode, recipe
identity, selected realizations, and widths. It rejects source/recipe mismatch
or an assumed norm unsupported by the physical recipe.

For a product of selected decompositions, sample branch j_t of each factor
with probability |a(t,j_t)|/gamma_t. Execute the corresponding fragments,
multiply their measurement scores and coefficient signs, and scale by Gamma.
For bounded product-Pauli scores, an individual trial has absolute value at
most Gamma, giving a variance bound Gamma squared. Observable coefficient
norms and the desired error/confidence add their own factors. Minimizing
Omega therefore compares an explicit estimator overhead, not a measured
hardware runtime or an exact observable-dependent variance.

At large N, neither all 2^N amplitudes nor the Cartesian product of all branch
choices is enumerated. Each saved final trajectory is checked by:

- Recomputing width feasibility and N + W_raw segment conservation.
- Compiling the selected physical recipes and matching their summed log norms
  to the evaluator's log overhead.
- Materializing one full branch and matching actual circuit register sizes to
  the reported fragment widths.

These checks establish resource and recipe consistency. Small-instance tests
provide the stronger numerical oracle: full reconstruction with entangled
spectators, incoming/outgoing boundary orientations, adjacent excisions, and
intervening local gates. A single large branch check does not establish an
entire large-instance observable reconstruction numerically.

## 8. Large-circuit benchmark survey and protocol

The selected families cover different graph and cost structures without a
variational parameter-training workload:

| Family | Construction and purpose | Identity / caveat |
|---|---|---|
| Graph-state preparation | H then CZ on a seeded degree-three regular graph; sparse nonlocal connectivity | Adaptation of the MQT Bench v2.3.0 generator; remove final measurements; not an author Table-I circuit |
| Approximate QFT | CP rotations through qubit distance five, no final swaps | Own explicitly truncated generator; not full QFT |
| Hamiltonian simulation on a grid | Three layers of fixed RZZ(0.37), followed by fixed RX rotations | Own bound simulation workload; tests geometric structure and cheaper gate cuts |
| Random brickwork | Six seeded random matchings of CX with fixed local RY rotations | Own stress test; tests temporal, nonlocal CX connectivity |
| Excision separator control | Four repeated interaction modules around a single CX | Deliberately favorable synthetic control; not representative evidence by itself |

The graph-state construction is traced to the pinned [MQT Bench generator](https://github.com/munich-quantum-toolkit/bench/blob/v2.3.0/src/mqt/bench/benchmarks/graphstate.py).
No MQT package or paper artifact is silently substituted for the frozen
local generators. The exact circuit identity and provenance are saved for
every case. AQFT accuracy relative to full QFT is not evaluated; all methods
receive the same truncated circuit.

Graph-state circuits here contain only H/CZ gates and serve as structural
partitioning benchmarks. They are not evidence of classical simulation
difficulty or quantum advantage. The other three workload families contain
fixed non-Clifford rotations.

Tuning uses five 64-qubit cases with generator seed 17 and three trajectories
per profile. Test cases use sizes 50, 100, 150, and 250 with generator seed 53;
the deterministic AQFT tuning size is also distinct from test sizes. There
are 16 ordinary test cases at M = floor(3N/4), four tighter AQFT/brickwork
cases at M = floor(N/3), and two controls at their module width. This is a
holdout split in sizes/seeds, not a claim of transfer to unrelated families.

The tuning capacities are 48 for graph states and random brickwork, 22 for
AQFT and the grid, and 17 for the separator control. Each profile sees the
same first three members of the shared five-state pool. Final runs use all
five members; therefore training and final seed mixtures are not identical.
The larger final budget also changes the cooling trajectory. Both are limits
on how strongly the tuning choice can be generalized.

The hyperparameter sweep includes ten profiles: original defaults, feasible
calibration, faster/slower cooling, more/fewer region proposals, longer short
intervals, more/fewer splits, and a combined setting. All tuning runs use
1600 cache misses at most, 960 main steps, and 48 pilots. Choose the profile
first by failure count and then by mean across cases of median-start
log10-overhead regret relative to the best median profile on that case.
An alphabetical tie-break is fixed in the runner. Test scores do not select
hyperparameters.

Final WG and WGE runs use the selected profile, five matched starts, at most
4000 cache misses each, 2400 main steps, and 48 pilots. The budget includes
polishing. Seed generation is shared and separately timed. These settings
are a bounded quality study, not exhaustive optimization.

### External baselines

**Qiskit:** call the installed official `qiskit_addon_cutting` 0.10.0
`find_cuts` with both gate and wire local operations enabled, max_backjumps =
10000 and max_gamma = 1e100. The latter is a bound on the unsquared norm,
not an excision discount. Results are independently materialized with
`cut_wires` and `partition_problem` and checked for widths and recipe norms.
See the [official cutting API](https://qiskit.github.io/qiskit-addon-cutting/apidocs/qiskit_addon_cutting.html)
and [automated-cut tutorial](https://qiskit.github.io/qiskit-addon-cutting/tutorials/04_automatic_cut_finding.html).

**2608.05287 without GG:** run our existing two-stage KL reconstruction:
interaction-graph spatial bisection with gate-count weights, then temporal
refinement with logarithmic independent gate/wire costs. Use balanced spatial
starts, max_iter = 10, and physical static-width guards. The full topology
includes explicit input/output vertices; it is the baseline's graph, not a
replacement for our contracted graph. The [paper, Sections V-A/B and VI](https://arxiv.org/html/2608.05287v1#S5),
specifies the architecture and a 50-run evaluation, but does not pin all
initialization and physical-capacity details. Accordingly these are results
of our explicit reconstruction, not the authors' executable or their claimed
Table-I numbers.

Both baseline methods request 50 restarts per case, with a 60-second worker
deadline per method/case. Only completed restarts contribute an incumbent.
Completed counts, errors, optimization time, and wall time are retained.
The dense temporal KL reconstruction and its deadline can constrain how many
restarts complete at larger sizes. This is a limit of this experiment and
implementation; it does not refute the paper's scalability claim.

For the tight test sizes, floor(N/3) times three is smaller than N. Even
without extra wire segments, these 100/250-qubit inputs require at least four
fragments; a maximum width is not an instruction to create exactly three.

The paper baseline is restricted to two blocks. When N > 2M it is reported as
proven infeasible for that formulation, rather than assigned an infinite
quality score. Official Qiskit and our SA permit multiple fragments. Even at
M = floor(3N/4), report actual fragment counts; a method is not guaranteed to
return exactly two fragments merely because two suffice.

Unequal baseline restart counts and different search budgets preclude an
equal-runtime ranking. Numerical comparisons use only valid completed plans;
errors, timeouts without incumbents, and two-way infeasibility are statuses,
not wins. Published small Table-I values are not extrapolated to 50–250-qubit
circuits for which the paper reports no corresponding overhead table.

## 9. Recorded results and interpretation

The completed study contains 150 tuning trajectories, 220 matched main
trajectories, and 20 separate default-profile confirmation trajectories.
Every main method/case has five feasible starts. The results, raw diagnostics,
compact winner plans, and audits are in
[`validation/excision_quality_v1`](../validation/excision_quality_v1/SUMMARY.md).
The [full report](../validation/excision_quality_v1/report.json),
[excision attribution](../validation/excision_quality_v1/analysis.json),
[tuning scores](../validation/excision_quality_v1/tuning.json), and
[physical audit](../validation/excision_quality_v1/audit.json) retain the
numbers behind the discussion below. All operations use physical recipe
norms; no gate grouping is enabled.

### 9.1 All test cases

Every numerical entry below is **log10(Omega)**. Baseline restart counts and
statuses are part of the result. The paper column refers to our no-GG
reconstruction. A dash with timeout/infeasibility is not a numerical loss.

| Circuit | N | M | WG | WGE | Excisions | Fragments | Qiskit | Paper reconstruction |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| graphstate_50_M37 | 50 | 37 | 6.6797 | 6.6797 | 0 | 2 | 25.7645 (4 starts; time-limit-with-incumbent) | 8.5882 (50 starts; completed) |
| graphstate_100_M75 | 100 | 75 | 15.2679 | 15.2679 | 0 | 2 | 42.9409 (1 starts; time-limit-with-incumbent) | 17.1764 (22 starts; time-limit-with-incumbent) |
| graphstate_150_M112 | 150 | 112 | 22.9018 | 22.9018 | 0 | 2 | 66.7970 (1 starts; time-limit-with-incumbent) | 27.6730 (8 starts; time-limit-with-incumbent) |
| graphstate_250_M187 | 250 | 187 | 45.8036 | 45.8036 | 0 | 2 | — (0 starts; timeout) | 54.3918 (2 starts; time-limit-with-incumbent) |
| aqft_50_M37 | 50 | 37 | 3.0618 | 3.0618 | 0 | 2 | 3.6398 (14 starts; time-limit-with-incumbent) | 3.6398 (11 starts; time-limit-with-incumbent) |
| aqft_100_M75 | 100 | 75 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (7 starts; time-limit-with-incumbent) | 3.6398 (2 starts; time-limit-with-incumbent) |
| aqft_150_M112 | 150 | 112 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (6 starts; time-limit-with-incumbent) | — (0 starts; timeout) |
| aqft_250_M187 | 250 | 187 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (5 starts; time-limit-with-incumbent) | — (0 starts; timeout) |
| trotter_grid_50_M37 | 50 | 37 | 9.9264 | 9.9264 | 0 | 2 | 9.9264 (6 starts; time-limit-with-incumbent) | 11.3445 (15 starts; time-limit-with-incumbent) |
| trotter_grid_100_M75 | 100 | 75 | 14.1806 | 14.1806 | 0 | 2 | 14.1806 (2 starts; time-limit-with-incumbent) | 14.1806 (2 starts; time-limit-with-incumbent) |
| trotter_grid_150_M112 | 150 | 112 | 17.0167 | 17.0167 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| trotter_grid_250_M187 | 250 | 187 | 21.2709 | 21.2709 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| random_brickwork_50_M37 | 50 | 37 | 36.0566 | 36.0566 | 0 | 2 | 47.7121 (1 starts; time-limit-with-incumbent) | 30.7856 (18 starts; time-limit-with-incumbent) |
| random_brickwork_100_M75 | 100 | 75 | 74.5215 | 74.5215 | 0 | 3 | 83.9733 (1 starts; time-limit-with-incumbent) | 61.0715 (3 starts; time-limit-with-incumbent) |
| random_brickwork_150_M112 | 150 | 112 | 110.7827 | 109.8284 | 1 | 2 | — (0 starts; timeout) | 103.0582 (1 starts; time-limit-with-incumbent) |
| random_brickwork_250_M187 | 250 | 187 | 197.6640 | 197.6640 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| aqft_100_M33 | 100 | 33 | 9.0618 | 9.0618 | 0 | 4 | 9.0618 (10 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| aqft_250_M83 | 250 | 83 | 9.0618 | 9.0618 | 0 | 4 | 9.0618 (3 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| random_brickwork_100_M33 | 100 | 33 | 129.2084 | 128.5041 | 3 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |
| random_brickwork_250_M83 | 250 | 83 | 356.8867 | 356.8867 | 0 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |
| control_100_M26 | 100 | 26 | 3.3625 | 2.4082 | 1 | 4 | 138.3652 (1 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| control_250_M63 | 250 | 63 | 3.3625 | 2.4082 | 1 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |

Across the 20 ordinary cases, WGE improves on WG twice and ties it 18 times;
two ordinary winners contain excision. Both synthetic controls improve from
2304 to 256. Increasing qubit count alone does not consistently improve the
relative usefulness of excision.

On the valid completed external comparisons, WGE is better/equal/worse than
Qiskit on 7/7/0 of 14 comparable cases (including one synthetic control),
and better/equal/worse than the KL
reconstruction on 6/2/3 of 11. In particular, the KL baseline remains better
on all three comparable random-brickwork cases at the wide bound. Thus these
results do not establish that our optimizer dominates the paper method.
Different budgets and completed restart counts further limit such rankings.

### 9.2 Where excision actually helps

The strongest search-attribution example is `random_brickwork_100_M33`:

| Quantity | WG | WGE |
|---|---:|---:|
| Independent gate cuts | 119 | 117 |
| Independent wire cuts | 13 | 8 |
| Two-qubit excisions | 0 | 3 |
| Raw structural gate cuts | 119 | 120 |
| Raw structural wire cuts | 13 | 14 |
| Fragment widths | 14, 33, 33, 33 | 15, 33, 33, 33 |
| log10 sampling overhead | 129.208418 | 128.504053 |

For this CX-only two-qubit gate set, the actual costs are

$$
\Omega_{\mathrm{WG}}=9^{119}16^{13},
\qquad
\Omega_{\mathrm{WGE}}=9^{117}16^8 256^3,
\qquad
\frac{\Omega_{\mathrm{WG}}}{\Omega_{\mathrm{WGE}}}
=\frac{81}{16}=5.0625.
$$

WGE accepts one additional structural gate cut and one additional wire cut,
which would multiply the independent-cut cost by 144. Three excision
substitutions save a factor of 729. The net result is a lower-cost partition
that independent edge weights would not prefer.

Postprocessing the WG winner with the excision LUT finds no eligible excision
and gives no reduction. The WGE result therefore improves beyond repricing
that winner. However, attribution inside the solver matters: both winning
trajectories start at log10 overhead 138.365164, and that is still their best
cost before polishing. The improvement is found by the existing deterministic
polishing stage with the excision-aware evaluator, not by the annealing
chain's incumbent improvements. This supports the integrated cost formulation
and refinement mechanism; it does not independently demonstrate an annealing
acceptance advantage.

For `random_brickwork_150_M112`, one excision gives a ninefold reduction, from
log10 overhead 110.782666 to 109.828423. The WG winner can achieve that same
cost through post-only excision selection, with widths 112 and 46. It is a
real decomposition saving, but not evidence of additional search benefit.

The 100- and 250-qubit separator controls give overhead 256 instead of 2304,
with widths 25/26/25/26 and 63/63/63/63 respectively. Their best seeds already
contain the useful separator pattern, and postprocessing the WG winner also
achieves 256. They validate physical integration and pricing, not discovery
of a new partition by SA. No known separator masks were supplied to seeding.

The post-only diagnostic for every case is in the result summary. Overall,
co-optimization improves beyond repricing the WG winner in one case.

### 9.3 Hyperparameter findings and recommendation

The sweep selected feasible-only pilot calibration with cooling rate 0.98.
The original profile's mean median regret is 0.61356 log10 units, compared
with 0.23186 for the selected profile. This is a training-set statistic;
zero regret would mean matching the best profile median on every case.
All 150 tuning runs returned a feasible incumbent.

| Profile | Changed settings (others default) | Mean median log10 regret |
|---|---|---:|
| slow_cooling | temperature_calibration=feasible, cooling_rate=0.98 | 0.23186 |
| long_intervals | temperature_calibration=feasible, block_probability=0.6, block_size=8 | 0.57516 |
| original | Original defaults | 0.61356 |
| combined | temperature_calibration=feasible, cooling_rate=0.95, initial_acceptance=0.6, block_probability=0.6, block_size=8, split_probability=0.12 | 0.62514 |
| feasible_temperature | temperature_calibration=feasible | 0.80702 |
| fewer_regions | temperature_calibration=feasible, block_probability=0.1 | 0.80702 |
| more_regions | temperature_calibration=feasible, block_probability=0.7 | 0.80702 |
| more_splits | temperature_calibration=feasible, split_probability=0.5 | 0.80702 |
| fast_cooling | temperature_calibration=feasible, cooling_rate=0.88 | 0.85700 |
| fewer_splits | temperature_calibration=feasible, split_probability=0.08 | 0.85700 |

The separate held-out confirmation gives a more cautious picture:

| 100-qubit case | Original best / median log10 | Selected best / median log10 |
|---|---:|---:|
| graphstate_100_M75 | 15.2679 / 19.0849 | 15.2679 / 19.0849 |
| aqft_100_M75 | 3.2493 / 3.6398 | 3.6398 / 3.9305 |
| trotter_grid_100_M75 | 14.1806 / 106.6727 | 14.1806 / 102.9207 |
| random_brickwork_100_M75 | 74.5215 / 108.6460 | 74.5215 / 102.4208 |

Best-of-five costs tie on three confirmation cases, while the original profile
is better on AQFT. Selected-profile medians improve on grid and brickwork,
tie on graph state, and worsen on AQFT. The original AQFT best is about
2.46 times cheaper than the selected best. Retain the original constructor
defaults, and treat feasible/0.98 as an explicit optional profile rather than
a universal replacement. The main ablation keeps the preselected profile;
confirmation trajectories are excluded from its five-start winners.

There is substantial sensitivity to seed family: the grid's median is many
orders of magnitude worse than its best start. Report both best and median
quality. A follow-up tuning study should use the same seed-family mixture
and budget in tuning and final evaluation, and measure annealing and polishing
contributions separately. The current study uses three of five seed-pool
members for tuning and all five for final runs, with different budgets.

At matched evaluation budgets, the sum of observed per-trajectory wall times
is 2089.6 seconds for WG and 15257.9 seconds for WGE; actual cache misses are
414785 and 414578 respectively. These are sums across concurrently executed
trajectories, not end-to-end elapsed times or controlled CPU-time rankings.
They nevertheless identify a significant implementation cost for realization
evaluation. Eligibility indexing and reuse of unchanged width allocations
are worthwhile next profiling targets while preserving the graph and search
kernel. No such acceleration is silently included in these results.

### 9.4 Limits and verification

The natural workloads with excision gains still have extremely large absolute
sampling overhead. These plans demonstrate optimizer/decomposition behavior,
not practical quantum execution at those costs. A high heuristic upper bound
also does not prove that every partition of that circuit is impractical.

All 44 main winner plans are independently recompiled and checked for recipe
norms, materialized widths, segment conservation, source identities, and
reproducible input regeneration. Every final trajectory also has its own
physical recipe/branch audit. All 11 valid paper-baseline winners are
materialized independently through Qiskit and their norms/widths checked.
The 14 valid official-Qiskit winners were materialized by the baseline adapter.
Large quantum states and all global branch combinations are not simulated.
The 157 existing tests and an additional large-cost overflow regression pass;
lint and whitespace checks also pass.

## 10. Reproduction and source map

```bash
conda activate quantum-partition
python examples/run_excision_quality.py --output results/excision-quality-rerun --workers 6
python examples/check_excision_quality.py --directory results/excision-quality-rerun
python -m pytest -q
ruff check src tests examples
```

Individual runner phases are `prepare`, `tune`, `final`, `baselines`, and
`summarize`, with a separate `confirmation` diagnostic phase. Completed
per-task checkpoints can be resumed with the same configuration; use a fresh output directory for changed seeds, budgets,
source code, or profiles. The metadata manifest records package versions,
source hashes, original-kernel checks, split definitions, and arguments.
Frozen QPY inputs and per-trajectory plan dumps are locally retained but
ignored by Git. Compact winning plans in `winners/`, diagnostics, and reports
remain reviewable. The audit can regenerate missing inputs from provenance.
The benchmark generators reconstruct the frozen inputs from recorded
provenance, and fingerprints verify that reconstruction.

| File | Responsibility |
|---|---|
| `src/qpartition/circuits.py` | Input validation, snapshots, semantic fingerprints, QPY I/O |
| `src/qpartition/partition_graph.py` | Contracted graph, label-to-cut conversion, components and segment widths |
| `src/qpartition/templates.py` | Independent cut recipes and checked physical excision recipe cache |
| `src/qpartition/excision.py` | PTM and product measure/prepare decomposition |
| `src/qpartition/grouping.py` | Shared realization catalogue/selector; this workflow enables only its single-gate `excision` family |
| `src/qpartition/label_moves.py` | Original unit/temporal-region pools and Hastings-corrected color swaps |
| `src/qpartition/multiway.py` | Original bridge and agglomerative seed helpers |
| `src/qpartition/annealing.py` | Original seeding, pilots, cooling, acceptance, caching, and polishing |
| `src/qpartition/hybrid.py` | Plan validation, physical compilation, lazy sampled reconstruction |
| `src/qpartition/large_benchmarks.py` | Frozen scalable workload/control generators |
| `src/qpartition/qiskit_baseline.py` | Official Qiskit optimizer adapter and independent materialization checks |
| `src/qpartition/paper_baseline.py` | Explicit two-stage KL reconstruction |
| `examples/run_excision_quality.py` | Holdout experiment, profile sweep, deadlines, checkpoints, and result tables |
| `examples/check_excision_quality.py` | Independent input regeneration, winner compilation, and SDK baseline checks |
| `tests/test_incidence_excision.py` | Small-instance physical/resource and integration correctness checks |

The module name `grouping.py` and legacy serialized field names are retained
for compatibility. Enabling only `PartitionGraph(problem, excision=True)`
activates individual-gate excision realizations; it does not enable
multi-gate grouping.
