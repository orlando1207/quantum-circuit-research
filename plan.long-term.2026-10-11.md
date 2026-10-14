# Quantum Circuit Partitioning: Research Plan

Date: 2026-10-11

This document summarizes the proposed research direction and literature survey.
See [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) for the detailed analysis and
[RESEARCH_FOUNDATION.md](RESEARCH_FOUNDATION.md) for the supplied-paper review.
The directions below are hypotheses to investigate; novelty and practical
advantage have not yet been established. Several recent references are preprints.

## 1. Research objective

The conventional problem takes a circuit and fragment width/depth constraints,
then minimizes sampling overhead. Our recommended direction is
**decision-oriented partitioning for related parameterized circuits**:

> Find partitions and measurement plans that minimize the cost of selecting a
> sufficiently good circuit candidate with a specified confidence.

Examples include choosing a variational parameter setting or deciding whether
an optimizer update improves the energy. Wire cutting, gate cutting, physical
gate excision, and eligible gate groups are available actions. Their combination
supports the research but is not sufficient as its novelty claim.

## 2. Survey summary

| Topic already explored | Representative primary references | Implication |
| --- | --- | --- |
| Mixed wire/gate cuts and gate groups | [Optimal Partitioning](https://arxiv.org/abs/2308.09567), [Gate-group framework](https://arxiv.org/abs/2608.05287) | A new graph heuristic needs an additional objective, mechanism, or guarantee. |
| Variance-aware shot allocation | [ShotQC](https://arxiv.org/html/2412.17704v3), [Noise-aware allocation](https://arxiv.org/html/2607.24704v1) | Allocation alone has direct overlap; ShotQC assumes a given partition. |
| Heterogeneous hardware and adaptive execution | [MOSAIQC](https://arxiv.org/abs/2607.18888), [MAESTROCUT](https://arxiv.org/abs/2509.00811) | Hardware awareness and adaptivity alone are crowded directions. |
| Observable-specific cutting and backpropagation | [Tomography-based knitting](https://arxiv.org/html/2512.19623v1), [OBP cutting](https://arxiv.org/abs/2510.19467) | Preserving only an observable or combining OBP with cutting is already studied. |
| Observable subspaces | [Dynamic Observable Subspaces](https://arxiv.org/abs/2609.09806) | Interface compression needs comparison with classical simulation methods. |
| Repeated structures and result reuse | [CiFold](https://arxiv.org/abs/2412.18705), [Semantic circuit cache](https://arxiv.org/abs/2604.26788) | Caching and repetition alone are insufficient novelty claims. |
| Classical/quantum execution and approximation | [SuperSim](https://arxiv.org/abs/2303.10788), [Qdislib](https://arxiv.org/abs/2505.01184), [Approximate cutting](https://arxiv.org/abs/2212.01270) | Any extension should contribute an explicit global cost/error formulation. |
| Application-level training and sampling | [CutReg](https://arxiv.org/abs/2506.14858), [DistributedEstimator](https://arxiv.org/abs/2602.16233), [QAOA sampling](https://arxiv.org/abs/2507.06649) | Applying cutting to an application is already explored. |

The tomography paper's favorable scaling concerns structured circuits and
depends on boundary dimension, learning resources, and conditional preparation.
It does not establish inexpensive cutting for arbitrary circuits.

## 3. Recommended formulation

**Input:** related circuits $C_{\theta_1},\ldots,C_{\theta_M}$, initial state
$\rho$, score observable $H$, fragment width/depth limits, a classical-memory
budget, acceptable suboptimality $\eta>0$, and failure probability $\delta$.

Define $E_k=\operatorname{Tr}[H C_{\theta_k}(\rho)]$.

**Output:** feasible partitions, executable decompositions, shared experiment
mappings, a shot-allocation/stopping policy, and selected candidate $\hat k$.

$$
\min_\pi \mathbb E[T_\pi]
\quad\text{subject to}\quad
\Pr\left(E_{\hat k}\leq\min_k E_k+\eta\right)\geq1-\delta.
$$

Begin with total physical fragment shots as the cost. Later, include planning,
compilation, pilot measurements, execution, reconstruction, and peak memory.

### Proposed mechanism

If an exact fixed physical experiment basis gives

$$
E(\theta)=\sum_i a_i(\theta)\mu_i,
$$

then a comparison can be reconstructed directly:

$$
\Delta=E(\theta_A)-E(\theta_B)
=\sum_i\bigl[a_i(\theta_A)-a_i(\theta_B)\bigr]\mu_i.
$$

Common contributions cancel. With shared data,

$$
\operatorname{Var}(\hat\Delta)
=\operatorname{Var}(\hat E_A)+\operatorname{Var}(\hat E_B)
-2\operatorname{Cov}(\hat E_A,\hat E_B).
$$

A cut or excision can expose reusable fragment responses, potentially changing
which partition is best. A common basis may also cost more than separate
decompositions; the optimizer must resolve this tradeoff.

**Potential contribution:** jointly selecting partitions and decompositions for
confidence-qualified comparisons, beyond fixed-partition allocation and caching.
The targeted survey did not identify an exact match, but citation tracing and
searches on correlated estimators, best-arm identification, parameter-shift
gradients, and multi-fidelity optimization remain necessary.

## 4. Other research directions

| Direction | Potential contribution | Priority |
| --- | --- | --- |
| Query-family boundary interfaces | Jointly choose boundaries and physically realizable operator interfaces sufficient for multiple, possibly noncommuting observables. Include interface construction cost. | Theoretical extension |
| Certified approximation with mixed execution | Allocate a global bias/variance budget across exact cutting, approximate transfer information, and classical/quantum execution. | After exact statistical validation |
| Feasibility certificates | Return a feasible partition or a justified lower bound within a declared protocol and decomposition model. [Side-information cutting](https://arxiv.org/abs/2503.22384) supplies relevant local formulations. | Longer-term theory |

## 5. Graph representations

Use a **temporal port graph for circuit semantics**, a compact graph/hypergraph
for partition search, and two derived graphs for reconstruction and shared
experiments. They share stable operation, port, fragment, and experiment IDs.
The graph is an optimization representation; the physical experiment compiler
remains responsible for validating the resulting plan.

### 5.1 Circuit semantics and search graph

For each operation $g$ acting on qubits $Q_g$, create an operation node and
input/output ports $(g,q,\mathrm{in})$, $(g,q,\mathrm{out})$ for every $q\in Q_g$.
Add directed wire edges from one operation's output port to the next operation's
input port on the same qubit. Include circuit input/output nodes. The original
unitary circuit has a directed acyclic dependency graph.

Store gate type, parameters, operand order, original operation ID, and allowed
decompositions on each operation. Wire edges store their qubit and position.
Keep single-qubit operations: they affect execution, observables, and group
eligibility even when they add no interaction edge.

The existing `temporal_graph()` in [circuits.py](src/qpartition/circuits.py)
already supplies this port structure. It needs richer optimization annotations,
not replacement with a qubit-only graph.

For initial standard-cut search, form a compact representation:

- A vertex $v_{g,q}$ represents the local input/output strand of gate $g$ on $q$.
- Temporal edges $E_w$ connect consecutive strands on each qubit.
- A gate factor $h_g=\{v_{g,q}:q\in Q_g\}$ couples the operands of $g$.
  Two-qubit factors can be ordinary interaction edges; larger gates should
  retain a hyperedge rather than be charged as unrelated pairwise interactions.

Thus the search representation is $\mathcal H=(V,E_w,\{h_g\})$. A partition label
$p:V\rightarrow\{1,\ldots,K\}$ assigns strands to fragments. Operation-factor
nodes are not qubits and must not count toward width. This compact view assumes
local input/output continuity; excision requires returning to separate ports.

For two consecutive two-qubit gates, the search view is:

```text
q0: input --> v(g1,q0) --> v(g2,q0) --> output
                  | h_g1       | h_g2
q1: input --> v(g1,q1) --> v(g2,q1) --> output
```

Horizontal connections are possible wire-cut locations. Vertical connections
are gate factors with gate-cut choices; they are not temporal wire edges.

Use the existing qubit interaction graph only for initialization: its vertices
are original qubits and its edges summarize interactions. It loses temporal
wire-cut choices, and its current interaction-count weights are not physical
sampling costs. A DAG formulation and exact bounded-width cut placement already
have direct precedent in [Quantum Circuit Cutting: Complexity and
Optimization](https://arxiv.org/abs/2604.23700).

### 5.2 Cutting operations as legal graph transformations

| Action | Graph interpretation | Required physical check |
| --- | --- | --- |
| Wire cut | Break a temporal edge and add a reconstruction factor between its two boundary ports. Different fragment labels across a wire require a cut. | Compile the upstream measurement and downstream preparation; account for the new wire segments. |
| Gate cut | Replace a crossing gate factor with its supported local instrument factors and a shared decomposition index. Local wire continuity is retained where the decomposition permits it. | Verify gate support, operand order, and permitted local operations. |
| Gate excision | Replace the entire operation by its input/output transfer factor; incoming and outgoing ports can become separate quantum segments. | Compile measured inputs and physically prepared outputs; deleting an interaction edge alone does not implement excision. |
| Joint gate-group cut | Replace several eligible gate decompositions with one joint factor. | Check the complete group template, intervening operations, partition assignment, and execution capabilities. |

The search state contains both fragment labels and explicit action/decomposition
choices. Labels alone cannot describe every useful excision or wire break.
An uncut multiqubit gate must remain wholly executable within one fragment;
every crossing gate needs a supported replacement. Do not force all original
operations to be atomic search vertices, since that would hide gate-cut choices.

After materialization, enforce

$$
\operatorname{width}(F_k)\leq Q_k,
\qquad
\operatorname{depth}(F_k)\leq D_k.
$$

Initially, width means allocated qubits without an additional qubit-reuse pass.
Count the actual cut-induced wire segments and ancillas, not gate vertices or
distinct original qubit names. Evaluate depth on executable fragment circuits,
including cut instruments and any declared compilation policy. Take width and
depth maxima over the generated experiments for each fragment. Original DAG
acyclicity does not guarantee an acyclic fragment quotient; impose additional
feed-forward constraints only when the selected communication protocol needs
them. Independently executed fragments can have cyclic reconstruction graphs.

### 5.3 Baseline weights and gate-group conflicts

For independent physical decompositions $a$, a baseline objective is

$$
L(P)=2\sum_{a\in A(P)}\log\gamma_a,
\qquad
\gamma_a=\sum_i|c_{a,i}|.
$$

This is the logarithm of the product estimator's worst-case sampling factor.
Use the actual physical decomposition norm; PTM nonzero counts are different.

For an eligible group $B$, replace its individual contributions by
$2\log\gamma_B(P)$. The cost depends on how the group intersects the partition.
Generate candidates from validated algebraic templates, with graph pattern
matching and dependency checks to locate them. Graph proximity alone does not
prove eligibility. This overlaps with the supplied
[gate-group framework](https://arxiv.org/abs/2608.05287).

Build a conflict graph whose vertices are candidate groups. Connect groups that
overlap in covered gates or have incompatible execution requirements. For fixed
partition and additive group savings, selection becomes a weighted set-packing
problem; an independent-set formulation applies when pairwise conflicts describe
all incompatibilities. Charge each gate exactly once. Joint placement and group
selection require reevaluating savings and validity after a partition move.

### 5.4 Reconstruction and experiment/query graphs

For each compiled plan, build a factor graph with fragment-response tensors and
decomposition factors. Its indices are cut settings/outcomes with recorded
dimensions. Use variable elimination or contraction-tree search to estimate
reconstruction work and intermediate memory. The relevant complexity depends
on index dimensions and elimination order, not simply the number of cuts.
[cotengra](https://cotengra.readthedocs.io/en/main/trees.html) provides contraction
trees and optimization mechanisms appropriate for this derived representation.

Build a separate bipartite graph between physical experiments and candidate
queries. Each experiment records its fragment circuit, input preparation,
measurement, score rules, shot batch, and execution context. Query edges record
how those results enter reconstruction. Reuse requires matching these semantics;
a shared graph hash is a lookup aid rather than an equivalence proof. Keep
independent multiplicative fragment estimates separate even when circuit
descriptions coincide.

### 5.5 Decision objective and refinement gains

For an exact linear comparison estimator based on independent experiment
batches, write $\hat\Delta=\sum_e\beta_e\bar X_e$. Experiment $e$ has per-shot
variance $s_e^2$, $N_e$ shots, and per-shot cost $t_e$. Then

$$
\operatorname{Var}(\hat\Delta)=\sum_e\frac{\beta_e^2s_e^2}{N_e}.
$$

In the continuous allocation model, ignoring fixed launch costs, the minimum
cost for variance at most $v$ is

$$
J_v(P)=\frac{\left(\sum_e|\beta_e|s_e\sqrt{t_e}\right)^2}{v}.
$$

For multiple scores obtained from one experiment, combine them into $X_e$ or
retain their within-experiment covariance. Product reconstruction needs a
separate analysis. This variance objective is a search surrogate, not by itself
a confidence or adaptive-stopping certificate. Use charged pilot data or valid
upper bounds for unknown variances, never the exact evaluation reference.

For a proposed graph move $m$, evaluate the gain

$$
\operatorname{gain}(m)=J(P)-J(P_m),
$$

subject to physical feasibility and the reconstruction-memory budget. Recompute
affected groups, response coefficients, and reuse relationships. This gain is
generally nonadditive; fixed edge weights cannot capture it completely.

### 5.6 Algorithms to consider

| Mechanism | Role and proposed extension | Priority |
| --- | --- | --- |
| Exact enumeration, MILP, or SMT | Establish optima on small instances under an explicit supported action set. Use enumeration with a compiled evaluator when the decision objective is nonlinear. | First baseline |
| Kernighan–Lin / Fiduccia–Mattheyses style refinement | Move or swap boundary strands, shift wire cuts, switch decompositions, and toggle groups. Replace ordinary edge-cut gain with the validated decision-cost gain. | Main initial algorithm |
| Multilevel graph/hypergraph partitioning | Coarsen, partition, then refine. Keep parameterized/query-sensitive boundaries available; use classical balance as a proxy followed by physical validation. | Scaling extension |
| Max-flow/min-cut and separator search | Generate or refine local boundary proposals for additive surrogates. Width constraints, groups, and covariance prevent treating ordinary min-cut as an exact solver for the full problem. | Supporting mechanism |
| Group conflict-graph optimization | Select compatible joint-cut templates using greedy selection or a small exact set-packing model. | After group execution is validated |
| Contraction ordering and separator-based elimination | Optimize classical reconstruction jointly with cut proposals; consider bounded-separator dynamic programming for restricted instances. DP states must retain all needed quantum/estimator boundary information. | Memory-aware extension |

Multilevel coarsening, local refinement, and flow refinement have strong general
precedent in [KaHyPar](https://arxiv.org/abs/2106.08696). Its standard cut-net and
connectivity objectives are useful baselines, but do not encode our complete
decision cost. [NetworkX minimum cut](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.minimum_cut.html)
solves the ordinary capacitated source–sink problem. Community or spectral
partitioning can supply additional seeds. Simulated annealing is an optional
search baseline; learning-based proposal policies should wait until we have
reliable labels, strong classical baselines, and evidence of a search bottleneck.

### 5.7 Concrete first graph experiment

Consider four qubits with local pairs $(q_0,q_1)$ and $(q_2,q_3)$. Apply local
state preparation and a CX within each pair, then a parameterized
$RZZ(\theta)$ on $(q_1,q_2)$, followed by fixed single-qubit rotations. Compare
two values of $\theta$ using the same observable.

- **Plan A:** keep the two local pairs as fragments and cut or excise the central
  parameterized gate. A fixed physical basis can expose shared responses.
- **Plan B:** wire-cut $q_1$ and $q_2$ before the central gate and execute its
  downstream portion in a third two-qubit fragment. Its changing parameter is
  now inside an executed fragment and generally prevents cross-candidate reuse
  of that fragment's measurements.

Both plans have at most two qubits per fragment. Compute their individual-score
costs and comparison costs; do not assume either is superior. Use this example
to check port accounting, legal transformations, reconstruction, and reuse.

The first search pipeline should be: interaction-graph or temporal seeds,
legal port-graph moves, materialized feasibility checks, common-experiment
mapping, decision-cost evaluation, and best-plan selection. Screen proposals
with inexpensive surrogates and compile promising ones. Benchmark this pipeline
against small-instance enumeration and conventional sampling-cost refinement.

The intended research contribution is the **decision-aware gain function and
its interaction with legal cut transformations**, supported by measured savings
and possibly restricted-model guarantees. Port graphs, hypergraphs, multilevel
partitioning, and local refinement are established ingredients.

## 6. First experiment

Use 4–8 qubit circuits with a 2–4 qubit fragment limit. Candidate circuits should
initially differ in one small parameterized block. Include local observables and
observables spanning partition boundaries; vary parameter separation and shared
circuit structure. Use exact uncut results only for evaluation.

Enumerate a small set of validated manual plans before building a new optimizer.
Compare:

1. Lowest realized physical QPD norm with separate candidate estimation.
2. The same plan with improved fixed-plan shot allocation.
3. The same plan with exact experiment reuse and covariance-correct comparisons.
4. A plan chosen for comparison cost, allowing extra cuts or excision.
5. OBP cutting, suitable classical simulation, and an uncut finite-shot reference
   where applicable.

Measure physical shots required for reliable selection, wrong-selection rate
over independent repetitions, confidence coverage, bias, unique experiments,
realized decomposition norm, timing, and peak memory. Charge pilot and rejected
plan costs when introducing adaptivity.

**Go/no-go criterion:** the decision objective should change the ranking of
feasible plans and reduce actual cost against the strong reuse-and-allocation
baseline. If it does not, revise the formulation before developing a large
optimizer.

## 7. Work sequence and current capabilities

1. Check the proposed novelty against the closest papers and adjacent statistics
   literature; fix the execution model and success criterion.
2. Validate a common physical basis and comparison estimator on small circuits.
3. Run the manual-plan experiment and isolate gains from allocation, reuse,
   reconstruction, and partition choice.
4. Develop a graph-based optimizer only after the experiment supports the
   formulation.
5. Extend to query families, certified approximation, and noisy hardware when
   justified by the results.

The current package supports manual wire/gate/mixed cutting, physical excision,
exact and finite-shot execution, experiment mappings, and covariance-aware
bootstrap. Cross-candidate experiment accounting, decision reconstruction, and
valid stopping rules still need development. Joint gate-group execution also
requires a validated implementation before inclusion in experiments.

## 8. Conditions for valid conclusions

- Reuse requires identical physical experiments; a changed fragment parameter
  generally prevents direct reuse.
- Preserve independence between multiplicative fragment estimates or derive an
  unbiased correction. Linear covariance formulas do not cover every product
  reconstruction.
- Bootstrap is an exploratory uncertainty tool, not a rigorous sequential
  stopping certificate.
- Report physical shots and empirical error; $\gamma^2$ is an estimator bound,
  not a universal measured sampling-cost ratio.
- Include approximation amplification, observable normalization, and classical
  processing costs.
- Fix the local-operation/communication model. Noisy-output confidence alone
  does not certify ideal-circuit accuracy.
- A high-cost heuristic solution cannot prove infeasibility; lower bounds must
  apply to the declared task and model.

Working title: **Decision-Oriented Quantum Circuit Partitioning with Shared
Fragment Measurements**.
