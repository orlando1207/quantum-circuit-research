# Design Proposal: Simulated Annealing for Mixed Circuit Cutting

This is a methodology proposal for review, not an implementation. It follows
the user's current [plan.md](plan.md) and the typed graph in
[GRAPH_REPRESENTATION_REVIEW.md](GRAPH_REPRESENTATION_REVIEW.md). The plan and
production code are unchanged. Wire cutting, gate cutting, and whole-gate
excision are included; grouping and query-dependent costs are excluded.

## 1. Recommended design

Use **action-space simulated annealing**. Its state specifies cutting actions;
the graph transformations determine connectivity, allocation segments, and
fragment count. Use the exact independent-template log-overhead as the primary
objective and width violations as penalties during search. Return only the best
validated feasible state.

This naturally supports multiple fragments. For the initial two-way experiment,
the proposed policy is **at most two executable static blocks**, obtained by
packing the connected components into A/B. If exactly two connected components
are intended instead, that is a different selectable constraint. Do not silently
equate either interpretation with two devices scheduling arbitrary fragments.

Fragment count is not an extra penalty in the baseline. Use it as a reported
metric and a tie-breaker, or impose a declared limit. An execution-cost extension
can give it a justified coefficient later.

## 2. Preprocessing and lookup tables

1. Validate the circuit and preserve its semantic ordering and stable IDs.
2. Derive the typed search graph: external wires, local-continuation edges, and
   two-qubit coupling edges, with complete excision footprints.
3. Enumerate compatible templates. Hash template semantics for cache lookup:
   gate type/matrix, ordered roles, fixed parameters, action, template version,
   normalization, and resource model. A hash identifies an entry; it does not
   estimate its cost or certify equivalence.
4. Precompute and validate bounded-score coefficients, norms, overheads,
   footprints, and local instructions. Reuse entries across compatible instances.
5. Create an action-adjacency graph: variables are adjacent when their operation
   scopes touch or are consecutive on a circuit wire. Freeze this graph for
   state-independent neighborhood proposals.

For an action template $a$, store

$$
\Gamma_a=\sum_j|c_{a,j}|,\qquad
\Omega_a=\Gamma_a^2,\qquad
\ell_a=\ln\Omega_a=2\ln\Gamma_a.
$$

These are physical, normalized template costs. Algebraic term count is not a
substitute. For retained gates and uncut wires, the action cost is zero.

Representative entries under the declared constructions:

| Action | $\Gamma$ | $\Omega$ | $\ell$ |
| --- | ---: | ---: | ---: |
| Standard single-wire cut | 4 | 16 | $\ln16$ |
| Standard CX gate cut | 3 | 9 | $\ln9$ |
| CX excision with the plan's unmerged Pauli-eigenstate expansion | 16 | 256 | $\ln256$ |

The CX excision value follows its signed-permutation PTM and the plan's specific
expansion. It is not an optimum over arbitrary excision templates. Standard
gate/wire norms and log-weight graph search have direct precedents in the
[supplied cutting framework](https://arxiv.org/html/2608.05287v1).

## 3. Annealing state: actions, not arbitrary edge masks

For every candidate external wire $w$:

$$
z_w\in\{\mathrm{uncut},\mathrm{wire\ template\ }t\}.
$$

For a one-qubit gate $g$:

$$
m_g\in\{\mathrm{retain},\mathrm{excision\ template\ }t\}.
$$

For a two-qubit gate $g$:

$$
m_g\in\{\mathrm{retain},\mathrm{gate\ cut\ template\ }t,
\mathrm{excision\ template\ }u\}.
$$

Only available templates appear in each alphabet. Categoricals enforce one mode
per gate. Reject any remaining mandatory conflicts before energy evaluation.
Adjacent wire-cut/excision combinations are not automatically forbidden; prune
them only after a valid dominance argument.

In the four-port gadget, gate cutting deletes $K_g$, while excision deletes
$L_0,L_1,K_g$ together. Excision is one variable and one charged action. Independent
toggles of these three internal edges would create unsupported partial excisions
and an incorrect cost model.

The complete state is $s=(\{z_w\},\{m_g\})$. It does not need a fixed number of
partition labels. Keep graph edges and action footprints as derived data.

## 4. Exact cost and resource evaluator

### 4.1 Sampling-overhead objective

For selected actions $A(s)$:

$$
C(s)=\sum_{a\in A(s)}\ell_a,
\qquad
\Omega_{\mathrm{total}}(s)=\exp C(s).
$$

$C$ is an exact evaluation of the chosen independent-template bound, not a
state-dependent variance estimate. Minimize $C$ rather than exponentiating it
inside SA. Changing a few action variables permits a local update of $C$.

### 4.2 Connectivity and static allocation

Apply action footprints to derive the graph. Compute connected components using
all surviving connectivity edges. Separately compute allocation segments using
surviving external-wire and local-continuation edges only. Count segments within
each component to obtain widths $w_1,\ldots,w_{K_c}$.

Retain input/output and boundary-only segments. The sum should satisfy the
no-ancilla, no-elimination invariant

$$
W(s)=\sum_i w_i
=n+N_{\mathrm{wire\ cuts}}+\sum_{g\ \mathrm{excised}}\operatorname{arity}(g).
$$

This is also a useful evaluator consistency check. Do not count graph vertices
or original qubits, or assume that restoring an edge can only reduce maximum
component width: it can merge large components.

### 4.3 Two-way versus multiple fragments

**Canonical multi-fragment mode:** execute each component separately. Width
feasibility is $w_i\leq M$ for every component; fragment count is $K_c$.

**Two-block mode:** assign entire components to A/B, merging disconnected
components within each block into one static circuit. A block's width is the
sum of its component widths. Let $\mathcal R(s)$ be reachable subset sums of
$\{w_i\}$, and $W=\sum_i w_i$. A feasible packing exists if

$$
\mathcal R(s)\cap[W-M,M]\neq\varnothing.
$$

Use integer subset-sum dynamic programming/bitsets to evaluate this exactly on
the initial small instances, and retain a packing witness for compilation.
The total bound $W\leq2M$ is necessary but insufficient: widths $(2,2,2)$ cannot
fit into two blocks of capacity 3. An exact-component-count restriction would
instead require $K_c=2$ and does not permit this packing interpretation.

Before a two-block run, $n>2M$ proves infeasibility under the static baseline.
When $n=2M$, no wire-cut or excision break can fit; select benchmarks with width
slack to investigate all three actions. If $n\leq M$, an uncut circuit already
has minimum baseline overhead one when one block is allowed.

## 5. Energy and constraint penalties

The scientific objective is minimum $C$ subject to width feasibility. The SA
energy extends that objective to temporarily infeasible states:

$$
E(s)=C(s)+\lambda_Q P_Q(s).
$$

For canonical fragments, use

$$
P_Q(s)=\sum_{i=1}^{K_c}\bigl[\max(0,w_i-M)\bigr]^2.
$$

This guides improvements in multiple oversized components even if the largest
width stays unchanged. All feasible states have zero penalty. Do not penalize
width below $M$ in the baseline: that would change the research objective.

For two-block mode, use the best packing's overflow instead:

$$
P_Q^{(2)}(s)=\min_{r\in\mathcal R(s)}
\left\{[\max(0,r-M)]^2+[\max(0,W-r-M)]^2\right\}.
$$

This vanishes exactly when the two-block capacity constraint is satisfied.
It replaces, rather than silently duplicates, the component-width penalty.

If an additional hard component-count cap is requested, add
$\lambda_K[\max(0,K_c-K_{\max})]^2$. For exactly $K_0$ components, use
$\lambda_K(K_c-K_0)^2$ and check equality at output. Neither term measures the
number of executable blocks after packing.

### Penalty calibration

All costs are finite and nonnegative for validated full-channel templates.
Let $\mathcal S_v$ be the alphabet of variable $v$, and define an upper bound

$$
C_U=\sum_v\max_{a\in\mathcal S_v}\ell_a.
$$

Since widths and count violations are integers, any violation has penalty at
least one. Choosing each required constraint coefficient greater than $C_U$
makes every infeasible energy larger than every feasible baseline cost. Thus a
global minimizer of the penalized problem is feasible if a feasible state exists.
This does not guarantee that a finite SA run finds that minimizer.

For the initial correctness baseline, use a fixed coefficient
$\lambda_Q=C_U+c_{\mathrm{scale}}$, with
$c_{\mathrm{scale}}=\max(1,\operatorname{median}(\ell_a:\ell_a>0))$ and value 1
when there are no positive costs. Calibrate temperatures to the resulting
energy changes and use coordinated moves to reduce penalty barriers. Benchmark
smaller or adaptive penalties later; archive the best feasible state independently
of any changing energy schedule.

## 6. Should fragment count have a soft penalty?

**Recommended baseline: no.** Report $K_c$, and prefer fewer executed families
when primary costs are equal. A hard count limit is appropriate when it is a
real requirement. Adding an arbitrary $\beta K_c$ can favor a high-overhead
two-fragment plan over a low-overhead three-fragment plan.

An explicit execution-cost extension has a physical interpretation. If every
independent reconstruction trial executes all $B$ final families once, then

$$
N\geq\frac{2\Omega_{\mathrm{total}}}{\epsilon^2}\ln\frac{2}{\delta}
\quad\Longrightarrow\quad
N_{\mathrm{fragment\ shots}}=B N.
$$

Ignoring integer rounding, its conservative cost proxy is

$$
C_{\mathrm{shot}}(s)=C(s)+\ln B(s).
$$

This optimizes a total-shot bound, which differs from the current sampling-factor
objective. Use $B=K_c$ for independently executed components, and $B=2$ for two
nonempty packed blocks. If per-trial execution costs differ, replace $B$ with a
defined sum of costs relative to a reference unit. Parallel latency, batching,
constant-score elimination, and configuration reuse require their own models.
Do not present this bound as measured optimal sampling cost.

If $C_{\mathrm{shot}}$ becomes the primary objective, recalibrate penalty upper
bounds to cover that objective too. Keep a $(C,B)$ Pareto report for comparison.

## 7. Simulated-annealing moves

Use a mixture with initially **70% single-action proposals and 30% small block
proposals**. These are tunable prototype settings, not claims of optimal choices.

### Single-action proposals

Choose a variable uniformly from all eligible variables and replace its value
uniformly by another value in its fixed alphabet. This permits wire cut/restore,
retain/gate-cut/excision switches, and template switches. The proposal is
symmetric. Mandatory conflicts produce rejection, not an untracked repair.

### Small block proposals

Choose a neighborhood of 2–4 variables from the precomputed action-adjacency
graph, using a state-independent rule. Redraw all their values uniformly from
their fixed alphabets, including current values. This creates reversible proposals
without unsupported partial gate states.

Such neighborhoods can shift a wire cut, replace gate cuts with wire cuts, or
switch an excision and nearby boundaries together. Evaluate the complete rewrite
once. A multi-action change can have a favorable final cost even if its individual
steps would be rejected at low temperature.

Selecting only currently expensive boundaries or greedily repairing proposals
is generally asymmetric. If those mechanisms are added, compute the appropriate
proposal ratio or explicitly describe the method as a heuristic variant.

With symmetric proposals, accept a legal proposal $s'$ with

$$
p_{\mathrm{accept}}=\min\{1,\exp[-(E(s')-E(s))/T]\}.
$$

For asymmetric proposals, use

$$
p_{\mathrm{accept}}=\min\left\{1,
\exp[-(E(s')-E(s))/T]\frac{q(s\mid s')}{q(s'\mid s)}\right\}.
$$

These are the standard annealing/Metropolis–Hastings mechanisms, not quantum
novelty. See [Kirkpatrick et al.](https://doi.org/10.1126/science.220.4598.671)
and [Hastings](https://doi.org/10.1093/biomet/57.1.97).

Permit width-infeasible intermediate states but never invalid decomposition
states. Rejecting all width violations can disconnect useful search paths.
For the plan's at-most-one and conflict constraints, legal selections can be
connected through the all-retained state; positive-temperature single-action
moves preserve access to that path even when it is width-infeasible.

## 8. Initialization, cooling, and stopping

Use several seeds rather than trusting one initial partition:

- A spatial qubit partition, cutting crossing gates when templates exist. With
  $n\leq2M$, supported gate cuts give a feasible two-block seed without wire breaks.
- All two-qubit gates gate-cut, with no wire cuts. This produces width-one
  components when every required template exists; pack them for two-block mode.
- For multi-fragment mode with $M\geq2$, cutting all candidate inter-gate wires
  isolates one-/two-qubit gates and gives a simple width-feasible seed.
- Greedy action restoration from a feasible seed, accepting cost reductions
  that preserve feasibility, provides a lower-cost starting point.

Unsupported templates can invalidate particular seed recipes. Use a feasibility
search or exact tiny-instance solver when needed. Failure to find a seed is not
an infeasibility certificate except for separately proved cases.

Pilot-sample legal proposals under the selected penalty model. For a representative
positive energy change $d_+$, set

$$
T_0=\frac{d_+}{-\ln p_0},\qquad p_0=0.8.
$$

This targets acceptance $p_0$ for the representative change, not necessarily the
whole proposal mixture. Use a median/quantile, measure actual acceptance, and
adjust the initial temperature if necessary. If no positive changes are observed,
use a declared positive template-cost scale instead.

Suggested initial geometric schedule: $T_{j+1}=0.95T_j$, with 5–10 sweeps per
temperature; one sweep is one proposal per action variable on average. Fix the
total evaluation/time budget and use multiple random restarts. These schedule
values need sensitivity analysis. Geometric finite-budget SA has no claimed
global-optimum guarantee.

Maintain the current state and a separate best feasible archive. Stop at the
budget or a declared low-temperature/stagnation criterion. Optionally finish
with greedy legal descent from the archive. Report no feasible plan if the
archive is empty, rather than returning the lowest penalized invalid state.

## 9. Evaluation cost and postprocessing

Initially recompute components and segments in $O(|V|+|E|)$ per proposal.
Two-block packing adds pseudo-polynomial subset-sum work in the total integer
segment width, with bitsets practical for the initial small instances. Update
the LUT cost incrementally, but do not assume ordinary union-find supports both
edge insertion and deletion. Profile before adding dynamic connectivity.

After search:

1. Revalidate all selected actions and exact costs independently of the SA cache.
2. Recompute graph components, segment widths, and the selected block-packing
   witness if applicable.
3. Compile physical fragment families with the declared static allocation.
4. Check all boundary operations, widths, template mappings, and resource rules.
5. Reconstruct exact small-circuit queries, then assess finite-shot estimates.
6. Report $C$, $\Omega$, component count, final family count, widths, action
   counts, search budget/seeds, and heuristic solution status.

Do not silently remove actions in postprocessing. Any simplification changes
the state and must be revalidated and recosted.

## 10. Required comparisons and research claim

Compare matched action libraries, constraints, and evaluation budgets for:

- Tiny-instance exact search.
- Random search and greedy descent.
- SA with single-action moves only.
- SA with the proposed small block moves.
- Gate-only, wire-only, mixed-without-excision, and full mixed action spaces.
- Fixed two-block mode versus canonical multi-fragment mode, clearly separated.

Measure feasibility rate, best feasible log cost, exact-model gaps where known,
search time/evaluation count, final widths/counts, and reconstruction correctness.
Use multiple seeds and circuit families; distinguish extra action-set gains from
better search. Excision may be dominated on some families under the current LUT,
which is a legitimate finding rather than a reason to force its selection.

The potential contribution is **quantum-action-aware state encoding, bundled
neighborhood moves, and validated segment-width evaluation**, supported by
comparative evidence. SA itself is established and has already appeared in
[OBP-based cutting research](https://arxiv.org/abs/2510.19467), although that paper
optimizes a different parameter. The proposed finite-budget method is a heuristic.
