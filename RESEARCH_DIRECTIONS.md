# Research directions for quantum circuit partitioning

Survey date: 2026-10-11. This note combines the supplied-paper review in
[RESEARCH_FOUNDATION.md](RESEARCH_FOUNDATION.md) with a targeted search of primary
papers. The closest statistical and observable-based papers were examined beyond
their abstracts; some systems papers below were screened at abstract level.
Several recent sources are preprints. Proposed contributions below are research
hypotheses, not established novelty claims or demonstrated improvements.

## Recommendation

Start with **decision-oriented partitioning for related parameterized circuits**.
Use query-family-preserving boundary compression as a possible theoretical
extension. The central question is whether a partition that is suboptimal for
estimating individual expectation values can be optimal for choosing a good
candidate with a specified confidence.

Wire cuts, gate cuts, physical gate excision, and eligible joint gate groups are
actions available to this problem. Their combination is useful infrastructure,
but should not be the central novelty claim.

## Where the literature already overlaps

| Proposed broad idea | Closest primary work | Consequence for our research |
| --- | --- | --- |
| Combine gate and wire cuts, optimize placement | [Optimal Partitioning](https://arxiv.org/abs/2308.09567); supplied [gate-group framework](https://arxiv.org/abs/2608.05287) | A new graph or heuristic needs a substantive new objective or guarantee. |
| Allocate shots according to measured variance | [ShotQC](https://arxiv.org/html/2412.17704v3), especially Section IV | Shot allocation alone is established; this paper assumes the partition is given. |
| Include hardware noise in shot allocation | [How Many Shots Does It Take?](https://arxiv.org/html/2607.24704v1) | Its evaluation combines allocation with an existing automatic cut finder. Joint placement could matter, but broad noise awareness is already crowded. |
| Heterogeneous partition sizes and topology-aware mapping | [MOSAIQC](https://arxiv.org/abs/2607.18888) | Different QPUs and topology constraints alone are weak novelty claims. |
| Adaptive recutting under device changes | [MAESTROCUT](https://arxiv.org/abs/2509.00811) | Adaptivity and drift alone have direct preprint overlap. |
| Preserve an observable rather than a whole channel | [Tomography-based circuit knitting](https://arxiv.org/html/2512.19623v1) | This change of semantics is already explicit in prior work. |
| Backpropagate observables to reduce cutting | [Low overhead circuit cutting with operator backpropagation](https://arxiv.org/pdf/2510.19467) | The combination, its search parameter, and truncation tradeoffs are already explored. |
| Use small observable subspaces | [Dynamic Observable Subspaces](https://arxiv.org/abs/2609.09806) | Observable-space compression also has adjacent classical-simulation literature. |
| Reuse repeated structures/results | [CiFold](https://arxiv.org/abs/2412.18705); [Semantic Quantum Circuit Cache](https://arxiv.org/abs/2604.26788) | Repeated fragments and caching alone are insufficient. |
| Mix classical and quantum execution | [SuperSim](https://arxiv.org/abs/2303.10788); [Qdislib](https://arxiv.org/abs/2505.01184) | Clifford islands and heterogeneous execution are established directions. |
| Optimize learning quality and cutting cost together | [CutReg](https://arxiv.org/abs/2506.14858); [DistributedEstimator](https://arxiv.org/abs/2602.16233) | Applying cutting to iterative training is already studied. |
| Change the application output to QAOA samples | [Optimized Circuit Cutting for QAOA Sampling Tasks](https://arxiv.org/abs/2507.06649) | Application-specific outputs are promising, but QAOA sampling itself has direct overlap. |

The tomography paper is particularly important: it constructs expectation-level
cuts and includes the cost of learning downstream operators. Its favorable
scaling results concern structured tree circuits and depend on boundary
dimension. Conditional preparation using classical communication is part of the
protocol. These results do not establish polynomial-cost cutting for arbitrary
circuits, or invalidate full-channel QPD lower bounds.

## Direction 1: decision-oriented partitioning

**Input:** a family of related circuits \(C_{\theta_1},\ldots,C_{\theta_M}\), a
Hamiltonian or score observable \(H\), width/depth and classical-memory budgets,
an acceptable suboptimality \(\eta>0\), and failure probability \(\delta\).
Define \(E_k=\operatorname{Tr}[H C_{\theta_k}(\rho)]\).

**Output:** an executable partition/decomposition plan, shared experiment map,
shot-allocation and stopping policy, and a selected candidate \(\hat k\).

The proposed objective is

\[
\min_{\pi}\;\mathbb E[T_\pi]
\quad\text{subject to}\quad
\Pr(E_{\hat k}\leq\min_k E_k+\eta)\geq1-\delta,
\]

with physical fragment feasibility and peak-memory constraints. Here \(\pi\)
includes partitioning and measurement decisions, and \(T_\pi\) includes planning,
compilation, pilot shots, execution, and reconstruction. Begin with total
physical fragment shots as the cost, then introduce time after measuring it.
For adaptive instance-specific policies, unknown fragment statistics must be
learned from charged pilot data; the optimizer cannot use exact-reference data.

### Why this can change the best partition

Suppose a small parameterized block is the only changing operation. An exact
decomposition into a fixed physical experiment basis can yield

\[
E(\theta)=\sum_i a_i(\theta)\mu_i.
\]

The fragment responses \(\mu_i\) are shared because the physical preparations,
measurements, and remaining gates are identical. Then

\[
\Delta=E(\theta_A)-E(\theta_B)
=\sum_i d_i\mu_i,\qquad d_i=a_i(\theta_A)-a_i(\theta_B).
\]

Estimate this contrast directly, using shared data. For linear reconstruction,
\(\operatorname{Var}(\hat\Delta)=d^\top\Sigma d\), where \(\Sigma\) is the
covariance of the fragment-response estimates. Equivalently,

\[
\operatorname{Var}(\hat E_A-\hat E_B)
=\operatorname{Var}(\hat E_A)+\operatorname{Var}(\hat E_B)
-2\operatorname{Cov}(\hat E_A,\hat E_B).
\]

When each term has a bounded unbiased physical score \(|X_i|\leq1\), the direct
contrast has the importance-sampling bound
\(\operatorname{Var}(Y)\leq\Gamma_\Delta^2\), with
\(\Gamma_\Delta=\sum_i|d_i|\). Observable coefficients must be included in the
weights or normalization. This is a bound for a specified estimator; every
sample may require multiple fragment executions.

Cancellation can make \(\Gamma_\Delta\) much smaller than the norms of two
separately estimated scores. Conversely, a common physical basis can be more
expensive than individually optimized decompositions. That tradeoff is exactly
what the partitioner should resolve. Excision of a parameterized gate can expose
reusable responses even when excision is unnecessary for width feasibility.

This is a proposed mechanism, not evidence of an empirical advantage. It requires
truly identical executable experiments; changing a parameter inside a fragment
usually changes its response and prevents direct reuse. A fixed small-block PTM
measure-and-prepare basis is one possible construction, but its physical norm
must be charged. Product-of-fragment reconstruction requires its own covariance
analysis; it cannot generally be replaced by the linear formula above.

### Proposed contribution and closest comparisons

The defensible hypothesis is **jointly choosing partitions and decompositions
to minimize the cost of a confidence-qualified comparison or candidate
selection**. Training with cutting, regularizing cutting angles, caching
identical circuits, control variates, and allocating shots are individual prior
ideas. Their names or combination do not establish novelty. ShotQC, CutReg,
DistributedEstimator, and semantic caching are mandatory comparisons.

The targeted search did not identify an exact match to this narrow joint
formulation. Before a submission-level novelty claim, follow citations and code
for these works and search adjacent best-arm identification, correlated
estimators, parameter-shift gradients, and multi-fidelity quantum optimization.

### Statistical requirements

- Use fixed batches first. Optional stopping later requires valid sequential
  bounds or confidence sequences, rather than repeatedly inspecting a fixed-shot
  bootstrap interval.
- Positive covariance helps a difference, but does not help every workload.
- Sharing data between different queries is distinct from multiplying estimates
  of two factors from the same data. Preserve independence between multiplicative
  factors or derive an unbiased correction.
- A nonzero \(\eta\) handles near ties. Exact best-candidate identification can
  require unbounded sampling as the gap approaches zero.
- A finite-shot confidence statement on noisy fragment outputs does not by
  itself certify the ideal uncut energy. Ideal-value claims need a justified
  device-bias bound or an explicitly different target.

## Direction 2: partitioning with query-family boundary interfaces

**Input:** one circuit, a query family \(\mathcal W=\{O_1,\ldots,O_m\}\), specified
initial-state assumptions, per-query accuracy/confidence, and physical/classical
resource budgets. **Output:** fragments and boundary interfaces sufficient for
this family, with exact reconstruction or a certified approximation.

At a boundary, let \(\mathcal S\) contain all operator factors needed to reconstruct
the downstream queries. A candidate replacement \(\mathcal M\) can be sufficient
without preserving the entire original channel \(\mathcal E\) if

\[
(\mathcal E^\dagger-\mathcal M^\dagger)(B)=0
\quad\text{for every }B\in\mathcal S.
\]

The operator factors must include the correlations with the remaining system;
matching an isolated reduced state or a few local expectations is insufficient.
The replacement may be a signed physical reconstruction rather than a single
channel. Physical realization costs must be derived separately from algebraic
rank or PTM sparsity.

The tentative gap is joint placement and physically realizable interface
selection for **multiple queries**, particularly noncommuting ones, with an
explicit complexity bound on constructing the interfaces. A theory result could
characterize when a small sufficient interface exists, or lower-bound its cost
as the query family expands. Do not claim that low-dimensional observable spaces,
single-query preservation, light cones, or OBP-plus-cutting are new. Compare
directly with tomography-based knitting, OBP cutting, DOS, and purely classical
Pauli propagation.

Globally, an operator approximation satisfying

\[
\|C^\dagger(O_j)-\widetilde C^\dagger(O_j)\|_\infty\leq b_j
\]

bounds the expectation bias by \(b_j\) for every density matrix. An approximate
plan can then require \(b_j+r_j\leq\epsilon_j\), where \(r_j\) is a valid
statistical radius. Local tolerances do not automatically sum: signed
decompositions and contractions can amplify them. Propagating the entire
observable can itself be exponentially expensive, so interface construction,
learning, contraction, and memory must be included in the objective.

This is a stronger theoretical direction but has higher overlap and feasibility
risk than the restricted decision experiment.

## Direction 3: certified approximation across mixed execution backends

Choose boundaries, exact versus approximate transfer information, and classical
versus quantum execution jointly, under a global error budget. The objective can
be minimum execution time with
\(\text{bias}^2+\operatorname{variance}\leq\epsilon^2\) for an MSE target, or a
bias-plus-confidence-radius constraint for a probability guarantee.

This naturally includes sparse gate excision, exact Clifford processing,
bounded operator truncation, and quantum treatment of difficult blocks. The
potential contribution is a composable global certificate and an optimizer that
spends the error budget where it saves the most measured cost. Approximate
cutting, classical/quantum division, and truncation themselves are established:
compare [Approximate Quantum Circuit Cutting](https://arxiv.org/abs/2212.01270),
SuperSim, Qdislib, and OBP cutting.

Count all approximation work. Small PTM entries are not automatically harmless
after contraction; low rank is not a low physical quasiprobability norm; removing
non-Clifford gates does not establish efficient classical execution if boundary
inputs inject difficult states. This direction is a plausible extension once
exact statistical behavior is understood, rather than the first experiment.

## Direction 4: partitioning with lower bounds and feasibility certificates

Given a circuit, query model, allowed local operations, and resource budget,
return either a feasible plan or a lower bound showing infeasibility within a
declared action set. This would help distinguish a poor heuristic from an
intrinsically expensive instance.

[Circuit cutting with classical side information](https://arxiv.org/abs/2503.22384)
develops instrument-based formulations and SDP bounds relevant to local
decomposition costs. A possible longer-term contribution is lifting suitable
local bounds into a restricted global partition model.

Be precise about the target: a lower bound on full-channel QPD norm is not a
lower bound on the instance-specific variance of one observable. A high-cost
heuristic plan gives an upper bound on the optimum, not an infeasibility proof.
Certificates from MILP/SDP relaxations apply to the stated model and supported
decompositions, not to all possible cutting protocols.

## Graph representation for these formulations

Keep three connected views:

1. **Circuit dependency graph:** operations, ports, time, parameters, legal cut
   locations, and physically checked group candidates.
2. **Reconstruction factor graph:** decomposition indices and fragment response
   tensors, exposing contraction work, intermediate dimensions, and memory.
3. **Experiment/query graph:** an experiment node connects to every query using
   its data, recording coefficients, shots, cost, and covariance provenance.

For the decision formulation, the third view exposes common experiments and
contrast cancellation. For query interfaces, boundary nodes carry sufficient
operator spaces. Candidate gate groups introduce nonadditive, partition-dependent
costs and conflicts. A conventional weighted min-cut is therefore a candidate
generator, not generally the complete objective.

Use a fixed, explicitly declared protocol initially: independent local fragment
execution, product input states, ideal execution, supported small-gate
decompositions, and physically materialized width constraints. Conditional
preparation/LOCC belongs to a separate capability model. Do not mix costs from
different communication assumptions.

## First experiment before developing an optimizer

Use 4–8 qubit circuits whose candidates differ only in one small parameterized
block. Impose a 2–4 qubit fragment limit. Include both local-energy observables
and observables spanning the boundary; vary the candidate separation and the
amount of shared circuit structure. Use exact uncut simulation only as the
evaluation reference.

Enumerate a small, validated set of manual plans first. Add a joint-group plan
only after its physical decomposition is implemented and verified; our current
framework does not yet provide joint gate-group execution.

Compare:

- Lowest realized physical QPD norm, with separate candidate estimation.
- The same plan with optimized allocation for its fixed reconstruction.
- The same plan with exact experiment reuse and covariance-correct contrast
  reconstruction.
- A plan selected for the contrast cost, allowing extra cuts or excision.
- OBP-plus-cutting and classical simulation where applicable and feasible.
- An uncut finite-shot reference when the test circuit fits the simulator.

These ablations separate benefits from allocation, reuse, changed estimators,
and partition placement. A new partitioning result requires improvement over
the strong reuse-and-allocation baseline, not merely equal shots per circuit.

Measure physical shots to reach the decision target, wrong-selection rate over
independent repetitions, confidence coverage, bias, realized QPD norm, unique
experiments, planning/compilation/execution/reconstruction time, and peak memory.
For adaptive selection, include pilot and rejected-plan costs. Predeclare the
candidate-gap and query sweeps to avoid reporting only favorable instances.

**Go/no-go question:** does the ranking of feasible partitions change when the
target changes from individual energies to a reliable energy comparison, and
does choosing by the decision objective reduce actual physical shots after
charging all required work? If not, refine the formulation before investing in
a large graph optimizer.

The present package provides manual cutting, physical excision, exact and
finite-shot execution, experiment mappings, and covariance-aware bootstrap.
It still needs cross-candidate experiment accounting, a common physical basis,
decision reconstruction, and valid stopping rules for this experiment. Bootstrap
alone is an exploratory diagnostic, not the required rigorous sequential
certificate. No code changes are made by this survey.

## Suggested reading order

1. [ShotQC, Section IV and statistical formulation](https://arxiv.org/html/2412.17704v3).
2. [OBP cutting, Sections 3–5](https://arxiv.org/pdf/2510.19467).
3. [Tomography-based knitting, Sections III–V and Appendix S6](https://arxiv.org/html/2512.19623v1).
4. [CutReg](https://arxiv.org/abs/2506.14858) and
   [Semantic Quantum Circuit Cache](https://arxiv.org/abs/2604.26788).
5. [Gate-group framework](https://arxiv.org/abs/2608.05287), revisited as a
   decomposition/placement baseline rather than the proposed research problem.

A possible working title is **Decision-Oriented Quantum Circuit Partitioning
with Shared Fragment Measurements**. The title describes the hypothesis; it is
not a claim of established priority.
