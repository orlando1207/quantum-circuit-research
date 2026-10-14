This note records the initial research review for quantum circuit partitioning, dated 2026-10-10. The five PDFs in `reference/` are the supplied evidence; framework recommendations also use the primary documentation linked below. The review preceded implementation. The subsequent executable foundation and its isolated conda environment are documented in [README.md](README.md); the environment inventory below is historical.

The recommended starting point is Python 3.12 with Qiskit, Qiskit Aer, the Qiskit circuit-cutting add-on, NumPy/SciPy, and NetworkX. Keep a small research-owned representation of cut plans and decompositions so that Qiskit, QCut, classical simulators, and future C++ components can be compared through the same interfaces. Begin with exact reconstruction of Pauli expectation values and independent local-operation cuts; then validate finite-shot estimators, excision, and joint decompositions.

**The research problem is to choose transformations together with an executable reconstruction strategy.** Given a circuit, an input state, a requested observable or probability query, and execution constraints, choose fragments, cuts, gate groups, excisions, and execution backends. The result must preserve the requested ideal computation and expose its statistical and classical costs. A graph partition is an intermediate result; the final object is an executable experiment and reconstruction plan.

The proposed combination of graph representations, gate cuts, wire cuts, and gate grouping overlaps directly with the supplied August 2026 paper. It should be treated as a central baseline. Adding gate excision is a promising extension, but the novelty and benefits remain hypotheses to evaluate rather than established conclusions.

**The supplied papers contribute different parts of that problem.**

| Supplied paper | Main contribution relevant to this project | What to carry into the foundation |
| --- | --- | --- |
| Peng, Harrow, Ozols, Wu, *Simulating Large Quantum Circuits on a Small Quantum Computer*, `1904.00102v2.pdf` | Tensor-network clusters and an eight-term identity-channel decomposition for wire cutting. The general estimator has an exponential cut cost; more structured reconstruction can exploit contraction complexity. | Represent time and boundary ports, preserve signed weights, and model reconstruction structure as well as cut count. |
| Mitarai, Fujii, *Constructing a virtual two-qubit gate by sampling single-qubit operations*, `1909.07534v2.pdf` | Channel decompositions of nonlocal gates into local rotations and measurements; combined gate/wire cutting. For the CZ cuts in Theorem 5, sample complexity scales as `9^Ms 16^Mt / epsilon^2`, with a confidence factor. | Gate type and angle affect cost. Keep measurement outcomes in the estimator. Separate the number of decomposition terms from variance overhead. |
| Tang et al., *CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations*, `2012.02333v3.pdf` | An end-to-end wire-cutting workflow, mixed-integer cut search, distribution reconstruction, and dynamic-definition queries. Section 4.1 optimizes a classical reconstruction-work estimate. | Boundary-induced qubits count toward fragment width. Classical memory, output size, and tensor/Kronecker contraction matter. Full-distribution reconstruction is a distinct workload. |
| Chen et al., *A Measure-and-Prepare Framework for Arbitrary Quantum Gate Excision and Its Applications*, QCE 2025, pp. 544–545 | An arbitrary-unitary Pauli measure/prepare expansion and a near-Clifford FPGA simulation application. | Use sparse Pauli transfer matrices to describe excision. Distinguish algebraic Pauli outputs from physically preparable states. Evaluate simulator cost and device sampling cost separately. |
| Fröhler et al., *Scalable Circuit Cutting: A Framework for Combined Gate and Wire Cuts Using Gate Groups*, `2608.05287v1.pdf` | Interaction-graph initialization, temporal topology refinement, logarithmic cut weights, and partition-dependent gate-group costs. | Reproduce the two-stage method as a baseline; extend group discovery, feasibility checking, and cost objectives. |

Peng's `K` counts intercluster quantum connections, and `d` bounds the qubits needed by a cluster. Qubit partitioning alone misses the more flexible temporal clusters in their examples. Their tensor contraction results also show why one scalar count of cuts cannot describe every reconstruction algorithm.

Mitarai and Fujii explicitly decompose the superoperator `rho -> U rho U†`. Decomposing the unitary matrix alone leaves cross terms when evaluating an observable, and those terms are not generally accessible through ordinary fragment measurements. Our decomposition interface should therefore operate at channel/instrument level.

CutQC combines measured upstream quantities with prepared downstream states. Its four Pauli-index contraction terms per wire cut explain the `4^K` reconstruction factor, but that is not the same as `16^K` sampling overhead for the standard bounded Monte Carlo wire-cut estimator. The paper's large runtime comparisons also omit cut-search and quantum-execution runtime in parts of the evaluation; the largest dynamic-definition scaling experiment uses uniform subcircuit distributions. These results motivate the architecture but do not establish an end-to-end device speedup for our future experiments.

The excision paper reports six nonzero Pauli-pair configurations for a T gate and 232 for a Toffoli, versus 64 and 262144 configurations for the surrounding-wire method it compares against. Those are counts in its algebraic representation. They are not equivalent to six or 232 hardware shots, and do not by themselves determine a physical quasiprobability norm. The authors' application removes non-Clifford operations so the remaining work can use a Clifford simulator. Its [PQSim artifact](https://github.com/NTU-ALComLab/PQSim) includes an FPGA workflow; it is a reference implementation rather than a required dependency for a CPU-first foundation.

The gate-group paper uses a qubit interaction graph first and a graph with gate ports and temporal wire edges second. Independent cut costs become additive through `log kappa`. Gate groups instead contribute a cost that depends on the partition; group gates are removed from the independent edge accounting to avoid charging twice. Evaluation is primarily cut placement and modeled sampling cost, not a demonstrated hardware reconstruction pipeline. Group identification is manual, the reported method is bipartitioning, and Table I leaves the larger grouped QFT case as future work. These are concrete extension opportunities.

Two qualifications should inform any reproduction of that paper. First, fragment feasibility should be verified from materialized wire segments, including cut-induced inputs; balancing original graph vertices is not a substitute for counting physical fragment qubits. Second, a high-cost heuristic solution is an upper bound on the best cost and cannot certify that every partition is impractical. A reliable impossibility claim needs a lower bound or exhaustive/exact reasoning under stated assumptions.

**A common algebraic framework can cover the three operations without erasing their differences.**

For a single-qubit wire cut, the Pauli expansion is

\[
A=\frac12\sum_{P\in\{I,X,Y,Z\}}\operatorname{Tr}(PA)P.
\]

Expressing the output Pauli operators as signed sums of their eigenstate projectors gives the familiar eight terms. Each term means a physical preparation plus a measured score; a signed score map need not itself be a completely positive channel. Correlations with untouched qubits must be retained by the instrument and tensor semantics.

For a two-fragment gate cut, write the channel as

\[
\mathcal U=\sum_i a_i\,\mathcal F_i^{(A)}\otimes\mathcal F_i^{(B)}.
\]

The local operations may include measurement instruments, outcome signs, and resets. The permitted local operations must be recorded. A decomposition that is local across one bipartition is not automatically executable across three or more independently run fragments.

For excision of an `n`-qubit unitary `G`, define its Pauli transfer matrix

\[
R_{QP}=2^{-n}\operatorname{Tr}(QGPG^\dagger).
\]

Then

\[
GAG^\dagger=2^{-n}\sum_{P,Q}R_{QP}\operatorname{Tr}(PA)Q.
\]

There are at most `16^n` Pauli-pair coefficients, and sparsity can substantially reduce this count. The output `Q` is an operator, not generally a density matrix. A device compiler must further express it using physical preparations and signed weights. A classical algebraic simulator may instead manipulate it directly. These two execution choices have different costs. Start arbitrary excision on one- and two-qubit operations; dense transfer matrices for large blocks are exponentially expensive.

Wire cutting excises an identity channel on a wire. Gate cutting removes a nonlocal interaction across chosen fragments while preserving local wire continuations. General gate excision removes the entire chosen operation and replaces its input/output relation. Excision can simplify simulator structure without disconnecting the circuit into smaller components.

**Sampling overhead needs an estimator definition.** For an implementable signed decomposition with coefficients `a_i`, let

\[
\gamma=\sum_i|a_i|,\qquad p_i=|a_i|/\gamma.
\]

If the score from a sampled term is bounded by `|X_i| <= 1`, the importance-sampled estimator `Y = gamma sign(a_i) X_i` is unbiased for the ideal target and satisfies `Var(Y) <= gamma^2`. To obtain additive error `epsilon` at failure probability `delta`, a Hoeffding bound gives a sample count proportional to

\[
\gamma^2\epsilon^{-2}\log(1/\delta).
\]

This is a worst-case variance factor relative to a bounded uncut estimator, not a guaranteed ratio to the empirical optimum of every uncut circuit. Observable normalization, fragment executions per sample, shot allocation, covariance, term reuse, device noise, and QPD-term sampling also matter.

| Operation under the stated local-operation model | Representative `gamma` | Worst-case factor `gamma^2` |
| --- | ---: | ---: |
| One standard LO wire cut | 4 | 16 |
| One CX/CZ cut | 3 | 9 |
| One `RZZ(theta) = exp(-i theta Z⊗Z/2)` cut | `1 + 2|sin theta|` | Its square |
| One SWAP cut, with an optimal permitted decomposition | 7 | 49 |
| Two eligible parallel CX-equivalent gates across the same bipartition, cut jointly | 7 | 49; independent cuts give 81 |

These values follow the supplied literature and the [Qiskit technical explanation](https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html). For arbitrary two-qubit gates, that documentation also notes that its generic KAK implementation is not always optimal. Always record both a theoretical bound and the norm of the decomposition actually generated.

For independent decompositions, the product construction gives `gamma_plan = product gamma_j`, so `log overhead = 2 sum log gamma_j`. Joint decompositions replace an entire eligible set's product cost. Groups can overlap, so selecting them requires conflict constraints or explicit plan validation; they cannot all be credited simultaneously. For eligible parallel rotation gates across a bipartition,

\[
\gamma_{\rm joint}=2\prod_s(1+|\sin\theta_s|)-1.
\]

Use this only when the assumptions and local operations of the chosen construction are satisfied. Zero-angle cases may give equality with the independent cost. The [joint-rotation paper](https://arxiv.org/abs/2312.09679) establishes an ancilla-free parallel construction without interfragment classical communication. More general grouped-unitary results are in [Schmitt, Piveteau, and Sutter](https://arxiv.org/abs/2312.11638).

Classical communication used to choose a later fragment's preparation is part of an LOCC protocol and differs from simply combining independently acquired results afterward. Use independent LO as the first experiment model. Keep LOCC as a separate capability for later evaluation; [optimal communicating wire-cutting protocols](https://arxiv.org/abs/2302.03366) have different attainable costs.

**The initial cost report should retain multiple quantities.** Record algebraic terms, nonzero physical decomposition terms, unique executable fragment circuits, QPD samples, actual shots per circuit, total fragment shots, `gamma`, `gamma^2`, empirical variance, bias, reconstruction time, and peak memory. A sampled global term may require one execution of each participating fragment. Shared fragment circuits may reduce execution counts but introduce covariance between reconstructed quantities.

For exact term enumeration with independent term estimates, `Var(sum a_i mu_hat_i) = sum a_i^2 sigma_i^2/N_i`. Equal-duration terms permit allocation `N_i` proportional to `|a_i| sigma_i`. If one sample of term `i` costs `t_i`, the optimal allocation under a fixed total time is proportional to `|a_i| sigma_i/sqrt(t_i)`. Fragment-product estimators and reused data require their own variance/covariance analysis rather than this simple formula.

Finite-shot reconstructed probabilities can be negative. Preserve raw estimates for validation. Clipping or renormalizing them changes the estimator and must be reported separately. Likewise, removing nonzero small coefficients introduces approximation bias; only truly zero terms can be removed without an additional error budget.

**Use three connected representations.**

1. The semantic circuit representation preserves operations, parameters, qubit order, initial state, observable, and dependencies. Keep original high-level gates for excision and grouping while making normalized one-/two-qubit views available for standard cutters. Record the mapping between these views.
2. A temporal graph with operation ports and directed wire segments exposes both gate-cut and wire-cut locations. Give operations and ports stable research IDs; gate-list indices change after transformations. The qubit interaction graph is a useful projection for initialization, but loses time, angles unless annotated, and intervening operations that can invalidate groups.
3. A reconstruction factor graph contains fragment data and decomposition indices. It exposes tensor contraction cost, shared measurements, and group correlations. A hypergraph can describe participation, but a fixed cut-net penalty usually cannot express a partition-dependent joint-decomposition cost without additional modeling.

Single-qubit gates can sometimes be omitted from an interaction graph, but must remain available for execution and group-validity reasoning. Noncommuting operations can break a cascade or prevent a proposed parallel scheduling. Partition validation must count newly prepared wire segments and required ancillas. A later qubit-reuse pass can reduce width, but should have its own scheduling and device-capability checks.

**The basic research cell should have seven independently replaceable components.**

| Component | Required responsibility |
| --- | --- |
| Circuit and query handling | Import/export, gate normalization, parameter binding, observables, dependency graph, stable IDs, and provenance. Start with product initial states and unitary input circuits; cut-generated measurements/resets are supported by execution adapters. |
| Decomposition registry | Wire, supported two-qubit gate, sparse PTM excision, and later joint decompositions. Store coefficient conventions, physical operations, measurement score rules, prerequisites, and cost bounds. |
| Cut plan and validator | Record partitions, wire ports, gate cuts, excisions, group membership, backend assignment, and constraints. Check width after materialization, operation support, group overlaps, and complete treatment of cross-fragment interactions. |
| Experiment compiler | Generate physical preparation/measurement circuits, map observables, retain result-bit meaning, deduplicate identical experiments, and choose enumeration or QPD sampling. Preserve term-to-experiment mappings. |
| Execution adapter | Provide exact small-circuit evaluation, finite-shot noiseless simulation, noisy simulation, Clifford execution where eligible, and eventually IBM execution. Return normalized result records and actual resource use. |
| Reconstructor | Evaluate the signed sum or factor-network contraction; expose uncertainties, covariance assumptions, and query-specific outputs. Support exact term summation and stochastic estimation as separate modes. |
| Cost and benchmark runner | Compare theoretical/realized norms and measured costs; record seeds, versions, source circuits, transformed circuits, plans, timing, memory, and results. Accept replaceable cut-search algorithms. |

Exact decomposition enumeration is not the same as exact circuit evaluation: all coefficients can be enumerated while each fragment still has shot noise. Keep these choices separate. A statevector primitive that samples final measurements is also not automatically an exact evaluator or suitable for mid-circuit cut instruments; use a branch-aware exact path or appropriate Aer support.

For early optimization, minimize `log gamma_plan` subject to fragment width, allowed operations, and a classical-memory budget. Compare the actual compiled plan as well as its fast graph surrogate. Later, consider minimizing end-to-end runtime subject to an MSE target, with `MSE = variance + bias^2`, and account for search, compilation, execution, and postprocessing. Avoid combining quantities with unrelated units through arbitrary weights when explicit constraints are available.

**The framework choice should reflect the work each tool already does.**

| Tool | Recommended role | Scope to verify before relying on it |
| --- | --- | --- |
| Qiskit + `qiskit-addon-cutting` | Primary circuit and standard cutting/reconstruction engine; automatic mixed-cut baseline. | The documented workflow is expectation-value focused. Generic two-qubit decompositions need not attain theoretical optimum. Custom excision and group protocols need explicit adapters. |
| Qiskit Aer | Initial execution engine and uncut reference; statevector, density-matrix, Clifford, and MPS options. | Choose methods that support the generated instruments; document approximation settings for approximate methods. |
| QCut, `JooNiv/QCut` | Additional Qiskit-compatible cutting baseline. | Indexed project documentation advertises joint rotation cuts, mixed cut finding, consolidation, and LOCC options. Verify the pinned release's features and reconstruction against small exact circuits before using its costs as evidence. |
| PennyLane `qcut` | Independent wire-cutting/reconstruction reference, especially for graph/tensor workflows and variational differentiation. | The documented entry points are wire-cutting transforms; it does not replace the custom unified plan layer. |
| Original CutQC | Historical wire-cut/distribution/MIP reproduction. | Use an isolated legacy environment if needed; its artifact requires Gurobi and should not dictate modern package versions. |
| DVLab-NTU Qsyn | Optional circuit synthesis, ZX/tableau rewrite, mapping, and verification tool. | Its documented functionality is a C++ compilation toolkit, rather than a complete cutting experiment/reconstruction workflow. Cross-tool export must preserve circuit semantics and research metadata. |
| MQT Bench | Reproducible benchmark inputs at controlled circuit levels. | Fix benchmark version, parameters, decomposition level, and mapping so comparisons are fair. |
| NetworkX; later METIS/KaHyPar | Inspectable initial graphs and KL baseline; faster graph/hypergraph candidate generation later. | Stock partition objectives generally need custom refinement to represent group costs and physical-width feasibility. |
| NumPy/SciPy; later `opt_einsum`/cotengra | Initial algebra and reconstruction; later contraction planning and memory estimates. | Contraction optimization does not itself define or reduce the variance of every estimator. |
| HiGHS/MILP, Gurobi, or an SMT solver | Small-instance exact baselines under an explicit model. | Exactness applies to the encoded action/decomposition set and completed solve; report time limits, gaps, and feasible incumbents. |

Primary tool references: [cutting repository](https://github.com/Qiskit/qiskit-addon-cutting), [cutting API](https://qiskit.github.io/qiskit-addon-cutting/apidocs/qiskit_addon_cutting.html), [Aer simulation methods](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html), [QCut options](https://jooniv.github.io/QCut/notebooks/QCutOptionsExample.html), [PennyLane wire cutting](https://docs.pennylane.ai/en/stable/code/api/pennylane.cut_circuit.html), [CutQC artifact](https://github.com/weiT1993/CutQC), [Qsyn](https://github.com/DVLab-NTU/qsyn), [MQT Bench](https://mqt.readthedocs.io/projects/bench/en/stable/), [KaHyPar](https://kahypar.org/), [SciPy MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), and [cotengra contraction objectives](https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/pathfinders/path_basic/index.html).

A Python research driver can still use compiled numerical and simulator kernels. Start in Python for transparent algebra, rapid experiments, and existing reconstruction support. Move profiled bottlenecks such as repeated local-search gain calculations or contractions into C++ only after correctness and costs are established. If existing Qsyn expertise or infrastructure makes it the natural optimizer host, retain a Qiskit execution/reconstruction adapter instead of rebuilding device integration there.

**The environment should be isolated and reproducible.** The current default interpreter is Python 3.12.9 with Qiskit 1.2.4, Aer 0.15.1, NetworkX 3.5, NumPy 2.2.5, SciPy 1.15.3, and rustworkx 0.17.1. The cutting add-on, PennyLane, QCut, MQT Bench, and cotengra were not installed in that interpreter when inspected. This inventory does not imply that the packages are compatible with a future combined environment.

The inspected [cutting add-on project metadata](https://raw.githubusercontent.com/Qiskit/qiskit-addon-cutting/main/pyproject.toml) identifies version 0.10.0 and declares Python >=3.10 and Qiskit >=1.3.1,<3. Thus the current Qiskit 1.2.4 is below its minimum. For implementation, create a separate Python 3.12 environment, resolve a compatible Qiskit 2.x/add-on/Aer combination, then freeze the complete resolved environment. Keep historical CutQC reproduction separate. This review did not resolve or install that environment.

Save circuits in QPY for Qiskit-native fidelity and export OpenQASM where interoperability is needed. Store stable IDs, cut plans, groups, queries, and mappings in explicit sidecar records; neither a plain QASM export nor arbitrary Python attributes should be trusted to retain every research annotation. [QPY documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qpy) explains the difference between native serialization and QASM export.

**Build the foundation through validation milestones before introducing a novel optimizer.**

1. Establish circuit/query conventions and an exact uncut reference. Verify qubit ordering, Pauli signs, high-level normalization, and circuit serialization. Use small Bell/GHZ, random, and parameterized examples.
2. Implement manual standard wire and gate plans through a cutting adapter. Validate one-cut and mixed-cut results against exact uncut expectation values; include entangled inputs to a cut operation and multiple wire cuts on the same original qubit. Verify materialized fragment widths.
3. Validate finite-shot reconstruction. Enumerate QPD terms first, then sample them. Measure empirical bias, variance versus shot count, confidence coverage, and actual fragment shots. Separate QPD sampling randomness from measurement randomness.
4. Add sparse PTM excision. Validate identity, T, representative rotations, and a small multiqubit gate. Compare algebraic reconstruction with physically expanded preparation/measurement experiments; keep Clifford-simulator eligibility explicit.
5. Add one validated joint-cut family, initially eligible parallel CX-equivalent rotations across two fragments. Confirm channel equality and the 81-to-49 worst-case overhead improvement for two CX gates, then investigate cascades and broader groups. Test illegal and overlapping candidates.
6. Add search baselines and benchmark automation. Compare manual plans, simple interaction-graph KL, mixed graph refinement, Qiskit's finder, documented QCut capabilities after validation, and exact small-instance solutions. Report feasible-plan rates, seeds, budgets, costs, and overhead/reconstruction errors together.
7. Add noisy and hardware execution after the statistical path is established. Use the same plan/result interfaces; keep calibration, routing, reset/measurement costs, job counts, and shot budgets in the experiment record.

The first milestone deliverable should be a small circuit whose manual cut plan produces legal fragments and reconstructs a specified expectation value exactly in simulation, followed by a finite-shot result with credible error bars. A large-circuit cut finder without this capability cannot validate the research objective.

Start benchmarks with chain/star/clustered circuits, QAOA with angle sweeps, near-Clifford Clifford+T circuits, and circuits with known eligible groups. Add QFT and dense random circuits as stress tests. Include uncut classical simulation, particularly stabilizer simulation for Clifford inputs; successfully cutting an easy classical circuit does not establish an advantage. Record both original and normalized gate counts, fragment width/depth, cut/group/excision details, realized norm, unique circuits, shots, errors, search/execute/reconstruct time, and peak memory.

**Three research extensions merit early experiments.**

First, make group discovery and partition search interact: detect eligible groups, resolve overlaps, and refine partitions using the realized cost of those groups. The supplied grouping paper's manual group input makes this a clear comparison target. QCut's advertised group logic is also relevant prior software and must be considered when assessing novelty.

Second, choose excision and backend assignment together. An excision may increase reconstruction work yet make a large residual fragment efficiently Clifford-simulatable. Optimizing this tradeoff requires a backend-aware cost model; a uniform physical-qubit constraint should not be imposed on an efficient classical stabilizer backend as though it were a small QPU.

Third, optimize partitioning jointly with reconstruction work and shot allocation. Similar `gamma` values can hide different circuit reuse, measurement grouping, contraction memory, and execution durations. Query-aware causal-cone reduction and locally equivalent rewrites are useful additional dimensions, but each must be assessed against existing literature and fair baselines.

For future hardware, use an execution adapter compatible with [IBM SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2). It exposes shot samples and batching modes useful for the cut-generated measurement circuits. Validate routing and result-bit mappings after transpilation, preserve cut measurement outcomes, and separately account for preparation/reset/measurement error. Independent fragments can be run sequentially on one QPU; multiple connected QPUs are not needed for the initial LO model. Evaluate communicating protocols only after confirming their scheduling and feedforward requirements.

The working default is therefore: exact Pauli expectation reconstruction first; independent LO wire/gate cuts first; sparse excision and a verified joint-cut family next; graph optimization after the execution and cost contracts are trustworthy. Full-distribution reconstruction remains a later, explicit target with its own memory and statistical requirements.
