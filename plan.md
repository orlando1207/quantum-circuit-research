# Mixed Circuit Cutting with Precomputed Sampling-Overhead Costs

## 1. Problem statement

Given a quantum circuit containing only single-qubit and two-qubit unitary gates, and a maximum subcircuit width of `M` qubits, choose a combination of wire cutting, gate cutting, and gate excision that produces executable subcircuits of width at most `M` while minimizing a certified sampling-overhead bound.

All cutting operations use a fixed library of physically executable decomposition templates. Their local costs are computed before the partition search and stored in a lookup table. The partition algorithm searches over operation locations and available templates; it does not recompute quantum decompositions at every search step.

The framework prohibits communication between subcircuits during quantum execution. Shared, preassigned configuration choices and centralized classical post-processing after execution are allowed.

The output is not a single smaller circuit equivalent to the original circuit. It is a collection of parameterized subcircuit families and a classical reconstruction rule that together provide unbiased estimates of the original circuit's target expectation values under ideal execution.

## 2. Scope and baseline assumptions

The initial version adopts the following assumptions:

1. **Gate arity:** Every original gate acts on one or two qubits. Toffoli and other higher-arity gates are outside the initial scope.
2. **Finite gate library:** Gates are drawn from a finite library, such as `H`, `S`, `T`, and `CNOT`. If parameterized gates are included, their parameters are part of the lookup key.
3. **Input preparation:** Initial states are independently preparable product states, with the default being the all-zero computational basis state. Entangled input preparation must be represented explicitly by gates in the input circuit.
4. **Circuit structure:** The input is a unitary circuit with terminal measurements. Original mid-circuit measurements and adaptive control are outside the initial scope. Cutting templates may introduce local measurements and classical records.
5. **No runtime communication:** A subcircuit's operations cannot depend on measurement outcomes obtained from another subcircuit during execution. Local control within one subcircuit is allowed when supported by its template.
6. **Template resources:** Baseline templates use no extra quantum ancillas and no entangled resources shared between subcircuits. Any extension that introduces ancillas must include them in the width accounting.
7. **Independent local decompositions:** Selected templates are composed independently. Cross-cut grouping, joint decomposition optimization, and coefficient cancellation across operations are not included in the baseline objective.
8. **Ideal execution:** Sampling error is considered, but device noise and mitigation are not included in the objective.
9. **Static qubit allocation:** Each resulting wire segment is assigned its own local qubit within a fragment. Dynamic reset-and-reuse scheduling is outside the baseline.
10. **Bounded queries:** The sampling guarantee is stated for a fixed product observable with operator norm at most one, or a separately estimated term of an observable expansion. Simultaneous estimation of many observables or an entire output distribution requires additional analysis.

The partition cost is query-independent under these assumptions. The observable need not be an input to the partition optimizer, but it must be specified when executing and reconstructing a query.

## 3. Inputs

### 3.1 Quantum circuit

Let

$$
C=(Q,\mathcal G,\prec),
$$

where:

- `Q` is the set of original logical qubits;
- `G` is the set of gate instances;
- each gate instance has a gate type, ordered operand list, and optional parameters;
- the relation `prec` records causal ordering along each qubit wire.

Operand order is retained. For example, the control and target of a CNOT are part of its metadata even though the connectivity graph used for partitioning is undirected.

### 3.2 Device width

An integer `M >= 1` specifies the maximum number of simultaneously allocated qubits in any executable subcircuit. It is fixed for a given optimization instance.

### 3.3 Decomposition-template library

The library contains the allowed wire-cutting, gate-cutting, and gate-excision templates. Each entry specifies:

- the operation and gate types it supports;
- its effect on circuit connectivity;
- executable local measurement and preparation instructions;
- real reconstruction coefficients and outcome-dependent signs;
- a coefficient-norm cost and its corresponding sampling-overhead factor;
- resource requirements and communication assumptions.

A method is a candidate only when a compatible template is available. In particular, a numerical gate-cutting cost must not be assigned to an arbitrary two-qubit gate without a supporting decomposition.

## 4. Port-based graph representation

The representation has three kinds of information:

1. A **connectivity graph** describing which circuit regions remain connected.
2. **Operation annotations**, including excision hyperedges, describing selectable modifications of that graph.
3. An **operation-relation graph** describing conflicts or justified pruning relations between candidate operations.

These can be stored in one data structure; they need not be separate software components.

### 4.1 Port vertices and external wire edges

For every gate `g` and operand `q`, create an input port and an output port:

$$
p^{\mathrm{in}}_{g,q},\qquad p^{\mathrm{out}}_{g,q}.
$$

Connect consecutive gates on the same qubit by an external wire edge:

$$
\left(p^{\mathrm{out}}_{g,q},p^{\mathrm{in}}_{h,q}\right).
$$

Add circuit-input and circuit-output terminals to retain wire paths that begin or end outside a gate. Preserve temporal direction as metadata for compilation, even when undirected connectivity is used to identify fragments.

Baseline wire-cut candidates are external edges between consecutive gates. Cutting a wire inside a gate gadget is not an independent wire-cut operation.

### 4.2 Single-qubit gate gadget

A single-qubit gate is represented by two ports and one internal structural edge:

```text
g_in -------- g_out
       g
```

Its excision action is annotated by a two-port hyperedge:

$$
h_g=\{p^{\mathrm{in}}_{g,q},p^{\mathrm{out}}_{g,q}\}.
$$

### 4.3 Two-qubit gate gadget

For a gate acting on ordered operands `q_a` and `q_b`, introduce two dummy nodes:

```text
a_in ---- u_a ---- a_out
           |
           | e_g
           |
b_in ---- u_b ---- b_out
```

The four horizontal edges are structural links. The vertical interaction edge is

$$
e_g=(u_a,u_b).
$$

The gate-excision annotation is a four-port hyperedge:

$$
h_g=\{a_{\mathrm{in}},a_{\mathrm{out}},b_{\mathrm{in}},b_{\mathrm{out}}\}.
$$

The interaction edge represents a gate-cut candidate. The hyperedge represents the scope of a whole-gate excision action.

**Important:** An excision hyperedge is an action annotation, not an additional connectivity path. Otherwise, removing the interaction edge would fail to disconnect the two wire paths because the hyperedge would still connect all four ports.

### 4.4 Graph transformations

| Operation | Selected object | Connectivity transformation | Execution interpretation |
|---|---|---|---|
| Wire cut | External wire edge | Remove that edge | Terminate the upstream segment with a measurement and initialize the downstream segment according to the template |
| Gate cut | Two-qubit interaction edge | Remove the vertical interaction edge; retain both horizontal wire paths | Replace the nonlocal gate by a weighted family of local operations |
| Gate excision | Gate's action hyperedge | Remove all internal links of that gate gadget and discard unused dummy nodes; retain its ports and external edges | Terminate the input ports with measurements and initialize the output ports with template-defined states |

A gate cut does not always replace a gate by two deterministic unitary gates. Its local instructions may include measurements, preparations, and signed classical outcome processing.

Gate excision acts jointly on all ports of the original gate. It is not charged as a collection of separate wire cuts around the gate.

### 4.5 Operation-relation graph

Each vertex of the relation graph represents a candidate operation, including its location and template. A conflict edge means that the two candidates cannot be selected together in the baseline model.

Mandatory conflicts include:

- different templates for the same selected operation;
- gate cutting and gate excision of the same gate;
- transformations that claim incompatible use of the same internal gate structure.

Potential redundancy rules, such as forbidding a wire cut immediately adjacent to an excised gate, are optional pruning rules rather than universal physical-validity conditions. Such rules should be enabled only after showing that they preserve a solution of no greater cost and feasible width under the chosen templates.

## 5. Subcircuits and qubit-width accounting

Apply all selected graph transformations and identify the connected components of the remaining connectivity graph. These components define the canonical circuit fragments.

Counting ports, dummy nodes, or original qubit labels is not a correct width calculation.

For the static-allocation baseline:

1. Follow each original qubit's input-to-output path through surviving external wire edges and surviving same-qubit internal links.
2. Break that path at each wire cut and at each excised gate.
3. The resulting maximal paths are **wire segments**. A gate cut retains the same-qubit paths through its gadget.
4. Each wire segment contained in a fragment requires one local qubit.

Thus,

$$
\operatorname{width}(F)
=\#\{\text{wire segments assigned to }F\}.
$$

Different segments of the same original logical qubit can belong to one connected fragment through other gates. Under static allocation, they must still be counted separately.

Preparation-only, measurement-only, or boundary-to-boundary segments must be retained when they carry a required template operation. They cannot be discarded solely because they contain no original gate. Any analytical elimination of such fragments must preserve both reconstruction and the reported cost model.

The feasibility constraint is

$$
\operatorname{width}(F)\leq M
\qquad\text{for every fragment }F.
$$

Local gate-cutting operations preserve their designated wire paths. For templates with configuration-dependent resource use, width must be checked for the worst-case executable configuration, not just one sampled instance.

## 6. Sampling-overhead metric

### 6.1 Local decomposition and normalization

Each candidate operation `a` has an exact linear decomposition

$$
\mathcal E_a=\sum_{j\in J_a} c_{a,j}\mathcal F_{a,j}.
$$

The maps `F` are implemented by allowed local operations and classical outcome processing. They need not individually be physical quantum channels if they are represented by signed measurement outcomes.

The template must provide an operational normalization: the unscaled single-trial score associated with each branch lies in `[-1,1]`. Any larger magnitude introduced by state normalization or outcome reweighting must be absorbed into the coefficients `c`.

Define the fixed-template coefficient norm and overhead factor:

$$
\Gamma_a=\sum_j|c_{a,j}|,
\qquad
\Omega_a=\Gamma_a^2.
$$

We use capital `Gamma` to emphasize that this is the cost of a specified template, not necessarily the minimum over all possible decompositions.

### 6.2 Composition across selected operations

Let `S` denote a compatible set of selected operations. Under independent composition of their decompositions,

$$
\Gamma_{\mathrm{total}}(S)=\prod_{a\in S}\Gamma_a,
$$

and the objective is

$$
\boxed{
\Omega_{\mathrm{total}}(S)
=\prod_{a\in S}\Omega_a
=\left(\prod_{a\in S}\Gamma_a\right)^2.
}
$$

The empty product equals one. Therefore, an uncut circuit that already satisfies the width bound has baseline overhead one.

For numerical stability and additive graph-search costs, use

$$
\boxed{
\log\Omega_{\mathrm{total}}(S)
=\sum_{a\in S}\log\Omega_a
=2\sum_{a\in S}\log\Gamma_a.
}
$$

This is an exact calculation of the chosen independent-template bound. It is not necessarily the lowest achievable overhead after grouping, merging terms, exploiting circuit context, or optimizing measurement allocation.

### 6.3 Operational meaning

For each selected operation, sample branch `j` with probability

$$
p_{a,j}=\frac{|c_{a,j}|}{\Gamma_a}.
$$

Execute the corresponding fragment configurations, combine their bounded scores and local measurement signs, and multiply by the global coefficient sign and `Gamma_total`.

The resulting estimator `Y` satisfies

$$
\mathbb E[Y]=\mu,
\qquad |Y|\leq\Gamma_{\mathrm{total}},
$$

where `mu` is the target expectation value of the original circuit. For independent reconstruction trials,

$$
\operatorname{Var}(\widehat\mu)
\leq\frac{\Omega_{\mathrm{total}}}{N}.
$$

A sufficient Hoeffding bound for additive error `epsilon` with failure probability at most `delta` is

$$
N\geq
\frac{2\Omega_{\mathrm{total}}}{\epsilon^2}
\ln\frac{2}{\delta}.
$$

This counts complete reconstruction trials, each of which may require executions of multiple fragments. It does not by itself count total device shots, wall-clock time, circuit-loading overhead, or classical reconstruction time. Actual variance may be smaller than this bound.

## 7. Precomputing gate-excision costs

### 7.1 Pauli expansion

For an `n`-qubit unitary gate `U`, with `n` equal to one or two, let `d=2^n` and use the unnormalized Hermitian Pauli basis

$$
\mathcal P_n=\{I,X,Y,Z\}^{\otimes n}.
$$

Define

$$
w_U(P,P')
=\frac{1}{d^2}\operatorname{Tr}(UPU^\dagger P').
$$

Then, for any operator `A`,

$$
UAU^\dagger
=\sum_{P,P'\in\mathcal P_n}
w_U(P,P')\operatorname{Tr}(AP)P'.
$$

The number of candidate Pauli pairs is

$$
|\mathcal P_n|^2=16^n,
$$

which is 16 for a single-qubit gate and 256 for a two-qubit gate.

Define the raw configuration count

$$
L_U=\#\{(P,P'):w_U(P,P')\neq0\}.
$$

This count is recorded as auxiliary information, not used as a substitute for sampling overhead.

### 7.2 Physically executable preparation template

A Pauli operator `P'` is generally not a density matrix. Expand it into product-state eigenprojectors:

$$
P'=\sum_{s=1}^{d}\eta_{P',s}\rho_{P',s},
\qquad \eta_{P',s}\in\{-1,+1\}.
$$

For identity factors, a fixed computational-basis expansion may be used. Every `rho` is a normalized, physically preparable product state.

The expanded coefficient is

$$
c_{P,P',s}=w_U(P,P')\eta_{P',s}.
$$

The associated signed map is estimated by measuring `P` on the incoming ports, preparing `rho` on the outgoing ports, and retaining the measurement sign in classical post-processing. Preparation does not depend on the realized upstream outcome, so the construction requires no runtime inter-fragment communication.

For this specific, unmerged template,

$$
\boxed{
\Gamma_{\mathrm{exc}}(U)
=d\sum_{P,P'}|w_U(P,P')|,
\qquad
\Omega_{\mathrm{exc}}(U)
=d^2\left(\sum_{P,P'}|w_U(P,P')|\right)^2.
}
$$

The raw expanded branch count is `d L_U`. Distinct executable configurations can sometimes be shared or merged, so this is not automatically the number of unique device circuits that must be loaded.

These formulas define the baseline excision template. They are not claims of optimal gate-excision overhead.

### 7.3 Relation to configuration count

For unitary `U` and the normalization above,

$$
\sum_{P,P'}|w_U(P,P')|^2=1.
$$

To see this, the Pauli transfer matrix

$$
R_{P',P}=\frac{1}{d}\operatorname{Tr}(P'UPU^\dagger)=d\,w_U(P,P')
$$

is a real orthogonal matrix of dimension `d^2`, and therefore its squared Frobenius norm is `d^2`.

Cauchy-Schwarz gives the template bound

$$
\Omega_{\mathrm{exc}}(U)\leq d^2L_U=4^nL_U.
$$

The optimizer should use the directly computed coefficient norm, not this looser count-only bound, because all coefficients are inexpensive to compute for the supported gate sizes.

### 7.4 Wire-cut and gate-cut entries

Wire cutting is an identity-channel decomposition. For example, the eight-branch single-qubit Peng-style template with coefficient magnitudes `1/2` has

$$
\Gamma_{\mathrm{wire}}=4,
\qquad \Omega_{\mathrm{wire}}=16.
$$

This is a template-specific cost, not a claim that every wire-cut protocol has overhead 16.

For gate cutting, store `Gamma_gate` and `Omega_gate` from the selected valid decomposition, using the same bounded-score convention. Gate-cutting and gate-excision entries for the same original gate must remain distinct because they induce different connectivity transformations.

### 7.5 Lookup-table schema

A suggested key is:

```text
(gate_type, ordered_operand_roles, gate_parameters,
 operation_type, template_id, resource_model)
```

Original qubit indices do not need to be part of the key when the template is invariant under relabeling. Operand roles, such as CNOT control and target, must still be preserved.

A suggested entry contains:

```text
arity
nonzero_pauli_coefficients          # excision entries
raw_pauli_pair_count                # excision entries
expanded_branch_description
local_measurement_and_preparation_instructions
coefficient_norm_Gamma
sampling_overhead_Omega
log_sampling_overhead
connectivity_transformation
extra_qubit_requirements
runtime_communication_required
normalization_convention
validation_status
```

Precomputation must use a consistent Pauli normalization. Numerical rounding must not silently discard nonzero coefficients: truncating terms can introduce bias. Prefer exact expressions for standard gates or explicitly controlled numerical tolerances with reconstruction-residual checks. Parameterized gates require parameter-aware cache keys; approximate parameter binning is outside the exact baseline unless its induced error is explicitly budgeted.

## 8. Decision variables and optimization problem

Let `A` be the set of candidate operations. Each candidate includes its location, operation type, and decomposition template. Introduce a binary variable

$$
x_a\in\{0,1\}\qquad(a\in\mathcal A).
$$

The selected set is `S(x) = {a : x_a = 1}`. The baseline optimization problem is

$$
\boxed{
\min_x\sum_{a\in\mathcal A}x_a\log\Omega_a
}
$$

subject to the following constraints.

### 8.1 Operation compatibility

At most one wire-cut template is selected for each external wire edge. At most one gate operation is selected for each original gate:

$$
\sum_{t\in T_g^{\mathrm{gate}}}x_{g,t}^{\mathrm{gate}}
+\sum_{t\in T_g^{\mathrm{exc}}}x_{g,t}^{\mathrm{exc}}
\leq1.
$$

The gate-cut template set is empty for a single-qubit gate. Selecting neither gate cutting nor excision means retaining that original gate; adjacent external wires may still be cut.

### 8.2 Relation constraints

For every mandatory conflict pair `(a,b)`,

$$
x_a+x_b\leq1.
$$

Any optional dominance or redundancy pruning must be documented separately from mandatory validity constraints.

### 8.3 Width constraints

After applying the selected transformations, every resulting fragment must satisfy

$$
\operatorname{width}(F)\leq M.
$$

These constraints depend on transformed graph connectivity and wire-segment accounting; they are not assumed to be linear in the decision variables.

### 8.4 Executability and exact reconstruction

All selected templates must be compatible with the no-communication resource model, compile into valid local circuits, and satisfy their exact linear reconstruction identities.

The goal is a global optimum within the fixed template library and baseline cost model. A heuristic may return a feasible approximate solution, but must not label it globally optimal without a certificate or an exact solver result.

## 9. Required output

The algorithm returns a partition plan with the following fields:

1. **Selected operations:** Locations, operation types, and template identifiers.
2. **Transformed graph:** Remaining connectivity and the mapping from original gates and ports to fragments.
3. **Subcircuit families:** For each fragment, its retained gates, local wire-segment allocation, and configuration-dependent measurement, preparation, and gate instructions. Explicit enumeration of all joint configurations is not required.
4. **Reconstruction specification:** Coefficients, branch-sampling distributions, measurement-sign rules, and the classical estimator used to reconstruct a requested expectation value.
5. **Resource report:** Fragment count, width of each fragment, and all required boundary operations.
6. **Cost report:** Each selected operation's `Gamma` and `Omega`, the total log-overhead, and the implied independent-template overhead bound.
7. **Solution status:** Feasibility, applicable pruning rules, and whether optimality is certified or the result is heuristic.

The reconstructed expectation value is exact in expectation under ideal sampling. Finite-shot estimates have statistical error. Independently executed fragments do not physically recreate the original global entangled state.

## 10. Validation requirements

Before using a template in the search, validate:

- **Algebra:** Its decomposition reconstructs the target channel, for example by checking its action on every Pauli basis operator.
- **Normalization:** Every claimed bounded branch score is actually bounded by one before coefficient reweighting.
- **Resources:** The template uses no prohibited inter-fragment communication or unaccounted ancillas.
- **Graph semantics:** The reported connectivity transformation matches the executable local operations.

For a returned partition, validate:

- all operation conflicts are respected;
- every original operation is retained or accounted for by exactly one selected gate transformation;
- fragment ordering respects the original circuit's causal structure;
- every required boundary measurement and preparation is present;
- all executable configurations satisfy the width bound;
- the reported total cost equals the product of selected template costs, or equivalently the sum of their log-costs.

These checks are part of the formulation's correctness contract, not a commitment to any particular search algorithm.

## 11. Deliberately excluded objectives and extensions

The baseline minimizes only the independent-template sampling-overhead bound. It does not directly minimize:

- the number of nonzero Pauli pairs or distinct executable configurations;
- exact, state-dependent estimator variance;
- total fragment shots or execution time;
- fragment depth, gate count, noise, or reconstruction-memory use.

Potential later extensions include gate grouping, circuit re-synthesis, observable-aware costs, optimized shot allocation, template optimization, dynamic qubit reuse, and multi-objective execution cost. Each extension changes the assumptions or objective and should be evaluated separately rather than silently included in the baseline.

The intended contribution of a graph-based algorithm is to search effectively over the different connectivity transformations and their calibrated costs. The formulation does not prescribe simulated annealing, a genetic algorithm, or any other solver as its novelty.

## 12. Theoretical grounding and attribution

- **General QPD sampling guarantees and signed measurement processing:** Piveteau, Schmitt, and Sutter, [Circuit cutting with classical side information](https://arxiv.org/html/2503.22384), 2025. Section 2.2 gives the coefficient-norm sampling guarantee; subsequent sections formalize classical measurement information through quantum instruments and distinguish communication models.
- **Optimization over decompositions:** Piveteau, Sutter, and Woerner, [Quasiprobability decompositions with reduced sampling overhead](https://arxiv.org/abs/2101.09290), published in 2022. Its primary setting is error mitigation; it provides methodological background rather than a gate-excision-specific optimality theorem.
- **Configuration count versus sampling overhead:** Harada, Wada, and Yamamoto, [Doubly Optimal Parallel Wire Cutting without Ancilla Qubits](https://doi.org/10.1103/PRXQuantum.5.040308), 2024. It studies both metrics, but in an LOCC setting; its optimal costs cannot automatically be imported into this no-runtime-communication framework.
- **Excision starting point:** The supplied paper, *A Measure-and-Prepare Framework for Arbitrary Quantum Gate Excision and its Applications*, supplies the Pauli measure-and-prepare expansion motivating the excision operation. The explicit physical-state expansion and fixed-template overhead derivation in Section 7 are the operational cost model adopted in this formulation, not an attribution of an optimal-overhead theorem to that paper.

## 13. Compact formulation

> **Input:** A quantum circuit of one- and two-qubit unitary gates, a maximum fragment width `M`, and a validated library of no-runtime-communication cutting templates with precomputed sampling-overhead costs.
>
> **Decision:** Select compatible wire cuts, gate cuts, and whole-gate excisions on a port-based graph.
>
> **Constraint:** Every resulting executable subcircuit uses at most `M` qubits under the stated allocation model.
>
> **Objective:** Minimize the product of selected templates' squared coefficient norms, equivalently the sum of their log-overhead costs.
>
> **Output:** Executable subcircuit families, their partition metadata, and an unbiased classical reconstruction procedure, together with a certified template-based sampling-overhead bound.
