"""Reproducible multi-start annealing over whole channel actions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
from math import exp, log
from statistics import median
from time import perf_counter

import numpy as np

from .partition_graph import GraphEvaluation, PartitionGraph


@dataclass(frozen=True)
class AnnealConfig:
    max_qubits: int
    partition_mode: str = "two_blocks"
    steps: int = 5000
    cooling_rate: float = 0.93
    temperature_levels: int = 120
    initial_acceptance: float = 0.8
    block_probability: float = 0.3
    block_size: int = 4
    pilot_steps: int = 64
    penalty_multiplier: float = 1.0
    allow_wire: bool = True
    allow_gate: bool = True
    search_space: str = "labels"
    interval_polish: bool = True
    proposal_balance: str = "variables"
    objective: str = "overhead"
    max_components: int | None = None
    cache_size: int = 4096
    split_probability: float = 0.25
    group_heatbath_probability: float = 0.0
    temperature_calibration: str = "all"
    max_evaluations: int | None = None

    def validate(self) -> None:
        if isinstance(self.max_qubits, bool) or not isinstance(self.max_qubits, int):
            raise ValueError("max_qubits must be an integer.")
        for name in ("max_qubits", "steps", "temperature_levels", "block_size", "cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.pilot_steps, int) or self.pilot_steps < 0:
            raise ValueError("pilot_steps must be a nonnegative integer.")
        if self.partition_mode not in ("two_blocks", "components", "multiway"):
            raise ValueError("Unknown partition mode.")
        if self.objective not in ("overhead", "fragment_shots"):
            raise ValueError("Unknown objective.")
        if self.search_space not in ("labels", "actions"):
            raise ValueError("Unknown search space.")
        if self.proposal_balance not in ("variables", "types"):
            raise ValueError("Unknown proposal balance.")
        if not 0 < self.cooling_rate < 1 or not 0 < self.initial_acceptance < 1:
            raise ValueError("Cooling rate and initial acceptance must lie strictly in (0,1).")
        if not 0 <= self.block_probability <= 1 or self.penalty_multiplier <= 0:
            raise ValueError("Invalid proposal probability or penalty multiplier.")
        if not np.isfinite(self.penalty_multiplier):
            raise ValueError("Penalty multiplier must be finite.")
        if not 0 < self.split_probability < 1:
            raise ValueError("Split probability must lie strictly in (0,1).")
        if not 0 <= self.group_heatbath_probability <= 1:
            raise ValueError("group_heatbath_probability must lie in [0,1].")
        if self.temperature_calibration not in {"all", "feasible"}:
            raise ValueError("temperature_calibration must be all or feasible.")
        if self.max_evaluations is not None and (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 2
        ):
            raise ValueError("max_evaluations must be an integer at least two.")
        if self.max_components is not None and (
            isinstance(self.max_components, bool)
            or not isinstance(self.max_components, int)
            or self.max_components < 1
        ):
            raise ValueError("max_components must be a positive integer.")


@dataclass(frozen=True)
class AnnealResult:
    seed: int
    config: AnnealConfig
    best: GraphEvaluation | None
    initial_log_overhead: float
    initial_temperature: float
    final_temperature: float
    penalty_weight: float
    proposals: int
    evaluations: int
    accepted_moves: int
    changed_proposals: int
    block_proposals: int
    elapsed_seconds: float
    trace: tuple[dict, ...]
    best_before_polish: float | None = None
    polish_evaluations: int = 0
    proposed_action_changes: dict | None = None
    accepted_action_changes: dict | None = None
    initial_feasible: bool = True
    heatbath_proposals: int = 0
    heatbath_improvements: int = 0
    budget_exhausted: bool = False
    calibration_samples: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.best:
            result["best"]["sampling_overhead"] = self.best.sampling_overhead
        result["solution_status"] = "heuristic-feasible" if self.best else "no-feasible-incumbent"
        result["penalty_dominance_guarantee"] = self.config.penalty_multiplier >= 1
        return result


def allowed_options(graph: PartitionGraph, config: AnnealConfig) -> tuple[tuple[int, ...], ...]:
    flags = {
        "wire": config.allow_wire,
        "gate": config.allow_gate,
    }
    return tuple(
        tuple(
            index
            for index, option in enumerate(variable.options)
            if option is None or flags.get(option.kind, False)
        )
        for variable in graph.variables
    )


def seed_states(
    graph: PartitionGraph, config: AnnealConfig, *, count: int = 16, seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Strong spatial seeds plus different feasible/all-retained initial states.

    Up to 14 qubits, score all spatial bipartitions, without calling the quantum
    simulator. This is a spatial upper-bound baseline, not a full optimum.
    """
    config.validate()
    n = graph.problem.circuit.num_qubits
    if config.partition_mode == "two_blocks" and n > 2 * config.max_qubits:
        raise ValueError("Two static blocks cannot hold even the original input wire segments.")
    allowed = allowed_options(graph, config)
    zero = (0,) * len(graph.variables)
    ranked = []
    crossings = []
    for index, variable in enumerate(graph.variables):
        if variable.kind != "gate":
            continue
        cuts = [
            value
            for value in allowed[index]
            if variable.options[value] and variable.options[value].kind == "gate"
        ]
        if not cuts:
            crossings = []
            break
        cut = min(cuts, key=lambda v: variable.options[v].log_overhead)
        crossings.append(
            (
                index,
                graph.nodes[variable.ports[0]]["qubit"],
                graph.nodes[variable.ports[1]]["qubit"],
                cut,
            )
        )
    rng = np.random.default_rng(seed)
    if crossings:
        if n <= 14 and n <= 2 * config.max_qubits:
            colors_iter = ((0,) + tail for tail in product((0, 1), repeat=n - 1))
        else:
            groups = max(2, (n + config.max_qubits - 1) // config.max_qubits)
            colors_iter = []
            for _ in range(max(64, count * 8)):
                permutation = rng.permutation(n)
                colors = [0] * n
                for position, q in enumerate(permutation):
                    colors[int(q)] = position * groups // n
                colors_iter.append(tuple(colors))
            if n <= 2 * config.max_qubits:
                colors_iter.extend(
                    tuple(int(q >= boundary) for q in range(n))
                    for boundary in range(
                        max(1, n - config.max_qubits), min(n, config.max_qubits + 1)
                    )
                )
                # Graph-traversal spatial seeds capture cycles/chains whose qubit
                # indices are shuffled, without relying on a benchmark-specific order.
                neighbors = [set() for _ in range(n)]
                for _, left, right, _ in crossings:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
                for root in range(min(n, 8)):
                    order, seen = [], set()
                    for start in (root, *range(n)):
                        stack = [start]
                        while stack:
                            q = stack.pop()
                            if q in seen:
                                continue
                            seen.add(q)
                            order.append(q)
                            stack.extend(sorted(neighbors[q] - seen, reverse=True))
                    for boundary in range(
                        max(1, n - config.max_qubits), min(n, config.max_qubits + 1)
                    ):
                        colors = [1] * n
                        for q in order[:boundary]:
                            colors[q] = 0
                        colors_iter.append(tuple(colors))
        for colors in colors_iter:
            sizes = [colors.count(color) for color in set(colors)]
            if max(sizes) > config.max_qubits:
                continue
            state = list(zero)
            cost = 0.0
            for index, left, right, cut in crossings:
                if colors[left] != colors[right]:
                    state[index] = cut
                    cost += graph.variables[index].options[cut].log_overhead
            if graph.grouping is not None:
                scored = graph.evaluate(
                    tuple(state),
                    max_qubits=config.max_qubits,
                    partition_mode=config.partition_mode,
                    max_components=config.max_components,
                    validate=False,
                )
                cost = scored.log_overhead
            ranked.append((cost, tuple(state)))
        ranked.sort()
    candidates = [state for _, state in ranked[: count * 2]]
    all_cut = []
    for index, variable in enumerate(graph.variables):
        choices = [
            value
            for value in allowed[index]
            if variable.options[value] and variable.options[value].kind == "gate"
        ]
        all_cut.append(choices[0] if choices else 0)
    candidates.extend((tuple(all_cut), zero))
    if config.allow_wire:
        candidates.append(
            tuple(1 if variable.kind == "wire" else 0 for variable in graph.variables)
        )
    # One-wire proposals are cheap, reproducible structural seeds. Record their
    # cost separately; matching a paper baseline at initialization is not an SA gain.
    if config.allow_wire:
        for index, variable in enumerate(graph.variables):
            if variable.kind == "wire":
                single = list(zero)
                single[index] = 1
                candidates.append(tuple(single))
    if config.partition_mode != "two_blocks":
        from .multiway import bridge_seed_state, merge_seed_states

        for include_gates in (False, True):
            bridge = bridge_seed_state(
                graph, allowed, max_qubits=config.max_qubits, include_gates=include_gates
            )
            if bridge is not None:
                candidates.append(bridge)
        candidates.extend(
            merge_seed_states(
                graph, allowed, max_qubits=config.max_qubits, count=min(count, 12), seed=seed
            )
        )
    unique = list(dict.fromkeys(candidates))
    feasible, infeasible, diverse = [], [], {}
    for state in unique:
        evaluation = graph.evaluate(
            state,
            max_qubits=config.max_qubits,
            partition_mode=config.partition_mode,
            max_components=config.max_components,
            validate=False,
        )
        score = evaluation.log_overhead + (
            log(len(evaluation.block_widths)) if config.objective == "fragment_shots" else 0
        )
        (feasible if evaluation.feasible else infeasible).append((score, state))
        if evaluation.feasible and config.partition_mode != "two_blocks":
            signature = (
                len(evaluation.block_widths),
                bool(evaluation.action_counts["wire"]),
                bool(evaluation.action_counts["gate"]),
            )
            diverse[signature] = min(diverse.get(signature, (float("inf"), ())), (score, state))
    feasible.sort()
    infeasible.sort()
    if feasible and diverse:
        # Preserve temporal/multi-fragment seeds even when many cheap spatial
        # partitions would otherwise occupy every slot of the initial pool.
        ordered = feasible[: max(1, count // 2)] + sorted(diverse.values()) + feasible
        selected = list(dict.fromkeys(ordered))[:count]
    else:
        selected = feasible[:count] or infeasible[:count] or [(0, zero)]
    return tuple(state for _, state in selected)


def anneal(
    graph: PartitionGraph,
    config: AnnealConfig,
    *,
    seed: int = 0,
    initial_state: tuple[int, ...] | None = None,
) -> AnnealResult:
    start = perf_counter()
    config.validate()
    if graph.graph_schema not in {"temporal-incidence-v2", "temporal-incidence-grouped-v3"}:
        raise ValueError("Core SA requires the gate/wire temporal incidence graph.")
    graph.problem.validate_snapshot()
    if (
        config.partition_mode == "two_blocks"
        and graph.problem.circuit.num_qubits > 2 * config.max_qubits
    ):
        raise ValueError("Two-block static allocation is provably infeasible.")
    allowed = allowed_options(graph, config)
    mutable = tuple(index for index, options in enumerate(allowed) if len(options) > 1)
    mutable_set = set(mutable)
    rng = np.random.default_rng(seed)
    state = initial_state or seed_states(graph, config, count=1, seed=seed)[0]
    graph.validate_state(state)
    if any(value not in choices for value, choices in zip(state, allowed)):
        raise ValueError("Initial state uses a disabled action.")
    costs = [
        graph.variables[index].options[value].log_overhead
        for index, choices in enumerate(allowed)
        for value in choices
        if value
    ]
    scale = max(1.0, median(costs)) if costs else 1.0
    upper = sum(
        max([0.0] + [graph.variables[i].options[v].log_overhead for v in choices if v])
        for i, choices in enumerate(allowed)
    )
    if config.objective == "fragment_shots":
        upper += log(max(1, len(graph.nodes)))
    penalty_weight = config.penalty_multiplier * (upper + scale)

    class BudgetExhausted(Exception):
        pass

    actual_evaluations = 0
    budget_exhausted = False

    @lru_cache(maxsize=config.cache_size)
    def evaluate(candidate):
        nonlocal actual_evaluations
        if config.max_evaluations is not None and actual_evaluations >= config.max_evaluations:
            raise BudgetExhausted
        actual_evaluations += 1
        return graph.evaluate(
            candidate,
            max_qubits=config.max_qubits,
            partition_mode=config.partition_mode,
            max_components=config.max_components,
            validate=False,
        )

    def objective(evaluation):
        value = evaluation.log_overhead
        if config.objective == "fragment_shots":
            value += log(len(evaluation.block_widths))
        return value

    def rank(evaluation):
        return objective(evaluation), len(evaluation.block_widths), evaluation.state

    def energy(evaluation):
        return objective(evaluation) + penalty_weight * evaluation.width_penalty

    neighborhoods = []
    for raw in getattr(graph, "macro_neighborhoods", ()):
        region = tuple(v for v in raw if v in mutable_set)
        if 1 < len(region) <= config.block_size:
            neighborhoods.append(region)
    neighborhoods = list(dict.fromkeys(neighborhoods))

    def proposal(current):
        if not mutable:
            return current, False, 0.0
        new = list(current)
        block = rng.random() < config.block_probability and config.block_size > 1
        if block:
            if neighborhoods and rng.random() < 0.7:
                region = neighborhoods[int(rng.integers(len(neighborhoods)))]
            else:
                region_set = {mutable[int(rng.integers(len(mutable)))]}
                while len(region_set) < config.block_size:
                    frontier = sorted(
                        set().union(*(graph.adjacency[v] for v in region_set))
                        & mutable_set - region_set
                    )
                    if not frontier:
                        break
                    region_set.add(frontier[int(rng.integers(len(frontier)))])
                region = tuple(sorted(region_set))
            for index in region:
                choices = allowed[index]
                new[index] = choices[int(rng.integers(len(choices)))]
        else:
            if config.proposal_balance == "types":
                groups = [
                    tuple(v for v in mutable if graph.variables[v].kind == kind)
                    for kind in ("gate", "wire")
                ]
                groups = [group for group in groups if group]
                group = groups[int(rng.integers(len(groups)))]
                index = group[int(rng.integers(len(group)))]
            else:
                index = mutable[int(rng.integers(len(mutable)))]
            choices = [v for v in allowed[index] if v != current[index]]
            new[index] = choices[int(rng.integers(len(choices)))]
        return tuple(new), block, 0.0

    search_labels = config.search_space == "labels"
    original_grouped_start = evaluate(state) if graph.grouping is not None else None
    if search_labels:
        from .label_moves import LabelMoves, MultiwayLabelMoves

        binary = config.partition_mode == "two_blocks"
        moves = (
            LabelMoves(graph, allowed, config.block_size)
            if binary
            else MultiwayLabelMoves(graph, allowed, config.block_size, config.split_probability)
        )
        state = graph.labels_from_evaluation(evaluate(state))

        def proposal(current):
            result = moves.propose(current, rng, config.block_probability)
            return (*result, 0.0) if binary else result

    def evaluate_proposal(candidate):
        return evaluate(graph.state_from_labels(candidate) if search_labels else candidate)

    current = evaluate_proposal(state)
    initial_feasible = current.feasible
    best = current if current.feasible else None
    initial_log = current.log_overhead
    if original_grouped_start is not None:
        initial_feasible = original_grouped_start.feasible
        initial_log = original_grouped_start.log_overhead
        if original_grouped_start.feasible and (
            best is None or rank(original_grouped_start) < rank(best)
        ):
            # Rebuilding labels can change the inner grouping/packing heuristic.
            # Preserve the supplied feasible incumbent even if normalization loses it.
            best = original_grouped_start
    positive = []
    pilot_proposals = 0
    for _ in range(config.pilot_steps if mutable else 0):
        pilot_proposals += 1
        candidate, _, _ = proposal(state)
        try:
            evaluation = evaluate_proposal(candidate)
        except BudgetExhausted:
            budget_exhausted = True
            break
        difference = energy(evaluation) - energy(current)
        eligible = config.temperature_calibration == "all" or (
            current.feasible and evaluation.feasible
        )
        if difference > 0 and eligible:
            positive.append(difference)
        if evaluation.feasible and (best is None or rank(evaluation) < rank(best)):
            best = evaluation
    initial_temperature = (median(positive) if positive else scale) / -log(
        config.initial_acceptance
    )
    temperature = initial_temperature
    accepted = changed = blocks = 0
    proposed_changes = {
        f"{kind}_{action}": 0 for kind in ("gate", "wire") for action in ("added", "removed")
    }
    accepted_changes = proposed_changes.copy()
    trace = []
    steps = config.steps if mutable and not budget_exhausted else 0
    completed_steps = heatbath_proposals = heatbath_improvements = 0
    for step in range(steps):
        level = min(config.temperature_levels - 1, step * config.temperature_levels // steps)
        if config.max_evaluations is not None:
            level = max(
                level,
                min(
                    config.temperature_levels - 1,
                    actual_evaluations * config.temperature_levels // config.max_evaluations,
                ),
            )
        temperature = max(1e-12, initial_temperature * config.cooling_rate**level)
        completed_steps += 1
        use_heatbath = (
            search_labels
            and config.group_heatbath_probability > 0
            and bool(moves.group_orbits)
            and rng.random() < config.group_heatbath_probability
        )
        if use_heatbath:
            from .group_heatbath import choose_pair, orbit, probabilities

            heatbath_proposals += 1
            pool = (
                moves.spatial_orbits
                if moves.spatial_orbits and rng.random() < 0.5
                else moves.group_orbits
            )
            region_pair = pool[int(rng.integers(len(pool)))]
            palette = 2 if binary else moves.palette_size
            pair = (0, 1) if binary else choose_pair(state, palette, rng, config.split_probability)
            states = orbit(state, region_pair, *pair)
            evaluations = []
            try:
                for proposed in states:
                    ev = evaluate_proposal(proposed)
                    evaluations.append(ev)
                    if ev.feasible and (best is None or rank(ev) < rank(best)):
                        best = ev
                        heatbath_improvements += 1
            except BudgetExhausted:
                budget_exhausted = True
                break
            weights = probabilities(
                states,
                [energy(ev) for ev in evaluations],
                temperature,
                pair,
                palette,
                binary,
                config.split_probability,
            )
            selected = int(rng.choice(len(states), p=weights))
            candidate, evaluation = states[selected], evaluations[selected]
            is_block, log_ratio = True, 0.0
        else:
            candidate, is_block, log_ratio = proposal(state)
        blocks += int(is_block)
        if candidate != state:
            changed += 1
            if not use_heatbath:
                try:
                    evaluation = evaluate_proposal(candidate)
                except BudgetExhausted:
                    budget_exhausted = True
                    break
            changes = proposed_changes.copy()
            for key in changes:
                changes[key] = 0
            for variable, before, after in zip(graph.variables, current.state, evaluation.state):
                if before != after:
                    changes[f"{variable.kind}_{'added' if after else 'removed'}"] += 1
            for key, value in changes.items():
                proposed_changes[key] += value
            if evaluation.feasible and (best is None or rank(evaluation) < rank(best)):
                best = evaluation
            difference = energy(evaluation) - energy(current)
            log_acceptance = 0.0 if use_heatbath else -difference / temperature + log_ratio
            if log_acceptance >= 0 or rng.random() < exp(max(-745, log_acceptance)):
                state, current = candidate, evaluation
                accepted += 1
                for key, value in changes.items():
                    accepted_changes[key] += value
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            trace.append(
                {
                    "step": step + 1,
                    "temperature": temperature,
                    "current_feasible": current.feasible,
                    "current_log_overhead": current.log_overhead,
                    "best_log_overhead": best.log_overhead if best else None,
                    "current_action_counts": current.action_counts,
                    "current_fragments": len(current.block_widths),
                    "best_fragments": len(best.block_widths) if best else None,
                }
            )
    before_polish = best.sampling_overhead if best else None
    polish_start = actual_evaluations
    try:
        if best and search_labels and not budget_exhausted:
            regions = moves.polish_regions if config.interval_polish else moves.units
            improved = True
            while improved:
                improved = False
                labels = graph.labels_from_evaluation(best)
                normalized = evaluate_proposal(labels)
                if normalized.feasible and rank(normalized) < rank(best):
                    best, improved = normalized, True
                    labels = graph.labels_from_evaluation(best)
                for region in regions:
                    alternatives = (
                        (moves.flip(labels, region),)
                        if binary
                        else moves.alternatives(labels, region)
                    )
                    for alternative in alternatives:
                        evaluation = evaluate_proposal(alternative)
                        if evaluation.feasible and rank(evaluation) < rank(best):
                            best, improved = evaluation, True
                            labels = graph.labels_from_evaluation(best)
        elif best and not budget_exhausted:
            # Deterministic coordinate descent is reported separately from annealing moves.
            improved = True
            while improved:
                improved = False
                for index in mutable:
                    for value in allowed[index]:
                        trial = list(best.state)
                        trial[index] = value
                        evaluation = evaluate(tuple(trial))
                        if evaluation.feasible and rank(evaluation) < rank(best):
                            best, improved = evaluation, True
    except BudgetExhausted:
        budget_exhausted = True
    return AnnealResult(
        seed,
        config,
        best,
        initial_log,
        initial_temperature,
        temperature,
        penalty_weight,
        completed_steps + pilot_proposals,
        actual_evaluations,
        accepted,
        changed,
        blocks,
        perf_counter() - start,
        tuple(trace),
        before_polish,
        actual_evaluations - polish_start,
        proposed_changes,
        accepted_changes,
        initial_feasible,
        heatbath_proposals,
        heatbath_improvements,
        budget_exhausted,
        len(positive),
    )


def _run_one(arguments):
    graph, config, seed, state = arguments
    return anneal(graph, config, seed=seed, initial_state=state)


@dataclass(frozen=True)
class RefinementResult:
    best: GraphEvaluation
    evaluations: int
    elapsed_seconds: float


def refine_partition(graph, config, initial_state) -> RefinementResult:
    """Deterministic feasible interval descent; report separately from SA gains."""
    from .label_moves import LabelMoves, MultiwayLabelMoves

    started = perf_counter()
    config.validate()
    graph.validate_state(initial_state)
    allowed = allowed_options(graph, config)
    if any(value not in choices for value, choices in zip(initial_state, allowed)):
        raise ValueError("Initial state uses a disabled action.")

    @lru_cache(maxsize=config.cache_size)
    def evaluate(state):
        return graph.evaluate(
            state,
            max_qubits=config.max_qubits,
            partition_mode=config.partition_mode,
            max_components=config.max_components,
            validate=False,
        )

    def rank(evaluation):
        cost = evaluation.log_overhead
        if config.objective == "fragment_shots":
            cost += log(len(evaluation.block_widths))
        return cost, len(evaluation.block_widths), evaluation.state

    best = evaluate(initial_state)
    if not best.feasible:
        raise ValueError("Refinement needs a feasible initial plan.")
    binary = config.partition_mode == "two_blocks"
    moves = (
        LabelMoves(graph, allowed, config.block_size)
        if binary
        else MultiwayLabelMoves(graph, allowed, config.block_size, config.split_probability)
    )
    improved = True
    while improved:
        improved = False
        labels = graph.labels_from_evaluation(best)
        normalized = evaluate(graph.state_from_labels(labels))
        if normalized.feasible and rank(normalized) < rank(best):
            best, improved = normalized, True
            labels = graph.labels_from_evaluation(best)
        for region in moves.polish_regions:
            alternatives = (
                (moves.flip(labels, region),) if binary else moves.alternatives(labels, region)
            )
            for alternative in alternatives:
                evaluation = evaluate(graph.state_from_labels(alternative))
                if evaluation.feasible and rank(evaluation) < rank(best):
                    best, improved = evaluation, True
                    labels = graph.labels_from_evaluation(best)
    return RefinementResult(best, evaluate.cache_info().misses, perf_counter() - started)


def multi_start(
    graph: PartitionGraph,
    config: AnnealConfig,
    *,
    seeds: tuple[int, ...] = tuple(range(10)),
    workers: int = 1,
    initial_states: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[AnnealResult, ...]:
    if not seeds or workers < 1:
        raise ValueError("Provide at least one seed and a positive worker count.")
    pool = initial_states or seed_states(graph, config)
    jobs = [(graph, config, seed, pool[i % len(pool)]) for i, seed in enumerate(seeds)]
    if workers == 1:
        return tuple(_run_one(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(_run_one, jobs))
