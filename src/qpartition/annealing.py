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
    allow_excision: bool = True
    objective: str = "overhead"
    max_components: int | None = None
    cache_size: int = 4096

    def validate(self) -> None:
        if isinstance(self.max_qubits, bool) or not isinstance(self.max_qubits, int):
            raise ValueError("max_qubits must be an integer.")
        for name in ("max_qubits", "steps", "temperature_levels", "block_size", "cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.pilot_steps, int) or self.pilot_steps < 0:
            raise ValueError("pilot_steps must be a nonnegative integer.")
        if self.partition_mode not in ("two_blocks", "components"):
            raise ValueError("Unknown partition mode.")
        if self.objective not in ("overhead", "fragment_shots"):
            raise ValueError("Unknown objective.")
        if not 0 < self.cooling_rate < 1 or not 0 < self.initial_acceptance < 1:
            raise ValueError("Cooling rate and initial acceptance must lie strictly in (0,1).")
        if not 0 <= self.block_probability <= 1 or self.penalty_multiplier <= 0:
            raise ValueError("Invalid proposal probability or penalty multiplier.")
        if not np.isfinite(self.penalty_multiplier):
            raise ValueError("Penalty multiplier must be finite.")


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
        "excision": config.allow_excision,
    }
    return tuple(
        tuple(
            index
            for index, option in enumerate(variable.options)
            if option is None or flags[option.kind]
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
        if variable.kind != "gate" or len(variable.ports) != 4:
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
                graph.nodes[variable.ports[2]]["qubit"],
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
        if variable.kind == "gate" and len(variable.ports) == 4 and not choices:
            choices = [
                value
                for value in allowed[index]
                if variable.options[value] and variable.options[value].kind == "excision"
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
    unique = list(dict.fromkeys(candidates))
    feasible, infeasible = [], []
    for state in unique:
        evaluation = graph.evaluate(
            state,
            max_qubits=config.max_qubits,
            partition_mode=config.partition_mode,
            max_components=config.max_components,
            validate=False,
        )
        (feasible if evaluation.feasible else infeasible).append((evaluation.log_overhead, state))
    feasible.sort()
    infeasible.sort()
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

    @lru_cache(maxsize=config.cache_size)
    def evaluate(candidate):
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
            return current, False
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
            index = mutable[int(rng.integers(len(mutable)))]
            choices = [v for v in allowed[index] if v != current[index]]
            new[index] = choices[int(rng.integers(len(choices)))]
        return tuple(new), block

    current = evaluate(state)
    best = current if current.feasible else None
    initial_log = current.log_overhead
    positive = []
    for _ in range(config.pilot_steps if mutable else 0):
        candidate, _ = proposal(state)
        evaluation = evaluate(candidate)
        difference = energy(evaluation) - energy(current)
        if difference > 0:
            positive.append(difference)
        if evaluation.feasible and (best is None or rank(evaluation) < rank(best)):
            best = evaluation
    initial_temperature = (median(positive) if positive else scale) / -log(
        config.initial_acceptance
    )
    temperature = initial_temperature
    accepted = changed = blocks = 0
    trace = []
    steps = config.steps if mutable else 0
    for step in range(steps):
        level = min(config.temperature_levels - 1, step * config.temperature_levels // steps)
        temperature = max(1e-12, initial_temperature * config.cooling_rate**level)
        candidate, is_block = proposal(state)
        blocks += int(is_block)
        if candidate != state:
            changed += 1
            evaluation = evaluate(candidate)
            if evaluation.feasible and (best is None or rank(evaluation) < rank(best)):
                best = evaluation
            difference = energy(evaluation) - energy(current)
            if difference <= 0 or rng.random() < exp(-min(745, difference / temperature)):
                state, current = candidate, evaluation
                accepted += 1
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            trace.append(
                {
                    "step": step + 1,
                    "temperature": temperature,
                    "current_feasible": current.feasible,
                    "current_log_overhead": current.log_overhead,
                    "best_log_overhead": best.log_overhead if best else None,
                }
            )
    if best:
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
    return AnnealResult(
        seed,
        config,
        best,
        initial_log,
        initial_temperature,
        temperature,
        penalty_weight,
        steps + (config.pilot_steps if mutable else 0),
        evaluate.cache_info().misses,
        accepted,
        changed,
        blocks,
        perf_counter() - start,
        tuple(trace),
    )


def _run_one(arguments):
    graph, config, seed, state = arguments
    return anneal(graph, config, seed=seed, initial_state=state)


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
