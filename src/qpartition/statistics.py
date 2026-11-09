"""Uncertainty from shot data; preserve covariance from reused experiments."""

from dataclasses import dataclass

import numpy as np

from .cutting import CompiledCut, CuttingResult
from .execution import ExecutionResults


@dataclass(frozen=True)
class BootstrapUncertainty:
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence: float
    replicates: int


def bootstrap_uncertainty(
    compiled: CompiledCut,
    result: CuttingResult,
    *,
    replicates: int = 300,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapUncertainty:
    """Multinomial bootstrap of joint bitstrings from each unique experiment.

    Intervals describe finite measurement sampling, not device bias. Standard
    cuts currently enumerate QPD coefficients, so there is no QPD sampling
    randomness here. Reused experiment results are resampled once, not once per
    term, retaining covariance between reconstructed terms and observables.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer of at least 2.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if (
        result.cost != compiled.cost
        or result.plan != compiled.plan
        or result.circuit_fingerprint != compiled.problem.circuit_fingerprint
        or result.query_fingerprint != compiled.problem.query_fingerprint
    ):
        raise ValueError("Result provenance does not match the compiled problem and plan.")
    if any(execution.shots_per_circuit is None for execution in result.execution.values()):
        raise ValueError("Bootstrap requires finite-shot results.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        resampled = {}
        for label, execution in result.execution.items():
            shots = execution.shots_per_circuit
            distributions = []
            for distribution in execution.distributions:
                outcomes = list(distribution)
                probabilities = np.array(list(distribution.values()))
                counts = rng.multinomial(shots, probabilities / probabilities.sum())
                distributions.append(dict(zip(outcomes, counts / shots, strict=True)))
            resampled[label] = ExecutionResults(distributions, shots, 0, "bootstrap")
        values.append(compiled.reconstruct(resampled))
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail])
    return BootstrapUncertainty(
        float(np.std(values, ddof=1)), (float(lower), float(upper)), confidence, replicates
    )
