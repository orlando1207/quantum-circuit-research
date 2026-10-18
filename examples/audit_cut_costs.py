"""Independent formula/channel/resource audit; no tuned operation multipliers."""

from __future__ import annotations

import argparse
import json
from math import pi, sin
from pathlib import Path

from qiskit.circuit.library import CPhaseGate, CXGate, RZZGate, SwapGate

from qpartition.benchmarks import qft_benchmark
from qpartition.templates import TemplateRegistry


def audit(output: Path) -> dict:
    registry = TemplateRegistry()
    records = []
    for angle in (0, pi / 2048, pi / 1024, 0.05, pi / 4, pi / 2, pi, -0.72, 2 * pi):
        for gate, expected in (
            (CPhaseGate(angle), 1 + 2 * abs(sin(angle / 2))),
            (RZZGate(angle), 1 + 2 * abs(sin(angle))),
        ):
            template, excision = registry.for_gate(gate)
            assert excision is None
            assert abs(template.gamma - expected) < 1e-12
            records.append(
                {
                    "gate": gate.name,
                    "angle": angle,
                    "formula_gamma": expected,
                    "physical_gamma": template.gamma,
                    "overhead": template.gamma**2,
                    "channel_residual": template.reconstruction_residual,
                }
            )
    for gate, gamma in ((CXGate(), 3), (SwapGate(), 7)):
        template, _ = registry.for_gate(gate)
        assert abs(template.gamma - gamma) < 1e-12
        records.append(
            {
                "gate": gate.name,
                "physical_gamma": template.gamma,
                "overhead": template.gamma**2,
                "channel_residual": template.reconstruction_residual,
            }
        )
    assert registry.wire.gamma == 4
    qft = qft_benchmark(12).problem
    phase_costs = [
        registry.for_gate(item.operation)[0].gamma ** 2
        for item in qft.circuit.data
        if item.operation.name == "cp"
    ]
    result = {
        "status": "passed",
        "metric": "Omega=(sum absolute physical coefficients)^2; C=ln(Omega)",
        "operation_cost_multipliers": {"gate": 1, "wire": 1},
        "wire": {
            "gamma": 4,
            "overhead": 16,
            "channel_residual": registry.wire.reconstruction_residual,
        },
        "gates": records,
        "qft12": {
            "phase_gates": len(phase_costs),
            "phase_cost_min": min(phase_costs),
            "phase_cost_max": max(phase_costs),
            "phases_cheaper_than_wire": sum(cost < 16 for cost in phase_costs),
            "swap_gates": 6,
        },
        "resource_model": "LO, no runtime communication, static segment allocation",
        "physical_capacity_effect": "Wire cuts add one allocated segment; gate cuts add none. "
        "The width penalty vanishes on feasible states, so it does not alter their cost ranking.",
        "formula_convention": "RZZ(theta): 1+2|sin(theta)|. CP(theta): 1+2|sin(theta/2)|. "
        "Paper log(kappa) weights and our 2*log(kappa) differ by a global factor two.",
        "limitations": "General KAK decompositions use the installed physical QPD bound; "
        "not every arbitrary-unitary SDK template is known optimal. The special gates audited "
        "here have documented norms. No grouping, LOCC wire discount, or variance estimate.",
        "sources": [
            "https://arxiv.org/html/2608.05287v1",
            "https://qiskit.github.io/qiskit-addon-cutting/explanation/",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/sa_benchmarks_v2/cost_audit.json")
    )
    print(json.dumps(audit(parser.parse_args().output), indent=2))
