"""Reusable LP for the optimal two-qubit gate excision cost (docs/second_stage_prompt.txt).

Model: a gate U acts on operands a, b, realized as four distinct, mutually
non-communicating fragments F1 (input a), F2 (input b), F3 (output a), F4
(output b). The only allowed quasi-probability term is

    T(rho) = tr[(O1 (x) O2) rho] * (s3 (x) s4),

where O_i is Hermitian with ||O_i|| <= 1 (measured in its own eigenbasis,
each outcome weighted by its eigenvalue) and s_i is a density matrix
(a fixed preparation, independent of any later choice). U(.)U^dagger must be
reproduced exactly as a signed sum of such terms:

    gamma_ex(U) = min sum_i |a_i|   s.t.   U(.)U^dagger = sum_i a_i T_i,

and the physical sampling overhead of excising U is gamma_ex(U)^2.

Choi/Pauli-coordinate formulation (qubit order a_in, b_in, a_out, b_out):
    J(U)  = (1/4) sum_P  P^T (x) (U P U^dagger),         P over the 16 two-qubit Paulis.
    J(T)  = O1^T (x) O2^T (x) s3 (x) s4                  (channel Phi(r)=tr[Kr]S has J=K^T(x)S).
    c(M)_Q = tr(M Q) for a 256-dim four-qubit Pauli string Q; c(A(x)B)=c(A) kron c(B).
    LP: min sum(a+ + a-)  s.t.  A (a+ - a-) = c(J(U)),  a+, a- >= 0,
        where each column of A is c(J(T_i)) for a candidate term T_i.

Extreme points of the (O, s) feasible sets suffice for an optimal term:
O in {I} union {n.sigma : unit vector n}; s in {(I +- n.sigma)/2 : unit n}.
The LP is therefore semi-infinite; it is solved by column generation:

  (a) Seeded finite pool: the Pauli axes, the 18 cube directions (axes and face
      diagonals), and the eigen-directions of the single-qubit factors of
      U P U^dagger (preparation side) and U^dagger Q U (measurement side),
      conditioned on each Pauli eigenstate of the other qubit. Pricing over
      this pool is exact: one tensor contraction of the dual w against every
      pool term; all violators |<w, v_T>| > 1 (up to a batch size) are added.
  (b) Continuous pricing: from the dual w, fix three factors and form the 2x2
      operator W_eff = w0 I + w.sigma; the best preparation is the top/bottom
      eigenvector of W_eff and the best measurement is I or the Pauli along
      w/|w|. Multi-start coordinate ascent over all four factors; the
      directions of every violating local maximum join the pool and (a) resumes.
  The LP is solved once per canonical KAK point exp(i(a XX + b YY + c ZZ))
  (gamma_ex is local-unitary invariant) and cached; `optimal_excision_
  decomposition` maps the canonical recipe through the KAK local unitaries
  to the actual gate and checks its physical terms against U(.)U^dagger.

Status per gate:
  EXACT        gamma_ex equals the universal lower bound 16 (composing an
               excision of U with one of U^-1 gives a two-wire cut, optimal
               cost 4^2 = 16), hence optimal.
  CERTIFIED    continuous pricing found no improving column (best effort; the
               pricing maximum is found heuristically).
  UPPER_BOUND  the time budget ended with improving columns remaining. The
               reported `lower_bound` is then the dual-scaled estimate
               gamma / max|<w, v_T>| (valid if that maximum is global).

Regression targets (tests/test_excision_lp_recipes.py): wire cut 4; CNOT 16;
CP(pi/2) 20 with Pauli axes, 16 enlarged; CP(pi/4), RZZ(0.37), RZZ(1.1),
B-gate 16; sqrt(iSWAP) <= 17.32; sqrt(SWAP) <= 19.00; gamma_ex(theta) ~
gamma_ex(pi/4 - theta) within the reported gaps.

Baselines (also provided by this module, no LP needed):
  gamma_cut_kak(a, b, c): optimal gate-cut cost from the canonical
    (KAK/Weyl-chamber) coordinates of exp(i(a XX + b YY + c ZZ)), via its
    normalized operator-Schmidt coefficients: gamma = 2*(sum_k sqrt(lambda_k))^2 - 1.
    Validated: CX -> 3, SWAP -> 7, iSWAP -> 7 (matches the existing registry).
  swap_peeled_gamma_cut(U): min(gamma_cut(U), gamma_cut(SWAP . U)) -- relabeling
    operands can be cheaper for near-SWAP gates.
  per_junction_gain(gamma_ex, gamma_eff) = (16 * gamma_eff / gamma_ex)^2 -- the
    fair, baseline-normalized measure of an excision junction's benefit.

Local-unitary invariance: gamma_ex depends only on the KAK canonical point
(a, b, c) of U (Local-unitary invariance of this whole optimization follows
because conjugating every O_i, s_i by the appropriate local unitary is a
bijection on the feasible term set that leaves ||.|| and purity unchanged).
`gate_excision_cost` canonicalizes every gate via
`qiskit.synthesis.two_qubit.two_qubit_decompose.TwoQubitWeylDecomposition`
and caches by the rounded (a, b, c) point.

Geometry rule: if both outputs (or both inputs) of an excision junction land
in the SAME fragment, no LP is needed -- gamma_ex = 16 for every U (fold the
pair into a single two-wire cut of cost 4^2, independent of U, since the
"gate" itself can then be applied directly on one side after reconstruction).

`TemplateRegistry(excision_recipe="lp")` (the default) uses these recipes for
every two-qubit excision; `"ptm"` keeps the earlier Pauli-transfer construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from scipy.optimize import linprog

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}
_AXES = "IXYZ"
_PAULI_LIST = [_PAULI[a] for a in _AXES]

LOWER_BOUND = 16.0


# --------------------------------------------------------------------------
# Pauli-coordinate / Choi machinery
# --------------------------------------------------------------------------


def _coords(M: np.ndarray) -> np.ndarray:
    """c(M)_k = tr(M sigma_k), k over I,X,Y,Z. Reconstruction: M=(1/2) sum c_k sigma_k."""
    return np.array([np.trace(M @ _PAULI[a]).real for a in _AXES])


def _two_qubit_paulis() -> list[np.ndarray]:
    return [np.kron(_PAULI[a], _PAULI[b]) for a, b in product(_AXES, repeat=2)]


def choi_matrix(U: np.ndarray) -> np.ndarray:
    """J(U) = (1/4) sum_P P^T kron (U P U^dagger), 16x16, qubit order (in, out).

    Independently verified (tests/test_excision_lp.py) against the textbook
    Choi definition sum_ij |i><j| kron (U|i><j|U^dagger).
    """
    J = np.zeros((16, 16), dtype=complex)
    for P in _two_qubit_paulis():
        J += np.kron(P.T, U @ P @ U.conj().T)
    return J / 4


_FOUR_QUBIT_PAULIS: np.ndarray | None = None


def _four_qubit_paulis() -> np.ndarray:
    global _FOUR_QUBIT_PAULIS
    if _FOUR_QUBIT_PAULIS is None:
        mats = [
            np.kron(np.kron(_PAULI[a], _PAULI[b]), np.kron(_PAULI[c], _PAULI[d]))
            for a, b, c, d in product(_AXES, repeat=4)
        ]
        _FOUR_QUBIT_PAULIS = np.array(mats)
    return _FOUR_QUBIT_PAULIS


def pauli_coords_4q(M: np.ndarray) -> np.ndarray:
    """The 256 four-qubit Pauli coordinates c(M)_Q = tr(M Q)."""
    return np.einsum("qij,ji->q", _four_qubit_paulis(), M).real


def term_coords(O1: np.ndarray, O2: np.ndarray, s3: np.ndarray, s4: np.ndarray) -> np.ndarray:
    """c(J(T)) for T(rho) = tr[(O1 (x) O2) rho] (s3 (x) s4)."""
    return np.kron(np.kron(_coords(O1.T), _coords(O2.T)), np.kron(_coords(s3), _coords(s4)))


def pauli_axis_pool() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """D = Pauli axes. Measurements {I,X,Y,Z}; preparations (I +- axis)/2."""
    measurements = [_I2, _X, _Y, _Z]
    preps = []
    for axis in "XYZ":
        preps.append((_I2 + _PAULI[axis]) / 2)
        preps.append((_I2 - _PAULI[axis]) / 2)
    return measurements, preps


def build_pool_columns(
    measurements: list[np.ndarray], preps: list[np.ndarray]
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    cols, meta = [], []
    for O1 in measurements:
        for O2 in measurements:
            for s3 in preps:
                for s4 in preps:
                    cols.append(term_coords(O1, O2, s3, s4))
                    meta.append((O1, O2, s3, s4))
    return np.array(cols).T, meta


def _solve_l1(A: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """min sum|a_i| s.t. A a = b. Returns (gamma, signed coefficients, dual w)."""
    n = A.shape[1]
    c = np.ones(2 * n)
    A_eq = np.hstack([A, -A])
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"Excision LP failed: {res.message}")
    coeffs = res.x[:n] - res.x[n:]
    gamma = float(np.sum(np.abs(coeffs)))
    # Solve the dual explicitly (max w.b s.t. -1<=A^T w<=1); more numerically
    # robust than scipy's primal-solve marginals for this highly degenerate LP.
    m = A.shape[0]
    dual_res = linprog(
        -b, A_ub=np.vstack([A.T, -A.T]), b_ub=np.ones(2 * n), bounds=(None, None), method="highs"
    )
    w = dual_res.x if dual_res.success else np.zeros(m)
    return gamma, coeffs, w


# --------------------------------------------------------------------------
# Column generation (experimental; see module docstring's known limitation)
# --------------------------------------------------------------------------


def _effective_operator(W: np.ndarray, slot: int, factor_coords: list[np.ndarray]) -> np.ndarray:
    """Contract the (4,4,4,4) dual tensor W over the other 3 slots, building
    the 2x2 Hermitian operator whose trace against this slot's candidate
    reproduces w . c(J(T)) exactly (verified in tests/test_excision_lp.py)."""
    letters = "ijkl"
    other_letters = [c for idx, c in enumerate(letters) if idx != slot]
    other_vecs = [factor_coords[idx] for idx in range(4) if idx != slot]
    spec = f"{letters}," + ",".join(other_letters) + f"->{letters[slot]}"
    w_eff = np.einsum(spec, W, *other_vecs)
    return sum(w_eff[i] * _PAULI_LIST[i] for i in range(4))


def _best_measurement(W_eff: np.ndarray) -> tuple[np.ndarray, float]:
    """Best O (I or n.sigma) maximizing |tr(O^T @ W_eff)|."""
    val_identity = np.trace(_I2.T @ W_eff).real
    u = np.array([np.trace(_PAULI_LIST[i].T @ W_eff).real for i in range(1, 4)])
    norm_u = np.linalg.norm(u)
    if norm_u >= abs(val_identity):
        n = u / norm_u if norm_u > 1e-14 else np.array([0.0, 0.0, 1.0])
        return n[0] * _X + n[1] * _Y + n[2] * _Z, norm_u
    return _I2, val_identity


def _best_preparation(W_eff: np.ndarray) -> tuple[np.ndarray, float]:
    """Best pure state s=(I+-n.sigma)/2 maximizing |tr(s @ W_eff)|."""
    u = np.array([np.trace(_PAULI_LIST[i] @ W_eff).real for i in range(1, 4)])
    norm_u = np.linalg.norm(u)
    base = np.trace(_I2 @ W_eff).real / 2
    n = u / norm_u if norm_u > 1e-14 else np.array([0.0, 0.0, 1.0])
    val_plus, val_minus = base + norm_u / 2, base - norm_u / 2
    if abs(val_plus) >= abs(val_minus):
        return (_I2 + n[0] * _X + n[1] * _Y + n[2] * _Z) / 2, val_plus
    return (_I2 - n[0] * _X - n[1] * _Y - n[2] * _Z) / 2, val_minus


def _alternate_maximize(
    W: np.ndarray, n_restarts: int = 12, n_iters: int = 25, seed: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
    """Multi-start coordinate ascent for max |w . c(J(T))| over (O1,O2,s3,s4)."""
    rng = np.random.default_rng(seed)
    best_val, best = 0.0, (_I2, _I2, (_I2 + _Z) / 2, (_I2 + _Z) / 2)
    for restart in range(n_restarts):
        if restart == 0:
            current = [_I2, _I2, (_I2 + _Z) / 2, (_I2 + _Z) / 2]
        else:
            current = []
            for kind in ("m", "m", "p", "p"):
                n = rng.normal(size=3)
                n /= np.linalg.norm(n)
                if kind == "m":
                    current.append(n[0] * _X + n[1] * _Y + n[2] * _Z)
                else:
                    current.append((_I2 + n[0] * _X + n[1] * _Y + n[2] * _Z) / 2)
        value = None
        for _ in range(n_iters):
            improved = False
            for slot in range(4):
                factors = [
                    _coords(current[0].T),
                    _coords(current[1].T),
                    _coords(current[2]),
                    _coords(current[3]),
                ]
                W_eff = _effective_operator(W, slot, factors)
                if slot < 2:
                    new_op, new_val = _best_measurement(W_eff)
                else:
                    new_op, new_val = _best_preparation(W_eff)
                prior = value
                current[slot], value = new_op, new_val
                if prior is None or abs(new_val) > abs(prior) + 1e-12:
                    improved = True
            if not improved:
                break
        if value is not None and abs(value) > abs(best_val):
            best_val, best = abs(value), tuple(current)
    return best, best_val


# --------------------------------------------------------------------------
# KAK canonicalization and baseline costs
# --------------------------------------------------------------------------


def kak_coordinates(U: np.ndarray) -> tuple[float, float, float]:
    """Canonical (a, b, c) with U locally equivalent to exp(i(a XX+b YY+c ZZ))."""
    decomposition = TwoQubitWeylDecomposition(Operator(U))
    return float(decomposition.a), float(decomposition.b), float(decomposition.c)


def gamma_cut_kak(a: float, b: float, c: float) -> float:
    """KAK-optimal independent gate-cut cost from canonical coordinates.

    gamma = 2*(sum_k sqrt(lambda_k))^2 - 1, lambda_k the NORMALIZED (sum=1)
    squared operator-Schmidt coefficients of exp(i(a XX+b YY+c ZZ)).
    Validated: (pi/4,0,0)->3 (CX/CZ), (pi/4,pi/4,pi/4)->7 (SWAP),
    (pi/4,pi/4,0)->7 (iSWAP), (0,0,0)->1 (identity).
    """
    from scipy.linalg import expm

    U = expm(1j * (a * np.kron(_X, _X) + b * np.kron(_Y, _Y) + c * np.kron(_Z, _Z)))
    M = U.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    s = np.linalg.svd(M, compute_uv=False)
    lam = s**2 / np.sum(s**2)
    return float(2 * np.sum(np.sqrt(lam)) ** 2 - 1)


def gamma_cut_gate(U: np.ndarray) -> float:
    a, b, c = kak_coordinates(np.asarray(U, dtype=complex))
    return gamma_cut_kak(a, b, c)


def swap_peeled_gamma_cut(U: np.ndarray) -> float:
    """min(gamma_cut(U), gamma_cut(SWAP . U)) -- operand relabeling baseline."""
    U = np.asarray(U, dtype=complex)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    return min(gamma_cut_gate(U), gamma_cut_gate(swap @ U))


def per_junction_gain(gamma_ex: float, gamma_eff: float) -> float:
    """(16 * gamma_eff / gamma_ex)^2 -- fair, baseline-normalized excision benefit."""
    return (LOWER_BOUND * gamma_eff / gamma_ex) ** 2


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcisionLPResult:
    gamma_ex: float
    status: str  # "EXACT" | "CERTIFIED" | "UPPER_BOUND"
    lower_bound: float
    pool_size: int
    kak: tuple[float, float, float]
    pauli_only: bool  # True if the optimal decomposition uses only Pauli-axis terms
    coefficients: np.ndarray = field(repr=False)
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(repr=False)
    residual: float = 0.0
    rounds: int = 0


# --------------------------------------------------------------------------
# Seeded direction pool and exact column generation
# --------------------------------------------------------------------------

_SIGMA = (_X, _Y, _Z)
_EIGENSTATES = tuple((_I2 + sign * P) / 2 for P in _SIGMA for sign in (1, -1))
_AXIS_UNITS = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


def canonical_unitary(a: float, b: float, c: float) -> np.ndarray:
    from scipy.linalg import expm

    return expm(1j * (a * np.kron(_X, _X) + b * np.kron(_Y, _Y) + c * np.kron(_Z, _Z)))


def _sigma_dot(d) -> np.ndarray:
    return d[0] * _X + d[1] * _Y + d[2] * _Z


def _bloch(H: np.ndarray) -> np.ndarray | None:
    d = np.array([np.trace(H @ P).real for P in _SIGMA])
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-9 else None


def _merge_directions(existing: list[np.ndarray], new, tol: float = 1e-7) -> list[np.ndarray]:
    """Append unit directions not already present up to sign (append-only, stable indices)."""
    for d in new:
        k = np.flatnonzero(np.abs(d) > 1e-9)[0]
        d = -d if d[k] < 0 else d
        if not any(np.allclose(d, e, atol=tol) for e in existing):
            existing.append(d)
    return existing


def _conditional_directions(M: np.ndarray) -> list[np.ndarray]:
    """Bloch directions of the single-qubit factors of a two-qubit Hermitian M,
    conditioned on each Pauli eigenstate of the other qubit."""
    M4 = M.reshape(2, 2, 2, 2)
    out = []
    for E in _EIGENSTATES:
        for H in (np.einsum("ijkl,lj->ik", M4, E), np.einsum("ijkl,ki->jl", M4, E)):
            d = _bloch(H)
            if d is not None:
                out.append(d)
    return out


def _cube_directions() -> list[np.ndarray]:
    """The 18 directions: 6 axes and 12 face diagonals."""
    return [
        np.array(v, dtype=float) / np.linalg.norm(v)
        for v in product((-1, 0, 1), repeat=3)
        if np.count_nonzero(v) in (1, 2)
    ]


def seeded_directions(U: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(measurement, preparation) directions: Pauli axes, the 18 cube directions,
    and the conditional factor eigen-directions of U^dagger Q U (measurement side)
    and U P U^dagger (preparation side). Axes come first, in X, Y, Z order."""
    base = _merge_directions(list(_AXIS_UNITS), _cube_directions())
    meas, prep = list(base), list(base)
    for P in _two_qubit_paulis()[1:]:
        _merge_directions(prep, _conditional_directions(U @ P @ U.conj().T))
        _merge_directions(meas, _conditional_directions(U.conj().T @ P @ U))
    return meas, prep


def _measurement_ops(dirs):
    return [_I2] + [_sigma_dot(d) for d in dirs]


def _preparation_ops(dirs):
    return [(_I2 + sign * _sigma_dot(d)) / 2 for d in dirs for sign in (1, -1)]


def _solve_master(A: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """min ||a||_1 s.t. A a = b, returning HiGHS's equality duals as the pricing vector.

    Retries with the dual simplex and interior-point methods if HiGHS returns an
    unknown status, and finally falls back to solving primal and dual separately."""
    n = A.shape[1]
    for method in ("highs", "highs-ds", "highs-ipm"):
        try:
            res = linprog(
                np.ones(2 * n), A_eq=np.hstack([A, -A]), b_eq=b, bounds=(0, None), method=method,
                options={"time_limit": _LP_TIME_LIMIT},
            )
        except MemoryError:
            raise
        except Exception:  # noqa: BLE001 -- try the next HiGHS method
            continue
        if res.status == 0 and res.x is not None and res.eqlin.marginals is not None:
            return float(res.fun), res.x[:n] - res.x[n:], np.asarray(res.eqlin.marginals)
    raise RuntimeError("Excision master LP failed with every HiGHS method.")


_LP_TIME_LIMIT = 30.0


def _prune_master(A, terms, coeffs, w, keep_threshold: float = 0.9, max_columns: int = 4000):
    """Drop non-basic columns far from binding once the master grows large."""
    if A.shape[1] <= max_columns:
        return A, terms
    score = np.abs(A.T @ w)
    keep = np.flatnonzero((np.abs(coeffs) > 1e-12) | (score >= keep_threshold))
    return A[:, keep], [terms[i] for i in keep]


_MAX_POOL_TERMS = 3_000_000


def _local_maxima(W: np.ndarray, n_restarts: int, seed: int, seeds=None):
    """Coordinate ascent of |w . c(J(T))| from many starts; returns every local maximum.

    Starts mix random directions with the seeded pool directions."""
    rng = np.random.default_rng(seed)
    found = []
    for restart in range(n_restarts):
        current = []
        for kind in ("m", "m", "p", "p"):
            if seeds is not None and rng.random() < 0.5:
                pool = seeds[0] if kind == "m" else seeds[1]
                n = pool[rng.integers(len(pool))] * rng.choice([-1, 1])
            else:
                n = rng.normal(size=3)
                n /= np.linalg.norm(n)
            current.append(_sigma_dot(n) if kind == "m" else (_I2 + _sigma_dot(n)) / 2)
        value = None
        for _ in range(40):
            improved = False
            for slot in range(4):
                factors = [
                    _coords(current[0].T), _coords(current[1].T),
                    _coords(current[2]), _coords(current[3]),
                ]
                W_eff = _effective_operator(W, slot, factors)
                op, val = _best_measurement(W_eff) if slot < 2 else _best_preparation(W_eff)
                if value is None or abs(val) > abs(value) + 1e-12:
                    improved = True
                current[slot], value = op, val
            if not improved:
                break
        found.append((tuple(current), abs(value)))
    return found


def _column_generation(
    U: np.ndarray, *, max_rounds: int, batch: int, continuous_rounds: int, restarts: int,
    tol: float, time_budget: float,
):
    """Column generation for the semi-infinite excision LP.

    Inner loop: exact pricing over a finite direction pool (one tensor contraction
    scores every pool term). Outer loop: continuous pricing over all pure
    directions (multi-start coordinate ascent); the directions of every violating
    local maximum join the pool, which is otherwise reset to the seeded directions
    plus those in the current support. Columns are kept as explicit operator terms.
    """
    from time import perf_counter

    started = perf_counter()
    b = pauli_coords_4q(choi_matrix(U))
    seed_meas, seed_prep = seeded_directions(U)
    meas_dirs, prep_dirs = list(seed_meas), list(seed_prep)
    measurements, preps = pauli_axis_pool()
    terms = [(O1, O2, s3, s4) for O1 in measurements for O2 in measurements
             for s3 in preps for s4 in preps]
    A = np.array([term_coords(*t) for t in terms]).T
    gamma, coeffs, w = _solve_master(A, b)
    present = {np.round(A[:, i], 8).tobytes() for i in range(A.shape[1])}
    rounds, certified, best_violation, stalled = 0, False, None, False
    for cont in range(continuous_rounds + 1):
        M_ops, P_ops = _measurement_ops(meas_dirs), _preparation_ops(prep_dirs)
        MC = np.array([_coords(op.T) for op in M_ops])
        PC = np.array([_coords(s) for s in P_ops])
        while rounds < max_rounds and perf_counter() - started <= time_budget:
            rounds += 1
            V = np.einsum("abcd,ia,jb,kc,ld->ijkl", w.reshape(4, 4, 4, 4), MC, MC, PC, PC,
                          optimize=True)
            flat = np.abs(V).ravel()
            order = np.argpartition(-flat, min(batch * 4, flat.size - 1))[: batch * 4]
            order = order[np.argsort(-flat[order])]
            new_terms, new_cols = [], []
            for f in order:
                if flat[f] <= 1 + tol or len(new_terms) >= batch:
                    break
                i, j, k, m = np.unravel_index(f, V.shape)
                col = np.kron(np.kron(MC[i], MC[j]), np.kron(PC[k], PC[m]))
                key = np.round(col, 8).tobytes()
                if key in present:
                    continue
                present.add(key)
                new_terms.append((M_ops[i], M_ops[j], P_ops[k], P_ops[m]))
                new_cols.append(col)
            if not new_terms:
                break
            A_try = np.hstack([A, np.array(new_cols).T])
            try:
                solved = _solve_master(A_try, b)
            except RuntimeError:
                stalled = True
                break
            A, terms = A_try, terms + new_terms
            gamma, coeffs, w = solved
            A_pruned, terms_pruned = _prune_master(A, terms, coeffs, w)
            if len(terms_pruned) != len(terms):
                try:
                    gamma, coeffs, w = _solve_master(A_pruned, b)
                    A, terms = A_pruned, terms_pruned
                    present = {np.round(A[:, i], 8).tobytes() for i in range(A.shape[1])}
                except RuntimeError:
                    stalled = True
                    break
        if stalled:
            break
        if abs(gamma - LOWER_BOUND) <= tol:
            break
        maxima = _local_maxima(w.reshape(4, 4, 4, 4), restarts, seed=cont,
                               seeds=(seed_meas, seed_prep))
        best_violation = max(value for _t, value in maxima)
        violators = [t for t, value in maxima if value > 1 + 1e-6]
        if not violators:
            certified = True
            break
        if cont == continuous_rounds or perf_counter() - started > time_budget:
            break
        # Lean pool for the next pass: axes + violator directions + current support,
        # capped so the pricing tensor (1+|M|)^2 (2|P|)^2 stays small.
        support = np.flatnonzero(np.abs(coeffs) > 1e-10)
        meas_dirs, prep_dirs = list(_AXIS_UNITS), list(_AXIS_UNITS)
        for O1, O2, s3, s4 in violators + [terms[i] for i in support]:
            if (1 + len(meas_dirs)) ** 2 * (2 * len(prep_dirs)) ** 2 > _MAX_POOL_TERMS:
                break
            _merge_directions(meas_dirs, [d for d in (_bloch(O1), _bloch(O2)) if d is not None])
            _merge_directions(prep_dirs, [_bloch(2 * s3 - _I2), _bloch(2 * s4 - _I2)])
        A_try, terms_try = A, list(terms)
        for t in violators:
            col = term_coords(*t)
            key = np.round(col, 8).tobytes()
            if key not in present:
                present.add(key)
                terms_try.append(t)
                A_try = np.hstack([A_try, col.reshape(-1, 1)])
        try:
            gamma, coeffs, w = _solve_master(A_try, b)
        except RuntimeError:
            break
        A, terms = A_try, terms_try
    pool_size = (1 + len(meas_dirs)) ** 2 * (2 * len(prep_dirs)) ** 2
    return gamma, coeffs, A, b, terms, certified, best_violation, pool_size, rounds


def _is_axis_operator(op: np.ndarray) -> bool:
    d = _bloch(op if abs(np.trace(op).real) < 1e-9 else 2 * op - _I2)
    return d is None or np.isclose(np.max(np.abs(d)), 1.0, atol=1e-9)


_CACHE: dict[tuple[float, float, float], ExcisionLPResult] = {}
CACHE_ENV = "QPARTITION_EXCISION_CACHE"


def kak_key(kak) -> tuple[float, float, float]:
    return tuple(round(v, 9) for v in kak)


def save_cache(path) -> None:
    """Persist solved canonical LPs (pickle), so worker processes can reuse them."""
    import os
    import pickle

    # Atomic replace: other processes may load the cache while it is being written.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        pickle.dump(dict(_CACHE), handle)
    os.replace(tmp, path)


def load_cache(path) -> int:
    import pickle

    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    _CACHE.update(loaded)
    return len(loaded)


def _autoload_cache() -> None:
    import os

    path = os.environ.get(CACHE_ENV)
    if path and os.path.exists(path):
        load_cache(path)


def _pauli_pool_result(U: np.ndarray, kak, tol: float) -> ExcisionLPResult:
    """Pauli-axis pool in the gate's own frame (the pre-enlargement reference)."""
    measurements, preps = pauli_axis_pool()
    A, pool = build_pool_columns(measurements, preps)
    b = pauli_coords_4q(choi_matrix(U))
    gamma, coeffs, _w = _solve_l1(A, b)
    nonzero = np.flatnonzero(np.abs(coeffs) > 1e-9)
    return ExcisionLPResult(
        gamma_ex=gamma,
        status="EXACT" if abs(gamma - LOWER_BOUND) <= tol else "UPPER_BOUND",
        lower_bound=LOWER_BOUND,
        pool_size=len(pool),
        kak=kak,
        pauli_only=True,
        coefficients=coeffs[nonzero],
        pool=[pool[i] for i in nonzero],
        residual=float(np.linalg.norm(A[:, nonzero] @ coeffs[nonzero] - b)),
    )


def gate_excision_cost(
    gate,
    *,
    enlarge: bool = True,
    max_rounds: int = 400,
    batch: int = 200,
    continuous_rounds: int = 40,
    restarts: int = 48,
    tol: float = 1e-7,
    time_budget: float = 60.0,
    use_cache: bool = True,
) -> ExcisionLPResult:
    """gamma_ex(U) with a certification status (see module docstring).

    With `enlarge` (default) the LP is solved for the canonical KAK representative
    exp(i(a XX + b YY + c ZZ)) -- gamma_ex is local-unitary invariant -- and cached
    by its rounded (a, b, c); `pool` holds the support terms in that canonical
    frame. `enlarge=False` returns the Pauli-axis pool value in the gate's own
    frame (not local-unitary invariant; kept as the reference value).
    """
    U = np.asarray(Operator(gate).data, dtype=complex)
    if U.shape != (4, 4):
        raise ValueError("gate_excision_cost requires a two-qubit (4x4) unitary.")
    kak = kak_coordinates(U)
    if not enlarge:
        return _pauli_pool_result(U, kak, tol)
    key = kak_key(kak)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    V = canonical_unitary(*kak)
    gamma, coeffs, A, b, all_terms, certified, best_violation, pool_size, rounds = (
        _column_generation(
            V, max_rounds=max_rounds, batch=batch, continuous_rounds=continuous_rounds,
            restarts=restarts, tol=tol, time_budget=time_budget,
        )
    )
    support = np.flatnonzero(np.abs(coeffs) > 1e-10)
    A_s = A[:, support]
    refined, *_ = np.linalg.lstsq(A_s, b, rcond=None)
    residual = float(np.linalg.norm(A_s @ refined - b))
    if residual < 1e-9 and np.sum(np.abs(refined)) <= gamma + 1e-7:
        coeffs_s = refined
    else:
        coeffs_s = coeffs[support]
        residual = float(np.linalg.norm(A_s @ coeffs_s - b))
    gamma = float(np.sum(np.abs(coeffs_s)))
    terms = [all_terms[i] for i in support]
    if abs(gamma - LOWER_BOUND) <= 1e-6:
        status = "EXACT"
    elif certified:
        status = "CERTIFIED"
    else:
        status = "UPPER_BOUND"
    pauli_only = all(
        all(_is_axis_operator(op) for op in term) for term in terms
    )
    # Dual scaling: w / max_T |w.v_T| is dual-feasible if the max is global, giving
    # gamma / best_violation as a lower-bound ESTIMATE (the max is found heuristically).
    lower = LOWER_BOUND
    if status != "EXACT" and best_violation is not None and best_violation > 1:
        lower = max(LOWER_BOUND, gamma / best_violation)
    elif status != "EXACT":
        lower = max(LOWER_BOUND, gamma if certified else LOWER_BOUND)
    result = ExcisionLPResult(
        gamma_ex=gamma,
        status=status,
        lower_bound=lower,
        pool_size=pool_size,
        kak=kak,
        pauli_only=bool(pauli_only),
        coefficients=np.asarray(coeffs_s),
        pool=terms,
        residual=residual,
        rounds=rounds,
    )
    if use_cache:
        _CACHE[key] = result
    return result


def optimal_excision_decomposition(gate, *, check_tolerance: float = 1e-9):
    """Physical `ExcisionDecomposition` realizing gamma_ex for this exact gate.

    Solves (or reuses) the canonical-frame LP and maps every term through the KAK
    local unitaries U = phase * (K1l (x) K1r) * Ud(a,b,c) * (K2l (x) K2r):
    measurements O -> K2^dagger O K2, preparations s -> K1 s K1^dagger. The result
    is verified at the level of its physical terms against U(.)U^dagger.
    """
    from .excision import ExcisionDecomposition, ExcisionTerm, decompose_excision

    U = np.asarray(Operator(gate).data, dtype=complex)
    weyl = TwoQubitWeylDecomposition(Operator(U))
    canonical = canonical_unitary(weyl.a, weyl.b, weyl.c)
    K1 = (np.asarray(weyl.K1l), np.asarray(weyl.K1r))
    K2 = (np.asarray(weyl.K2l), np.asarray(weyl.K2r))
    rebuilt = np.kron(*K1) @ canonical @ np.kron(*K2)
    phase = np.vdot(rebuilt.ravel(), U.ravel()) / 4
    if not np.allclose(phase * rebuilt, U, atol=1e-8):
        raise RuntimeError("KAK reconstruction mismatch; cannot map the canonical recipe.")
    result = gate_excision_cost(U)

    terms = []
    for coefficient, (O1, O2, s3, s4) in zip(result.coefficients, result.pool, strict=True):
        # Kron slot 0 acts on local qubit 1 (Qiskit little-endian), slot 1 on local qubit 0.
        meas = []
        for op, k in ((O2, K2[1]), (O1, K2[0])):
            d = _bloch(k.conj().T @ op @ k)
            meas.append(None if d is None else tuple(float(v) for v in d))
        prep = []
        for s, k in ((s4, K1[1]), (s3, K1[0])):
            prep.append(tuple(float(v) for v in _bloch(2 * (k @ s @ k.conj().T) - _I2)))
        terms.append(
            ExcisionTerm("", "", (), float(coefficient), tuple(meas), tuple(prep))
        )
    reference = decompose_excision(U, max_qubits=2, tolerance=0)
    decomposition = ExcisionDecomposition(
        reference.pauli_labels, reference.transfer_matrix, tuple(terms), len(terms), 0.0
    )
    paulis = [np.kron(_PAULI[a], _PAULI[b]) for a, b in product(_AXES, repeat=2)]
    error = max(
        float(np.max(np.abs(decomposition.apply_terms(P) - U @ P @ U.conj().T))) for P in paulis
    )
    if error > check_tolerance:
        raise RuntimeError(f"Optimal excision recipe failed its channel check ({error:.2e}).")
    return decomposition, result


def wire_cut_gamma() -> float:
    """The 1-qubit analogue (2 fragments: in, out). Known value: 4."""
    J = np.zeros((4, 4), dtype=complex)
    for P in _PAULI_LIST:
        J += np.kron(P.T, P)
    J /= 2
    two_q_paulis = _two_qubit_paulis()
    b = np.array([np.trace(m @ J).real for m in two_q_paulis])
    measurements, preps = pauli_axis_pool()
    cols = [np.kron(_coords(meas.T), _coords(s)) for meas in measurements for s in preps]
    A = np.array(cols).T
    gamma, _coeffs, _w = _solve_l1(A, b)
    return gamma


_autoload_cache()
