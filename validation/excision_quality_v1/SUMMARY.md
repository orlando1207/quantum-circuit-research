# Large-circuit excision quality study

All values in the comparison table are **log10 sampling overhead** (squared coefficient norm). Physical PTM excision recipes; no assumed discounts. Only wire cuts, gate cuts, and individual two-qubit excisions are enabled.

Training selected `slow_cooling`. Starts use the original generic seed pool, shared between WG and WGE. No hand-supplied separators or baseline winners. The synthetic control is deliberately favorable and is not an application workload.

| Circuit | N | M | WG | WGE | Excisions | Fragments | Qiskit | Paper reconstruction |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| graphstate_50_M37 | 50 | 37 | 6.6797 | 6.6797 | 0 | 2 | 25.7645 (4 starts; time-limit-with-incumbent) | 8.5882 (50 starts; completed) |
| graphstate_100_M75 | 100 | 75 | 15.2679 | 15.2679 | 0 | 2 | 42.9409 (1 starts; time-limit-with-incumbent) | 17.1764 (22 starts; time-limit-with-incumbent) |
| graphstate_150_M112 | 150 | 112 | 22.9018 | 22.9018 | 0 | 2 | 66.7970 (1 starts; time-limit-with-incumbent) | 27.6730 (8 starts; time-limit-with-incumbent) |
| graphstate_250_M187 | 250 | 187 | 45.8036 | 45.8036 | 0 | 2 | — (0 starts; timeout) | 54.3918 (2 starts; time-limit-with-incumbent) |
| aqft_50_M37 | 50 | 37 | 3.0618 | 3.0618 | 0 | 2 | 3.6398 (14 starts; time-limit-with-incumbent) | 3.6398 (11 starts; time-limit-with-incumbent) |
| aqft_100_M75 | 100 | 75 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (7 starts; time-limit-with-incumbent) | 3.6398 (2 starts; time-limit-with-incumbent) |
| aqft_150_M112 | 150 | 112 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (6 starts; time-limit-with-incumbent) | — (0 starts; timeout) |
| aqft_250_M187 | 250 | 187 | 3.6398 | 3.6398 | 0 | 2 | 3.6398 (5 starts; time-limit-with-incumbent) | — (0 starts; timeout) |
| trotter_grid_50_M37 | 50 | 37 | 9.9264 | 9.9264 | 0 | 2 | 9.9264 (6 starts; time-limit-with-incumbent) | 11.3445 (15 starts; time-limit-with-incumbent) |
| trotter_grid_100_M75 | 100 | 75 | 14.1806 | 14.1806 | 0 | 2 | 14.1806 (2 starts; time-limit-with-incumbent) | 14.1806 (2 starts; time-limit-with-incumbent) |
| trotter_grid_150_M112 | 150 | 112 | 17.0167 | 17.0167 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| trotter_grid_250_M187 | 250 | 187 | 21.2709 | 21.2709 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| random_brickwork_50_M37 | 50 | 37 | 36.0566 | 36.0566 | 0 | 2 | 47.7121 (1 starts; time-limit-with-incumbent) | 30.7856 (18 starts; time-limit-with-incumbent) |
| random_brickwork_100_M75 | 100 | 75 | 74.5215 | 74.5215 | 0 | 3 | 83.9733 (1 starts; time-limit-with-incumbent) | 61.0715 (3 starts; time-limit-with-incumbent) |
| random_brickwork_150_M112 | 150 | 112 | 110.7827 | 109.8284 | 1 | 2 | — (0 starts; timeout) | 103.0582 (1 starts; time-limit-with-incumbent) |
| random_brickwork_250_M187 | 250 | 187 | 197.6640 | 197.6640 | 0 | 2 | — (0 starts; timeout) | — (0 starts; timeout) |
| aqft_100_M33 | 100 | 33 | 9.0618 | 9.0618 | 0 | 4 | 9.0618 (10 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| aqft_250_M83 | 250 | 83 | 9.0618 | 9.0618 | 0 | 4 | 9.0618 (3 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| random_brickwork_100_M33 | 100 | 33 | 129.2084 | 128.5041 | 3 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |
| random_brickwork_250_M83 | 250 | 83 | 356.8867 | 356.8867 | 0 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |
| control_100_M26 | 100 | 26 | 3.3625 | 2.4082 | 1 | 4 | 138.3652 (1 starts; time-limit-with-incumbent) | — (0 starts; proven-infeasible-two-way) |
| control_250_M63 | 250 | 63 | 3.3625 | 2.4082 | 1 | 4 | — (0 starts; timeout) | — (0 starts; proven-infeasible-two-way) |

## Hyperparameter sweep

| Profile | Mean median log10 regret | Median acceptance | Median polish evaluations |
|---|---:|---:|---:|
| slow_cooling | 0.23186 | 0.850 | 1131 |
| long_intervals | 0.57516 | 0.486 | 971 |
| original | 0.61356 | 0.579 | 1024 |
| combined | 0.62514 | 0.339 | 786 |
| feasible_temperature | 0.80702 | 0.438 | 950 |
| fewer_regions | 0.80702 | 0.433 | 920 |
| more_regions | 0.80702 | 0.478 | 959 |
| more_splits | 0.80702 | 0.495 | 949 |
| fast_cooling | 0.85700 | 0.236 | 820 |
| fewer_splits | 0.85700 | 0.367 | 858 |

## Interpretation and limits

Qiskit is the installed official addon, using gate and wire cuts. The paper baseline is our two-stage KL reconstruction with physical static-width guards, not author code. Its two-way infeasibility under tight bounds is not a quality victory. Deadlines preserve only completed restarts; interrupted runs do not supply an incumbent. Errors/timeouts do not receive a numerical score.

Each SA run has a fixed cache-miss budget, including pilots and polishing. Preprocessing and shared original seed construction are timed separately in `report.json`. Baseline time budgets and restart counts differ, so these are bounded quality comparisons, not equal-runtime rankings or optimality claims. Large plans are checked by physical recipe norms and a materialized branch; large quantum states and all QPD branches are not simulated.


## Excision attribution

`WG + post` reprices the WG winner with the same physical excision LUT and performs no additional search. It distinguishes an available decomposition saving from a benefit of changing the search trajectory. These diagnostics do not add starts to WGE in the main comparison.

| Case | WG log10 | WG + post log10 | WGE search log10 | WGE excisions |
|---|---:|---:|---:|---:|
| graphstate_50_M37 | 6.6797 | 6.6797 | 6.6797 | 0 |
| graphstate_100_M75 | 15.2679 | 15.2679 | 15.2679 | 0 |
| graphstate_150_M112 | 22.9018 | 22.9018 | 22.9018 | 0 |
| graphstate_250_M187 | 45.8036 | 45.8036 | 45.8036 | 0 |
| aqft_50_M37 | 3.0618 | 3.0618 | 3.0618 | 0 |
| aqft_100_M75 | 3.6398 | 3.6398 | 3.6398 | 0 |
| aqft_150_M112 | 3.6398 | 3.6398 | 3.6398 | 0 |
| aqft_250_M187 | 3.6398 | 3.6398 | 3.6398 | 0 |
| trotter_grid_50_M37 | 9.9264 | 9.9264 | 9.9264 | 0 |
| trotter_grid_100_M75 | 14.1806 | 14.1806 | 14.1806 | 0 |
| trotter_grid_150_M112 | 17.0167 | 17.0167 | 17.0167 | 0 |
| trotter_grid_250_M187 | 21.2709 | 21.2709 | 21.2709 | 0 |
| random_brickwork_50_M37 | 36.0566 | 36.0566 | 36.0566 | 0 |
| random_brickwork_100_M75 | 74.5215 | 74.5215 | 74.5215 | 0 |
| random_brickwork_150_M112 | 110.7827 | 109.8284 | 109.8284 | 1 |
| random_brickwork_250_M187 | 197.6640 | 197.6640 | 197.6640 | 0 |
| aqft_100_M33 | 9.0618 | 9.0618 | 9.0618 | 0 |
| aqft_250_M83 | 9.0618 | 9.0618 | 9.0618 | 0 |
| random_brickwork_100_M33 | 129.2084 | 129.2084 | 128.5041 | 3 |
| random_brickwork_250_M83 | 356.8867 | 356.8867 | 356.8867 | 0 |
| control_100_M26 | 3.3625 | 2.4082 | 2.4082 | 1 |
| control_250_M63 | 3.3625 | 2.4082 | 2.4082 | 1 |

Among completed ordinary cases, WGE is better/equal/worse than WG on 2/18/0 cases; 2 ordinary winners contain excision.

A finite-budget WGE trajectory can lose to WG despite pointwise cheaper realizations: changed costs change acceptance and the visited states. Its comparison with the post-only column makes that distinction explicit.

## Held-out profile confirmation

Same five starts and 4000-evaluation limits on four 100-qubit cases. This diagnostic does not retune the selected profile or enter the main table.

| Case | Original best / median log10 | Selected best / median log10 |
|---|---:|---:|
| graphstate_100_M75 | 15.2679 / 19.0849 | 15.2679 / 19.0849 |
| aqft_100_M75 | 3.2493 / 3.6398 | 3.6398 / 3.9305 |
| trotter_grid_100_M75 | 14.1806 / 106.6727 | 14.1806 / 102.9207 |
| random_brickwork_100_M75 | 74.5215 / 108.6460 | 74.5215 / 102.4208 |