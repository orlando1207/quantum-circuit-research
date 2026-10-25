# Original incidence-graph excision validation

Matched starting states, 3 label-SA starts per model. Only excision factors change. Large known-witness starts test integration, not search quality. Qiskit and reconstructed paper KL each receive one bounded run.

| Circuit | N | M | Gate/wire | With excision | Qiskit | Paper reconstruction |
|---|---:|---:|---:|---:|---:|---:|
| excision_butterfly_6_L3_s19 | 6 | 2 | 2304 | 256 | 2304 | infeasible-two-way |
| excision_butterfly_18_L3_s19 | 18 | 5 | 2304 | 256 | 7.17898e+23 | infeasible-two-way |
| excision_butterfly_100_L3_s19 | 100 | 26 | 2304 | 256 | 2.31827e+138 | infeasible-two-way |
| excision_butterfly_1000_L3_s19 | 1000 | 251 | 2304 | 256 | error | infeasible-two-way |
| qft6 | 6 | 4 | 2521.75 | 2521.75 | 133568 | 133568 |
| qft12 | 12 | 9 | 796656 | 796656 | 1.50138e+07 | 2.765e+11 |
| trotter_chain_100_L3_s19 | 100 | 75 | 26.1855 | 26.1855 | 26.1855 | timeout |
| trotter_chain_1000_L3_s19 | 1000 | 750 | 26.1855 | 26.1855 | timeout | local-dense-reconstruction-resource-limit |
