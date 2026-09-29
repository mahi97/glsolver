# Competitive comparison — glsolver vs the best available software

504 runs, 300 s wall budget each, one subprocess per run, every answer re-checked by `glsolver.verify.verify_instance_parts`.

## 1. Full result matrix

Wall-clock seconds when the method answered; `✗` marks an answer the independent verifier rejected. `**t/o**` = timeout, `lic` = licence size limit, `·` = not attempted (the method had already failed at a smaller n of the same family).

| instance | glsolver | bruteforce | cpsat-ortools | ilp-highs-scipy | ilp-gurobi | ilp-highs-pulp | ilp-scip-pulp | ilp-cbc-pulp | glsolver-reference | metis | kahip | mtkahypar | scotch | networkit-spectral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `harary/n=10` | 0.00282s | 0.171s | 0.128s | 0.358s | 0.0154s | 0.236s | 0.639s | 0.274s | 0.00932s | 0.285s ✗ | 0.343s ✗ | 0.433s | 0.158s ✗ | 1.14s ✗ |
| `harary/n=20` | 0.0049s | 0.164s | 0.142s | 0.517s | 0.0248s | 0.276s | 0.684s | 0.362s | 0.0875s | 0.567s ✗ | 0.355s ✗ | 0.418s | 0.156s | 1.14s ✗ |
| `harary/n=50` | 0.00524s | 0.605s | 0.194s | 30.3s | 1.14s | 21s | 4.8s | 4.16s | 1.22s | 0.356s ✗ | 0.354s ✗ | 0.412s ✗ | 0.162s ✗ | 1.15s ✗ |
| `harary/n=100` | 0.00941s | **t/o** | 1.29s | 285s | lic | **t/o** | 30.3s | **t/o** | 10.1s | 0.352s ✗ | 0.374s ✗ | 0.419s ✗ | 0.169s ✗ | 1.15s ✗ |
| `harary/n=200` | 0.0258s | · | 25.5s | **t/o** | · | · | **t/o** | · | 128s | 0.356s ✗ | 0.379s ✗ | 0.416s ✗ | 0.261s ✗ | 1.15s ✗ |
| `harary/n=500` | 0.1s | · | 13.6s | · | · | · | · | · | **t/o** | 0.349s ✗ | 0.39s ✗ | 0.428s ✗ | 0.293s ✗ | 1.25s |
| `harary/n=1000` | 0.491s | · | **t/o** | · | · | · | · | · | · | 0.496s ✗ | 0.382s ✗ | 0.526s ✗ | 0.351s ✗ | 1.8s ✗ |
| `harary/n=2000` | 3.71s | · | · | · | · | · | · | · | · | 0.329s ✗ | 0.385s ✗ | 0.436s ✗ | 0.362s ✗ | 3.53s ✗ |
| `random_regular/n=10` | 0.00273s | 0.163s | 0.318s | 0.165s | 0.0101s | 0.264s | 0.245s | 0.282s | 0.019s | 0.349s ✗ | 0.352s ✗ | 0.488s | 0.166s ✗ | 1.03s ✗ |
| `random_regular/n=20` | 0.00396s | 0.164s | 0.278s | 0.357s | 0.0428s | 0.417s | 0.276s | 0.365s | 0.124s | 0.395s ✗ | 0.355s ✗ | 0.532s | 0.163s | 0.858s ✗ |
| `random_regular/n=50` | 0.00497s | 0.157s | 0.373s | 0.809s | lic | 0.959s | 0.371s | 0.704s | 2.04s | 0.359s ✗ | 0.397s ✗ | 0.492s | 0.157s ✗ | 0.851s ✗ |
| `random_regular/n=100` | 0.00857s | **t/o** | 0.499s | 1.31s | · | 1.54s | 107s | 1.61s | 15.9s | 0.36s ✗ | 0.424s ✗ | 0.633s | 0.148s | 0.824s ✗ |
| `random_regular/n=200` | 0.0195s | · | 0.922s | 11.7s | · | 12.1s | 96.7s | 4.32s | 129s | 0.353s ✗ | 0.521s ✗ | 1.02s | 0.156s ✗ | 0.625s ✗ |
| `random_regular/n=500` | 0.0468s | · | 0.801s | 21.6s | · | 112s | **t/o** | 42.7s | **t/o** | 0.352s ✗ | 0.71s ✗ | 1.03s | 0.154s | 0.615s ✗ |
| `random_regular/n=1000` | 0.255s | · | 10.4s | 287s | · | **t/o** | · | **t/o** | · | 0.374s ✗ | 0.895s ✗ | 0.959s ✗ | 0.227s ✗ | 0.648s ✗ |
| `random_regular/n=2000` | 0.639s | · | 109s | **t/o** | · | · | · | · | · | 0.365s ✗ | 1.36s ✗ | 0.597s | 0.177s ✗ | 0.674s ✗ |
| `erdos_renyi/n=10` | 0.00407s | 0.169s | 0.34s | 0.164s | 0.018s | 3.48s | 0.234s | 0.315s | 0.0375s | 0.369s ✗ | 0.364s ✗ | 1.05s | 0.146s ✗ | 0.568s ✗ |
| `erdos_renyi/n=20` | 0.00364s | 0.173s | 0.369s | 0.186s | 0.0282s | 0.779s | 0.293s | 0.373s | 0.35s | 0.35s ✗ | 0.391s ✗ | 0.52s | 0.158s | 0.595s ✗ |
| `erdos_renyi/n=50` | 0.0077s | 0.152s | 0.506s | 0.798s | lic | 4.16s | 0.41s | 1.15s | 8.85s | 0.349s ✗ | 0.425s ✗ | 0.556s | 0.174s ✗ | 0.612s ✗ |
| `erdos_renyi/n=100` | 0.00468s | 0.152s | 0.802s | 0.605s | · | 1.91s | 17.1s | 3.99s | 84.4s | 0.348s ✗ | 0.508s ✗ | 1.6s | 0.161s | 0.591s ✗ |
| `erdos_renyi/n=200` | 0.00851s | 0.151s | 0.972s | 1.55s | · | 4.33s | 1.7s | 21.8s | **t/o** | 0.356s ✗ | 0.851s | 1.31s | 0.154s ✗ | 0.623s ✗ |
| `erdos_renyi/n=500` | 0.0532s | 0.152s | 6.67s | 21.9s | · | 66.1s | 11.2s | **t/o** | · | 0.346s ✗ | 3.84s ✗ | 1.15s | 0.162s ✗ | 0.672s ✗ |
| `erdos_renyi/n=1000` | 0.196s | 0.192s | 33.2s | **t/o** | · | **t/o** | 48.1s | · | · | 0.359s ✗ | 10.7s ✗ | 1.71s | 0.223s ✗ | 1.07s ✗ |
| `erdos_renyi/n=2000` | 2.22s | err | 222s | · | · | · | 199s | · | · | 0.489s ✗ | 19.4s ✗ | 2.01s | 0.505s ✗ | 1.29s ✗ |
| `sparse_k_connected/n=10` | 0.00347s | 0.165s | 0.293s | 0.178s | 0.0183s | 0.464s | 0.264s | 0.369s | 0.0211s | 0.178s ✗ | 0.25s ✗ | 0.428s | 0.168s ✗ | 0.475s ✗ |
| `sparse_k_connected/n=20` | 0.00411s | 0.164s | 0.145s | 0.681s | 0.0398s | 0.953s | 0.269s | 0.438s | 0.0774s | 0.176s ✗ | 0.373s ✗ | 0.408s ✗ | 0.184s ✗ | 0.478s ✗ |
| `sparse_k_connected/n=50` | 0.00868s | **t/o** | 0.17s | 4.23s | 0.536s | 16s | 0.408s | 11.2s | 1.44s | 0.188s ✗ | 0.408s ✗ | 0.422s | 0.192s ✗ | 0.475s ✗ |
| `sparse_k_connected/n=100` | 0.0179s | · | 0.271s | 213s | lic | 280s | 58.8s | 12.9s | 9.41s | 0.206s ✗ | 0.405s ✗ | 0.424s ✗ | 0.19s ✗ | 0.526s ✗ |
| `sparse_k_connected/n=200` | 0.016s | · | 0.433s | **t/o** | · | **t/o** | **t/o** | 24.3s | 69.8s | 0.186s | 1.51s | 0.426s | 0.178s ✗ | 0.493s ✗ |
| `sparse_k_connected/n=500` | 0.16s | · | 23s | · | · | · | · | **t/o** | **t/o** | 0.199s ✗ | 0.644s ✗ | 0.433s ✗ | 0.227s ✗ | 0.498s ✗ |
| `sparse_k_connected/n=1000` | 0.14s | · | **t/o** | · | · | · | · | · | · | 0.277s ✗ | 0.74s ✗ | 0.447s ✗ | 0.553s | 0.532s ✗ |
| `sparse_k_connected/n=2000` | 0.574s | · | · | · | · | · | · | · | · | 0.43s | 1.03s ✗ | 0.508s ✗ | 0.291s | 0.602s ✗ |
| `paper_running_example` | 0.00317s | 0.161s | 0.324s | 0.197s | 0.0138s | 0.619s | 0.258s | 0.38s | 0.00717s | 0.371s ✗ | 0.554s | 0.416s | 0.17s | 0.516s |
| `paper_contract_counterexample` | 0.00318s | 0.161s | 0.332s | 0.17s | 0.0167s | 0.628s | 0.234s | 0.353s | 0.00423s | 0.321s | 0.351s | 0.41s | 0.196s | 0.463s |
| `paper_essential_example` | 0.00576s | 0.161s | 0.323s | 0.175s | 0.0153s | 0.643s | 0.264s | 0.339s | 0.00397s | 0.337s ✗ | 0.47s ✗ | 0.415s | 0.184s ✗ | 0.472s ✗ |
| `counterexample(copies=1)` | 0.0123s | **t/o** | 1.11s | 0.352s | lic | 2.46s | 1.22s | 8.19s | **t/o** | 0.333s ✗ | 3.1s ✗ | 0.458s ✗ | 0.171s ✗ | 0.554s ✗ |

## 2. Crossover for the exact solvers

`wall` is the whole adapter call and therefore includes the one-off cost of importing the solver's Python library (0.15–0.35 s, which dominates the small instances). `model+solve only` excludes it: it is the time to build the model and solve it. glsolver solved **all 36 instances** and so has no row of its own.

| method | family | largest n solved ≤300 s | wall at that n | model+solve only | glsolver (wall / solve) | speed-up | first n it failed | how it failed |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bruteforce | harary | 50 | 0.605 s | 0.605 s | 0.00524 / 0.00522 s | 116× | 100 | timeout |
| bruteforce | random_regular | 50 | 0.157 s | 0.157 s | 0.00497 / 0.00496 s | 32× | 100 | timeout |
| bruteforce | erdos_renyi | 1000 | 0.192 s | 0.192 s | 0.196 / 0.196 s | 1× | 2000 | error |
| bruteforce | sparse_k_connected | 20 | 0.164 s | 0.164 s | 0.00411 / 0.00409 s | 40× | 50 | timeout |
| cpsat-ortools | harary | 500 | 13.6 s | 13.3 s | 0.1 / 0.0998 s | 133× | 1000 | timeout |
| cpsat-ortools | random_regular | 2000 | 109 s | 108 s | 0.639 / 0.639 s | 170× | — | never (n≤2000) |
| cpsat-ortools | erdos_renyi | 2000 | 222 s | 221 s | 2.22 / 2.22 s | 100× | — | never (n≤2000) |
| cpsat-ortools | sparse_k_connected | 500 | 23 s | 22.9 s | 0.16 / 0.159 s | 144× | 1000 | timeout |
| ilp-highs-scipy | harary | 100 | 285 s | 285 s | 0.00941 / 0.00939 s | 30317× | 200 | timeout |
| ilp-highs-scipy | random_regular | 1000 | 287 s | 287 s | 0.255 / 0.255 s | 1128× | 2000 | timeout |
| ilp-highs-scipy | erdos_renyi | 500 | 21.9 s | 21.7 s | 0.0532 / 0.0532 s | 409× | 1000 | timeout |
| ilp-highs-scipy | sparse_k_connected | 100 | 213 s | 212 s | 0.0179 / 0.0179 s | 11847× | 200 | timeout |
| ilp-gurobi | harary | 50 | 1.14 s | 1.12 s | 0.00524 / 0.00522 s | 215× | 100 | license_limit |
| ilp-gurobi | random_regular | 20 | 0.0428 s | 0.0319 s | 0.00396 / 0.00395 s | 8× | 50 | license_limit |
| ilp-gurobi | erdos_renyi | 20 | 0.0282 s | 0.0171 s | 0.00364 / 0.00362 s | 5× | 50 | license_limit |
| ilp-gurobi | sparse_k_connected | 50 | 0.536 s | 0.525 s | 0.00868 / 0.00867 s | 61× | 100 | license_limit |
| ilp-highs-pulp | harary | 50 | 21 s | 20.8 s | 0.00524 / 0.00522 s | 3974× | 100 | timeout |
| ilp-highs-pulp | random_regular | 500 | 112 s | 112 s | 0.0468 / 0.0468 s | 2398× | 1000 | timeout |
| ilp-highs-pulp | erdos_renyi | 500 | 66.1 s | 62.7 s | 0.0532 / 0.0532 s | 1179× | 1000 | timeout |
| ilp-highs-pulp | sparse_k_connected | 100 | 280 s | 279 s | 0.0179 / 0.0179 s | 15580× | 200 | timeout |
| ilp-scip-pulp | harary | 100 | 30.3 s | 29.7 s | 0.00941 / 0.00939 s | 3160× | 200 | timeout |
| ilp-scip-pulp | random_regular | 200 | 96.7 s | 96.4 s | 0.0195 / 0.0195 s | 4949× | 500 | timeout |
| ilp-scip-pulp | erdos_renyi | 2000 | 199 s | 189 s | 2.22 / 2.22 s | 85× | — | never (n≤2000) |
| ilp-scip-pulp | sparse_k_connected | 100 | 58.8 s | 58.5 s | 0.0179 / 0.0179 s | 3264× | 200 | timeout |
| ilp-cbc-pulp | harary | 50 | 4.16 s | 3.89 s | 0.00524 / 0.00522 s | 745× | 100 | timeout |
| ilp-cbc-pulp | random_regular | 500 | 42.7 s | 42.4 s | 0.0468 / 0.0468 s | 907× | 1000 | timeout |
| ilp-cbc-pulp | erdos_renyi | 200 | 21.8 s | 21.6 s | 0.00851 / 0.00849 s | 2545× | 500 | timeout |
| ilp-cbc-pulp | sparse_k_connected | 200 | 24.3 s | 23.9 s | 0.016 / 0.016 s | 1492× | 500 | timeout |
| glsolver-reference | harary | 200 | 128 s | 128 s | 0.0258 / 0.0258 s | 4964× | 500 | timeout |
| glsolver-reference | random_regular | 200 | 129 s | 129 s | 0.0195 / 0.0195 s | 6625× | 500 | timeout |
| glsolver-reference | erdos_renyi | 100 | 84.4 s | 84.4 s | 0.00468 / 0.00467 s | 18077× | 200 | timeout |
| glsolver-reference | sparse_k_connected | 200 | 69.8 s | 69.8 s | 0.016 / 0.016 s | 4356× | 500 | timeout |

## 3. Partitioner failure modes

Counted over every instance each tool was run on, in units of *parts* (k parts per instance). These tools do not solve this problem and were not built to; the numbers say how far their output is from a valid Győri–Lovász partition after the most generous mapping we could give them.

| partitioner | runs | valid GL partitions | parts disconnected (of k·runs) | parts of wrong size | terminals in the wrong part | total size error Σ\|Δ\| |
|---|---:|---:|---:|---:|---:|---:|
| metis | 36 | 3 | 13 / 147 | 39 / 147 | 50 / 147 | 378 |
| kahip | 36 | 4 | 11 / 147 | 80 / 147 | 43 / 147 | 668 |
| mtkahypar | 36 | 23 | 25 / 147 | 0 / 147 | 0 / 147 | 0 |
| scotch | 36 | 10 | 24 / 147 | 63 / 147 | 0 / 147 | 404 |
| networkit-spectral | 36 | 3 | 34 / 147 | 23 / 147 | 43 / 147 | 306 |

### Valid answers by instance size (how many of the 4 families it got right)

| partitioner | n=10 | n=20 | n=50 | n=100 | n=200 | n=500 | n=1000 | n=2000 | paper examples (n=9,13) | counterexample (n=333) |
|---|---|---|---|---|---|---|---|---|---|---|
| metis | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 0/4 | 0/4 | 1/4 | 1/3 | 0/1 |
| kahip | 0/4 | 0/4 | 0/4 | 0/4 | 2/4 | 0/4 | 0/4 | 0/4 | 2/3 | 0/1 |
| mtkahypar | valid | 3/4 | 3/4 | 2/4 | 3/4 | 2/4 | 1/4 | 2/4 | 3/3 | 0/1 |
| scotch | 0/4 | 3/4 | 0/4 | 2/4 | 0/4 | 1/4 | 1/4 | 1/4 | 2/3 | 0/1 |
| networkit-spectral | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 0/4 | 0/4 | 2/3 | 0/1 |
