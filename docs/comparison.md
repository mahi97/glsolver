# Comparison against existing software

*Measured 2026-09-29 on aarch64 Linux, 20 cores, CPython 3.12.13. 504 runs, one
subprocess per run, 300 s wall budget each, every answer re-checked by the
independent verifier (`glsolver.verify.verify_instance_parts`).*

Harness: [`benchmarks/compare/`](../benchmarks/compare/).
Full tables: [`benchmarks/results/compare/compare.md`](../benchmarks/results/compare/compare.md).
Raw rows: `benchmarks/results/compare/compare.jsonl` (504 records).
Figures: `fig_runtime_vs_n.png`, `fig_failure_modes.png`, `fig_crossover.png`.

## 1. The headline is not "fastest"

No third-party software solves the Győri–Lovász problem. An exhaustive search
(§5) found no implementation of the polynomial-time algorithm, of Győri's
exponential cascade, or of any of the published k ≤ 4 special cases, in any
general-purpose graph library or anywhere else. The paper's own authors
published five files that check one fixed counterexample, and nothing more.

So the honest claim is not that this implementation is the fastest. It is that
it appears to be the only one. What the benchmark below establishes is the
second-best thing: **how far general-purpose optimisation gets on the same
instances, and exactly where it stops.**

## 2. What was compared

Thirteen methods across four categories, on 36 instances: four families
(`harary`, `random_regular`, `erdos_renyi`, `sparse_k_connected`) at
n ∈ {10, 20, 50, 100, 200, 500, 1000, 2000} with k = 4, seed 1; the three paper
examples; and the authors' official counterexample (n = 333, k = 9).

| category | methods |
|---|---|
| this project | `glsolver` (C++ core), `glsolver-reference` (pure Python), `bruteforce` (repo oracle) |
| CP | OR-Tools CP-SAT 9.15 |
| MILP | HiGHS 1.15 (via `highspy` and via scipy 1.18), SCIP 10.0, CBC (git 146ce89), Gurobi 13.0.3 |
| graph partitioners | METIS, KaHIP 3.25, Mt-KaHyPar 1.7, Scotch 7.0.13, NetworKit spectral |

Every partitioner was configured as generously as its API allows: exact target
block weights, fixed vertices pinning each terminal to its block, and the
strongest quality preset. Output blocks were relabelled onto terminals by a
maximum-weight bipartite matching.

## 3. Results

| method | solved **and verified** /36 | verifier rejected | timeout | licence | error |
|---|---:|---:|---:|---:|---:|
| **glsolver** | **36** | 0 | 0 | 0 | 0 |
| cpsat-ortools | 32 | 0 | 2 | 0 | 0 |
| ilp-highs-scipy | 25 | 0 | 4 | 0 | 0 |
| ilp-scip-pulp | 25 | 0 | 3 | 0 | 0 |
| ilp-highs-pulp | 23 | 0 | 4 | 0 | 0 |
| ilp-cbc-pulp | 23 | 0 | 4 | 0 | 0 |
| glsolver-reference | 22 | 0 | 5 | 0 | 0 |
| bruteforce | 18 | 0 | 4 | 0 | 1 |
| ilp-gurobi | 13 | 0 | 0 | 5 | 0 |
| mtkahypar | 23 | 13 | 0 | 0 | 0 |
| scotch | 10 | 26 | 0 | 0 | 0 |
| kahip | 4 | 32 | 0 | 0 | 0 |
| metis | 3 | 33 | 0 | 0 | 0 |
| networkit-spectral | 3 | 33 | 0 | 0 | 0 |

No exact solver ever returned an answer the verifier rejected. glsolver's total
wall time over all 36 instances is **8.77 s**; its slowest instance is
`harary/n=2000` at 3.71 s.

### Where each method stops

Harary graphs $H_{4,n}$ — the minimum 4-connected graph, no slack anywhere — are
the hardest family for every general-purpose method.

| method | largest Harary n inside 300 s | its time there | glsolver, same instance |
|---|---:|---:|---:|
| cpsat-ortools | 500 | 13.3 s | 0.0998 s |
| ilp-scip-pulp | 100 | 29.7 s | 0.0094 s |
| ilp-highs-scipy | 100 | 285 s | 0.0094 s |
| ilp-cbc-pulp | 50 | 3.89 s | 0.0052 s |
| ilp-gurobi | 50 | 1.12 s | 0.0052 s (licence wall at n = 100) |
| glsolver | **2000** | 3.71 s | — |

On dense Erdős–Rényi the gap narrows to roughly 85–100×, because a random
balanced split of a dense graph is almost always connected and every method
finds one immediately.

## 4. Where the competitors are better

This section is not a formality. Four of these are real.

1. **CP-SAT and the MILPs can prove non-existence; this solver cannot.** They
   need no connectivity precondition, and an `Infeasible` answer is a genuine
   certificate that no partition exists. When glsolver reports
   `precondition_failed` it has proved nothing — 48 of 1518 non-FEAC instances
   in the repository's own corpus admit a partition anyway. This is a
   structural capability gap, not a performance one.
2. **CP-SAT is a serious competitor, not a straw man.** It reached n = 2000 on
   two of the four families inside the budget. Anyone who needed this before
   2026 and had no algorithm should have reached for CP-SAT, and it would often
   have worked.
3. **Mt-KaHyPar came far closer than expected.** Given the roots as fixed
   vertices and exact block weights it never got a size or a terminal wrong —
   connectivity was its *only* failure mode — and it produced a fully valid
   Győri–Lovász partition on 23 of 36 instances in a fraction of a second. Its
   13 failures cluster exactly where connectivity is hardest to hit by
   accident: every Harary size from n = 50 up (3 of 4 parts disconnected at
   n = 2000), most Harary+chords sizes, and the official counterexample. What
   it cannot do is guarantee, certify, or tell you when it failed.
4. **The pure-Python path loses to several MILP solvers.** `glsolver-reference`
   tops out at n = 100–200 and is slower than CBC and SCIP on random 5-regular
   graphs at n = 200–500. The 36/36 result depends on the C++ core being built.

A fifth item was a defect in this repository rather than a competitor's
advantage: the brute-force oracle died with `RecursionError` on
`erdos_renyi/n=2000`, because the search nests one generator frame per
branching decision and CPython's default recursion limit is 1000. Found by this
comparison, fixed, and covered by a regression test; the oracle now solves that
instance in 0.35 s and reports a `depth_limit` status instead of crashing when
the ceiling genuinely binds.

## 5. Literature and software search

| # | Looked for | Verdict |
|---|---|---|
| 1 | Any code computing a Győri–Lovász partition | **Only this repository.** GitHub repo search `gyori+lovasz` returns one hit, the authors' counterexample checker; `lovasz+partition`, `connected+partition+terminals` and `k-connected+partition` return nothing. Zenodo and Software Heritage return nothing. Absent from NetworkX, SageMath, igraph and LEMON. |
| 2 | Győri's exponential cascade (1976/78) | No implementation found. Chandran–Cheung–Issac (ICALP 2018, [arXiv:1802.07632](https://arxiv.org/abs/1802.07632)) give an O*(4ⁿ) constructive proof, paper only. Casel et al. ([arXiv:2207.09262](https://arxiv.org/abs/2207.09262)) state that polynomial algorithms are known only for k ≤ 4, with no code. |
| 3 | The k = 2/3/4 polynomial special cases | No implementation of any of them (Suzuki–Takahashi–Nishizeki 1990; Wada–Kawaguchi 1993; Nakano–Rahman–Nishizeki 1997). General k = 4 is Hoyer's 2019 Georgia Tech PhD thesis. |
| 4 | Confluent flow rounding | One stale implementation: [loicseguin/confluent](https://github.com/loicseguin/confluent), Python, 2013, no licence, requires NetworkX 1.x (`nx.ford_fulkerson`, `G.node[...]`) so it does not run today. |
| 5 | Code from the paper's authors | [mahdi-jfri/Gyori-Lovasz-Codes](https://github.com/mahdi-jfri/Gyori-Lovasz-Codes), the only code link in the PDF: 5 files, 6 KB, created 2026-08-28. Builds one fixed 9-terminal digraph and checks a structural condition. No partition algorithm of any kind. |

Worth noting by contrast: the *NP-hard* cousin, Balanced Connected Partition
(Dyer–Frieze 1985), has plenty of software — several GitHub solvers plus
GerryChain's ReCom sampler. The problem that became polynomial in 2026 has none.

**Search gap.** GitHub full-text *code* search requires a login and was not
available, so a file buried in an unrelated repository with no matching name,
description or README would have been missed. Repository-level, README-level,
Zenodo, Software Heritage and general web search agree independently, so the
residual risk is low but not zero.

## 6. Caveats on the numbers

Single seed, single machine, k = 4 for the scaled families, one run per cell.
No repetitions, so the CP-SAT and Mt-KaHyPar figures carry run-to-run variance
(CP-SAT solved `harary/n=500` in 13.6 s but `harary/n=200` in 25.5 s, which is
noise, not a trend). Mt-KaHyPar ran 8-threaded on the non-deterministic QUALITY
preset. The ladder rule — stop climbing a family after the first failure —
is generous to competitors: it never lowers the largest n any of them reached.

Two environment notes for anyone re-running this. `ortools` and `highspy`
cannot share an interpreter here: both ship `libhighs.so.1` and OR-Tools links
an older HiGHS, so whichever loads second fails on an undefined symbol. The
harness runs every (instance, method) pair in its own subprocess, which the
hard timeout needed anyway. And Gurobi's pip licence is capped at 2000
variables and 2000 linear constraints, which for this model shape binds at
n ≈ 50–100.
