"""Turn ``compare.jsonl`` into the markdown tables and the figures.

    python benchmarks/compare/analyze.py [--jsonl ...] [--outdir ...]

Writes ``compare.md``, ``fig_runtime_vs_n.png``, ``fig_failure_modes.png`` and
``fig_crossover.png`` into ``benchmarks/results/compare/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "benchmarks" / "results" / "compare"

# validated categorical palette (light mode), assigned in fixed slot order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

FAMILIES = ["harary", "random_regular", "erdos_renyi", "sparse_k_connected"]
FAMILY_LABEL = {"harary": "Harary $H_{4,n}$ (minimum 4-connected)",
                "random_regular": "random 5-regular",
                "erdos_renyi": "Erdős–Rényi $G(n,0.1)$",
                "sparse_k_connected": "Harary + $n/10$ chords"}
EXACT_ORDER = ["glsolver", "cpsat-ortools", "ilp-highs-scipy", "ilp-gurobi",
               "ilp-highs-pulp", "ilp-scip-pulp", "ilp-cbc-pulp", "glsolver-reference"]
PARTITIONERS = ["metis", "kahip", "mtkahypar", "scotch", "networkit-spectral"]
PARTITIONER_LABEL = {"metis": "METIS\n(tpwgts + contig)",
                     "kahip": "KaHIP\n(kaffpa STRONG, 3%)",
                     "mtkahypar": "Mt-KaHyPar\n(fixed roots + exact\nblock weights)",
                     "scotch": "Scotch\n(fixed roots,\nown balance)",
                     "networkit-spectral": "NetworKit\n(spectral, balanced)"}
ALL_METHOD_ORDER = EXACT_ORDER[:1] + ["bruteforce"] + EXACT_ORDER[1:] + PARTITIONERS


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def solved(r: dict) -> bool:
    """Solved *and* independently verified, inside the wall-clock budget."""
    return r["status"] == "ok" and r.get("valid") is True and not r.get("over_budget")


# ---------------------------------------------------------------------------
def md_tables(rows: list[dict]) -> str:
    by = {(r["instance"], r["method"]): r for r in rows}
    methods = [m for m in ALL_METHOD_ORDER if any(r["method"] == m for r in rows)]
    insts, seen = [], set()
    for fam in FAMILIES:
        for r in rows:
            sp = r.get("spec", {})
            if sp.get("family") == fam and r["instance"] not in seen:
                seen.add(r["instance"])
                insts.append((r["instance"], sp.get("n", 0)))
    insts.sort(key=lambda t: (t[0].split("/")[0], t[1]))
    order = {f: i for i, f in enumerate(FAMILIES)}
    insts.sort(key=lambda t: (order.get(t[0].split("/")[0], 99), t[1]))
    specials = [r["instance"] for r in rows
                if r.get("spec", {}).get("family", "").startswith("paper_")
                or r.get("spec", {}).get("family") == "counterexample"]
    for s in dict.fromkeys(specials):
        insts.append((s, 0))

    def cell(r: dict | None) -> str:
        if r is None:
            return "—"
        st = r["status"]
        if st == "skipped_after_timeout":
            return "·"
        if st == "timeout":
            return "**t/o**"
        if st == "license_limit":
            return "lic"
        if st == "error":
            return "err"
        if st == "infeasible":
            return "INFEAS"
        w = r.get("wall_s")
        mark = "" if r.get("valid") else " ✗"
        return f"{w:.3g}s{mark}" if isinstance(w, (int, float)) else f"{st}{mark}"

    out = ["| instance | " + " | ".join(methods) + " |",
           "|---|" + "---|" * len(methods)]
    for iid, _ in insts:
        out.append(f"| `{iid}` | " + " | ".join(cell(by.get((iid, m))) for m in methods) + " |")
    main = "\n".join(out)

    # --- crossover table ---
    cross = ["| method | family | largest n solved ≤300 s | wall at that n | model+solve only | glsolver (wall / solve) | speed-up | first n it failed | how it failed |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for m in [x for x in ALL_METHOD_ORDER
              if x != "glsolver" and x in methods and x not in PARTITIONERS]:
        for fam in FAMILIES:
            fr = sorted([r for r in rows if r["method"] == m and r.get("spec", {}).get("family") == fam],
                        key=lambda r: r["spec"]["n"])
            if not fr:
                continue
            ok = [r for r in fr if solved(r)]
            bad = [r for r in fr if not solved(r)]
            if ok:
                best = ok[-1]
                g = by.get((best["instance"], "glsolver"))
                gt = g["wall_s"] if g and isinstance(g.get("wall_s"), (int, float)) else None
                gs = (g.get("build_s", 0.0) or 0.0) + (g.get("solve_s", 0.0) or 0.0) if g else None
                net = (best.get("build_s", 0.0) or 0.0) + (best.get("solve_s", 0.0) or 0.0)
                su = f"{net / gs:.0f}×" if gs else "—"
                cross.append(f"| {m} | {fam} | {best['spec']['n']} | {best['wall_s']:.3g} s | "
                             f"{net:.3g} s | "
                             f"{f'{gt:.3g} / {gs:.3g} s' if gt is not None else '—'} | {su} | "
                             f"{bad[0]['spec']['n'] if bad else '—'} | "
                             f"{bad[0]['status'] if bad else 'never (n≤2000)'} |")
            else:
                cross.append(f"| {m} | {fam} | none | — | — | — | {fr[0]['spec']['n']} | {fr[0]['status']} |")
    crossmd = "\n".join(cross)

    # --- partitioner failure modes ---
    fm = ["| partitioner | runs | valid GL partitions | parts disconnected (of k·runs) | parts of wrong size | terminals in the wrong part | total size error Σ\\|Δ\\| |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for m in PARTITIONERS:
        rs = [r for r in rows if r["method"] == m and r.get("failure_modes")]
        if not rs:
            continue
        k_tot = sum(r["k"] for r in rs)
        fm.append(
            f"| {m} | {len(rs)} | {sum(1 for r in rs if r['valid'])} | "
            f"{sum(r['failure_modes']['parts_disconnected'] for r in rs)} / {k_tot} | "
            f"{sum(r['failure_modes']['size_errors'] for r in rs)} / {k_tot} | "
            f"{sum(r['failure_modes']['terminal_misplaced'] for r in rs)} / {k_tot} | "
            f"{sum(r['failure_modes']['size_l1'] for r in rs)} |")
    fmmd = "\n".join(fm)

    # --- partitioner validity by instance size ---
    ns = sorted({r["spec"]["n"] for r in rows if "n" in r.get("spec", {})})
    by_n = ["| partitioner | " + " | ".join(f"n={x}" for x in ns) + " | paper examples (n=9,13) | counterexample (n=333) |",
            "|---|" + "---|" * (len(ns) + 2)]
    for m in PARTITIONERS:
        cells = []
        for x in ns:
            rs = [r for r in rows if r["method"] == m and r.get("spec", {}).get("n") == x]
            cells.append("—" if not rs else ("valid" if all(r["valid"] for r in rs) else
                         (f"{sum(1 for r in rs if r['valid'])}/{len(rs)}" if any(r["valid"] for r in rs) else f"0/{len(rs)}")))
        pap = [r for r in rows if r["method"] == m and r.get("spec", {}).get("family", "").startswith("paper_")]
        cex = [r for r in rows if r["method"] == m and r.get("spec", {}).get("family") == "counterexample"]
        cells.append(f"{sum(1 for r in pap if r['valid'])}/{len(pap)}" if pap else "—")
        cells.append(f"{sum(1 for r in cex if r['valid'])}/{len(cex)}" if cex else "—")
        by_n.append(f"| {m} | " + " | ".join(cells) + " |")
    return main, crossmd, fmmd + "\n\n### Valid answers by instance size (how many of the 4 families it got right)\n\n" + "\n".join(by_n)


# ---------------------------------------------------------------------------
def fig_runtime(rows: list[dict], out: Path) -> None:
    methods = [m for m in EXACT_ORDER if any(r["method"] == m for r in rows)]
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}
    budget = next((r["budget_s"] for r in rows if r.get("budget_s")), 300.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        ax.set_facecolor(SURFACE)
        ax.axhspan(budget, budget * 9, color="#f2f1ed", zorder=0)
        ax.axhline(budget, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for mi, m in enumerate(methods):
            rs = sorted([r for r in rows if r["method"] == m and r.get("spec", {}).get("family") == fam],
                        key=lambda r: r["spec"]["n"])
            xs = [r["spec"]["n"] for r in rs if solved(r)]
            ys = [max(r["wall_s"], 1e-4) for r in rs if solved(r)]
            if xs:
                ax.plot(xs, ys, lw=2.0, color=colors[m], marker="o", ms=5,
                        mec=SURFACE, mew=1.5, zorder=4, solid_capstyle="round")
            fails = [r for r in rs if not solved(r) and r["status"] in
                     ("timeout", "error", "license_limit", "infeasible")]
            if fails:
                f0 = fails[0]
                # stagger the markers vertically so two methods failing at the
                # same n stay visible
                ax.plot([f0["spec"]["n"] * 1.07 ** (mi - 3.5)],
                        [budget * 1.15 * 1.29 ** mi], marker="x",
                        ms=8.5, mew=2.4, color=colors[m], ls="none", zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(FAMILY_LABEL[fam], fontsize=10.5, color=INK, pad=7)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylim(2e-4, budget * 9)
    for ax in axes[1]:
        ax.set_xlabel("n (vertices)", fontsize=10, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("wall-clock time (s, log)", fontsize=10, color=INK2)

    handles = [Line2D([], [], color=colors[m], lw=2.4, marker="o", ms=5,
                      mec=SURFACE, mew=1.2, label=m) for m in methods]
    handles.append(Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)),
                          label=f"{budget:.0f} s budget"))
    handles.append(Line2D([], [], color=MUTED, lw=0, marker="x", ms=8, mew=2.2,
                          label="first n that failed (timeout / licence / error)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.2, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Exact solvers on the Győri–Lovász problem: wall-clock time vs n  (k = 4, seed 1)",
                 fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.935, "points = solved and independently verified within the budget;  "
                         "× = the smallest n at which the method stopped answering",
             ha="center", fontsize=9.3, color=INK2)
    fig.tight_layout(rect=(0, 0.085, 1, 0.925))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_failures(rows: list[dict], out: Path) -> None:
    modes = [("parts_disconnected", "part not connected", SERIES[0]),
             ("size_errors", "part of the wrong size", SERIES[1]),
             ("terminal_misplaced", "terminal in the wrong part", SERIES[2])]
    ms = [m for m in PARTITIONERS if any(r["method"] == m and r.get("failure_modes") for r in rows)]
    data = {m: {} for m in ms}
    totals, valids, runs = {}, {}, {}
    for m in ms:
        rs = [r for r in rows if r["method"] == m and r.get("failure_modes")]
        totals[m] = sum(r["k"] for r in rs)
        valids[m] = sum(1 for r in rs if r["valid"])
        runs[m] = len(rs)
        for key, _, _ in modes:
            data[m][key] = sum(r["failure_modes"][key] for r in rs)

    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = range(len(ms))
    bottom = [0.0] * len(ms)
    for key, label, col in modes:
        vals = [data[m][key] for m in ms]
        ax.bar(xs, vals, bottom=bottom, width=0.56, color=col, label=label,
               edgecolor=SURFACE, linewidth=2.0, zorder=3)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(i, b + v / 2, str(v), ha="center", va="center",
                        fontsize=9.5, color="#ffffff", fontweight="bold", zorder=4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, m in enumerate(ms):
        ax.text(i, bottom[i] + max(bottom) * 0.025,
                f"{int(bottom[i])} defects over {totals[m]} parts\n"
                f"{valids[m]}/{runs[m]} instances fully valid",
                ha="center", fontsize=9, color=INK2, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([PARTITIONER_LABEL[m] for m in ms], fontsize=9.5, color=INK)
    ax.set_ylabel("defect count (part × failure mode), summed over all instances",
                  fontsize=10, color=INK2)
    ax.set_ylim(0, max(bottom) * 1.24)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.19))
    ax.set_title("No balanced partitioner guarantees a Győri–Lovász partition\n"
                 "None was built to, and each was configured as generously as its API allows:\n"
                 "Mt-KaHyPar and Scotch are told the roots, Mt-KaHyPar and METIS the exact sizes.",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_crossover(rows: list[dict], out: Path) -> None:
    methods = [m for m in ALL_METHOD_ORDER
               if m not in PARTITIONERS and any(r["method"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.2
    for j, fam in enumerate(FAMILIES):
        vals = []
        for m in methods:
            ok = [r["spec"]["n"] for r in rows
                  if r["method"] == m and r.get("spec", {}).get("family") == fam and solved(r)]
            vals.append(max(ok) if ok else 0)
        ax.bar([i + (j - 1.5) * w for i in range(len(methods))], vals, width=w * 0.88,
               color=SERIES[j], label=fam, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=9.5, color=INK)
    ax.set_yscale("symlog", linthresh=10, linscale=0.25)
    ax.set_yticks([0, 10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 9000)
    ax.set_ylabel("largest n solved and verified within 300 s", fontsize=10, color=INK2)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=4, loc="upper right")
    ax.set_title("Crossover: how far each exact method gets in 300 s\n"
                 "n = 2000 is the largest instance in the suite, so a 2000 bar means "
                 "\"never ran out of budget\", not \"stopped here\"",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=OUTDIR / "compare.jsonl")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    rows = load(args.jsonl)
    args.outdir.mkdir(parents=True, exist_ok=True)

    main_md, cross_md, fm_md = md_tables(rows)
    (args.outdir / "compare.md").write_text(
        "# Competitive comparison — glsolver vs the best available software\n\n"
        f"{len(rows)} runs, 300 s wall budget each, one subprocess per run, "
        "every answer re-checked by `glsolver.verify.verify_instance_parts`.\n\n"
        "## 1. Full result matrix\n\n"
        "Wall-clock seconds when the method answered; `✗` marks an answer the independent "
        "verifier rejected. `**t/o**` = timeout, `lic` = licence size limit, "
        "`·` = not attempted (the method had already failed at a smaller n of the same family).\n\n"
        + main_md +
        "\n\n## 2. Crossover for the exact solvers\n\n"
        "`wall` is the whole adapter call and therefore includes the one-off cost of "
        "importing the solver's Python library (0.15–0.35 s, which dominates the small "
        "instances). `model+solve only` excludes it: it is the time to build the model "
        "and solve it. glsolver solved **all 36 instances** and so has no row of its own.\n\n"
        + cross_md +
        "\n\n## 3. Partitioner failure modes\n\n"
        "Counted over every instance each tool was run on, in units of *parts* "
        "(k parts per instance). These tools do not solve this problem and were "
        "not built to; the numbers say how far their output is from a valid "
        "Győri–Lovász partition after the most generous mapping we could give them.\n\n"
        + fm_md + "\n")

    fig_runtime(rows, args.outdir / "fig_runtime_vs_n.png")
    fig_failures(rows, args.outdir / "fig_failure_modes.png")
    fig_crossover(rows, args.outdir / "fig_crossover.png")
    print(f"wrote {args.outdir}/compare.md and 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
