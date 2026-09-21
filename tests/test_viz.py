"""Tests of the visualization layer (glsolver.viz): state replay, frames, HTML, GIF, SVG, CLI."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from glsolver import glpartition
from glsolver.generators import paper_running_example
from glsolver.io import load_instance, save_instance
from glsolver.viz import (
    TraceRenderer,
    animate_solution,
    compute_layout,
    counterexample_counts,
    final_figure,
    partition_svg,
    render_html,
    replay,
    visualize_counterexample,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "examples" / "curated"


@pytest.fixture(scope="module")
def paper():
    inst = paper_running_example()
    res = glpartition(inst, algorithm="reference", trace=True)
    assert res.status == "ok" and res.valid and res.trace
    return inst, res


@pytest.fixture(scope="module")
def renderer(paper):
    inst, res = paper
    return TraceRenderer(inst, res.trace)


# --------------------------------------------------------------------------- layout
@pytest.mark.parametrize("method", ["auto", "grid", "kamada", "spring"])
def test_layout_deterministic_with_terminals_on_bottom_row(paper, method):
    inst, _ = paper
    a = compute_layout(inst, seed=0, method=method)
    b = compute_layout(inst, seed=0, method=method)
    assert a == b and set(a) == set(range(inst.n))
    for t in inst.terminals:
        assert a[t][1] == 0.0
    for v in range(inst.n):
        if v not in inst.terminals:
            assert a[v][1] > 0.0
    xs = sorted(a[t][0] for t in inst.terminals)
    assert len(set(xs)) == inst.k  # distinct abscissae on the row


# --------------------------------------------------------------------------- replay
def test_replay_reconstructs_the_final_partition_and_counters(paper):
    inst, res = paper
    states = replay(inst, res.trace)
    assert len(states) == len(res.trace)
    last = states[-1]
    assert last.type == "done"
    assert {t: sorted(vs) for t, vs in last.parts.items()} == {t: sorted(p) for t, p in zip(inst.terminals, res.parts)}
    assert last.num_nonterminals() == 0 and not last.arcs
    for key in ("contractions", "deletions", "cycle_shifts", "terminal_removals"):
        assert last.counters[key] == res.stats[key], key
    # every deletion moves the arc from the live set to the dead set
    for i, st in enumerate(states):
        if st.type == "delete_arc":
            u, v = st.event["u"], st.event["v"]
            assert (u, v) not in st.arcs and (u, states[i - 1].arcs[(u, v)]) in st.dead_arcs
        if st.type == "cycle_shift":  # [Lem 7.11] the potential strictly decreases
            assert st.potential_before is not None and st.potential_after is not None
            assert st.potential_after < st.potential_before
            for v, _old, new in st.changes:
                assert st.phi[v] == new
        if st.type == "contract":
            p = st.event["p"]
            assert p not in st.live and st.part_of[p] == st.event["t"] and p not in st.phi
    # the witness always assigns essential terminals (A1 of paper_notes §7.2)
    for st in states:
        for v, t in st.phi.items():
            assert t in st.ess[v], (st.index, v, t, st.ess[v])


def test_replay_recomputes_essential_sets_after_terminal_removal():
    inst = load_instance(CURATED / "zero_capacity_removal.json")
    res = glpartition(inst, algorithm="reference", trace=True)
    assert res.status == "ok" and res.valid
    states = replay(inst, res.trace)
    exp = inst.meta["expected"]
    idx = next(i for i, s in enumerate(states) if s.type == "remove_terminal")
    v, t_new = exp["new_essential_vertex"], exp["new_essential_terminal"]
    before, after = set(states[idx - 1].ess[v]), set(states[idx].ess[v])
    assert t_new not in before and t_new in after and before < after  # paper_notes §13.3
    assert not states[idx].ess_stale


# --------------------------------------------------------------------------- frames
def test_frames_svg_png_pdf_are_written(renderer, tmp_path: Path):
    i_rg = next(i for i, s in enumerate(renderer.states) if s.type == "reassignment_graph")
    for fmt in ("svg", "png", "pdf"):
        p = renderer.save_frame(i_rg, tmp_path / f"step.{fmt}")
        assert p.exists() and p.stat().st_size > 1000, fmt
    svg = (tmp_path / "step.svg").read_text()
    assert "<svg" in svg and "reassignment graph" in svg
    assert renderer.states[i_rg + 1].type == "cycle_shift"
    fig = renderer.frame(i_rg + 1)  # cycle shift frame: matplotlib Figure with graph, legend, panel, inset
    assert len(fig.axes) == 4


def test_render_all_writes_one_frame_per_event(renderer, tmp_path: Path):
    paths = renderer.render_all(tmp_path / "frames", fmt="svg")
    assert len(paths) == len(renderer) == len(renderer.trace)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_trace_renderer_accepts_a_glresult(paper):
    _inst, res = paper
    r = TraceRenderer(res)
    assert len(r) == len(res.trace)
    with pytest.raises(ValueError):
        TraceRenderer(res.instance, [])


# --------------------------------------------------------------------------- html
def test_html_contains_every_step(paper, tmp_path: Path):
    inst, res = paper
    p = render_html(inst, res.trace, tmp_path / "paper.html")
    s = p.read_text(encoding="utf-8")
    assert s.startswith("<!DOCTYPE html>") and "cdn" not in s.lower() and "<script src" not in s
    assert f'<meta name="gl-steps" content="{len(res.trace)}">' in s
    m = re.search(r"window\.__GL_DATA__ = (.*?);</script>", s, re.S)
    assert m
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(data["steps"]) == len(res.trace)
    assert [st["type"] for st in data["steps"]] == [ev["type"] for ev in res.trace]
    shift = next(st for st in data["steps"] if st["type"] == "cycle_shift")
    assert shift["panel"]["pot"][1] < shift["panel"]["pot"][0]
    reass = next(st for st in data["steps"] if st["type"] == "reassignment_graph")
    assert reass["reass"] and any(on for *_rest, on in reass["reass"]["arcs"])


def test_cut_shading_from_events_carrying_cuts(paper, tmp_path: Path):
    """Events may carry ``cuts`` (paper_notes §14); the tightest cut of v4 from docs/algorithm.md is
    L={t3,v6,v7,v9}, S={t1,t2}, R={v4,v5,v8} (ids 2,5,6,8 / 0,1 / 3,4,7)."""
    inst, res = paper
    trace = [dict(ev) for ev in res.trace]
    trace[1]["cuts"] = {"3": {"L": [2, 5, 6, 8], "S": [0, 1], "R": [3, 4, 7]}}
    r = TraceRenderer(inst, trace)
    st = r.state(1)
    assert st.cuts == {3: {"L": [2, 5, 6, 8], "S": [0, 1], "R": [3, 4, 7]}}
    assert not r.state(2).cuts  # cuts belong to the event that carries them
    p = r.save_frame(1, tmp_path / "cut.svg", cut_vertex=3)
    assert p.stat().st_size > 1000
    html_path = render_html(inst, trace, tmp_path / "cut.html")
    m = re.search(r"window\.__GL_DATA__ = (.*?);</script>", html_path.read_text(encoding="utf-8"), re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    hulls = data["steps"][1]["cuts"]["3"]
    assert len(hulls["L"]) >= 3 and len(hulls["S"]) == 2 and len(hulls["R"]) == 3  # hull polygons in px
    assert data["steps"][1]["panel"]["cutinfo"] == {"3": [4, 2, 3]}
    assert data["steps"][2]["cuts"] is None


# --------------------------------------------------------------------------- gif / mp4
def test_gif_animation_and_mp4_fallback(paper, tmp_path: Path):
    from PIL import Image

    inst, res = paper
    p = animate_solution(res, path=tmp_path / "paper.gif", fps=4)
    assert p.suffix == ".gif" and p.stat().st_size > 10_000
    with Image.open(p) as im:
        assert im.n_frames == len(res.trace)  # one GIF frame per trace event
        im.seek(1)
        base = im.info["duration"]
        im.seek(im.n_frames - 1)
        assert base == 250 and im.info["duration"] == 2 * base  # fps=4, last frame held twice as long
    try:
        import imageio_ffmpeg  # noqa: F401

        have_ffmpeg = True
    except ImportError:
        have_ffmpeg = False
    if have_ffmpeg:
        q = animate_solution(inst, res.trace, tmp_path / "paper.mp4", fps=4)
        assert q.suffix == ".mp4" and q.stat().st_size > 0
    else:
        with pytest.warns(RuntimeWarning, match="ffmpeg"):
            q = animate_solution(inst, res.trace, tmp_path / "paper.mp4", fps=4)
        assert q == tmp_path / "paper.gif" and q.stat().st_size > 0


# --------------------------------------------------------------------------- partition svg / final figure
def test_partition_svg_is_a_standalone_svg(paper):
    inst, res = paper
    svg = partition_svg(inst, res.parts, width=400, parents=res.certificate["parents"])
    assert svg.startswith("<svg") and svg.endswith("</svg>") and 'width="400"' in svg
    assert svg.count("<circle") == inst.num_nonterminals and svg.count("<rect ") == inst.k + 1  # + background
    assert "<script" not in svg
    fig = final_figure(inst, res.parts, parents=res.certificate["parents"])
    assert fig.axes


# --------------------------------------------------------------------------- counterexample
def test_counterexample_figure_renders(tmp_path: Path):
    from glref.counterexample import build_counterexample_instance

    out = visualize_counterexample(1, tmp_path / "cex.svg")
    assert out["path"].exists() and out["path"].stat().st_size > 10_000
    c = out["counts"]
    assert (c["pre_terminals"], c["forcing_vertices"], c["n"], c["m"]) == (108, 216, 333, 2160)
    ci = build_counterexample_instance(1)
    assert (ci.n, ci.m) == (c["n"], c["m"])
    assert "108 pre-terminals" in out["summary"]
    assert counterexample_counts(17)["n"] == 3789


# --------------------------------------------------------------------------- CLI
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "glsolver.cli", *args], capture_output=True, text=True)


def test_cli_visualize_html_and_svg_step(paper, tmp_path: Path):
    inst, res = paper
    p = tmp_path / "inst.json"
    save_instance(inst, p)
    r = _run_cli("visualize", str(p), "--format", "html", "-o", str(tmp_path / "out.html"))
    assert r.returncode == 0, r.stderr
    s = (tmp_path / "out.html").read_text(encoding="utf-8")
    assert f'content="{len(res.trace)}"' in s
    r2 = _run_cli("visualize", str(p), "--format", "svg", "--step", "3", "-o", str(tmp_path / "step3.svg"))
    assert r2.returncode == 0, r2.stderr
    assert (tmp_path / "step3.svg").stat().st_size > 1000 and "wrote step 3" in r2.stdout
    r3 = _run_cli("visualize", str(p), "--format", "svg", "--step", "999")
    assert r3.returncode != 0


# --------------------------------------------------------------------------- curated examples
def _load_render_all():
    spec = importlib.util.spec_from_file_location("render_all", ROOT / "examples" / "render_all.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", sorted(CURATED.glob("*.json")), ids=lambda p: p.stem)
def test_curated_examples_have_their_expected_properties(path: Path):
    mod = _load_render_all()
    inst = load_instance(path)
    res = glpartition(inst, algorithm=inst.meta["viz_algorithm"], trace=True)
    assert res.status == "ok" and res.valid is True
    assert mod.check_expected(inst.meta, res) == []
    assert inst.n <= 16


def test_reassignment_cycle_arcs_follow_the_arc_direction():
    """The reference lists ``cycle`` in backward-walk order (§13.5); the cycle arcs must still be found."""
    from glsolver.viz.render import cycle_arcs

    inst = load_instance(CURATED / "multi_cycle_shift.json")
    res = glpartition(inst, algorithm="reference", trace=True)
    states = replay(inst, res.trace)
    seen_long = False
    for st in states:
        if st.type != "reassignment_graph":
            continue
        arcs = cycle_arcs(st.reassignment)
        cyc = set(st.event["cycle"])
        r_arcs = {(a, b) for a, b, _ in st.reassignment["arcs"]}
        assert len(arcs) == len(cyc) >= 2 and set(arcs) <= r_arcs
        assert {b for _, b in arcs} == cyc and {a for a, _ in arcs} == cyc
        for (_a, b), (c, _d) in zip(arcs, arcs[1:] + arcs[:1]):
            assert b == c  # consecutive arcs chain into a closed cycle
        seen_long = seen_long or len(cyc) >= 3
    assert seen_long  # the curated example has a 3-cycle, where orientation matters


def test_weighted_and_dag_traces_render(tmp_path: Path):
    w = load_instance(CURATED / "weighted_rounding.json")
    res = glpartition(w, algorithm="reference-weighted", trace=True)
    r = TraceRenderer(w, res.trace)
    idx = next(i for i, s in enumerate(r.states) if s.type == "round_and_remove")
    st = r.states[idx]
    assert st.counters["roundings"] == 1
    for t, p in st.event["pairs"]:
        assert st.part_of[p] == t and p not in st.live
    psi_state = next(s for s in r.states if s.type == "min_cost_split")
    v = next(v for v in range(w.n) if v not in w.terminals and v in psi_state.live)
    assert abs(sum(f for _, f in psi_state.ring_of(v)) - 1.0) < 1e-9
    r.save_frame(idx, tmp_path / "round.svg")
    d = load_instance(CURATED / "small_dag.json")
    res_d = glpartition(d, algorithm="reference-dag", trace=True)
    rd = TraceRenderer(d, res_d.trace)
    assert sum(1 for s in rd.states if s.type == "dag_contract") == d.num_nonterminals
    assert rd.states[-1].counters["contractions"] == d.num_nonterminals
    rd.save_frame(3, tmp_path / "dag.png")
    assert (tmp_path / "dag.png").stat().st_size > 1000
