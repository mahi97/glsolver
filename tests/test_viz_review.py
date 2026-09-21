"""Review tests of the visualization layer (``glsolver.viz``).

Every test parses the JSON state embedded in the self-contained HTML viewer
(``window.__GL_DATA__``) and/or the replayed :class:`StepState` objects and
asserts facts *against the trace events* (docs/paper_notes.md §14) of the
paper's running example (``glsolver.generators.paper_running_example``):

1. after a ``cycle_shift`` the witness rings of the shifted vertices show the
   new terminals (and the old colour lingers as a dashed marker on that frame only);
2. after ``delete_arc`` the arc is drawn dashed/grey, is no longer live and the
   live-arc count decreases;
3. after ``contract`` the vertex is drawn as part of the terminal's part, the
   arborescence arc appears and the capacity in the panel decreases;
4. the essential sectors equal the ``essential`` events (the tracer re-emits
   ``Ess``/``κ`` after every recomputation) and an independent NetworkX
   max-flow computation [Def 4.1]; the replay imports no solver internals;
5. matching / secondary arcs and the reassignment inset are highlighted only
   inside a ShiftAssignment call (``matching`` … ``delete_arc`` inclusive).

Further checks: one frame per event, byte-determinism (PNG / HTML / partition
SVG / SVG and PDF frames / counterexample SVG), valid PNG/GIF images, no
external resources in the HTML, escaped labels, rejected renderer kwargs and
foreign traces, the counterexample figure (9 terminals, 108 pre-terminal
markers), the CLI on every format and its ``meta['viz_algorithm']`` default,
the git-ignored ``examples/output/``, and the import surface of the viz package.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import pytest

from glsolver import glpartition
from glsolver.generators import paper_running_example
from glsolver.io import load_instance
from glsolver.viz import (
    TraceRenderer,
    animate_solution,
    counterexample_figure,
    partition_svg,
    render_html,
    replay,
    visualize_counterexample,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "examples" / "curated"
VIZ_DIR = ROOT / "src" / "glsolver" / "viz"

# ----------------------------------------------------------------------------- helpers
DATA_RE = re.compile(r"window\.__GL_DATA__ = (.*?);</script>", re.S)


def viewer_data(html_path: Path) -> dict:
    """The JSON state embedded in a viewer written by :func:`render_html`."""
    text = html_path.read_text(encoding="utf-8")
    m = DATA_RE.search(text)
    assert m, "no embedded __GL_DATA__ in the HTML viewer"
    return json.loads(m.group(1).replace("<\\/", "</"))


def node_of(step: dict, v: int) -> dict:
    return next(n for n in step["nodes"] if n["v"] == v)


def arc_classes(step: dict) -> set[str]:
    return {cls for _key, cls, _label in step["arcs"]}


def shift_windows(trace: list[dict]) -> list[tuple[int, int]]:
    """``(matching index, following delete_arc index)`` of every ShiftAssignment call."""
    out = []
    for i, ev in enumerate(trace):
        if ev["type"] == "matching":
            j = next(k for k in range(i + 1, len(trace)) if trace[k]["type"] == "delete_arc")
            out.append((i, j))
    return out


def kappa_nx(arcs: list[tuple[int, int]], terminals: list[int], live: set[int], v: int, drop: int | None = None) -> int:
    """``κ_G(v)`` by one NetworkX max-flow on the vertex-split network [Prop 4.2];
    ``drop`` removes a terminal (for the ``κ_{G\\{t}}`` cross-check of [Def 4.1])."""
    K = len(terminals) + 1
    G = nx.DiGraph()
    G.add_nodes_from(["s", "z", f"{v}i"])  # keep the sink when the last terminal is dropped
    for x in live:
        if x != drop:
            G.add_edge(f"{x}i", f"{x}o", capacity=K if x == v else 1)
    for u, w in arcs:
        if drop not in (u, w):
            G.add_edge(f"{u}o", f"{w}i", capacity=K)
    G.add_edge("s", f"{v}i", capacity=K)
    for t in terminals:
        if t != drop:
            G.add_edge(f"{t}o", "z", capacity=K)
    return int(nx.maximum_flow_value(G, "s", "z"))


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def paper():
    inst = paper_running_example()
    res = glpartition(inst, algorithm="reference", trace=True)
    assert res.status == "ok" and res.valid and res.trace
    return inst, res


@pytest.fixture(scope="module")
def paper_html(paper, tmp_path_factory) -> tuple[list[dict], dict]:
    inst, res = paper
    path = render_html(inst, res.trace, tmp_path_factory.mktemp("html") / "paper.html")
    return res.trace, viewer_data(path)


@pytest.fixture(scope="module")
def dag():
    inst = load_instance(CURATED / "small_dag.json")
    res = glpartition(inst, algorithm="reference-dag", trace=True)
    assert res.status == "ok" and res.valid and res.trace
    return inst, res


# ----------------------------------------------------------------------------- frames per event
def test_every_event_index_has_a_step_and_a_frame(paper, paper_html, tmp_path: Path):
    inst, res = paper
    trace, data = paper_html
    steps = data["steps"]
    assert [s["i"] for s in steps] == list(range(len(trace)))
    assert [s["type"] for s in steps] == [ev["type"] for ev in trace]
    assert data["n"] == inst.n and data["k"] == inst.k
    paths = TraceRenderer(inst, res.trace).render_all(tmp_path / "frames", "svg")
    assert [p.name for p in paths] == [f"step_{i:03d}.svg" for i in range(len(trace))]
    for i, p in enumerate(paths):
        root = ET.parse(p).getroot()  # well-formed XML
        assert root.tag.endswith("svg")
        text = p.read_text(encoding="utf-8")
        # Matplotlib writes every text string as a comment before its glyph paths
        assert f"step {i}/{len(trace) - 1}" in text, p.name


# ----------------------------------------------------------------------------- (1) cycle shift
def test_cycle_shift_changes_rings_to_the_new_terminals(paper_html):
    trace, data = paper_html
    steps, tidx = data["steps"], {int(k): v for k, v in data["tidx"].items()}
    shifts = [i for i, ev in enumerate(trace) if ev["type"] == "cycle_shift"]
    assert len(shifts) == 2  # examples/README.md: two cycle shifts
    for i in shifts:
        ev = trace[i]
        assert ev["changes"]
        for v, old_t, new_t in ev["changes"]:
            before, after = node_of(steps[i - 1], v), node_of(steps[i], v)
            assert before["ring"] == [[tidx[old_t], 1.0]], (i, v)
            assert after["ring"] == [[tidx[new_t], 1.0]], (i, v)
            assert after["old"] == [tidx[old_t]] and after["glow"], (i, v)
            # the dashed "previous colour" marker is shown on the shift frame only
            assert node_of(steps[i + 1], v)["old"] == []
            # the full witness of the event is what the panel shows
            assert f"{v}→t{tidx[new_t] + 1}" in steps[i]["panel"]["witness"]
        # every ring at the shift step equals the phi carried by the event
        for v, t in ev["phi"].items():
            assert node_of(steps[i], int(v))["ring"] == [[tidx[t], 1.0]]
        # [Lem 7.11] the potential Φ decreases (computed from the criticality event)
        pb, pa = steps[i]["panel"]["pot"]
        assert pb is not None and pa is not None and pa < pb
        # the cycle in the inset is exactly the event's cycle, in arc direction
        rg = trace[i - 1]
        assert rg["type"] == "reassignment_graph"
        on = [(a, b) for a, b, _lab, flag in steps[i - 1]["reass"]["arcs"] if flag]
        names = steps[i - 1]["reass"]["nodes"]
        terms = [next(int(t) for t, j in data["tidx"].items() if j == ti) for ti, _ in names]
        on_terms = {(terms[a], terms[b]) for a, b in on}
        cyc = set(rg["cycle"])
        assert {a for a, _ in on_terms} == cyc and {b for _, b in on_terms} == cyc
        assert on_terms <= {(a, b) for a, b, _v in rg["arcs"]}


# ----------------------------------------------------------------------------- (2) delete_arc
def test_delete_arc_is_drawn_dead_and_not_counted_live(paper, paper_html):
    inst, res = paper
    trace, data = paper_html
    steps = data["steps"]
    states = replay(inst, res.trace)
    dels = [i for i, ev in enumerate(trace) if ev["type"] == "delete_arc"]
    assert len(dels) == res.stats["deletions"] == 3
    for i in dels:
        u, v = trace[i]["u"], trace[i]["v"]
        head = states[i - 1].arcs[(u, v)]  # draw head before the deletion
        key = f"{u}-{head}"
        before = {k: c for k, c, _ in steps[i - 1]["arcs"]}
        now = {k: c for k, c, _ in steps[i]["arcs"]}
        later = {k: c for k, c, _ in steps[i + 1]["arcs"]}
        assert before[key] in ("arc", "sec", "match")  # live before (here: a secondary arc)
        assert before[key] == "sec"
        assert now[key] == "deadnow" and later[key] in ("dead", "deadfaint")
        assert (u, v) not in states[i].arcs and (u, head) in states[i].dead_arcs
        assert steps[i]["panel"]["arcs"] == steps[i - 1]["panel"]["arcs"] - 1 == len(states[i].arcs)
        assert "✕" not in json.dumps(steps[i - 1])  # the ✕ marker is drawn by the viewer for deadnow only
    # the viewer draws "dead" classes dashed and grey (style table + JS)
    assert data["style"]["dead"].lower() == "#bdbdbd"


# ----------------------------------------------------------------------------- (3) contract
def test_contract_moves_vertex_into_part_and_decreases_capacity(paper, paper_html):
    inst, res = paper
    trace, data = paper_html
    steps, tidx = data["steps"], {int(k): v for k, v in data["tidx"].items()}
    contracts = [i for i, ev in enumerate(trace) if ev["type"] == "contract"]
    assert len(contracts) == res.stats["contractions"] == 6
    for i in contracts:
        ev = trace[i]
        p, t, parent = ev["p"], ev["t"], ev["parent"]
        before, after = node_of(steps[i - 1], p), node_of(steps[i], p)
        assert before["kind"] == "n" and after["kind"] == "c" and after["ti"] == tidx[t] and after["glow"]
        # capacity panel: c_t decreased by one, part size increased by one, other terminals untouched
        caps_before = {row[1]: row[2] for row in steps[i - 1]["panel"]["terminals"]}
        caps_after = {row[1]: row[2] for row in steps[i]["panel"]["terminals"]}
        size_before = {row[1]: row[3] for row in steps[i - 1]["panel"]["terminals"]}
        size_after = {row[1]: row[3] for row in steps[i]["panel"]["terminals"]}
        assert caps_after[t] == caps_before[t] - 1 and size_after[t] == size_before[t] + 1
        for x in caps_before:
            if x != t:
                assert caps_after[x] == caps_before[x] and size_after[x] == size_before[x]
        # the panel equals the capacities carried by the event
        assert {int(k): c for k, c in ev["capacities"].items()} == {x: c for x, c in caps_after.items() if c != "done"}
        assert steps[i]["panel"]["left"] == steps[i - 1]["panel"]["left"] - 1
        # the arborescence arc (p, parent) appears in the part colour, highlighted on this frame
        tree_now = [(k, ti, now) for k, ti, now in steps[i]["tree"]]
        assert (f"{p}-{parent}", tidx[t], 1) in tree_now
        assert (f"{p}-{parent}", tidx[t], 0) in [(k, ti, now) for k, ti, now in steps[i + 1]["tree"]]
        # p left the witness and the essential table
        assert f"{p}→" not in steps[i]["panel"]["witness"]
    # the final parts of the done step equal the solver's parts; every terminal is either
    # removed ("done") or, for the last one standing, has capacity 0 (no remove_terminal event)
    done = steps[-1]
    for row in done["panel"]["terminals"]:
        assert row[2] in ("done", 0)
    assert sum(1 for row in done["panel"]["terminals"] if row[2] == "done") == res.stats["terminal_removals"] == 2
    assert {n["v"]: n["ti"] for n in done["nodes"] if n["kind"] == "c"} == {
        v: i for i, part in enumerate(res.parts) for v in part if v != inst.terminals[i]
    }


# ----------------------------------------------------------------------------- (4) essential sectors
def test_essential_sectors_match_the_essential_event(paper_html):
    trace, data = paper_html
    steps, tidx = data["steps"], {int(k): v for k, v in data["tidx"].items()}
    e = next(i for i, ev in enumerate(trace) if ev["type"] == "essential")
    first_mutation = next(i for i, ev in enumerate(trace) if ev["type"] in ("delete_arc", "contract", "remove_terminal"))
    ess = {int(v): [tidx[t] for t in ts] for v, ts in trace[e]["ess"].items()}
    assert ess and all(len(s) >= 1 for s in ess.values())
    for i in range(e, first_mutation):  # the graph is unchanged until the first mutation
        for v, sectors in ess.items():
            nd = node_of(steps[i], v)
            assert nd["kind"] == "n" and nd["ess"] == sectors and not nd["stale"], (i, v)
            assert f"κ = {trace[e]['kappa'][str(v)]}" in nd["tip"]
    for v in range(data["n"]):
        if v not in ess and str(v) not in data["tidx"]:
            pytest.fail(f"non-terminal {v} has no essential set in the event")
    # sectors never show a terminal that is not essential: ring ⊆ sectors (witness property A1)
    for st in steps:
        for nd in st["nodes"]:
            if nd["kind"] == "n" and nd["ring"]:
                assert {ti for ti, _ in nd["ring"]} <= set(nd["ess"]), (st["i"], nd["v"])


def test_tracer_emits_essential_after_every_recomputation(paper):
    """paper_notes §14: one ``essential`` event initially and one right after every
    ``delete_arc`` / ``remove_terminal`` / ``round_and_remove`` (the reference solver
    recomputes ``Ess`` exactly there, §13.3) — for the unweighted and the weighted tracer."""
    inst, res = paper
    for algo in ("reference", "reference-weighted"):
        trace = res.trace if algo == "reference" else glpartition(inst, algorithm=algo, trace=True).trace
        types = [ev["type"] for ev in trace]
        ess_idx = [i for i, t in enumerate(types) if t == "essential"]
        mutations = [i for i, t in enumerate(types) if t in ("delete_arc", "remove_terminal", "round_and_remove")]
        assert ess_idx[0] == 1 and ess_idx[1:] == [i + 1 for i in mutations], algo
        assert len(ess_idx) == 1 + res.stats["deletions"] + res.stats["terminal_removals"] + res.stats["roundings"] or algo != "reference"
        for i in mutations:  # contractions leave Ess unchanged (§13.2) and emit nothing
            assert types[i + 1] == "essential"
        for i, t in enumerate(types):
            if t == "contract":
                assert types[i + 1] != "essential"
        for i in ess_idx:
            assert set(trace[i]) == {"type", "ess", "kappa"} and trace[i]["ess"] and trace[i]["kappa"]


def test_replayed_essential_sets_match_an_independent_max_flow(paper):
    """Every state from the first ``essential`` on carries current sets (the mutation
    frame takes them from the ``essential`` event that follows it); check them against
    ``Ess(v) = {t : κ_{G\\{t}}(v) = κ_G(v) − 1}`` [Def 4.1] computed with NetworkX."""
    inst, res = paper
    states = replay(inst, res.trace)
    e = next(i for i, s in enumerate(states) if s.type == "essential")
    checked = 0
    for st in states[e:]:
        assert not st.ess_stale, st.index
        tset = set(st.terminals)
        arcs = list(st.arcs.keys())
        for v in sorted(st.live - tset):
            k = kappa_nx(arcs, st.terminals, st.live, v)
            ess = sorted(t for t in st.terminals if kappa_nx(arcs, st.terminals, st.live, v, drop=t) == k - 1)
            assert st.kappa[v] == k and st.ess[v] == ess, (st.index, st.type, v)
            checked += 1
    assert checked > 50


def test_recomputed_essential_frames_highlight_the_changes():
    """zero_capacity_removal (paper_notes §13.3): the ``essential`` frame after the
    ``remove_terminal`` shows ``Ess(3): {t1} → {t1,t3}`` and glows exactly the changed vertices."""
    inst = load_instance(CURATED / "zero_capacity_removal.json")
    res = glpartition(inst, algorithm="reference", trace=True)
    states = replay(inst, res.trace)
    exp = inst.meta["expected"]
    v, t_new = exp["new_essential_vertex"], exp["new_essential_terminal"]
    rm = next(i for i, s in enumerate(states) if s.type == "remove_terminal")
    st = states[rm + 1]
    assert st.type == "essential" and st.ess_recomputed and not st.ess_stale
    before, after = st.ess_changes[v]
    assert before == states[rm - 1].ess[v] and after == st.ess[v] == states[rm].ess[v]  # look-ahead on the mutation frame
    assert t_new not in before and t_new in after
    assert st.highlight_vertices == sorted(set(st.ess_changes) | set(st.kappa_changes)) and v in st.highlight_vertices
    assert f"Ess({v}) {{t1}}→{{t1,t3}}" in st.description and "[§13.3]" in st.description
    assert not states[1].ess_recomputed and states[1].highlight_vertices == []  # the initial computation
    # a recomputation that changes nothing says so
    unchanged = [s for s in states if s.ess_recomputed and not s.ess_changes and not s.kappa_changes]
    for s in unchanged:
        assert "unchanged" in s.description and s.highlight_vertices == []


def test_trace_without_reemitted_essential_is_flagged_stale(paper, tmp_path: Path):
    """A backend that does not re-emit ``essential`` (only the initial one) gets stale
    sectors from the first mutation on — the replay must not fall back to an oracle."""
    inst, res = paper
    first = next(i for i, ev in enumerate(res.trace) if ev["type"] == "essential")
    trace = [ev for i, ev in enumerate(res.trace) if ev["type"] != "essential" or i == first]
    assert len(trace) == len(res.trace) - res.stats["deletions"] - res.stats["terminal_removals"]
    states = replay(inst, trace)
    first_mut = next(i for i, s in enumerate(states) if s.type in ("delete_arc", "remove_terminal"))
    assert all(not s.ess_stale for s in states[:first_mut]) and all(s.ess_stale for s in states[first_mut:])
    data = viewer_data(render_html(inst, trace, tmp_path / "stale.html"))
    assert data["steps"][first_mut]["panel"]["stale"] is True and not data["steps"][first_mut - 1]["panel"]["stale"]
    assert all(n["stale"] for n in data["steps"][first_mut]["nodes"] if n["kind"] == "n")
    TraceRenderer(inst, trace).save_frame(first_mut, tmp_path / "stale.svg")  # still renders (faded sectors)
    # an essential event re-emitted later clears the flag again
    trace2 = trace[: first_mut + 1] + [res.trace[first_mut + 1]] + trace[first_mut + 1 :]
    assert res.trace[first_mut + 1]["type"] == "essential"
    states2 = replay(inst, trace2)
    assert states2[first_mut].ess_stale is False and states2[first_mut + 1].type == "essential"  # look-ahead


def test_foreign_or_malformed_traces_are_rejected(paper, tmp_path: Path):
    """A trace of another instance (different ``init``) and a trace naming a non-terminal
    as a terminal raise ``ValueError`` instead of being drawn with terminal 0's colour."""
    from glsolver.generators import paper_essential_example

    inst, res = paper
    other = paper_essential_example()
    with pytest.raises(ValueError, match="different instance"):
        replay(other, res.trace)
    bad = [dict(ev) for ev in res.trace]
    e = next(i for i, ev in enumerate(bad) if ev["type"] == "essential")
    non_terminal = next(v for v in range(inst.n) if v not in inst.terminals)
    bad[e] = dict(bad[e], ess={**bad[e]["ess"], str(non_terminal): [non_terminal]})  # a non-terminal as 'essential terminal'
    with pytest.raises(ValueError, match="as a terminal"):
        render_html(inst, bad, tmp_path / "bad.html")
    with pytest.raises(ValueError, match="as a terminal"):
        TraceRenderer(inst, bad).frame(e)
    bad2 = [dict(ev) for ev in res.trace]
    w = next(i for i, ev in enumerate(bad2) if ev["type"] == "witness")
    bad2[w] = dict(bad2[w], phi={**bad2[w]["phi"], str(non_terminal): non_terminal})  # phi to a non-terminal
    with pytest.raises(ValueError, match="as a terminal"):
        render_html(inst, bad2, tmp_path / "bad2.html")


# ----------------------------------------------------------------------------- (5) matching / secondary
def test_matching_and_secondary_arcs_only_during_shift_assignment(paper, paper_html):
    inst, res = paper
    trace, data = paper_html
    steps = data["steps"]
    states = replay(inst, res.trace)
    windows = shift_windows(trace)
    assert len(windows) == res.stats["shift_calls"] == 3
    active = {i for a, b in windows for i in range(a, b + 1)}
    for i, st in enumerate(steps):
        cls = arc_classes(st)
        if i in active:
            assert "match" in cls and "sec" in cls, (i, st["type"])
        else:
            assert not ({"match", "sec"} & cls), (i, st["type"], cls)
            assert st["reass"] is None and st["panel"]["crit"] == ""
    for _a, b in windows:  # the delete_arc frame keeps the call's context (documented); the next event drops it
        assert steps[b]["type"] == "delete_arc" and steps[b + 1]["type"] == "essential"
        assert "stay shown" in steps[b]["desc"] and re.search(r"delete e\d = ", steps[b]["desc"])
        assert not ({"match", "sec"} & arc_classes(steps[b + 1])) and steps[b + 1]["reass"] is None
    for a, b in windows:
        ev = trace[a]
        st = states[a]
        sec_keys = {f"{p}-{st.draw_arc(p, q)[1]}": f"e{j + 1}" for j, (p, q) in enumerate(ev["secondary"])}
        match_keys = {f"{p}-{st.draw_arc(p, t)[1]}" for p, t in ev["pairs"]}
        for i in range(a, b):  # the deleted secondary arc leaves the set at step b
            labelled = {k: lab for k, cls, lab in steps[i]["arcs"] if cls == "sec"}
            assert labelled == sec_keys, (i, labelled, sec_keys)
            assert {k for k, cls, _ in steps[i]["arcs"] if cls == "match"} == match_keys, i
        # the reassignment inset exists exactly from the reassignment_graph event to the end of the call
        rg = [i for i in range(a, b + 1) if trace[i]["type"] == "reassignment_graph"]
        for i in range(a, b + 1):
            assert (steps[i]["reass"] is not None) == (bool(rg) and i >= rg[0]), i
        # the criticality table is listed from the criticality event onwards
        c = next(i for i in range(a, b + 1) if trace[i]["type"] == "criticality")
        for i in range(a, b + 1):
            assert (steps[i]["panel"]["crit"] != "") == (i >= c), i
        ncrit = len(trace[c]["crit"])
        assert steps[c]["panel"]["crit"].count("(") == ncrit
        assert steps[c]["desc"].startswith(f"{ncrit} critical triple")


# ----------------------------------------------------------------------------- determinism
def test_html_png_and_partition_svg_are_deterministic(paper, tmp_path: Path):
    inst, res = paper
    r1, r2 = TraceRenderer(inst, res.trace), TraceRenderer(inst, res.trace)
    assert r1.layout == r2.layout
    h1 = render_html(inst, res.trace, tmp_path / "a.html").read_bytes()
    h2 = render_html(inst, res.trace, tmp_path / "b.html").read_bytes()
    assert h1 == h2 and len(h1) > 20_000
    p1 = r1.save_frame(10, tmp_path / "a.png").read_bytes()
    p2 = r2.save_frame(10, tmp_path / "b.png").read_bytes()
    assert p1 == p2 and p1[:8] == b"\x89PNG\r\n\x1a\n"
    assert partition_svg(inst, res.parts, parents=res.certificate["parents"]) == partition_svg(
        inst, res.parts, parents=res.certificate["parents"]
    )
    # two solver runs give the same trace, hence the same viewer data
    res2 = glpartition(inst, algorithm="reference", trace=True)
    assert res2.trace == res.trace


_SVG_NOISE = (
    re.compile(r"<dc:date>[^<]*</dc:date>"),  # Matplotlib's timestamp
    re.compile(r"\b[pmh][0-9a-f]{10}\b"),  # hashed clip-path / marker ids (random salt)
)


def _svg_content(raw: bytes) -> str:
    s = raw.decode("utf-8")
    for pat in _SVG_NOISE:
        s = pat.sub("", s)
    return s


def test_svg_frames_have_deterministic_content(paper, tmp_path: Path):
    inst, res = paper
    for i in (5, 10):
        a = TraceRenderer(inst, res.trace).save_frame(i, tmp_path / f"a{i}.svg").read_bytes()
        b = TraceRenderer(inst, res.trace).save_frame(i, tmp_path / f"b{i}.svg").read_bytes()
        assert _svg_content(a) == _svg_content(b) and len(a) > 50_000


def test_svg_pdf_and_counterexample_bytes_are_identical(paper, tmp_path: Path):
    """Fresh renderers (and fresh figures) give byte-identical SVG/PDF: no ``<dc:date>``,
    no ``CreationDate``, ids salted with a fixed ``svg.hashsalt`` (was a review finding)."""
    inst, res = paper
    a = TraceRenderer(inst, res.trace).save_frame(9, tmp_path / "a.svg").read_bytes()
    b = TraceRenderer(inst, res.trace).save_frame(9, tmp_path / "b.svg").read_bytes()
    assert a == b and b"<dc:date>" not in a and len(a) > 50_000
    pa = TraceRenderer(inst, res.trace).save_frame(9, tmp_path / "a.pdf").read_bytes()
    pb = TraceRenderer(inst, res.trace).save_frame(9, tmp_path / "b.pdf").read_bytes()
    assert pa == pb and pa.startswith(b"%PDF") and b"CreationDate" not in pa
    ca = visualize_counterexample(1, tmp_path / "ca.svg")["path"].read_bytes()
    cb = visualize_counterexample(1, tmp_path / "cb.svg")["path"].read_bytes()
    assert ca == cb and b"<dc:date>" not in ca
    # the fixed salt does not leak into other Matplotlib output (rc_context restores the default)
    import matplotlib

    assert matplotlib.rcParams["svg.hashsalt"] is None


# ----------------------------------------------------------------------------- images
def test_png_frames_and_gif_are_valid_images(dag, tmp_path: Path):
    from PIL import Image

    inst, res = dag
    r = TraceRenderer(inst, res.trace)
    pngs = r.render_all(tmp_path / "png", "png")
    assert len(pngs) == len(res.trace) == 9
    for p in pngs:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            assert im.format == "PNG" and im.size == (1200, 700)
    gif = animate_solution(inst, res.trace, tmp_path / "dag.gif", fps=2, renderer=r)
    with Image.open(gif) as im:
        im.verify()
    with Image.open(gif) as im:
        assert im.format == "GIF" and im.n_frames == len(res.trace) and im.size == (1200, 700)
        im.seek(im.n_frames - 1)
        assert im.info["duration"] == 2 * 500  # last frame held twice as long at fps=2
    # frames differ (the animation is not a single repeated image)
    assert len({p.read_bytes() for p in pngs}) == len(pngs)


# ----------------------------------------------------------------------------- html hygiene
def test_html_escapes_custom_labels_and_rejects_ignored_renderer_kwargs(paper, tmp_path: Path):
    """Labels are text: the cut selector builds its options with DOM APIs (no innerHTML
    with a label); the JSON payload cannot close its script tag. ``render_html`` /
    ``animate_solution`` refuse renderer kwargs next to an explicit renderer."""
    inst, res = paper
    trace = [dict(ev) for ev in res.trace]
    trace[1]["cuts"] = {"3": {"L": [2, 5, 6, 8], "S": [0, 1], "R": [3, 4, 7]}}
    lab = {3: "<b>x</b><script>alert(1)</script>"}
    r = TraceRenderer(inst, trace, labels=lab)
    text = render_html(inst, trace, tmp_path / "lab.html", renderer=r).read_text(encoding="utf-8")
    # the label sits inside the JSON payload (a JS string): its closing tag cannot end the script block
    assert text.count("</script>") == 2 and "<\\/script>" in text and "alert(1)</script>" not in text
    js = text[text.rindex("<script>"):]
    assert not re.search(r"innerHTML\s*=[^;]*D\.labels", js)  # no label reaches innerHTML
    assert "createElement('option')" in js and "o.textContent = 'cut of ' + D.labels[k]" in js
    data = viewer_data(tmp_path / "lab.html")
    assert data["labels"]["3"] == lab[3]  # stored verbatim as JSON text, rendered via textContent
    with pytest.raises(TypeError, match="cannot be combined"):
        render_html(inst, trace, tmp_path / "kw.html", renderer=r, layout_method="spring")
    with pytest.raises(TypeError, match="cannot be combined"):
        animate_solution(inst, trace, tmp_path / "kw.gif", renderer=r, layout_method="spring")


def test_html_viewer_has_no_external_resources(paper_html, tmp_path: Path):
    for name in sorted(p.stem for p in CURATED.glob("*.json")) + ["paper"]:
        if name == "paper":
            inst = paper_running_example()
            algo = "reference"
        else:
            inst = load_instance(CURATED / f"{name}.json")
            algo = inst.meta["viz_algorithm"]
        res = glpartition(inst, algorithm=algo, trace=True)
        text = render_html(inst, res.trace, tmp_path / f"{name}.html").read_text(encoding="utf-8")
        urls = set(re.findall(r"https?://[^\"'\s<>)]+", text))
        assert urls <= {"http://www.w3.org/2000/svg"}, (name, urls)  # the SVG namespace only
        assert "<script src" not in text and "<link" not in text and "@import" not in text
        assert "cdn" not in text.lower() and "url(http" not in text and "fetch(" not in text
        assert text.count("<script>") == 2  # the data blob and the inline viewer
        assert "</script>" not in DATA_RE.search(text).group(1)  # payload cannot close the script tag
        data = viewer_data(tmp_path / f"{name}.html")
        assert len(data["steps"]) == len(res.trace)


# ----------------------------------------------------------------------------- capacities across examples
@pytest.mark.parametrize("path", sorted(CURATED.glob("*.json")), ids=lambda p: p.stem)
def test_panel_capacities_and_parts_track_every_curated_trace(path: Path, tmp_path: Path):
    inst = load_instance(path)
    res = glpartition(inst, algorithm=inst.meta["viz_algorithm"], trace=True)
    assert res.status == "ok" and res.valid
    data = viewer_data(render_html(inst, res.trace, tmp_path / "v.html"))
    steps = data["steps"]
    for i, ev in enumerate(res.trace):
        rows = {row[1]: row for row in steps[i]["panel"]["terminals"]}
        if isinstance(ev.get("capacities"), dict) and ev["type"] != "round_and_remove":
            for t, c in ev["capacities"].items():
                assert rows[int(t)][2] == c, (path.stem, i, ev["type"], t)
        if ev["type"] == "dag_contract":
            assert rows[ev["t"]][2] == ev["residual"]
        if ev["type"] == "remove_terminal":
            assert rows[ev["t"]][2] == "done" and rows[ev["t"]][4] is False
            assert node_of(steps[i], ev["t"])["removed"] is True
        if ev["type"] == "round_and_remove":
            for t, p in ev["pairs"]:
                assert node_of(steps[i], p)["kind"] == "c" and node_of(steps[i], p)["ti"] == data["tidx"][str(t)]
            for t in ev["S"]:
                assert rows[t][2] == "done"
    done = steps[-1]
    assert done["type"] == "done" and done["panel"]["left"] == 0 and done["panel"]["arcs"] == 0
    sizes = {row[1]: row[3] for row in done["panel"]["terminals"]}
    assert sizes == {t: len(part) for t, part in zip(inst.terminals, res.parts)}
    assert not arc_classes(done) & {"arc", "match", "sec"}


# ----------------------------------------------------------------------------- counterexample
def test_counterexample_figure_has_9_terminals_and_108_pre_terminal_markers(tmp_path: Path):
    from matplotlib.patches import Circle, FancyBboxPatch, Wedge

    out = visualize_counterexample(1, tmp_path / "cex.svg")
    assert out["counts"]["k"] == 9 and out["counts"]["pre_terminals"] == 108 and out["counts"]["copies"] == 1
    assert len(out["structure"]["pre_terminals"]) == 108
    assert out["path"].exists() and ET.parse(out["path"]).getroot().tag.endswith("svg")
    fig = counterexample_figure(1, structure=out["structure"])
    left = fig.axes[0]
    terminals = [p for p in left.patches if isinstance(p, FancyBboxPatch)]
    assert len(terminals) == 9
    pre_circles = [p for p in left.patches if isinstance(p, Circle) and abs(p.get_radius() - 0.02) < 1e-9]
    assert len(pre_circles) == 2 * 108  # white disc + outline per pre-terminal marker
    centres = {(round(c.center[0], 6), round(c.center[1], 6)) for c in pre_circles}
    assert len(centres) == 108  # 108 distinct marker positions
    pre_wedges = [p for p in left.patches if isinstance(p, Wedge) and abs(p.r - 0.02) < 1e-9]
    assert len(pre_wedges) == 2 * 108  # each pre-terminal split into its two terminals' colours
    # markers sit on the chords between terminal pairs (36 pairs × 3 = 108)
    assert out["counts"]["pairs"] * 3 == 108
    # the forcing gadget inset shows the nine out-neighbours of v_e and six terminals
    inset = fig.axes[1]
    assert len([p for p in inset.patches if isinstance(p, FancyBboxPatch)]) == 6
    assert len([p for p in inset.patches if isinstance(p, Circle) and abs(p.get_radius() - 0.2) < 1e-9]) == 2 * 9


# ----------------------------------------------------------------------------- CLI, every format
def test_cli_visualize_every_format(tmp_path: Path):
    from PIL import Image

    src = CURATED / "small_dag.json"
    n_events = 9
    for fmt in ("svg", "png", "pdf", "html", "gif", "mp4"):
        out = tmp_path / (f"{fmt}_dir" if fmt in ("svg", "png", "pdf") else f"out.{fmt}")
        r = subprocess.run(
            [sys.executable, "-m", "glsolver.cli", "visualize", str(src), "--algorithm", "reference-dag",
             "--format", fmt, "-o", str(out)],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert r.returncode == 0, (fmt, r.stdout, r.stderr)
        if fmt in ("svg", "png", "pdf"):
            files = sorted(out.glob(f"step_*.{fmt}"))
            assert len(files) == n_events and f"wrote {n_events} frames" in r.stdout
            if fmt == "png":
                with Image.open(files[0]) as im:
                    im.verify()
        elif fmt == "html":
            assert len(viewer_data(out)["steps"]) == n_events
        elif fmt == "gif":
            with Image.open(out) as im:
                assert im.n_frames == n_events
        else:  # mp4 falls back to a GIF without imageio-ffmpeg, and says so
            try:
                import imageio_ffmpeg  # noqa: F401

                assert out.exists() and out.stat().st_size > 0
            except ImportError:
                assert out.with_suffix(".gif").exists() and "ffmpeg" in (r.stderr + r.stdout)


def test_cli_visualize_defaults_to_the_instance_viz_algorithm(tmp_path: Path):
    """Without ``--algorithm`` the curated DAG example runs its ``meta['viz_algorithm']``
    (``reference-dag``, 9 events) like ``examples/render_all.py``; an explicit
    ``--algorithm reference`` still overrides it (the general algorithm's longer trace)."""
    src = CURATED / "small_dag.json"
    assert load_instance(src).meta["viz_algorithm"] == "reference-dag"
    r = subprocess.run([sys.executable, "-m", "glsolver.cli", "visualize", str(src), "--format", "html",
                        "-o", str(tmp_path / "meta.html")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "meta['viz_algorithm'] = 'reference-dag'" in r.stdout and "wrote interactive viewer with 9 steps" in r.stdout
    assert len(viewer_data(tmp_path / "meta.html")["steps"]) == 9
    r2 = subprocess.run([sys.executable, "-m", "glsolver.cli", "visualize", str(src), "--algorithm", "reference",
                         "--format", "html", "-o", str(tmp_path / "ref.html")], capture_output=True, text=True, cwd=tmp_path)
    assert r2.returncode == 0, r2.stderr
    n_ref = len(viewer_data(tmp_path / "ref.html")["steps"])
    assert n_ref > 9 and "meta['viz_algorithm']" not in r2.stdout
    assert [s["type"] for s in viewer_data(tmp_path / "meta.html")["steps"]].count("dag_contract") == 7


def test_examples_output_is_git_ignored():
    """examples/README.md calls ``examples/output/`` disposable: git must ignore it."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(["git", "check-ignore", "-q", "examples/output/paper/step_000.svg"], cwd=ROOT)
    assert r.returncode == 0
    assert "output/" in (ROOT / "examples" / ".gitignore").read_text().splitlines()
    r2 = subprocess.run(["git", "check-ignore", "-q", "examples/curated/small_dag.json"], cwd=ROOT)
    assert r2.returncode == 1  # the curated inputs are not ignored


# ----------------------------------------------------------------------------- import surface
ALLOWED_PREFIXES = (
    "glsolver.api", "glsolver.instance", "glsolver.viz", "glsolver.io",
    "glref.counterexample",  # the official counterexample builder (structure only)
    "glref.trace",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return {m for m in mods if m.startswith(("glsolver", "glref"))}


def test_viz_package_imports_only_public_api_and_trace_builders():
    found = {p.name: _imports(p) for p in VIZ_DIR.glob("*.py")}
    assert found  # the package exists
    for name in ("layout.py", "html.py", "animate.py", "counterexample.py", "__init__.py"):
        bad = {m for m in found[name] if not m.startswith(ALLOWED_PREFIXES)}
        assert not bad, (name, bad)


def test_render_replay_does_not_import_solver_internals():
    """The replay is trace-driven: no ``glref.essential`` / ``glref.graph`` (was a review finding)."""
    bad = {m for m in _imports(VIZ_DIR / "render.py") if not m.startswith(ALLOWED_PREFIXES)}
    assert not bad, bad
    src = (VIZ_DIR / "render.py").read_text(encoding="utf-8")
    assert "glref.essential" not in src and "DiGraphState" not in src and "recompute_essential" not in src
