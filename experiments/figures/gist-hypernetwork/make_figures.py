"""Generate the 4 summary figures for the gist-hypernetwork experiment line.

Outputs SVG (source of truth) + PNG (for chat delivery) into this directory.
Palette: dataviz reference instance, light mode, slots blue/orange/aqua
(validated: all checks pass; aqua carries direct labels as contrast relief).
Color semantics held constant across figures:
  blue  #2a78d6 = sentence content / hypernetwork (iteration 1)
  orange #eb6834 = gist tokens / fixed-gist channel
  aqua  #1baf7a = engram memory (iteration 2)
  inks           = reference / baseline
"""
import os

DIR = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_L = "#cde2fb"   # sequential blue 100 (light wash)
FONT = "DejaVu Sans, system-ui, -apple-system, 'Segoe UI', sans-serif"


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>')


def text(x, y, s, size=13, fill=INK, weight="normal", anchor="start", style=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" {style}>{s}</text>')


def rect(x, y, w, h, fill, rx=0, opacity=1.0, stroke="none", sw=0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" rx="{rx}" '
            f'opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, stroke=GRID, sw=1, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" {d}/>'


def arrow(x1, y1, x2, y2, stroke=INK2, sw=1.6):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" marker-end="url(#arr)"/>')


ARROW_DEF = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{INK2}"/></marker></defs>')


def chip(x, y, color, label, out, label_fill=INK2):
    out.append(rect(x, y - 9, 12, 12, color, rx=3))
    out.append(text(x + 18, y + 1, label, 12, label_fill))


# ===========================================================================
# FIG 1 — problem setup: strict block-causal + gist mask
# ===========================================================================
def fig1():
    W, H = 1240, 660
    o = [svg_open(W, H)]
    o.append(text(28, 40, "Sentence attention — the strict block-causal + gist regime", 21, INK, "bold"))
    o.append(text(28, 64, "d12 / 10k steps / K gist tokens inserted at NLTK sentence boundaries; the attention mask is derived from token ids in the forward pass.", 13, INK2))

    # --- token sequence: 14 toy tokens (K=2 shown; experiments use K=8) ---
    toks = ["B", "t", "t", "t", "g", "g", "t", "t", "t", "g", "g", "t", "t", "t"]
    kinds = ["bos", "s", "s", "s", "g", "g", "s", "s", "s", "g", "g", "s", "s", "s"]
    boundary_before = {}  # most recent gist-run END strictly before q (index), else -1
    for q in range(14):
        b = -1
        for p in range(q):
            if kinds[p] == "g" and (p + 1 >= 14 or kinds[p + 1] != "g"):
                b = p
        boundary_before[q] = b

    cs = 27  # cell size
    gx, gy = 96, 150
    # sequence strip above the matrix
    for k in range(14):
        col = {"bos": BASELINE, "s": BLUE, "g": ORANGE}[kinds[k]]
        o.append(rect(gx + k * cs + 2, gy - 46, cs - 4, 24, col, rx=4, opacity=0.9 if kinds[k] != "bos" else 1.0))
        o.append(text(gx + k * cs + cs / 2, gy - 29, toks[k], 12, "#ffffff", "bold", "middle"))
    o.append(text(gx - 8, gy - 29, "keys →", 11, MUTED, anchor="end"))

    # mask matrix
    for q in range(14):
        for k in range(14):
            if k > q:
                continue  # future: leave surface
            allowed_gist = kinds[k] in ("g",) or kinds[k] == "bos"
            in_block = k >= boundary_before[q] and boundary_before[q] >= 0 or (boundary_before[q] < 0)
            in_block = k >= max(boundary_before[q], 0)
            if allowed_gist:
                col, op = ORANGE, (1.0 if kinds[k] == "g" else 0.45)
            elif in_block or k == q:
                col, op = BLUE, 0.9
            else:
                col, op = GRID, 1.0  # visible in full causal, masked here
            o.append(rect(gx + k * cs + 2, gy + q * cs + 2, cs - 4, cs - 4, col, rx=3, opacity=op))
    for q in range(14):
        col = {"bos": BASELINE, "s": BLUE, "g": ORANGE}[kinds[q]]
        o.append(rect(gx - 30, gy + q * cs + 2, 22, cs - 4, col, rx=4, opacity=0.9 if kinds[q] != "bos" else 1.0))
        o.append(text(gx - 19, gy + q * cs + cs / 2 + 4, toks[q], 11, "#ffffff", "bold", "middle"))
    o.append(text(gx - 42, gy + 7 * cs, "queries ↓", 11, MUTED, anchor="end",
                  style=f'transform="rotate(-90 {gx - 42} {gy + 7 * cs})"'))

    my = gy + 14 * cs + 30
    chip(gx, my, BLUE, "own sentence block (visible)", o)
    chip(gx + 240, my, ORANGE, "earlier gist tokens (always visible)", o)
    chip(gx, my + 24, GRID, "masked — visible in full causal, lost here", o)

    # --- right panel ---
    rx0 = 620
    o.append(text(rx0, 150, "Mechanism", 15, INK, "bold"))
    for i, s in enumerate([
        "A token attends only to (a) the tokens of its own current",
        "sentence block and (b) the K gist tokens emitted at every",
        "earlier sentence boundary in the same document.",
        "Gists are the ONLY cross-sentence channel — all long-range",
        "information must be written into K slots per boundary by the trunk.",
    ]):
        o.append(text(rx0, 176 + i * 20, s, 13, INK2))

    o.append(text(rx0, 306, "Where the line stood before this experiment (val BPB, lower = better)", 13.5, INK, "bold"))
    rows = [
        ("full-causal baseline  d12_sa_baseline", "0.8022", INK),
        ("strict + gists, best of K sweep (K=8)  d12_sa_nltk_k8", "0.8191", ORANGE),
        ("the gap the program wants to close", "+0.0169", INK),
    ]
    for i, (lbl, v, c) in enumerate(rows):
        y = 336 + i * 26
        o.append(rect(rx0, y - 14, 8, 8, c, rx=2))
        o.append(text(rx0 + 16, y - 6, lbl, 12.5, INK2))
        o.append(text(rx0 + 560, y - 6, v, 12.5, INK, "bold", "end"))

    o.append(text(rx0, 440, "Question posed to the hypernetwork line", 15, INK, "bold"))
    for i, s in enumerate([
        "The K gist embeddings are fixed learned rows — identical for every",
        "sentence. Does making them CONTENT-CONDITIONED (a hypernetwork",
        "reading the completed sentence) recover part of the 0.017 gap?",
        "Win gate: val BPB ≤ 0.8161 (−0.003 vs the fixed-gist control;",
        "≈3× any static-init effect ever measured, ≫ seed noise ~1e-4).",
    ]):
        o.append(text(rx0, 466 + i * 20, s, 13, INK2))

    o.append(text(28, H - 22, "Matrix shown for K=2 gists/boundary for legibility; all experiments use K=8. BOS column shown at reduced opacity (always visible, per-document segmenting).", 11, MUTED))
    o.append("</svg>")
    return "".join(o)


# ===========================================================================
# FIG 2 — iteration 1: the gist hypernetwork
# ===========================================================================
def fig2():
    W, H = 1240, 640
    o = [svg_open(W, H), ARROW_DEF]
    o.append(text(28, 40, "Iteration 1 — Gist Hypernetwork: content-conditioned gist embeddings", 21, INK, "bold"))
    o.append(text(28, 64, "One masked cross-attention pass replaces the content-blind fixed gist rows. Everything else — trunk, mask, dataloader, schedule — identical to the control.", 13, INK2))

    # sequence strip
    sy = 120
    o.append(text(28, sy - 10, "completed sentence", 12, BLUE, "bold"))
    o.append(text(475, sy - 10, "its K=8 gist slots", 12, ORANGE, "bold"))
    for i in range(7):
        o.append(rect(28 + i * 60, sy, 54, 30, BLUE, rx=5, opacity=0.9))
        o.append(text(55 + i * 60, sy + 20, f"t{i+1}", 12, "#ffffff", "bold", "middle"))
    for i in range(4):
        o.append(rect(460 + i * 44, sy, 38, 30, ORANGE, rx=5))
        o.append(text(479 + i * 44, sy + 20, f"g{i+1}", 12, "#ffffff", "bold", "middle"))
    o.append(text(645, sy + 20, "…g8", 13, ORANGE, "bold"))

    # flow boxes
    def box(x, y, w, h, title, lines, accent=INK2):
        o.append(rect(x, y, w, h, "#ffffff", rx=8, stroke=GRID, sw=1.5))
        o.append(rect(x, y, 4, h, accent, rx=2))
        o.append(text(x + 14, y + 24, title, 13.5, INK, "bold"))
        for i, s in enumerate(lines):
            o.append(text(x + 14, y + 44 + i * 17, s, 12, INK2))

    box(28, 190, 300, 96, "keys / values", [
        "shared wte embeddings of the",
        "sentence tokens → RMSNorm",
        "(gradients flow back into wte)"], BLUE)
    box(28, 306, 300, 96, "queries", [
        "K=8 learned per-slot vectors,",
        "directly in projected-Q space",
        "(slot k = gist id − gist_start)"], ORANGE)
    box(392, 234, 330, 130, "masked cross-attention", [
        "own-sentence mask (cummax boundary",
        "machinery; no gathers — compile-safe),",
        "rotary + QK-norm as in the trunk,",
        "c_proj output → h₁ … h₈  (content)"], BLUE)
    o.append(arrow(328, 238, 392, 270))
    o.append(arrow(328, 354, 392, 330))
    o.append(arrow(722, 300, 790, 300))

    # arms
    box(790, 190, 420, 100, "GATED arm   d12_sa_nltk_k8_hnet_gated", [
        "gist embedding = wte[gₖ] + αₖ · hₖ,  per-slot αₖ = 0 at init",
        "→ bit-exact equal to the fixed-gist control at step 0",
        "→ final |α| is a free mechanistic readout (did it engage?)"], BLUE)
    box(790, 310, 420, 84, "FORCED arm   d12_sa_nltk_k8_hnet_forced", [
        "gist embedding = hₖ outright — no fixed row, no gate",
        "→ the model MUST live off content (falsification arm)"], BLUE)

    # facts strip
    fy = 440
    o.append(line(28, fy, W - 28, fy, GRID, 1))
    o.append(text(28, fy + 30, "Why two arms (ADR 0002)", 14, INK, "bold"))
    for i, s in enumerate([
        "gated ≈ control with α≈0  →  channel unused (honest null)",
        "forced < control while gated ≈ control  →  content harmful/useless — direction falsified",
        "either arm ≤ 0.8161  →  content-conditioning wins",
    ]):
        o.append(text(28, fy + 54 + i * 20, s, 12.5, INK2))
    o.append(text(660, fy + 30, "Budget & bookkeeping", 14, INK, "bold"))
    for i, s in enumerate([
        "+1.8M params (~1%) — projections in Muon, queries/gates in AdamW",
        "hypernet excluded from the scaling-params horizon → schedule identical to control",
        "c_proj deliberately NOT zero-init: zero gate × zero proj = dead path (known bug class)",
    ]):
        o.append(text(660, fy + 54 + i * 20, s, 12.5, INK2))
    o.append("</svg>")
    return "".join(o)


# ===========================================================================
# FIG 3 — iteration 2: engram sparse bigram memory
# ===========================================================================
def fig3():
    W, H = 1240, 680
    o = [svg_open(W, H), ARROW_DEF]
    o.append(text(28, 40, "Iteration 2 — Engram-style sparse bigram memory in the hypernetwork", 21, INK, "bold"))
    o.append(text(28, 64, "Iteration 1 verdict: context-derived content is redundant. This injects the one thing the trunk cannot compute from context — stored n-gram associations.", 13, INK2))

    def box(x, y, w, h, title, lines, accent=INK2):
        o.append(rect(x, y, w, h, "#ffffff", rx=8, stroke=GRID, sw=1.5))
        o.append(rect(x, y, 4, h, accent, rx=2))
        o.append(text(x + 14, y + 24, title, 13.5, INK, "bold"))
        for i, s in enumerate(lines):
            o.append(text(x + 14, y + 44 + i * 17, s, 12, INK2))

    box(28, 110, 250, 92, "bigram stream", [
        "every sentence position j:",
        "(t₍ⱼ₋₁₎ , tⱼ)  — surface form,",
        "no learned computation"], INK2)
    box(322, 110, 300, 92, "hash (engram-lite recipe)", [
        "(36313·cur ⊕ 27191·prev) mod 2ᵇ",
        "kept eager via compiler.disable",
        "(known Inductor int32 gather trap)"], AQUA)
    o.append(arrow(278, 156, 322, 156))

    # table graphic
    tx, ty = 668, 96
    o.append(text(tx, ty - 4, "memory table  2ᵇ × 128, ZERO-init", 13, INK, "bold"))
    for r in range(12):
        touched = r in (2, 5, 9)
        o.append(rect(tx, ty + 6 + r * 10, 150, 8, AQUA if touched else GRID, rx=2, opacity=0.95 if touched else 0.8))
    o.append(text(tx, ty + 148, "O(1) sparse lookups — few rows touched", 11.5, INK2))
    o.append(text(tx, ty + 164, "per example; params ≫ compute", 11.5, INK2))
    o.append(text(tx, ty + 180, "(the Engram thesis)", 11.5, INK2))
    o.append(arrow(622, 156, 668, 156))

    box(940, 110, 272, 92, "into the SAME cross-attn", [
        "proj 128→768 (nonzero init)",
        "added to the RMSNormed KV input;",
        "slot queries select memories"], BLUE)
    o.append(arrow(826, 156, 940, 156))

    # guarantees strip
    gy = 320
    o.append(text(28, gy, "Safety of the composition", 14, INK, "bold"))
    for i, s in enumerate([
        "zero-init table  →  each arm starts bit-exact equal to the plain gated arm (test-enforced)",
        "not a dead path: α gates open first (measured |α|≈1.15 in iteration 1), then gradients reach the table through the nonzero projection",
        "own-sentence mask keeps retrieval sentence-local; table sits in the AdamW embedding group, excluded from the scaling-params horizon",
    ]):
        o.append(text(28, gy + 26 + i * 20, s, 12.5, INK2))

    # arms
    ay = 440
    o.append(text(28, ay, "Capacity sweep — 4 arms on top of GATED (win gate unchanged: val BPB ≤ 0.8161)", 14, INK, "bold"))
    arms = [("eng_b17", "2¹⁷ × 128", "≈ 17M table params"),
            ("eng_b18", "2¹⁸ × 128", "≈ 34M"),
            ("eng_b19", "2¹⁹ × 128", "≈ 67M"),
            ("eng_b20", "2²⁰ × 128", "≈ 134M")]
    for i, (tag, size, params) in enumerate(arms):
        x = 28 + i * 300
        o.append(rect(x, ay + 16, 280, 74, "#ffffff", rx=8, stroke=GRID, sw=1.5))
        o.append(rect(x, ay + 16, 4, 74, AQUA, rx=2))
        o.append(text(x + 14, ay + 40, tag, 13, INK, "bold"))
        o.append(text(x + 14, ay + 60, size, 12.5, INK2))
        o.append(text(x + 14, ay + 78, params, 12.5, INK2))
    o.append(text(28, ay + 130, "Prior support: token-level engram-lite was this repo's best low-dim result (0.7937 vs 0.8027 baseline, monotone in table size) — dev/LOG 2026-01-27.", 12, MUTED))
    o.append("</svg>")
    return "".join(o)


# ===========================================================================
# FIG 4 — results + mechanism
# ===========================================================================
def fig4():
    W, H = 1960, 730
    o = [svg_open(W, H)]
    o.append(text(28, 40, "Results — the gist-input channel is double-falsified", 21, INK, "bold"))
    o.append(text(28, 64, "Iteration 1: content is used but redundant. Iteration 2: memory is used but harmful. The win gate was never approached.", 13, INK2))

    # (label, val bpb, core, core std over 5 eval seeds or None, color)
    rows = [
        ("full-causal baseline", 0.80221, 0.18304, None, MUTED),
        ("fixed-gist control (K=8)", 0.81906, 0.14352, None, ORANGE),
        ("hypernet gated  (it.1)", 0.81826, 0.14644, 0.0018, BLUE),
        ("hypernet forced  (it.1)", 0.81863, 0.14880, 0.0020, BLUE),
        ("+ engram 2¹⁷  (it.2)", 0.81919, 0.13770, 0.0012, AQUA),
        ("+ engram 2¹⁸  (it.2)", 0.81948, 0.11632, 0.0014, AQUA),
        ("+ engram 2¹⁹  (it.2)", 0.81968, 0.12631, 0.0020, AQUA),
        ("+ engram 2²⁰  (it.2)", 0.81879, 0.14352, 0.0016, AQUA),
    ]
    py0, rh = 130, 40
    pyb = py0 + len(rows) * rh  # plot bottom

    # ---- panel A: val BPB dot plot (primary metric) ----
    ax0, ax1 = 300, 680
    v0, v1 = 0.800, 0.8210
    def X(v):
        return ax0 + (v - v0) / (v1 - v0) * (ax1 - ax0)
    o.append(text(ax0, 104, "val BPB (primary, lower is better)", 14, INK, "bold"))
    for gv in [0.800, 0.805, 0.810, 0.815, 0.820]:
        o.append(line(X(gv), py0 - 8, X(gv), pyb + 4, GRID, 1))
        o.append(text(X(gv), pyb + 22, f"{gv:.3f}", 11, MUTED, anchor="middle"))
    o.append(line(X(0.8161), py0 - 8, X(0.8161), pyb + 4, INK2, 1.4, dash="5,4"))
    o.append(text(X(0.8161), py0 - 16, "win gate 0.8161", 11.5, INK2, "bold", "middle"))
    for i, (lbl, v, _, _, c) in enumerate(rows):
        y = py0 + i * rh + rh // 2
        o.append(text(ax0 - 12, y + 4, lbl, 12.5, INK, anchor="end"))
        o.append(line(ax0, y, X(v), y, GRID, 1))
        o.append(f'<circle cx="{X(v)}" cy="{y}" r="7" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        o.append(text(X(v) + 13, y + 4, f"{v:.5f}", 12, INK, "bold"))

    # ---- panel B: CORE dot plot (reference-only), same rows ----
    cx0, cx1 = 800, 1080
    c0, c1 = 0.10, 0.19
    def XC(v):
        return cx0 + (v - c0) / (c1 - c0) * (cx1 - cx0)
    o.append(text(cx0, 104, "CORE (higher is better; reference-only)", 14, INK, "bold"))
    for gv in [0.10, 0.12, 0.14, 0.16, 0.18]:
        o.append(line(XC(gv), py0 - 8, XC(gv), pyb + 4, GRID, 1))
        o.append(text(XC(gv), pyb + 22, f"{gv:.2f}", 11, MUTED, anchor="middle"))
    for i, (lbl, _, cv, cs2, c) in enumerate(rows):
        y = py0 + i * rh + rh // 2
        o.append(line(cx0, y, XC(cv), y, GRID, 1))
        if cs2 is not None:  # ± s.d. whisker over the 5 eval seeds
            o.append(line(XC(cv - cs2), y, XC(cv + cs2), y, INK2, 2))
        o.append(f'<circle cx="{XC(cv)}" cy="{y}" r="7" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        if cv > 0.17:
            o.append(text(XC(cv) - 13, y + 4, f"{cv:.4f}", 12, INK, "bold", "end"))
        else:
            o.append(text(XC(cv) + 13, y + 4, f"{cv:.4f}", 12, INK, "bold"))
    o.append(text(cx0, pyb + 76, "whiskers: ± s.d. over the 5 eval seeds (where evaluated with the", 10.5, MUTED))
    o.append(text(cx0, pyb + 90, "seeded protocol). Training-seed noise on CORE is ~±0.01.", 10.5, MUTED))

    ly = pyb + 50
    chip(ax0, ly, MUTED, "reference", o)
    chip(ax0 + 105, ly, ORANGE, "fixed gists", o)
    chip(ax0 + 215, ly, BLUE, "it.1 content hypernet", o)
    chip(ax0 + 390, ly, AQUA, "it.2 + engram memory", o)

    # ---- panel C: alpha gates ----
    bx0, bw, bgap = 1215, 40, 12
    bY0, bY1 = 130, 400
    amax = 1.3
    def Y(a):
        return bY1 - a / amax * (bY1 - bY0)
    o.append(text(bx0 - 45, 104, "gate readout: mean |α| per arm", 14, INK, "bold"))
    for gv in [0.0, 0.5, 1.0]:
        o.append(line(bx0 - 40, Y(gv), bx0 + 5 * (bw + bgap) - bgap + 8, Y(gv), GRID, 1))
        o.append(text(bx0 - 46, Y(gv) + 4, f"{gv:.1f}", 11, MUTED, anchor="end"))
    bars = [("gated", 1.146, BLUE), ("b17", 0.424, AQUA), ("b18", 0.509, AQUA),
            ("b19", 0.526, AQUA), ("b20", 0.581, AQUA)]
    for i, (lbl, a, c) in enumerate(bars):
        x = bx0 + i * (bw + bgap)
        o.append(rect(x, Y(a), bw, bY1 - Y(a), c, rx=4))
        o.append(rect(x, bY1 - 2, bw, 2, SURFACE))
        o.append(text(x + bw / 2, Y(a) - 8, f"{a:.2f}", 11.5, INK, "bold", "middle"))
        o.append(text(x + bw / 2, bY1 + 18, lbl, 11.5, INK2, anchor="middle"))
    o.append(line(bx0 - 40, bY1, bx0 + 5 * (bw + bgap) - bgap + 8, bY1, BASELINE, 1.5))
    for i, s in enumerate([
        "Adding memory to the shared KV stream",
        "made the model TURN DOWN the whole",
        "channel (α: 1.15 → 0.42–0.58), despite",
        "every table being written at full breadth",
        "(100% of rows, norms ≈ wte scale).",
        "Used, but harmful: collision noise",
        "displaced the content signal.",
    ]):
        o.append(text(bx0 - 45, 448 + i * 18, s, 12, INK2))

    # ---- panel D: attention tokens saved (measured on val rows, K=8) ----
    # Numbers from experiments/figures/gist-hypernetwork/measure_token_savings.py
    # (200 val rows x 2048 tokens, the exact GPT._build_sentence_mask mask).
    dx0, dx1 = 1580, 1900
    vmax = 1100.0
    def XT(v):
        return dx0 + v / vmax * (dx1 - dx0)
    o.append(text(dx0 - 12, 104, "attention tokens per query (K=8, val rows)", 14, INK, "bold"))
    for gv in [0, 250, 500, 750, 1000]:
        o.append(line(XT(gv), 130, XT(gv), 420, GRID, 1))
        o.append(text(XT(gv), 436, f"{gv}", 11, MUTED, anchor="middle"))
    bars = [
        ("full causal (baseline)", 1024.5, BASELINE),
        ("full causal within doc", 593.1, GRID),
        ("sentence attention (K=8)", 169.4, ORANGE),
    ]
    for i, (lbl, v, c) in enumerate(bars):
        y = 210 + i * 70
        o.append(text(dx0 - 12, y + 4, lbl, 12, INK2, anchor="end"))
        o.append(rect(dx0, y - 9, XT(v) - dx0, 18, c, rx=4))
        o.append(text(XT(v) + 10, y + 4, f"{v:,.0f}", 12, INK, "bold"))
        if lbl.startswith("sentence"):
            o.append(text(XT(v) + 10, y - 24, "−83.5%", 15, ORANGE, "bold"))
    for i, s in enumerate([
        "83.5% fewer attended keys per query than the full-causal baseline",
        "(169 vs 1024; −71.4% even against a same-doc causal reference).",
        "Attention pairs ≈ prefill FLOPs and KV cache — both scale 1:1 with them.",
        "Cost: inserted gist tokens occupy 25% of sequence capacity (40% at",
        "K=16, where the saving falls to 72.4%).",
    ]):
        o.append(text(dx0 - 12, 470 + i * 17, s, 11, INK2))

    # ---- takeaway strip ----
    ty = 600
    o.append(line(28, ty, W - 28, ty, GRID, 1))
    o.append(text(28, ty + 28, "Takeaway", 14, INK, "bold"))
    for i, s in enumerate([
        "Neither recomputed content (it.1) nor stored n-gram memory (it.2) improves the strict regime through the gist INPUT channel — content was consumed at full gate strength",
        "yet bought −0.0008 BPB; memory was written at full breadth yet cost +0.0005…+0.0014. CORE agrees: no arm recovers the strict regime's CORE cost (0.183 → 0.14x) and the",
        "engram arms trend lower still. The 0.017 BPB gap to full causal lives in channel bandwidth / block-causality itself; deferred TTT-on-gist-inputs is deprioritized by the same evidence.",
        "Efficiency survives: the strict mask attends 169 vs 1024 keys per query (K=8, val rows) — 83.5% fewer attention tokens / prefill FLOPs — at the cost of gists filling 25% of sequence capacity.",
    ]):
        o.append(text(28, ty + 52 + i * 19, s, 12.5, INK2))
    o.append("</svg>")
    return "".join(o)


if __name__ == "__main__":
    import cairosvg
    for name, fn in [("fig1_setup", fig1), ("fig2_hypernet", fig2),
                     ("fig3_engram", fig3), ("fig4_results", fig4)]:
        svg = fn()
        svg_path = os.path.join(DIR, name + ".svg")
        png_path = os.path.join(DIR, name + ".png")
        with open(svg_path, "w") as f:
            f.write(svg)
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2.0, background_color=SURFACE)
        print("wrote", svg_path, "and .png")
