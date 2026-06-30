"""
Synthetic-language generator for the Polysemy × Context experiment (component 1).

The generator emits a controlled toy language in which **one form = one token id** (no
BPE), so the lexical sense-ambiguity ``H(S|W)`` is exact and analytically known. The
pipeline is:

    PCFG over POS-typed senses  ->  Zipfian sense frequencies  ->  sense->form layer
    (merge senses to a target H(S|W) under a context-overlap regime, paired with splits
    that hold |V| constant)  ->  render derivations to whitespace-separated form symbols.

Design decisions (see docs/adr/0003-*, docs/adr/0004-* and
run/deep-interview/deep-interview-polysemy-context.md):

- A *form* is a surface token the LM sees; the form->token map is the identity (no
  merges/splits) so the controlled H(S|W) survives end to end. The corpus is written as
  text (whitespace-separated form symbols) and consumed later by an identity tokenizer.
- The sense stream (syntax) is generated ONCE and reused across all conditions in a run,
  so global / syntactic entropy is held constant by construction; only the lexical
  (sense->form) layer varies between conditions.
- v1 ENFORCES only |V| (held equal across conditions via paired merge/split) and exports
  the analytic PCFG entropy. Every other confound (unigram entropy, gzip size, H_m,
  per-form H(S|W), synonymy) is MEASURED and recorded as a covariate, not matched.

Nothing in this module imports torch or the training harness, so it is cheap to unit-test.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Some envs ship a pandas compiled against a different numpy ABI; importing it raises a
# ValueError (not ImportError) mid-import. pyarrow's array path lazily imports pandas and
# would crash on it. We never use pandas here, so if pandas is broken we install a finder
# that makes an actual `import pandas` raise a clean ImportError, which pyarrow's pandas
# shim catches and skips. This is a no-op where pandas imports fine.
#
# The finder raises ONLY for a real import (driven by importlib's _find_and_load); a bare
# existence probe via importlib.util.find_spec (e.g. torch._dynamo's trace_rules, which
# checks whether 'pandas' is installed) gets a clean "not found" (None) instead. Raising on
# the probe would propagate out of torch and break any torch._dynamo / torch.compile use in
# the same process — fatal for the component-3 probe + its tests, which import the generator
# (installing this finder) alongside the GPT/torch stack.
if "pandas" not in sys.modules:
    try:
        import pandas as _pandas_probe  # noqa: F401
    except Exception:
        import importlib.abc

        class _BrokenPandasBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if not (name == "pandas" or name.startswith("pandas.")):
                    return None
                # Decide by the nearest enclosing import frame:
                #  - if it is `_find_and_load(_unlocked)` actually loading pandas -> real
                #    import -> raise a clean ImportError (pyarrow's shim catches it);
                #  - if we first hit importlib.util.find_spec -> a bare existence probe
                #    (e.g. torch._dynamo's trace_rules, even while it is itself being
                #    imported) -> report "not found" (None) so it never blows up.
                f = sys._getframe(1)
                while f is not None:
                    code = f.f_code
                    if code.co_name in ("_find_and_load", "_find_and_load_unlocked"):
                        loading = f.f_locals.get("name", "")
                        if isinstance(loading, str) and (loading == "pandas" or loading.startswith("pandas.")):
                            raise ImportError("pandas is broken in this env; disabled for nanochat.polysemy")
                        return None  # importing something else; our pandas lookup is nested -> probe
                    if code.co_name == "find_spec" and "importlib" in (code.co_filename or ""):
                        return None  # importlib.util.find_spec existence probe
                    f = f.f_back
                return None

        sys.modules.pop("pandas", None)
        sys.meta_path.insert(0, _BrokenPandasBlocker())

import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# Grammar definition (PCFG over POS-typed senses)

# Terminal POS classes. Senses are partitioned into these classes; the syntax is defined
# over the classes and the lexical choice (which sense within a class) is Zipfian.
POS_CLASSES = ("N", "V", "DET", "P")


@dataclass(frozen=True)
class Rule:
    """A single PCFG production ``lhs -> rhs`` with probability ``prob``."""
    lhs: str
    rhs: Tuple[str, ...]
    prob: float


@dataclass
class PCFG:
    """A probabilistic context-free grammar over nonterminals + POS-class terminals."""
    start: str
    rules: List[Rule]

    def rules_for(self, nonterminal: str) -> List[Rule]:
        return [r for r in self.rules if r.lhs == nonterminal]

    @property
    def nonterminals(self) -> List[str]:
        # preserve a stable, deterministic order
        seen: List[str] = []
        for r in self.rules:
            if r.lhs not in seen:
                seen.append(r.lhs)
        return seen


def build_default_pcfg() -> PCFG:
    """The v1 default grammar (see the spec).

    S  -> NP VP
    NP -> DET N | N | NP PP
    VP -> V NP | V NP PP | V
    PP -> P NP

    Recursion enters via ``NP -> NP PP`` and ``PP -> P NP``; a depth cap (applied at
    sampling time) forces the short rules near the cap so derivations terminate at a
    bounded length (~10-40 senses).
    """
    rules = [
        Rule("S", ("NP", "VP"), 1.0),
        Rule("NP", ("DET", "N"), 0.45),
        Rule("NP", ("N",), 0.35),
        Rule("NP", ("NP", "PP"), 0.20),
        Rule("VP", ("V", "NP"), 0.45),
        Rule("VP", ("V", "NP", "PP"), 0.25),
        Rule("VP", ("V",), 0.30),
        Rule("PP", ("P", "NP"), 1.0),
    ]
    return PCFG(start="S", rules=rules)


# Rules that can grow the derivation by re-introducing a nonterminal (forbidden at the
# recursion-depth cap so sampling always terminates).
_RECURSIVE_RHS = {("NP", "PP"), ("V", "NP", "PP")}


# -----------------------------------------------------------------------------
# Sense inventory (POS class -> senses with Zipfian frequencies)


@dataclass
class SenseInventory:
    """Maps each sense id to its POS class, with a Zipfian within-class weight.

    Sense ids are contiguous 0..K-1. ``class_of[s]`` is the POS class of sense s;
    ``within_class_weight[s]`` is its (unnormalized) Zipf weight 1/rank used to sample it
    when its class is emitted by the grammar.
    """
    class_of: List[str]
    within_class_weight: List[float]
    senses_by_class: Dict[str, List[int]]

    @property
    def num_senses(self) -> int:
        return len(self.class_of)


def build_sense_inventory(class_sizes: Dict[str, int], zipf_exponent: float = 1.0) -> SenseInventory:
    """Build K = sum(class_sizes) senses, Zipf-weighted (1/rank**exponent) within class."""
    class_of: List[str] = []
    within_class_weight: List[float] = []
    senses_by_class: Dict[str, List[int]] = {c: [] for c in class_sizes}
    sid = 0
    for cls in POS_CLASSES:
        n = class_sizes.get(cls, 0)
        for rank in range(1, n + 1):
            class_of.append(cls)
            within_class_weight.append(1.0 / (rank ** zipf_exponent))
            senses_by_class[cls].append(sid)
            sid += 1
    return SenseInventory(class_of=class_of, within_class_weight=within_class_weight, senses_by_class=senses_by_class)


# -----------------------------------------------------------------------------
# Sampling the sense stream from the PCFG


def _sample_pos_sequence(pcfg: PCFG, rng: np.random.Generator, max_depth: int) -> List[str]:
    """Sample one derivation and return its terminal POS-class sequence."""
    # Pre-index rules and cumulative probabilities per nonterminal for fast sampling.
    out: List[str] = []
    stack: List[Tuple[str, int]] = [(pcfg.start, 0)]
    # We expand depth-first, left-to-right, using an explicit stack to avoid recursion limits.
    # Because we want left-to-right terminal order, push RHS in reverse.
    while stack:
        sym, depth = stack.pop()
        if sym in POS_CLASSES:
            out.append(sym)
            continue
        rules = pcfg.rules_for(sym)
        if depth >= max_depth:
            # Forbid recursive RHS at the cap so the derivation terminates.
            rules = [r for r in rules if r.rhs not in _RECURSIVE_RHS] or rules
        probs = np.array([r.prob for r in rules], dtype=np.float64)
        probs = probs / probs.sum()
        choice = rules[int(rng.choice(len(rules), p=probs))]
        for child in reversed(choice.rhs):
            stack.append((child, depth + 1))
    return out


def _sample_senses_for_pos(pos_seq: Sequence[str], inventory: SenseInventory,
                           class_cum: Dict[str, np.ndarray], rng: np.random.Generator) -> List[int]:
    """Map a POS sequence to concrete sense ids by Zipf-sampling within each class."""
    senses: List[int] = []
    for cls in pos_seq:
        ids = inventory.senses_by_class[cls]
        cum = class_cum[cls]
        r = rng.random()
        idx = int(np.searchsorted(cum, r, side="right"))
        idx = min(idx, len(ids) - 1)
        senses.append(ids[idx])
    return senses


def generate_sense_corpus(pcfg: PCFG, inventory: SenseInventory, *, num_tokens: int,
                          max_depth: int = 5, min_len: int = 4, max_len: int = 60,
                          seed: int = 0) -> List[List[int]]:
    """Generate documents (each a list of sense ids) until ~num_tokens senses are produced.

    Each document is one PCFG derivation. Documents whose length falls outside
    [min_len, max_len] are rejected and resampled (a soft length control). The sense
    stream is the *syntactic* layer and is reused across all lexical conditions.
    """
    rng = np.random.default_rng(seed)
    # Precompute normalized cumulative Zipf distributions per class.
    class_cum: Dict[str, np.ndarray] = {}
    for cls, ids in inventory.senses_by_class.items():
        if not ids:
            continue
        w = np.array([inventory.within_class_weight[s] for s in ids], dtype=np.float64)
        w = w / w.sum()
        class_cum[cls] = np.cumsum(w)

    docs: List[List[int]] = []
    total = 0
    while total < num_tokens:
        pos_seq = _sample_pos_sequence(pcfg, rng, max_depth)
        if not (min_len <= len(pos_seq) <= max_len):
            continue
        senses = _sample_senses_for_pos(pos_seq, inventory, class_cum, rng)
        docs.append(senses)
        total += len(senses)
    return docs


# -----------------------------------------------------------------------------
# Entropy helpers (all in bits)


def entropy_bits(probs: Sequence[float]) -> float:
    """Shannon entropy in bits of a (sub)distribution; zero-mass entries ignored."""
    p = np.asarray(list(probs), dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _counts_entropy_bits(counts: Sequence[int]) -> float:
    c = np.asarray(list(counts), dtype=np.float64)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def block_entropies(docs: Sequence[Sequence[int]], max_k: int) -> List[float]:
    """Within-document block entropies H_block(k) for k=1..max_k (bits), exact from counts."""
    out: List[float] = []
    for k in range(1, max_k + 1):
        counts: Counter = Counter()
        for doc in docs:
            if len(doc) < k:
                continue
            for i in range(len(doc) - k + 1):
                counts[tuple(doc[i:i + k])] += 1
        out.append(_counts_entropy_bits(list(counts.values())))
    return out


def m_local_entropies(docs: Sequence[Sequence[int]], ms: Sequence[int]) -> Dict[int, float]:
    """m-local entropy H_m = H_block(m) - H_block(m-1) for each m in ms (bits).

    H_1 is the unigram entropy. This is Someya et al.'s instrument applied directly to a
    token stream; we record it on both the form stream (total) and the sense stream
    (syntactic baseline) so component 3 can do the lexical-vs-total decomposition.
    """
    max_k = max(ms)
    hb = [0.0] + block_entropies(docs, max_k)  # hb[0] = H_block(0) = 0
    return {m: hb[m] - hb[m - 1] for m in ms}


# -----------------------------------------------------------------------------
# Sense -> form layer (the polysemy lever)


@dataclass
class SenseFormMap:
    """The lexical layer for one condition.

    ``sense_to_forms[s]`` is the list of allomorph form symbols sense s may render as
    (length 1 normally; >1 if the sense was *split* into synonyms). ``form_to_senses[w]``
    is the set of senses a form covers (size 1 normally; >1 if senses were *merged* onto
    a shared form -> polysemy). The two are inverse views.
    """
    sense_to_forms: Dict[int, List[str]]
    form_to_senses: Dict[str, List[int]]

    @property
    def vocab(self) -> List[str]:
        return sorted(self.form_to_senses.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.form_to_senses)


def corpus_hsw(form_to_senses: Dict[str, List[int]], sense_prob: Dict[int, float]) -> Tuple[float, Dict[str, float]]:
    """Corpus-level H(S|W) = sum_w P(w) H(S|W=w), plus the per-form H(S|W=w) map (bits)."""
    per_form: Dict[str, float] = {}
    total = 0.0
    for w, senses in form_to_senses.items():
        ps = np.array([sense_prob.get(s, 0.0) for s in senses], dtype=np.float64)
        pw = float(ps.sum())
        h = entropy_bits(ps) if pw > 0 else 0.0
        per_form[w] = float(h)
        total += pw * h
    return float(total), per_form


def _group_corpus_hsw(groups: Sequence[Sequence[int]], prob: np.ndarray) -> float:
    """Corpus H(S|W) for a partition of senses into form-groups (bits)."""
    total = 0.0
    for g in groups:
        ps = prob[list(g)]
        pw = float(ps.sum())
        if pw > 0:
            total += pw * entropy_bits(ps)
    return total


def _partition_within_class(inventory: SenseInventory, prob: np.ndarray, scale: float,
                            pmax: float, d_max: int) -> List[List[int]]:
    """Overlapping-polysemy partition: merge senses WITHIN each POS class.

    Each class is swept high-frequency first; a form's degree (number of merged senses)
    grows with sqrt(freq) (Ferrer-i-Cancho's meaning-frequency law), and its members are
    adjacent in frequency so the merge is balanced (meaningful per-form entropy).
    """
    groups: List[List[int]] = []
    for ids in inventory.senses_by_class.values():
        if not ids:
            continue
        order = sorted(ids, key=lambda s: prob[s], reverse=True)
        i = 0
        while i < len(order):
            lead = order[i]
            d = 1 + int(round(scale * math.sqrt(prob[lead] / pmax)))
            d = max(1, min(d_max, d))
            groups.append(list(order[i:i + d]))
            i += d
    return groups


def _partition_cross_class(inventory: SenseInventory, prob: np.ndarray, scale: float,
                           pmax: float, d_max: int, window: int = 64) -> List[List[int]]:
    """Homonymy partition: merge senses from DIFFERENT POS classes (disjoint contexts).

    Senses are swept high-frequency first; each form gathers up to ``d`` members of
    *distinct* classes scanning forward within a bounded frequency window, so merged
    senses are similar-frequency homonyms whose grammatical slots disambiguate them.
    """
    K = inventory.num_senses
    order = sorted(range(K), key=lambda s: prob[s], reverse=True)
    used = [False] * K
    groups: List[List[int]] = []
    for i in range(K):
        if used[i]:
            continue
        lead = order[i]
        used[i] = True
        d = 1 + int(round(scale * math.sqrt(prob[lead] / pmax)))
        d = max(1, min(d_max, d))
        grp = [lead]
        classes_in = {inventory.class_of[lead]}
        j = i + 1
        scanned = 0
        while len(grp) < d and j < K and scanned < window:
            if not used[j]:
                t = order[j]
                if inventory.class_of[t] not in classes_in:
                    grp.append(t)
                    used[j] = True
                    classes_in.add(inventory.class_of[t])
            j += 1
            scanned += 1
        groups.append(grp)
    return groups


def _fine_topup(groups: List[List[int]], h: float, target: float, tolerance: float,
                inventory: SenseInventory, prob: np.ndarray, overlap: str) -> List[List[int]]:
    """Close the gap to ``target`` by merging small compatible pairs of monosemous senses.

    The band partition lands just below the target with coarse, quantized steps; this adds
    degree-2 forms from the (numerous) monosemous tail — each a tiny, fine-grained H(S|W)
    increment and a *distinct* new polysemous form — so we hit the target precisely without
    collapsing polysemy onto a few forms. Pairs respect the overlap regime (cross-class for
    homonymy, same-class for overlapping).
    """
    if abs(h - target) <= tolerance:
        return groups
    singles = sorted([g[0] for g in groups if len(g) == 1], key=lambda s: prob[s], reverse=True)
    used: set = set()
    pairs: List[List[int]] = []
    n = len(singles)
    for i in range(n):
        if abs(h - target) <= tolerance:
            break
        s = singles[i]
        if s in used:
            continue
        for j in range(i + 1, n):
            t = singles[j]
            if t in used:
                continue
            same_class = inventory.class_of[t] == inventory.class_of[s]
            ok = (overlap == "partial" and same_class) or (overlap == "none" and not same_class)
            if not ok:
                continue
            ps = prob[[s, t]]
            delta = float(ps.sum() * entropy_bits(ps))
            if h + delta <= target + tolerance:  # lower-freq t -> smaller delta; first fit is fine
                used.add(s)
                used.add(t)
                pairs.append([s, t])
                h += delta
                break
    merged = [g for g in groups if len(g) > 1]
    remaining = [[s] for s in singles if s not in used]
    return merged + pairs + remaining


def build_sense_form_map(inventory: SenseInventory, sense_prob: Dict[int, float], *,
                         target_hsw: float, overlap: str, tolerance: float = 0.05,
                         seed: int = 0) -> SenseFormMap:
    """Build the sense->form map for one condition.

    Polysemy is spread across many forms via a frequency-band partition: a form's degree
    (number of merged senses) grows with sqrt(freq) — Ferrer-i-Cancho's meaning-frequency
    law — and merged senses are frequency-balanced so each polysemous form carries
    meaningful H(S|W=w). A single scalar ``scale`` controls the overall amount of polysemy;
    we scan it to land corpus H(S|W) on ``target_hsw`` within ``tolerance``.

    ``overlap`` selects the regime:
    - ``"none"``  (homonymy):  merge senses from DIFFERENT POS classes -> disjoint
      grammatical contexts -> context fully resolves them.
    - ``"partial"`` (overlapping polysemy): merge senses from the SAME POS class ->
      shared contexts -> residual H(S|W,C) > 0.

    Merging shrinks the distinct-form count; an equal number of *splits* (a monosemous
    sense rendered via several synonym forms) restores |V| to exactly K. Splits add
    synonymy (a recorded covariate) but no sense-ambiguity, and never touch a merged sense
    (which would perturb the measured H(S|W)).
    """
    assert overlap in ("none", "partial"), f"overlap must be 'none' or 'partial', got {overlap}"
    rng = np.random.default_rng(seed)
    K = inventory.num_senses
    prob = np.array([sense_prob.get(s, 0.0) for s in range(K)], dtype=np.float64)
    pmax = float(prob.max()) if prob.max() > 0 else 1.0
    n_classes = sum(1 for ids in inventory.senses_by_class.values() if ids)

    if target_hsw <= 0:
        groups: List[List[int]] = [[s] for s in range(K)]
    else:
        if overlap == "partial":
            d_max = 6
            build = lambda sc: _partition_within_class(inventory, prob, sc, pmax, d_max)
        else:
            d_max = max(2, n_classes)  # a homonym group spans distinct classes
            build = lambda sc: _partition_cross_class(inventory, prob, sc, pmax, d_max)
        # Coarse: corpus H(S|W) rises monotonically with scale; take the largest-scale band
        # partition that stays at/under the target (closest from below).
        coarse_groups: List[List[int]] = [[s] for s in range(K)]
        coarse_h = 0.0
        for sc in np.linspace(0.0, 40.0, 401):
            g = build(float(sc))
            h = _group_corpus_hsw(g, prob)
            if h <= target_hsw + 1e-9:
                coarse_groups, coarse_h = g, h
            else:
                break
        # Fine: top up with small tail pairs to land on target within tolerance.
        groups = _fine_topup(coarse_groups, coarse_h, target_hsw, tolerance, inventory, prob, overlap)

    # ---- assign symbols, restore |V| to K via splits on monosemous senses ----
    # Guarantee at least one monosemous sense (to host the split synonyms): if every sense
    # got merged (only happens at very small K with a high target), free the lowest-freq
    # member of the least-contributing group. Splits must never touch a merged sense, since
    # that would perturb the measured H(S|W).
    if K - len(groups) > 0 and not any(len(g) == 1 for g in groups):
        def _contrib(g):
            ps = prob[list(g)]
            return float(ps.sum() * entropy_bits(ps))
        victim = min((g for g in groups if len(g) > 1), key=_contrib)
        victim_sorted = sorted(victim, key=lambda s: prob[s], reverse=True)
        freed = victim_sorted[-1]
        victim[:] = victim_sorted[:-1]
        groups.append([freed])

    groups = sorted(groups, key=lambda g: (-float(prob[list(g)].sum()), min(g)))
    n_merges = K - len(groups)  # senses absorbed by merging = forms to add back as synonyms

    sense_to_forms: Dict[int, List[str]] = {}
    form_to_senses: Dict[str, List[int]] = {}
    next_form_idx = 0

    def new_symbol() -> str:
        nonlocal next_form_idx
        sym = f"w{next_form_idx:05d}"
        next_form_idx += 1
        return sym

    for g in groups:
        sym = new_symbol()
        form_to_senses[sym] = list(g)
        for s in g:
            sense_to_forms[s] = [sym]

    singleton_senses = [g[0] for g in groups if len(g) == 1]
    if n_merges > 0:
        assert singleton_senses, "no monosemous senses available for |V|-restoring splits"
        order = list(singleton_senses)
        rng.shuffle(order)
        for k in range(n_merges):
            s = order[k % len(order)]
            sym = new_symbol()
            sense_to_forms[s].append(sym)
            form_to_senses[sym] = [s]

    return SenseFormMap(sense_to_forms=sense_to_forms, form_to_senses=form_to_senses)


# -----------------------------------------------------------------------------
# Rendering and a minimal identity vocab (for generation-side validation)


class IdentityVocab:
    """Trivial 1-form = 1-id vocab; the generation-side stand-in for the identity tokenizer."""

    def __init__(self, symbols: Sequence[str]):
        self.itos: List[str] = list(symbols)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[t] for t in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def __len__(self) -> int:
        return len(self.itos)


def render_documents(sense_docs: Sequence[Sequence[int]], smap: SenseFormMap, *, seed: int = 0) -> List[str]:
    """Render each sense document to a whitespace-separated form-symbol string.

    A sense with multiple allomorph forms (a split synonym) picks one uniformly at random
    per occurrence (seeded), which adds form entropy but no sense-ambiguity.
    """
    rng = np.random.default_rng(seed)
    out: List[str] = []
    for doc in sense_docs:
        toks: List[str] = []
        for s in doc:
            forms = smap.sense_to_forms[s]
            toks.append(forms[0] if len(forms) == 1 else forms[int(rng.integers(len(forms)))])
        out.append(" ".join(toks))
    return out


def render_documents_with_senses(sense_docs: Sequence[Sequence[int]], smap: SenseFormMap, *,
                                 seed: int = 0) -> List[Dict[str, list]]:
    """Render docs to forms while keeping the aligned ground-truth sense id at each position.

    Returns one ``{"forms": [...], "senses": [...]}`` record per document (parallel arrays).
    This is the sense-labeled probe set: the form stream is what the LM sees, the sense
    stream is the latent label a representation probe (component 3) tries to decode from the
    model's hidden state — the test of whether context resolves polysemy.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict[str, list]] = []
    for doc in sense_docs:
        forms: List[str] = []
        for s in doc:
            allo = smap.sense_to_forms[s]
            forms.append(allo[0] if len(allo) == 1 else allo[int(rng.integers(len(allo)))])
        out.append({"forms": forms, "senses": [int(s) for s in doc]})
    return out


# -----------------------------------------------------------------------------
# Analytic PCFG entropy (calibration baseline; uncapped grammar)


def analytic_pcfg_entropy(pcfg: PCFG, inventory: SenseInventory, zipf_exponent: float = 1.0) -> Optional[Dict[str, float]]:
    """Source entropy rate (bits/sense) of the sense stream for the uncapped grammar.

    Uses expected expansion counts E = e_start (I - M)^-1, where M[i][j] is the expected
    number of nonterminal j produced by one expansion of nonterminal i. Returns None if
    the grammar is supercritical (I - M singular / non-positive expectations).
    """
    nts = pcfg.nonterminals
    idx = {nt: i for i, nt in enumerate(nts)}
    n = len(nts)
    M = np.zeros((n, n), dtype=np.float64)
    rule_entropy = np.zeros(n, dtype=np.float64)
    # expected count of each POS class produced directly by one expansion of nt i
    term_per_nt = {c: np.zeros(n, dtype=np.float64) for c in POS_CLASSES}

    for nt in nts:
        rules = pcfg.rules_for(nt)
        probs = np.array([r.prob for r in rules], dtype=np.float64)
        probs = probs / probs.sum()
        rule_entropy[idx[nt]] = entropy_bits(probs)
        for r, p in zip(rules, probs):
            for sym in r.rhs:
                if sym in idx:
                    M[idx[nt], idx[sym]] += p
                elif sym in POS_CLASSES:
                    term_per_nt[sym][idx[nt]] += p

    try:
        fundamental = np.linalg.inv(np.eye(n) - M)
    except np.linalg.LinAlgError:
        return None
    e_start = np.zeros(n, dtype=np.float64)
    e_start[idx[pcfg.start]] = 1.0
    E = e_start @ fundamental  # expected expansions of each nonterminal per derivation
    if not np.all(np.isfinite(E)) or np.any(E < 0):
        return None

    total_rule_entropy = float(np.sum(E * rule_entropy))
    expected_terms = {c: float(np.sum(E * term_per_nt[c])) for c in POS_CLASSES}
    class_term_entropy = {}
    for c in POS_CLASSES:
        ids = inventory.senses_by_class.get(c, [])
        if ids:
            w = np.array([inventory.within_class_weight[s] for s in ids], dtype=np.float64)
            class_term_entropy[c] = entropy_bits(w)
        else:
            class_term_entropy[c] = 0.0
    total_term_entropy = float(sum(expected_terms[c] * class_term_entropy[c] for c in POS_CLASSES))
    expected_senses = float(sum(expected_terms.values()))
    if expected_senses <= 0:
        return None
    per_sense = (total_rule_entropy + total_term_entropy) / expected_senses
    return {
        "bits_per_sense": per_sense,
        "expected_senses_per_doc": expected_senses,
        "rule_entropy_per_doc": total_rule_entropy,
        "terminal_entropy_per_doc": total_term_entropy,
    }


# -----------------------------------------------------------------------------
# Condition orchestration + metadata


@dataclass
class GeneratorConfig:
    """Top-level generation config; serialized into every condition's metadata."""
    class_sizes: Dict[str, int]
    num_tokens: int
    seed: int = 0
    zipf_exponent: float = 1.0
    max_depth: int = 5
    min_len: int = 4
    max_len: int = 60
    tolerance: float = 0.05
    hm_ms: Tuple[int, ...] = (1, 2, 3)
    hm_max_tokens: int = 2_000_000

    @property
    def num_senses(self) -> int:
        return sum(self.class_sizes.values())


@dataclass
class Condition:
    """One (target H(S|W), overlap) cell to generate."""
    slug: str
    target_hsw: float
    overlap: str  # "none" | "partial" | "mono"


def default_conditions() -> List[Condition]:
    """The v1 grid: H(S|W) in {0, low=0.5, high=1.5} x overlap in {homonymy, overlapping}."""
    return [
        Condition("mono", 0.0, "mono"),
        Condition("hsw0p5_homonymy", 0.5, "none"),
        Condition("hsw0p5_overlap", 0.5, "partial"),
        Condition("hsw1p5_homonymy", 1.5, "none"),
        Condition("hsw1p5_overlap", 1.5, "partial"),
    ]


def _sense_probabilities(sense_docs: Sequence[Sequence[int]], num_senses: int) -> Dict[int, float]:
    counts = np.zeros(num_senses, dtype=np.float64)
    for doc in sense_docs:
        for s in doc:
            counts[s] += 1.0
    total = counts.sum()
    if total <= 0:
        return {s: 0.0 for s in range(num_senses)}
    return {s: counts[s] / total for s in range(num_senses)}


def _truncate_docs(docs: Sequence[Sequence[int]], max_tokens: int) -> List[List[int]]:
    out: List[List[int]] = []
    total = 0
    for doc in docs:
        out.append(list(doc))
        total += len(doc)
        if total >= max_tokens:
            break
    return out


def build_condition(cfg: GeneratorConfig, pcfg: PCFG, inventory: SenseInventory,
                    sense_docs: Sequence[Sequence[int]], sense_prob: Dict[int, float],
                    cond: Condition) -> Tuple[List[str], SenseFormMap, dict]:
    """Build one condition: lexical map, rendered corpus, and the metadata sidecar dict."""
    overlap = "none" if cond.overlap == "mono" else cond.overlap
    smap = build_sense_form_map(inventory, sense_prob, target_hsw=cond.target_hsw,
                                overlap=overlap, tolerance=cfg.tolerance, seed=cfg.seed)
    documents = render_documents(sense_docs, smap, seed=cfg.seed)

    # --- measurements (covariates) ---
    measured_hsw, per_form_hsw = corpus_hsw(smap.form_to_senses, sense_prob)

    # token (form) streams for entropy stats
    form_vocab = IdentityVocab(smap.vocab)
    form_docs = [form_vocab.encode(d) for d in documents]
    sample_form_docs = _truncate_docs(form_docs, cfg.hm_max_tokens)
    sample_sense_docs = _truncate_docs(sense_docs, cfg.hm_max_tokens)

    hm_forms = m_local_entropies(sample_form_docs, cfg.hm_ms)
    hm_senses = m_local_entropies(sample_sense_docs, cfg.hm_ms)

    # unigram entropies
    form_counter: Counter = Counter()
    for d in form_docs:
        form_counter.update(d)
    unigram_form_entropy = _counts_entropy_bits(list(form_counter.values()))
    sense_counter: Counter = Counter()
    for d in sense_docs:
        sense_counter.update(d)
    unigram_sense_entropy = _counts_entropy_bits(list(sense_counter.values()))

    # synonymy covariate
    n_split_senses = sum(1 for s, forms in smap.sense_to_forms.items() if len(forms) > 1)
    forms_per_sense = np.mean([len(f) for f in smap.sense_to_forms.values()]) if smap.sense_to_forms else 0.0
    # polysemy structure
    n_poly_forms = sum(1 for w, ss in smap.form_to_senses.items() if len(ss) > 1)

    # gzip compressibility (proxy for total corpus complexity)
    corpus_bytes = ("\n".join(documents)).encode("utf-8")
    n_tokens = sum(len(d) for d in form_docs)
    gzip_bytes = len(gzip.compress(corpus_bytes, compresslevel=6))

    analytic = analytic_pcfg_entropy(pcfg, inventory, cfg.zipf_exponent)

    metadata = {
        "condition": {"slug": cond.slug, "target_hsw_bits": cond.target_hsw, "overlap": cond.overlap},
        "vocab_size": smap.vocab_size,
        "num_senses": inventory.num_senses,
        "num_documents": len(documents),
        "num_tokens": n_tokens,
        "h_s_given_w": {
            "target_bits": cond.target_hsw,
            "measured_bits": float(measured_hsw),
            "within_tolerance": bool(abs(measured_hsw - cond.target_hsw) <= cfg.tolerance),
            "per_form_bits": {w: float(h) for w, h in per_form_hsw.items()},
        },
        "h_m_bits": {
            "ms": list(cfg.hm_ms),
            "forms_total": {int(m): hm_forms[m] for m in cfg.hm_ms},
            "senses_syntactic": {int(m): hm_senses[m] for m in cfg.hm_ms},
            "estimated_on_tokens": sum(len(d) for d in sample_form_docs),
            "note": "forms_total = m-local entropy of the form stream; senses_syntactic = "
                    "the held-constant syntactic baseline. The formal lexical-vs-total H_m "
                    "decomposition is computed in component 3 from these recorded streams.",
        },
        "unigram_entropy_bits": {"forms": unigram_form_entropy, "senses": unigram_sense_entropy},
        "gzip": {"bytes": gzip_bytes, "bits_per_token": (gzip_bytes * 8.0 / n_tokens) if n_tokens else None},
        "synonymy": {"split_senses": n_split_senses, "mean_forms_per_sense": float(forms_per_sense)},
        "polysemy": {"polysemous_forms": n_poly_forms},
        "analytic_pcfg_entropy": analytic,
        "generator_config": asdict(cfg),
        "note": "Forms map 1:1 to token ids via an identity tokenizer (component 2); "
                "special tokens (e.g. <bos>) are owned by that tokenizer, not this vocab.",
    }
    return documents, smap, metadata


# -----------------------------------------------------------------------------
# Writers


def _write_text_parquet(docs: Sequence[str], path: str, row_group_size: int) -> None:
    """Write a 'text' column of strings to one zstd parquet file.

    Builds the column with an explicit ``pa.array(..., type=string)`` rather than
    ``Table.from_pydict``: the latter routes through pyarrow's array-like inference, which
    imports pandas, and a broken pandas/numpy ABI in some envs would crash the write.
    """
    arr = pa.array(list(docs), type=pa.string())
    table = pa.Table.from_arrays([arr], names=["text"])
    pq.write_table(table, path, row_group_size=row_group_size, use_dictionary=False,
                   compression="zstd", compression_level=3, write_statistics=False)


def write_parquet_shards(documents: Sequence[str], out_dir: str, *, shard_chars: int = 50_000_000,
                         row_group_size: int = 1024) -> List[str]:
    """Write documents to zstd parquet shards (single 'text' column), matching the trainer's format."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    shard_docs: List[str] = []
    shard_chars_count = 0
    shard_index = 0

    def flush():
        nonlocal shard_docs, shard_chars_count, shard_index
        if not shard_docs:
            return
        path = os.path.join(out_dir, f"shard_{shard_index:05d}.parquet")
        _write_text_parquet(shard_docs, path, row_group_size)
        paths.append(path)
        shard_docs = []
        shard_chars_count = 0
        shard_index += 1

    for doc in documents:
        shard_docs.append(doc)
        shard_chars_count += len(doc)
        if shard_chars_count >= shard_chars and len(shard_docs) % row_group_size == 0:
            flush()
    flush()
    # Guarantee at least 2 shards so the trainer's "last shard = val" split has a train shard.
    if len(paths) == 1 and len(documents) >= 2:
        os.remove(paths[0])
        paths = []
        mid = len(documents) // 2
        for part_idx, part in enumerate((documents[:mid], documents[mid:])):
            path = os.path.join(out_dir, f"shard_{part_idx:05d}.parquet")
            _write_text_parquet(list(part), path, row_group_size)
            paths.append(path)
    return paths


# Seed offset for the held-out probe stream, so its derivations are disjoint from the
# training sense stream (which uses the run seed) while staying deterministic.
PROBE_SEED_OFFSET = 1_000_003


def write_probe_jsonl(probe_records: Sequence[Dict[str, list]], out_dir: str,
                      filename: str = "probe.jsonl") -> str:
    """Write the sense-labeled probe set (one JSON object per line: forms + senses)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for rec in probe_records:
            f.write(json.dumps(rec) + "\n")
    return path


def write_vocab(smap: SenseFormMap, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vocab = {sym: i for i, sym in enumerate(smap.vocab)}
    path = os.path.join(out_dir, "vocab.json")
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)
    return path


def _json_default(o):
    """Make numpy scalars/arrays JSON-serializable."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_metadata(metadata: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    return path
