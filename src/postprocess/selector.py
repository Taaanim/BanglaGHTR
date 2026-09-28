"""
BanglaHTR Post-Processing Selector
====================================
``select_best_prediction`` — the central public API.

Given three candidate (text, confidence) pairs from:
  - CTC Beam Search
  - CTC Greedy
  - Attention Decoder

it computes a calibrated Combined Score for each candidate and returns the
winner, along with full diagnostics.

Combined_Score formula
----------------------
  combined = w_model * model_conf
           + w_lm    * lm_score
           - hallucination_penalty
           - length_mismatch_penalty
           - repetition_penalty
           - novel_token_penalty

All scores are kept in [0, 100] so the formula is directly interpretable
as a percentage confidence.

Weights (defaults — see ``SelectorConfig`` for tuning)
-------------------------------------------------------
  w_model = 0.55   (raw model confidence dominates, it is well-calibrated)
  w_lm    = 0.35   (language model plausibility)
  w_agree = 0.10   (cross-decoder agreement bonus)

Each weight, threshold, and penalty constant is documented in ``SelectorConfig``
so practitioners can tune without touching the logic.
"""

from __future__ import annotations

import re
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# We import the LM lazily so the module is importable even if not used.
_LM_SINGLETON = None  # module-level LM cache


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration dataclass  (all tunable hyperparameters in one place)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectorConfig:
    """
    All hyperparameters for the post-processing selector.

    Weight tuning guide
    -------------------
    - Increase ``w_model`` if your models are well-calibrated and you trust
      their confidence scores.  Decrease if confidence is over-confident.
    - Increase ``w_lm`` if you have a good external LM.  With only the
      built-in bigram scorer, keep it ≤ 0.40.
    - Increase ``w_agree`` if your three decoders are diverse and agreement is
      a reliable signal.  On short/simple text it tends to be noisy.

    Penalty tuning guide
    --------------------
    - ``repetition_penalty``: 30–50 is safe for suppressing attention
      hallucinations like "কককককক".
    - ``length_coverage_threshold``: 0.5 means a candidate must be at least
      50% as long as the longest non-degenerate candidate to avoid penalty.
    - ``novel_ngram_penalty``: 0.12 per novel trigram is fairly aggressive;
      lower to 0.05 if your decoders produce many legitimate OOV words.
    - ``low_conf_threshold``: below this (0-100 scale) a candidate is treated
      as "uncertain" and can trigger fallback logic.
    """

    # ── Ensemble weights (must sum to 1.0) ─────────────────────────────────
    w_model: float = 0.55   # Raw model confidence contribution
    w_lm:    float = 0.35   # LM plausibility contribution
    w_agree: float = 0.10   # Cross-decoder agreement bonus

    # ── Penalties (subtracted from Combined Score, 0-100 scale) ────────────
    repetition_run_len: int   = 4     # ≥ N identical chars in a row → degenerate
    repetition_bigram_run: int = 3    # ≥ N identical bigrams in a row → degenerate
    repetition_penalty: float = 40.0  # Hard penalty applied to degenerate candidates

    length_coverage_threshold: float = 0.50  # Min (len / max_len) ratio before penalty
    length_penalty_strength:   float = 0.55  # Score multiplier for too-short candidates

    novel_ngram_n:       int   = 3     # N-gram size for hallucination detection
    novel_ngram_penalty: float = 0.15  # Fractional penalty per novel N-gram (max 0.60)

    prefix_superset_min_ratio:   float = 0.70  # Min in-order char overlap to be a "prefix"
    prefix_superset_min_longer:  float = 0.20  # Candidate B must be ≥20% longer than A
    prefix_superset_penalty:     float = 0.50  # Score multiplier for detected truncations

    # ── Agreement scoring ───────────────────────────────────────────────────
    agreement_boost_max:   float = 8.0   # Max bonus for full positional agreement
    agreement_rate_min:    float = 0.40  # Below this rate → 0 boost
    odd_one_out_penalty:   float = 0.60  # Multiplier when two others fully agree and you don't

    # ── Fallback thresholds ─────────────────────────────────────────────────
    low_conf_threshold:    float = 25.0  # Below this → "low confidence" flag
    all_low_conf_fallback: str   = "ctc_beam"  # Winner when ALL are low confidence

    # ── LM scoring scale ────────────────────────────────────────────────────
    lm_scale: float = 100.0  # Multiplies [0,1] LM score to match 0-100 conf scale


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else ""


def _norm_ws(s: str) -> str:
    """Strips whitespace for comparison purposes."""
    return re.sub(r"\s+", "", s or "")


def _is_degenerate(text: str, run_len: int = 4, bigram_run: int = 3) -> bool:
    """
    Detects attention hallucination patterns:
      1. N or more of the same character consecutively (e.g. "কককক")
      2. N or more of the same bigram consecutively  (e.g. "কাকাকা" with bigram_run=3)
      3. Empty / whitespace-only string
    """
    if not text or not text.strip():
        return True

    s = _norm_ws(text)

    # Rule 1: Long single-char run
    for ch in set(s):
        if ch * run_len in s:
            return True

    # Rule 2: Long bigram run
    for i in range(len(s) - 1):
        bigram = s[i: i + 2]
        if bigram * bigram_run in s:
            return True

    return False


def _positional_agreement_rate(a: str, b: str) -> float:
    """
    Position-aware character overlap between two normalised strings.
    Returns [0, 1]; 1.0 = identical, 0.0 = disjoint / unrelated.
    ±1 position slack handles small insertions/deletions.
    """
    na, nb = _norm_ws(a), _norm_ws(b)
    if not na or not nb:
        return 0.0
    n = min(len(na), len(nb))
    matched = 0.0
    for i in range(n):
        if na[i] == nb[i]:
            matched += 1.0
        elif i + 1 < len(nb) and na[i] == nb[i + 1]:
            matched += 0.5
        elif i + 1 < len(na) and na[i + 1] == nb[i]:
            matched += 0.5
    return matched / max(len(na), len(nb))


def _is_prefix_superset(short: str, long: str,
                         min_ratio: float, min_longer: float) -> bool:
    """
    Returns True if ``short`` appears to be a truncated prefix of ``long``:
      - ``long`` must be ≥ (1 + min_longer) × len(short)  longer
      - ≥ min_ratio of ``short``'s characters appear in-order inside ``long``
    """
    ns, nl = _norm_ws(short), _norm_ws(long)
    if not ns or not nl or ns == nl:
        return False
    if len(nl) < len(ns) * (1.0 + min_longer):
        return False
    if len(ns) > len(nl):
        return False

    matched = 0
    j = 0
    for ch in ns:
        while j < len(nl) and nl[j] != ch:
            j += 1
        if j < len(nl):
            matched += 1
            j += 1
    return (matched / max(1, len(ns))) >= min_ratio


def _novel_ngrams(text: str, others: List[str], n: int = 3) -> List[str]:
    """
    Returns N-grams in ``text`` that do NOT appear in any string in ``others``.
    These are potential hallucinated tokens.
    """
    nt = _norm_ws(text)
    if len(nt) < n:
        return []
    grams = {nt[i: i + n] for i in range(len(nt) - n + 1)}
    others_norm = [_norm_ws(o) for o in others]
    return [g for g in grams if not any(g in o for o in others_norm)]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BanglaPostProcessor class  (stateful, LM held inside)
# ─────────────────────────────────────────────────────────────────────────────

class BanglaPostProcessor:
    """
    Stateful post-processor.  Create once per session; the LM is initialised
    lazily on the first call.

    Parameters
    ----------
    vocab_json_path : str
        Path to the model's ``vocab.json``.  Used to build the internal
        Bangla LM.  If None, LM scoring is skipped (w_lm effectively 0).
    kenlm_path : str, optional
        Path to a KenLM binary model for superior n-gram LM scoring.
    config : SelectorConfig, optional
        Override any default weight / threshold / penalty.

    Example
    -------
    >>> processor = BanglaPostProcessor(vocab_json_path="exports/.../vocab.json")
    >>> result = processor.select_best_prediction({
    ...     "ctc_beam":   {"text": "বাংলা", "confidence": 72.0},
    ...     "ctc_greedy": {"text": "বাঙলা", "confidence": 68.0},
    ...     "attn":       {"text": "বাংলা", "confidence": 65.0},
    ... })
    >>> print(result["best_text"], result["best_source"])
    """

    def __init__(
        self,
        vocab_json_path: Optional[str] = None,
        kenlm_path: Optional[str] = None,
        config: Optional[SelectorConfig] = None,
    ):
        self.config = config or SelectorConfig()
        self._lm = None
        self._vocab_json_path = vocab_json_path
        self._kenlm_path      = kenlm_path

    def _get_lm(self):
        """Lazy-initialises the LM on first call."""
        if self._lm is not None:
            return self._lm
        if self._vocab_json_path is None:
            return None
        try:
            from .bangla_lm import build_lm_from_vocab_json
            self._lm = build_lm_from_vocab_json(
                self._vocab_json_path,
                kenlm_path=self._kenlm_path,
            )
        except Exception as e:
            # Non-fatal: LM won't be used, but selector still works
            print(f"[BanglaPostProcessor] LM init failed (non-fatal): {e}")
            self._lm = None
        return self._lm

    def select_best_prediction(self, candidates: dict) -> dict:
        """
        Main entry point.  See module docstring for the Combined Score formula.

        Parameters
        ----------
        candidates : dict
            Must have at least these keys; additional keys are ignored::

                {
                    "ctc_beam":   {"text": str, "confidence": float},  # 0–100
                    "ctc_greedy": {"text": str, "confidence": float},
                    "attn":       {"text": str, "confidence": float},
                }

        Returns
        -------
        dict with keys:
            best_text       : str     – winning transcription
            best_source     : str     – "ctc_beam" | "ctc_greedy" | "attn"
            best_score      : float   – combined score of winner (0–100)
            selection_reason: str     – human-readable explanation
            model_confs     : dict    – raw model confidences {name: float}
            lm_scores       : dict    – LM scores {name: float}  (0–100)
            combined_scores : dict    – final combined scores {name: float}
            is_low_conf     : bool    – True when all candidates are uncertain
            is_degenerate   : dict    – {name: bool} degenerate flags
        """
        cfg = self.config
        lm  = self._get_lm()

        # ── Unpack candidates ─────────────────────────────────────────────
        REQUIRED = ("ctc_beam", "ctc_greedy", "attn")
        parsed: Dict[str, Tuple[str, float]] = {}
        for key in REQUIRED:
            entry = candidates.get(key, {})
            if isinstance(entry, dict):
                text = str(entry.get("text", "")).strip()
                conf = float(entry.get("confidence", 0.0))
            else:
                # Accept (text, confidence) tuple for convenience
                text, conf = str(entry[0]).strip(), float(entry[1])
            parsed[key] = (_nfc(text), max(0.0, min(100.0, conf)))

        texts = {k: v[0] for k, v in parsed.items()}
        model_confs = {k: v[1] for k, v in parsed.items()}

        # ── Step 1: Degenerate / empty gate ──────────────────────────────
        # Degenerate candidates get their model confidence zeroed out so
        # downstream scoring naturally demotes them.
        degen_flags: Dict[str, bool] = {}
        for name, text in texts.items():
            degen_flags[name] = _is_degenerate(
                text,
                run_len=cfg.repetition_run_len,
                bigram_run=cfg.repetition_bigram_run,
            )

        # Working copy of model confidence (mutable)
        adj_conf: Dict[str, float] = dict(model_confs)
        for name in REQUIRED:
            if degen_flags[name]:
                adj_conf[name] = 0.0
            # Very short text is suspicious for non-tiny input
            if texts[name] and len(_norm_ws(texts[name])) < 2:
                adj_conf[name] = min(adj_conf[name], 30.0)

        # ── Step 2: LM scoring ────────────────────────────────────────────
        lm_scores: Dict[str, float] = {}
        for name, text in texts.items():
            if lm is None or degen_flags[name]:
                lm_scores[name] = 0.0
            else:
                lm_scores[name] = lm.score(text) * cfg.lm_scale  # → 0-100

        # ── Step 3: Length-coverage penalty ──────────────────────────────
        # The longest non-degenerate candidate sets the reference length.
        non_degen_lens = [
            max(1, len(_norm_ws(texts[n])))
            for n in REQUIRED
            if not degen_flags[n] and adj_conf[n] > 0
        ]
        max_len = max(non_degen_lens) if non_degen_lens else 1

        length_coverage: Dict[str, float] = {}
        for name in REQUIRED:
            L = max(1, len(_norm_ws(texts[name])))
            coverage = max(cfg.length_coverage_threshold, L / max_len)
            length_coverage[name] = coverage

        # ── Step 4: Cross-decoder agreement bonus ────────────────────────
        pairs = [
            ("ctc_beam", "ctc_greedy"),
            ("ctc_beam", "attn"),
            ("ctc_greedy", "attn"),
        ]
        agree_bonus: Dict[str, float] = {n: 0.0 for n in REQUIRED}

        for a, b in pairs:
            rate = _positional_agreement_rate(texts[a], texts[b])
            # Smooth bonus: 0 at rate≤min, cfg.agreement_boost_max at rate=1
            bonus = max(0.0, min(
                cfg.agreement_boost_max,
                (rate - cfg.agreement_rate_min) * (cfg.agreement_boost_max / (1.0 - cfg.agreement_rate_min))
            ))
            agree_bonus[a] += bonus
            agree_bonus[b] += bonus

        # Odd-one-out penalty: if two others fully agree and you don't, penalise.
        for name in REQUIRED:
            others = [x for x in REQUIRED if x != name]
            a, b   = others[0], others[1]
            na, nb = _norm_ws(texts[a]), _norm_ws(texts[b])
            if na and na == nb and na != _norm_ws(texts[name]):
                adj_conf[name] = adj_conf[name] * cfg.odd_one_out_penalty

        # ── Step 5: Prefix-superset rule (truncation detection) ──────────
        for name in REQUIRED:
            others = [m for m in REQUIRED if m != name]
            is_prefix_of = [
                m for m in others
                if _is_prefix_superset(
                    texts[name], texts[m],
                    min_ratio=cfg.prefix_superset_min_ratio,
                    min_longer=cfg.prefix_superset_min_longer,
                )
            ]
            if is_prefix_of:
                adj_conf[name] = adj_conf[name] * cfg.prefix_superset_penalty

        # ── Step 6: Novel N-gram hallucination penalty ────────────────────
        for name in REQUIRED:
            others_texts = [texts[m] for m in REQUIRED if m != name]
            novel = _novel_ngrams(texts[name], others_texts, n=cfg.novel_ngram_n)
            if novel:
                factor = max(1.0 - cfg.novel_ngram_penalty, 1.0 - cfg.novel_ngram_penalty * len(novel))
                factor = max(0.40, factor)  # floor at 40%
                adj_conf[name] = adj_conf[name] * factor

        # ── Step 7: Combined Score ────────────────────────────────────────
        combined: Dict[str, float] = {}
        for name in REQUIRED:
            if degen_flags[name]:
                # Degenerate candidates are hard-blocked
                combined[name] = 0.0
                continue

            score = (
                cfg.w_model * adj_conf[name]
                + cfg.w_lm  * lm_scores[name]
                + cfg.w_agree * agree_bonus[name]
            )

            # Apply length-coverage as a multiplicative damper
            score = score * length_coverage[name]

            combined[name] = max(0.0, min(100.0, score))

        # ── Step 8: Fallback — all low-confidence ────────────────────────
        all_low = all(c < cfg.low_conf_threshold for c in combined.values())
        is_low_conf = all_low

        if all_low:
            # Trust the fallback decoder; bump its score so it wins
            fb = cfg.all_low_conf_fallback
            if fb in combined:
                combined[fb] = max(combined[fb], cfg.low_conf_threshold)

        # ── Step 9: Pick the winner ───────────────────────────────────────
        # Tie-break: prefer ctc_beam > ctc_greedy > attn
        priority = {"ctc_beam": 0, "ctc_greedy": 1, "attn": 2}
        best_name = max(
            REQUIRED,
            key=lambda n: (combined[n], -priority[n]),
        )
        best_text  = texts[best_name] or "(empty)"
        best_score = round(combined[best_name], 1)

        # ── Step 10: Build explanation ────────────────────────────────────
        reason_parts = []
        if all_low:
            reason_parts.append(
                f"all decoders below low-conf threshold ({cfg.low_conf_threshold:.0f}); "
                f"fallback to {cfg.all_low_conf_fallback}"
            )
        else:
            # Agreement notes
            na_beam = _norm_ws(texts["ctc_beam"])
            na_grdy = _norm_ws(texts["ctc_greedy"])
            na_attn = _norm_ws(texts["attn"])
            if na_beam and na_beam == na_grdy == na_attn:
                reason_parts.append("all three decoders agree")
            elif na_beam and na_beam == na_grdy:
                reason_parts.append("CTC beam + greedy agree")
            elif na_beam and na_beam == na_attn:
                reason_parts.append("CTC beam + attention agree")
            elif na_grdy and na_grdy == na_attn:
                reason_parts.append("greedy + attention agree")

            # Degenerate notes
            degen_names = [n for n, d in degen_flags.items() if d]
            if degen_names:
                reason_parts.append(f"degenerate output blocked: {', '.join(degen_names)}")

            if not reason_parts:
                reason_parts.append(
                    f"no full agreement; {best_name} wins by combined score "
                    f"(model_conf={model_confs[best_name]:.1f}, "
                    f"lm={lm_scores[best_name]:.1f})"
                )

        return {
            "best_text":        best_text,
            "best_source":      best_name,
            "best_score":       best_score,
            "selection_reason": "; ".join(reason_parts),
            "model_confs":      {k: round(v, 1) for k, v in model_confs.items()},
            "lm_scores":        {k: round(v, 1) for k, v in lm_scores.items()},
            "combined_scores":  {k: round(v, 1) for k, v in combined.items()},
            "is_low_conf":      is_low_conf,
            "is_degenerate":    degen_flags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Functional API  (module-level convenience, mirrors class API)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singleton processor (created on first call)
_PROCESSOR: Optional[BanglaPostProcessor] = None


def select_best_prediction(
    candidates: dict,
    vocab_json_path: Optional[str] = None,
    kenlm_path: Optional[str] = None,
    config: Optional[SelectorConfig] = None,
) -> dict:
    """
    Functional entry point for ``select_best_prediction``.

    Wraps ``BanglaPostProcessor`` with a module-level singleton so you can
    call it directly without managing a class instance.

    Parameters
    ----------
    candidates : dict
        ``{ "ctc_beam": {"text": ..., "confidence": ...}, ... }``
        Same format as ``BanglaPostProcessor.select_best_prediction``.
    vocab_json_path : str, optional
        Path to ``vocab.json``.  Only needed on the FIRST call; cached after.
    kenlm_path : str, optional
        Path to KenLM model.  Optional; fall back to built-in bigram LM.
    config : SelectorConfig, optional
        Override default weights/thresholds.  Applies globally if set.

    Returns
    -------
    dict  — same as ``BanglaPostProcessor.select_best_prediction``

    Example
    -------
    >>> from src.postprocess import select_best_prediction
    >>>
    >>> result = select_best_prediction({
    ...     "ctc_beam":   {"text": "বাংলা লেখা",  "confidence": 74.5},
    ...     "ctc_greedy": {"text": "বাঙলা লেখা",  "confidence": 68.2},
    ...     "attn":       {"text": "বাংলা লেখা",  "confidence": 61.0},
    ... }, vocab_json_path="exports/banghtr_x_v2_production/vocab.json")
    >>>
    >>> print(result["best_text"])     # → "বাংলা লেখা"
    >>> print(result["best_source"])   # → "ctc_beam"
    >>> print(result["best_score"])    # → combined score
    """
    global _PROCESSOR

    if _PROCESSOR is None or config is not None:
        _PROCESSOR = BanglaPostProcessor(
            vocab_json_path=vocab_json_path,
            kenlm_path=kenlm_path,
            config=config,
        )
    elif vocab_json_path and _PROCESSOR._vocab_json_path != vocab_json_path:
        # vocab path changed — reinitialise
        _PROCESSOR = BanglaPostProcessor(
            vocab_json_path=vocab_json_path,
            kenlm_path=kenlm_path,
            config=_PROCESSOR.config,
        )

    return _PROCESSOR.select_best_prediction(candidates)
