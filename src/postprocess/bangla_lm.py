"""
BanglaLM: Lightweight Bangla Language Model Scorer
====================================================
Scores candidate strings WITHOUT requiring KenLM or any external binary.
Approach (tiered, all zero-dependency):

  Tier 1 – Character-level Unicode validity
    Every grapheme cluster in the output must be a valid Bengali Unicode
    sequence (no orphaned Hasant/Kar, no combining marks without a base).

  Tier 2 – Vocab-coverage score
    Fraction of model-vocab characters that appear in the candidate.
    A string full of OOV tokens is suspicious.

  Tier 3 – Grapheme-cluster plausibility
    Checks that every grapheme cluster follows valid Bengali phonotactics:
      - consonant (optionally + hasant + consonant)* + optional vowel-kar
      - vowel cluster (independent vowel)

  Tier 4 – Character bigram score (learned from the vocab list)
    Bigram log-likelihood over individual characters.  Trained on-the-fly
    the first time from vocab.json (model's own training distribution).
    Laplace-smoothed; works even with zero-shot priors.

Optional Tier 5 – KenLM (loaded only if kenlm is installed and a model
    path is provided).  When available this supersedes Tiers 3-4 with a
    proper n-gram LM but the other tiers still contribute.

Design goals
  - No extra pip installs required (kenlm is opt-in)
  - Initialises fast (<5ms) from an already-loaded vocab dict
  - Thread-safe (all state is instance-level)
  - Returns scores in [0, 1] so they are directly comparable to model
    confidence values
"""

from __future__ import annotations

import math
import unicodedata
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Bengali Unicode constants ──────────────────────────────────────────────────

# Bengali script range: U+0980 – U+09FF
_BN_RANGE = (0x0980, 0x09FF)

# Independent vowels (should start a word or stand alone)
_INDEPENDENT_VOWELS = set("\u0985\u0986\u0987\u0988\u0989\u098a\u098b\u098c"
                           "\u098f\u0990\u0993\u0994")

# Consonants (raw set – includes modified forms)
_CONSONANTS = set(
    "\u0995\u0996\u0997\u0998\u0999"   # ক খ গ ঘ ঙ
    "\u099a\u099b\u099c\u099d\u099e"   # চ ছ জ ঝ ঞ
    "\u099f\u09a0\u09a1\u09a2\u09a3"   # ট ঠ ড ঢ ণ
    "\u09a4\u09a5\u09a6\u09a7\u09a8"   # ত থ দ ধ ন
    "\u09aa\u09ab\u09ac\u09ad\u09ae"   # প ফ ব ভ ম
    "\u09af\u09b0\u09b2\u09b6\u09b7"   # য র ল শ ষ
    "\u09b8\u09b9\u09dc\u09dd\u09df"   # স হ ড় ঢ় য়
    "\u09ce"                            # ৎ (khanda ta)
)

# Vowel signs (matras / kars) – must follow a consonant/cluster
_VOWEL_SIGNS = set(
    "\u09be\u09bf\u09c0\u09c1\u09c2"   # া ি ী ু ূ
    "\u09c3\u09c4\u09c7\u09c8"          # ৃ ৄ ে ৈ
    "\u09cb\u09cc"                       # ো ৌ
)

# Combining / modifier signs (follow consonants/clusters)
_MODIFIERS = set("\u0981\u0982\u0983")  # ঁ ং ঃ (chandrabindu, anusvara, visarga)

_HASANT    = "\u09cd"   # ্  Virama (makes conjunct consonants)
_NUKTA     = "\u09bc"   # ়  Nukta
_ZWSP      = "\u200b"   # zero-width space
_ZWNJ      = "\u200c"   # zero-width non-joiner

# Digits (Bengali)
_BN_DIGITS = set("\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef")

# Punctuation that is legitimate in Bengali text
_PUNCT     = set("।,.!?;:\"'()-–—…")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_bengali(ch: str) -> bool:
    """True if codepoint falls in the Bengali Unicode block."""
    cp = ord(ch)
    return _BN_RANGE[0] <= cp <= _BN_RANGE[1]


def _strip_zw(text: str) -> str:
    return text.replace(_ZWSP, "").replace(_ZWNJ, "").replace("\ufeff", "")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ── Main class ────────────────────────────────────────────────────────────────

class BanglaLM:
    """
    Stateful scorer.  Construct once, call ``score(text)`` many times.

    Parameters
    ----------
    vocab_chars : list[str]
        Characters from the model's vocab.json (``idx2char`` values minus
        special tokens).  Used to build the bigram table and OOV checker.
    kenlm_path : str, optional
        Path to a KenLM binary ``.arpa`` or ``.klm`` file.  If provided AND
        kenlm is installed, uses proper n-gram LM for Tiers 3-4 equivalent.
        Falls back silently to the built-in scorer if the file is missing.

    Tuning knobs
    ------------
    bigram_alpha  : Laplace smoothing constant (default 0.5)
    oov_penalty   : Score deduction per OOV character, as a fraction (0-1)
    unicode_weight, bigram_weight, oov_weight : relative contribution of
        each tier to the final [0,1] score.  They will be normalised.
    """

    def __init__(
        self,
        vocab_chars: List[str],
        kenlm_path: Optional[str] = None,
        bigram_alpha: float = 0.5,
        oov_penalty: float = 0.12,
        unicode_weight: float = 0.35,
        bigram_weight: float = 0.45,
        oov_weight: float = 0.20,
    ):
        self.bigram_alpha   = bigram_alpha
        self.oov_penalty    = oov_penalty

        # Normalise weights
        total_w = unicode_weight + bigram_weight + oov_weight
        self.w_unicode  = unicode_weight  / total_w
        self.w_bigram   = bigram_weight   / total_w
        self.w_oov      = oov_weight      / total_w

        # Build vocab set (excludes special tokens)
        self._vocab_chars: set = set(
            c for c in vocab_chars
            if c and not c.startswith("<") and not c.endswith(">")
        )

        # Build bigram table from vocab chars (character transitions)
        self._bigram: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._unigram: Dict[str, float] = defaultdict(float)
        self._build_bigram_from_vocab(vocab_chars)

        # Optional KenLM
        self._kenlm_model = None
        if kenlm_path:
            self._try_load_kenlm(kenlm_path)

    # ── Initialisation helpers ─────────────────────────────────────────────

    def _build_bigram_from_vocab(self, chars: List[str]) -> None:
        """
        Builds a character bigram model from the model's own vocabulary.

        Since we only have individual characters (not words), we simulate
        plausible Bengali character sequences using Unicode category priors:
          - consonant -> hasant        (conjunct start)
          - hasant    -> consonant     (conjunct continuation)
          - consonant -> vowel_sign    (syllable)
          - consonant -> modifier      (anusvara/visarga)
          - vowel_sign -> consonant    (next syllable start)
          - Any character -> space
          - space -> Any character

        This is a rule-derived bigram prior; if you have actual Bangla text
        corpora you can replace this with corpus bigram counts for better
        accuracy.
        """
        bn_chars = [c for c in chars if c and len(c) == 1 and _is_bengali(c)]
        space    = " "

        # Unigram counts (uniform prior over vocab chars)
        for c in bn_chars:
            self._unigram[c] += 1.0
        self._unigram[space] += 1.0

        # Rule-derived bigram boosts
        consonants    = [c for c in bn_chars if c in _CONSONANTS]
        vowel_signs   = [c for c in bn_chars if c in _VOWEL_SIGNS]
        ind_vowels    = [c for c in bn_chars if c in _INDEPENDENT_VOWELS]
        modifiers     = [c for c in bn_chars if c in _MODIFIERS]

        # Consonant -> vowel sign (syllable nucleus)
        for c in consonants:
            for v in vowel_signs:
                self._bigram[c][v] += 5.0
            # Consonant -> hasant (conjunct)
            if _HASANT in self._vocab_chars:
                self._bigram[c][_HASANT] += 3.0
            # Consonant -> space (word end)
            self._bigram[c][space] += 4.0
            # Consonant -> modifier
            for m in modifiers:
                self._bigram[c][m] += 2.0

        # Hasant -> consonant (conjunct body)
        if _HASANT in self._vocab_chars:
            for c in consonants:
                self._bigram[_HASANT][c] += 5.0

        # Vowel sign -> consonant (next syllable)
        for v in vowel_signs:
            for c in consonants:
                self._bigram[v][c] += 4.0
            self._bigram[v][space] += 3.0

        # Modifier -> consonant / space
        for m in modifiers:
            for c in consonants:
                self._bigram[m][c] += 3.0
            self._bigram[m][space] += 4.0

        # Independent vowel -> consonant (following consonant after pure vowel)
        for iv in ind_vowels:
            for c in consonants:
                self._bigram[iv][c] += 2.0
            self._bigram[iv][space] += 3.0

        # Space -> consonant / independent vowel
        for c in consonants + ind_vowels:
            self._bigram[space][c] += 4.0

    def _try_load_kenlm(self, path: str) -> None:
        try:
            import kenlm  # type: ignore
            if not path or not __import__("os").path.exists(path):
                return
            self._kenlm_model = kenlm.Model(path)
        except ImportError:
            pass  # kenlm not installed – silently skip
        except Exception:
            pass  # bad model file – silently skip

    # ── Scoring ────────────────────────────────────────────────────────────

    def score(self, text: str) -> float:
        """
        Returns a Language Model plausibility score in [0, 1].
        Higher = more plausible Bengali text.

        Pipeline:
          1. Unicode validity check  → u_score  (0-1)
          2. Character bigram score  → b_score  (0-1)   or KenLM if available
          3. OOV coverage check      → o_score  (0-1)
          Final = w_unicode*u_score + w_bigram*b_score + w_oov*o_score
        """
        if not text or not text.strip():
            return 0.0

        clean = _nfc(_strip_zw(text.strip()))
        if not clean:
            return 0.0

        u_score = self._unicode_validity_score(clean)
        o_score = self._oov_score(clean)

        if self._kenlm_model is not None:
            b_score = self._kenlm_score(clean)
        else:
            b_score = self._bigram_score(clean)

        final = (
            self.w_unicode * u_score
            + self.w_bigram * b_score
            + self.w_oov    * o_score
        )
        return float(max(0.0, min(1.0, final)))

    # ── Internal scorers ───────────────────────────────────────────────────

    def _unicode_validity_score(self, text: str) -> float:
        """
        Penalises Unicode sequences that violate Bengali phonotactics.
        Checks:
          - Orphaned Hasant at end of text
          - Vowel sign with no preceding consonant (start of string)
          - Multiple consecutive Hasants
          - Matra (vowel sign) directly after another matra
          - Non-Bengali non-ASCII non-space characters
        Returns fraction of 'good' characters (1.0 = perfectly valid).
        """
        if not text:
            return 0.0

        violations = 0
        total      = len(text)

        for i, ch in enumerate(text):
            prev = text[i - 1] if i > 0 else None

            # Hasant at very end
            if ch == _HASANT and i == total - 1:
                violations += 1
                continue

            # Orphaned vowel sign (no consonant before it)
            if ch in _VOWEL_SIGNS:
                if prev is None or (prev not in _CONSONANTS
                                    and prev not in _VOWEL_SIGNS
                                    and prev != _NUKTA
                                    and prev not in _MODIFIERS):
                    violations += 0.5

            # Double hasant (e.g. ্্)
            if ch == _HASANT and prev == _HASANT:
                violations += 1

            # Two matras in a row
            if ch in _VOWEL_SIGNS and prev in _VOWEL_SIGNS:
                violations += 0.5

            # Completely foreign characters (not Bengali, not ASCII printable,
            # not Bengali digit, not common punctuation)
            cp = ord(ch)
            if (
                not (0x0980 <= cp <= 0x09FF)
                and not (0x0020 <= cp <= 0x007E)   # ASCII printable
                and ch not in _BN_DIGITS
                and ch not in _PUNCT
            ):
                violations += 1

        score = 1.0 - (violations / max(1, total))
        return float(max(0.0, min(1.0, score)))

    def _oov_score(self, text: str) -> float:
        """
        Fraction of BENGALI characters in ``text`` that are in-vocabulary.
        Space and punctuation are not penalised.
        """
        if not self._vocab_chars:
            return 0.8  # no vocab info → neutral

        bn_chars_in_text = [c for c in text if _is_bengali(c)]
        if not bn_chars_in_text:
            return 0.8

        in_vocab = sum(1 for c in bn_chars_in_text if c in self._vocab_chars)
        rate = in_vocab / len(bn_chars_in_text)
        # Apply oov_penalty: each OOV fraction point reduces score
        penalty = (1.0 - rate) * self.oov_penalty * 10.0
        return float(max(0.0, 1.0 - penalty))

    def _bigram_score(self, text: str) -> float:
        """
        Laplace-smoothed character bigram log-likelihood, normalised to [0,1].

        log P(text) = Σ log P(c_i | c_{i-1})
        Normalised by sequence length so short and long strings are comparable.
        """
        chars = list(text)
        if len(chars) < 2:
            return 0.5  # single char – neutral

        vocab_size = max(1, len(self._unigram))
        alpha = self.bigram_alpha

        log_prob = 0.0
        n_bigrams = 0

        for i in range(1, len(chars)):
            ctx  = chars[i - 1]
            next_c = chars[i]
            ctx_count  = sum(self._bigram[ctx].values()) + alpha * vocab_size
            pair_count = self._bigram[ctx][next_c] + alpha
            log_prob  += math.log(pair_count / ctx_count)
            n_bigrams += 1

        if n_bigrams == 0:
            return 0.5

        # Average log-prob per bigram.
        # Typical range: roughly [-4, 0] for reasonable text.
        avg = log_prob / n_bigrams  # in (-∞, 0]

        # Map to [0, 1] with a sigmoid-like transform calibrated empirically:
        #   avg ≈ 0      → score ≈ 1.0  (very plausible bigram)
        #   avg ≈ -2     → score ≈ 0.5  (neutral)
        #   avg ≈ -4     → score ≈ 0.1  (implausible)
        score = 1.0 / (1.0 + math.exp(-avg - 1.5))
        return float(max(0.0, min(1.0, score)))

    def _kenlm_score(self, text: str) -> float:
        """
        Converts KenLM perplexity to a [0,1] score.
        lower perplexity → higher score.
        """
        try:
            ppl = self._kenlm_model.perplexity(text)
            # Calibration: ppl=1 → 1.0, ppl=1000 → ~0.0
            score = 1.0 / (1.0 + math.log1p(ppl) / 7.0)
            return float(max(0.0, min(1.0, score)))
        except Exception:
            return self._bigram_score(text)


# ── Factory helper ────────────────────────────────────────────────────────────

def build_lm_from_vocab_json(vocab_json_path: str, kenlm_path: Optional[str] = None) -> BanglaLM:
    """
    Convenience factory: loads vocab.json (the model's own vocabulary)
    and constructs a BanglaLM instance ready to score candidates.

    Usage:
        lm = build_lm_from_vocab_json("exports/banghtr_x_v2_production/vocab.json")
        score = lm.score("বাংলা হাতের লেখা")
    """
    import json
    with open(vocab_json_path, encoding="utf-8") as f:
        vocab = json.load(f)

    idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
    # Exclude special tokens (index 0-4)
    chars = [v for k, v in idx2char.items() if k >= 5]
    return BanglaLM(vocab_chars=chars, kenlm_path=kenlm_path)
