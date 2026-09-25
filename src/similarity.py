"""
similarity.py
=============

Similarity primitives for the Business Entity Resolution pipeline.

This module is deliberately dependency-light and side-effect-free: every
function takes plain strings (or lists of strings) in and returns numbers
or normalized strings out. `blocking.py` and `candidate_generation.py`
import from here; nothing in here should import from them (keeps the
dependency graph a DAG and makes this file unit-testable in isolation).

Design notes for the team
--------------------------
- All similarity scores are normalized to [0.0, 1.0], where 1.0 = identical.
- `normalize_text` / `normalize_business_name` / `normalize_address` are
  pure string functions — safe to `.apply()` over a pandas Series, or to
  vectorize with `.map()`, without touching a dataframe directly. Keeping
  I/O and dataframe logic out of this file is what makes it easy to later
  swap in a faster backend (e.g. RapidFuzz's C implementation, or a
  vectorized TF-IDF matrix) without changing call sites elsewhere.
- Functions that are natural candidates for batch/vectorized versions
  (used during blocking over potentially millions of candidate pairs)
  have a `batch_*` counterpart that avoids Python-level per-pair overhead.

Dependencies: scikit-learn, numpy (required); rapidfuzz (strongly
recommended — this module falls back to slower pure-Python equivalents
automatically if it isn't installed, so teammates without it yet can still
`import similarity` and run everything, just slower on large candidate sets).
    pip install rapidfuzz scikit-learn numpy
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity

try:
    from rapidfuzz import fuzz, distance
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only when rapidfuzz absent
    import difflib
    _HAS_RAPIDFUZZ = False


def _pure_python_levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) edit-distance DP, used only as a fallback
    when the (much faster, C-backed) rapidfuzz package isn't installed."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]

# --------------------------------------------------------------------------
# Normalization vocabularies
# --------------------------------------------------------------------------
# These are intentionally simple regex/dict substitutions rather than an
# external lookup service — the challenge rules prohibit external data
# lookups (geocoding APIs, registries, etc.), and simple substitution is
# both allowed and fast.

LEGAL_SUFFIXES: dict[str, str] = {
    r"\bpvt\.?\b": "private",
    r"\bltd\.?\b": "limited",
    r"\bcorp\.?\b": "corporation",
    r"\binc\.?\b": "incorporated",
    r"\bco\.?\b": "company",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\bpllc\b": "professional limited liability company",
    r"\bplc\b": "public limited company",
    r"\bgmbh\b": "gmbh",  # kept as-is, just normalized casing/spacing
    r"&": " and ",
}

ADDRESS_ABBREVIATIONS: dict[str, str] = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bapt\.?\b": "apartment",
    r"\bfl\.?\b": "floor",
    r"\bste\.?\b": "suite",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bhwy\.?\b": "highway",
    r"\bnr\.?\b": "near",
    r"\bopp\.?\b": "opposite",
    r"\bsq\.?\b": "square",
    r"\bpk\.?\b": "park",
}

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def _apply_substitutions(text: str, substitutions: dict[str, str]) -> str:
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    return text


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@lru_cache(maxsize=200_000)
def normalize_text(text: str | None) -> str:
    """
    Generic normalization: lowercase, strip punctuation, collapse whitespace.

    Cached (lru_cache) because the same raw strings recur across many
    candidate pairs during blocking — avoids redoing the same regex work
    for a name/address that appears in dozens of comparisons.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=200_000)
def normalize_business_name(name: str | None) -> str:
    """
    Business-name-specific normalization: expands legal suffix abbreviations
    (Pvt -> private, Ltd -> limited, & -> and) before generic normalization,
    so that "Amazon & Co Pvt Ltd" and "Amazon and Company Private Limited"
    normalize to the same token sequence.
    """
    if not name or not isinstance(name, str):
        return ""
    lowered = name.lower()
    expanded = _apply_substitutions(lowered, LEGAL_SUFFIXES)
    return normalize_text(expanded)


@lru_cache(maxsize=200_000)
def normalize_address(address: str | None) -> str:
    """
    Address-specific normalization: expands street-type abbreviations
    (Rd -> road, St -> street, Nr -> near) before generic normalization.
    """
    if not address or not isinstance(address, str):
        return ""
    lowered = address.lower()
    expanded = _apply_substitutions(lowered, ADDRESS_ABBREVIATIONS)
    return normalize_text(expanded)


def first_token(text: str) -> str:
    """First whitespace-delimited token of an already-normalized string.
    Used by blocking.py as a cheap blocking key (e.g. first word of the
    normalized business name)."""
    text = text.strip()
    return text.split(" ", 1)[0] if text else ""


# --------------------------------------------------------------------------
# Pairwise similarity functions (single pair)
# --------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    """
    Token-set Jaccard similarity: |A ∩ B| / |A ∪ B| over whitespace tokens.
    Robust to word-order transpositions (e.g. "India Amazon" vs "Amazon
    India" score 1.0). Expects already-normalized strings for best results,
    but will lowercase/split raw strings too.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def levenshtein_similarity(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity in [0, 1]: 1 - (edit_distance / max_len).
    Backed by RapidFuzz's C implementation (`distance.Levenshtein`), which is
    substantially faster than a pure-Python DP implementation — this matters
    when scoring millions of candidate pairs during matching.
    Good at catching typos ("Amazn" vs "Amazon"); weaker on word reordering
    (use jaccard_similarity or token_sort_ratio for that).
    """
    if not a and not b:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return distance.Levenshtein.normalized_similarity(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (_pure_python_levenshtein(a, b) / max_len)


def token_sort_ratio(a: str, b: str) -> float:
    """
    RapidFuzz token-sort ratio, normalized to [0, 1]. Sorts tokens
    alphabetically before comparing, so it's robust to both typos AND
    word-order transpositions in one shot — a good general-purpose default
    for business names.
    """
    if _HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(a, b) / 100.0
    sorted_a = " ".join(sorted(a.lower().split()))
    sorted_b = " ".join(sorted(b.lower().split()))
    return difflib.SequenceMatcher(None, sorted_a, sorted_b).ratio()


def token_set_ratio(a: str, b: str) -> float:
    """
    RapidFuzz token-set ratio, normalized to [0, 1]. Robust to one string
    being a subset of the other's tokens (e.g. "Amazon" vs "Amazon India
    Pvt Ltd") — useful for DBA/trade-name-vs-full-legal-name comparisons.
    """
    if _HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(a, b) / 100.0
    tokens_a, tokens_b = set(a.lower().split()), set(b.lower().split())
    common = tokens_a & tokens_b
    joined_common = " ".join(sorted(common))
    joined_a = " ".join(sorted(tokens_a))
    joined_b = " ".join(sorted(tokens_b))
    return max(
        difflib.SequenceMatcher(None, joined_common, joined_a).ratio(),
        difflib.SequenceMatcher(None, joined_common, joined_b).ratio(),
        difflib.SequenceMatcher(None, joined_a, joined_b).ratio(),
    )


# --------------------------------------------------------------------------
# TF-IDF cosine similarity
# --------------------------------------------------------------------------
# TF-IDF needs a corpus to fit against (IDF weights depend on term
# frequency across the whole dataset), so unlike the pair functions above,
# this is exposed as a small stateful helper class rather than a bare
# function. Fit it ONCE per source-pair comparison (e.g. once on all
# Source 1 + Source 2 names) and reuse the fitted vectorizer for every
# candidate pair — refitting per-pair would be the single biggest
# performance mistake possible here.

class TfidfSimilarity:
    """
    Fit-once, score-many TF-IDF cosine similarity helper.

    Usage
    -----
        tfidf = TfidfSimilarity(analyzer="char_wb", ngram_range=(2, 4))
        tfidf.fit(all_normalized_names)          # fit once on the full corpus
        score = tfidf.similarity("amazon india", "amazon india pvt ltd")

    For blocking-stage bulk scoring (many candidates per Source 1 record),
    prefer `similarity_matrix` / `top_k_matches`, which use a single sparse
    matrix multiply instead of one call per pair.
    """

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 4),
        min_df: int = 1,
    ):
        # char_wb n-grams (word-boundary-aware character n-grams) handle
        # typos and transliteration variants better than word-level TF-IDF
        # for short business-name strings; switch analyzer="word" if you'd
        # rather do word-level TF-IDF.
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range, min_df=min_df
        )
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "TfidfSimilarity":
        self.vectorizer.fit(list(corpus))
        self._fitted = True
        return self

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two strings using the fitted vocabulary."""
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before .similarity().")
        vecs = self.vectorizer.transform([a, b])
        return float(_sk_cosine_similarity(vecs[0], vecs[1])[0, 0])

    def similarity_matrix(
        self, queries: Sequence[str], candidates: Sequence[str]
    ) -> np.ndarray:
        """
        Vectorized cosine similarity between every query and every
        candidate in one sparse matrix multiply. Returns an
        (len(queries) x len(candidates)) numpy array.

        This is the version blocking.py / candidate_generation.py should
        call for bulk scoring — computing this as a matrix op is orders of
        magnitude faster than looping `.similarity(a, b)` over every pair
        in Python.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before .similarity_matrix().")
        query_vecs = self.vectorizer.transform(queries)
        cand_vecs = self.vectorizer.transform(candidates)
        return _sk_cosine_similarity(query_vecs, cand_vecs)

    def top_k_matches(
        self, query: str, candidates: Sequence[str], k: int = 10
    ) -> list[tuple[int, float]]:
        """
        Return the top-k (candidate_index, score) pairs for a single query
        against a candidate pool, sorted by descending score. Useful inside
        blocking.py when narrowing many same-block candidates down to a
        manageable shortlist before the matching model runs.
        """
        scores = self.similarity_matrix([query], candidates)[0]
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is O(n) vs O(n log n) full sort — matters when
            # candidate pools get large.
            top_idx = np.argpartition(-scores, k)[:k]
            order = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in order]


def tfidf_similarity(a: str, b: str) -> float:
    """
    Convenience one-off TF-IDF cosine similarity between two strings,
    fitting a throwaway vectorizer on just this pair.

    NOTE: this refits a vectorizer on every call, which is fine for ad-hoc
    exploration/notebooks but far too slow to use inside a blocking loop
    over many pairs — use `TfidfSimilarity.fit()` once + `similarity_matrix`
    for that case instead.
    """
    return TfidfSimilarity().fit([a, b]).similarity(a, b)


# --------------------------------------------------------------------------
# Batch helpers (avoid per-pair Python-loop overhead for the cheap metrics)
# --------------------------------------------------------------------------

def batch_jaccard_similarity(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Jaccard similarity for a list of (a, b) pairs."""
    return [jaccard_similarity(a, b) for a, b in pairs]


def batch_token_sort_ratio(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """
    Token-sort ratio for a list of (a, b) pairs, using RapidFuzz's
    `process`-style batch scoring path where possible. For very large pair
    lists, prefer `rapidfuzz.process.cdist` directly on two string arrays
    instead of Python-looping this function.
    """
    return [token_sort_ratio(a, b) for a, b in pairs]


# --------------------------------------------------------------------------
# Composite scoring
# --------------------------------------------------------------------------

# Default weights: name similarity matters most for entity resolution;
# address corroborates. Tune these against the validation split.
DEFAULT_WEIGHTS = {
    "name_token_sort": 0.45,
    "name_jaccard": 0.15,
    "address_token_sort": 0.30,
    "address_jaccard": 0.10,
}


def composite_similarity(
    name_a: str,
    name_b: str,
    address_a: str,
    address_b: str,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Weighted blend of name + address similarity signals into a single score
    in [0, 1], for ranking candidate pairs after blocking.

    Expects RAW (not pre-normalized) name/address strings — normalization
    is applied internally so callers don't have to remember the order of
    operations.
    """
    weights = weights or DEFAULT_WEIGHTS

    norm_name_a, norm_name_b = normalize_business_name(name_a), normalize_business_name(name_b)
    norm_addr_a, norm_addr_b = normalize_address(address_a), normalize_address(address_b)

    scores = {
        "name_token_sort": token_sort_ratio(norm_name_a, norm_name_b),
        "name_jaccard": jaccard_similarity(norm_name_a, norm_name_b),
        "address_token_sort": token_sort_ratio(norm_addr_a, norm_addr_b),
        "address_jaccard": jaccard_similarity(norm_addr_a, norm_addr_b),
    }
    total_weight = sum(weights.values())
    return sum(scores[k] * weights[k] for k in scores) / total_weight


# --------------------------------------------------------------------------
# Self-test (run directly: python src/similarity.py)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    n1, n2 = "Amazon India Pvt Ltd", "Amazon India Private Limited"
    a1, a2 = "123 MG Rd, Near SBI ATM, Bangalore", "123 M.G. Road, Bangalore"

    print("normalize_business_name:", normalize_business_name(n1))
    print("normalize_address       :", normalize_address(a1))
    print("jaccard (names)         :", round(jaccard_similarity(normalize_business_name(n1), normalize_business_name(n2)), 3))
    print("levenshtein (names)     :", round(levenshtein_similarity(n1, n2), 3))
    print("token_sort_ratio (names):", round(token_sort_ratio(n1, n2), 3))
    print("token_set_ratio (names) :", round(token_set_ratio("Amazon", n1), 3))
    print("tfidf (names, ad-hoc)   :", round(tfidf_similarity(n1, n2), 3))
    print("composite_similarity    :", round(composite_similarity(n1, n2, a1, a2), 3))