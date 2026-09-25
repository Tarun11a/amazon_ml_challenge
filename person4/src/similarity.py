"""
similarity.py
Version 2.0

High-performance similarity utilities for
Amazon ML Challenge – Business Entity Resolution

Author: Team Person 4
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ============================================================
# Regex
# ============================================================

SPACE_RE = re.compile(r"\s+")
NON_ALPHA = re.compile(r"[^a-z0-9\s]")


# ============================================================
# Dictionaries
# ============================================================

LEGAL_SUFFIX = {

    "pvt":"private",
    "pvt.":"private",

    "ltd":"limited",
    "ltd.":"limited",

    "co":"company",
    "co.":"company",

    "corp":"corporation",
    "corp.":"corporation",

    "inc":"incorporated",
    "inc.":"incorporated",

    "&":"and"

}


ADDRESS_MAP = {

    "rd":"road",
    "rd.":"road",

    "st":"street",
    "st.":"street",

    "ave":"avenue",

    "blvd":"boulevard",

    "apt":"apartment",

    "nr":"near",

    "opp":"opposite"

}


# ============================================================
# Basic Cleaning
# ============================================================

@lru_cache(maxsize=500000)
def normalize_text(text):

    if text is None:
        return ""

    text=str(text).lower()

    text=NON_ALPHA.sub(" ",text)

    text=SPACE_RE.sub(" ",text)

    return text.strip()


# ============================================================
# Business Name Cleaning
# ============================================================

@lru_cache(maxsize=500000)
def normalize_business_name(name):

    if not name:
        return ""

    name=normalize_text(name)

    words=[]

    for w in name.split():

        words.append(
            LEGAL_SUFFIX.get(w,w)
        )

    return " ".join(words)


# ============================================================
# Address Cleaning
# ============================================================

@lru_cache(maxsize=500000)
def normalize_address(address):

    if not address:
        return ""

    address=normalize_text(address)

    words=[]

    for w in address.split():

        words.append(
            ADDRESS_MAP.get(w,w)
        )

    return " ".join(words)


# ============================================================
# Blocking Keys
# ============================================================

def first_token(text):

    if not text:

        return ""

    return text.split()[0]


def first_two_tokens(text):

    if not text:

        return ""

    x=text.split()

    return " ".join(x[:2])


def first_three_char(text):

    if not text:

        return ""

    return text[:3]


def address_prefix(text):

    if not text:

        return ""

    return text[:8]
# ============================================================
# Jaccard Similarity
# ============================================================

def jaccard_similarity(a: str, b: str) -> float:

    sa = set(normalize_text(a).split())
    sb = set(normalize_text(b).split())

    if len(sa) == 0 and len(sb) == 0:
        return 1.0

    if len(sa) == 0 or len(sb) == 0:
        return 0.0

    return len(sa & sb) / len(sa | sb)


# ============================================================
# RapidFuzz Similarity
# ============================================================

def token_sort_similarity(a: str, b: str) -> float:

    a = normalize_business_name(a)
    b = normalize_business_name(b)

    if RAPIDFUZZ_AVAILABLE:

        return fuzz.token_sort_ratio(a, b) / 100.0

    return jaccard_similarity(a, b)


def token_set_similarity(a: str, b: str) -> float:

    a = normalize_business_name(a)
    b = normalize_business_name(b)

    if RAPIDFUZZ_AVAILABLE:

        return fuzz.token_set_ratio(a, b) / 100.0

    return jaccard_similarity(a, b)


def partial_similarity(a: str, b: str) -> float:

    a = normalize_business_name(a)
    b = normalize_business_name(b)

    if RAPIDFUZZ_AVAILABLE:

        return fuzz.partial_ratio(a, b) / 100.0

    return jaccard_similarity(a, b)


# ============================================================
# Address Similarity
# ============================================================

def address_similarity(a: str, b: str) -> float:

    a = normalize_address(a)
    b = normalize_address(b)

    if RAPIDFUZZ_AVAILABLE:

        return fuzz.token_sort_ratio(a, b) / 100.0

    return jaccard_similarity(a, b)


# ============================================================
# TF-IDF Similarity
# ============================================================

class TFIDFSimilarity:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(

            analyzer="char_wb",

            ngram_range=(2,4),

            lowercase=True

        )

        self.fitted=False


    def fit(self, corpus):

        corpus=[normalize_text(x) for x in corpus]

        self.vectorizer.fit(corpus)

        self.fitted=True


    def similarity(self,a,b):

        if not self.fitted:

            raise RuntimeError("TF-IDF model not fitted.")

        vectors=self.vectorizer.transform(

            [

                normalize_text(a),

                normalize_text(b)

            ]

        )

        return cosine_similarity(

            vectors[0],

            vectors[1]

        )[0][0]


# ============================================================
# Composite Similarity
# ============================================================

def composite_similarity(

        name1,

        name2,

        addr1,

        addr2

):

    name_score = (

        0.45 * token_sort_similarity(name1,name2)

        +

        0.25 * token_set_similarity(name1,name2)

        +

        0.10 * partial_similarity(name1,name2)

    )

    address_score = (

        0.20 * address_similarity(addr1,addr2)

    )

    return name_score + address_score
# ============================================================
# Batch Similarity Functions
# ============================================================

def batch_token_sort(names1, names2):
    """
    Batch token sort similarity.
    """

    scores = []

    for a, b in zip(names1, names2):

        scores.append(
            token_sort_similarity(a, b)
        )

    return scores


def batch_token_set(names1, names2):

    scores = []

    for a, b in zip(names1, names2):

        scores.append(
            token_set_similarity(a, b)
        )

    return scores


def batch_jaccard(names1, names2):

    scores = []

    for a, b in zip(names1, names2):

        scores.append(
            jaccard_similarity(a, b)
        )

    return scores


# ============================================================
# Candidate Ranking
# ============================================================

def rank_candidates(

    source_name,

    source_address,

    candidate_df

):
    """
    Rank all candidates by composite similarity.
    """

    scores = []

    for _, row in candidate_df.iterrows():

        score = composite_similarity(

            source_name,

            row["business_name"],

            source_address,

            row["business_address"]

        )

        scores.append(score)

    candidate_df = candidate_df.copy()

    candidate_df["score"] = scores

    candidate_df = candidate_df.sort_values(

        by="score",

        ascending=False

    )

    return candidate_df


# ============================================================
# Top K Candidates
# ============================================================

def top_k_candidates(

    source_name,

    source_address,

    candidate_df,

    k=20

):

    ranked = rank_candidates(

        source_name,

        source_address,

        candidate_df

    )

    return ranked.head(k)


# ============================================================
# Threshold Filtering
# ============================================================

def filter_candidates(

    ranked_df,

    threshold=0.75

):

    return ranked_df[

        ranked_df["score"] >= threshold

    ]


# ============================================================
# Utility
# ============================================================

def similarity_report(

    name1,

    name2,

    addr1,

    addr2

):

    return {

        "token_sort":

            token_sort_similarity(name1,name2),

        "token_set":

            token_set_similarity(name1,name2),

        "partial":

            partial_similarity(name1,name2),

        "jaccard":

            jaccard_similarity(name1,name2),

        "address":

            address_similarity(addr1,addr2),

        "composite":

            composite_similarity(

                name1,

                name2,

                addr1,

                addr2

            )

    }


# ============================================================
# Self Test
# ============================================================

if __name__ == "__main__":

    n1 = "Amazon India Pvt Ltd"

    n2 = "Amazon India Private Limited"

    a1 = "123 MG Rd Bangalore"

    a2 = "123 MG Road Bengaluru"

    print("\n=========== Similarity Test ===========\n")

    report = similarity_report(

        n1,

        n2,

        a1,

        a2

    )

    for k, v in report.items():

        print(f"{k:15} : {round(v,4)}")

    print("\n=======================================\n")