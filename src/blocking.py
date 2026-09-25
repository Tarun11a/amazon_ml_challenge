"""
blocking.py

High-performance blocking engine for
Amazon ML Challenge - Business Entity Resolution

Version: 2.0
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

from similarity import (
    normalize_business_name,
    normalize_address,
    composite_similarity,
)

from block_keys import (
    generate_name_keys,
    generate_address_keys,
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize business names and addresses.
    """

    df = df.copy()

    print("Normalizing business names...")

    df["norm_name"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_business_name)
    )

    print("Normalizing addresses...")

    df["norm_address"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )

    return df


# ============================================================
# BLOCK INDEX
# ============================================================

class BlockIndex:
    """
    Stores multiple blocking indexes.

    Every key maps to a list of candidate row indices.
    """

    def __init__(self):

        self.name_index = defaultdict(list)

        self.address_index = defaultdict(list)

    def add_record(self, idx: int, row: pd.Series):

        country = str(row["country"])

        # -------------------------
        # Name Keys
        # -------------------------

        for key in generate_name_keys(row["norm_name"]):

            self.name_index[(country, key)].append(idx)

        # -------------------------
        # Address Keys
        # -------------------------

        for key in generate_address_keys(row["norm_address"]):

            self.address_index[(country, key)].append(idx)

    def build(self, df: pd.DataFrame):

        print("Building blocking indexes...")

        for idx, row in df.iterrows():

            self.add_record(idx, row)

        print(
            f"Name Blocks    : {len(self.name_index)}"
        )

        print(
            f"Address Blocks : {len(self.address_index)}"
        )


# ============================================================
# BUILD INDEX
# ============================================================

def build_indexes(
        
    source2: pd.DataFrame,
    source3: pd.DataFrame,
):

    source2 = preprocess(source2)

    source3 = preprocess(source3)

    combined = pd.concat(

        [source2, source3],

        ignore_index=True

    )

    block_index = BlockIndex()

    block_index.build(combined)

    return combined, block_index
# ============================================================
# GET CANDIDATE INDICES
# ============================================================

def get_candidate_indices(
    row: pd.Series,
    block_index: BlockIndex,
) -> Set[int]:
    """
    Retrieve candidate row indices using
    multiple blocking keys.
    """

    country = str(row["country"])

    candidate_indices = set()

    # -------------------------
    # Name Blocking
    # -------------------------

    for key in generate_name_keys(row["norm_name"]):

        candidate_indices.update(

            block_index.name_index.get(

                (country, key),

                []

            )

        )

    # -------------------------
    # Address Blocking
    # -------------------------

    for key in generate_address_keys(row["norm_address"]):

        candidate_indices.update(

            block_index.address_index.get(

                (country, key),

                []

            )

        )

    return candidate_indices


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_candidates(
    source_row: pd.Series,
    candidate_df: pd.DataFrame,
):

    if candidate_df.empty:

        return candidate_df

    scores = []

    for _, row in candidate_df.iterrows():

        score = composite_similarity(

            source_row["business_name"],

            row["business_name"],

            source_row["business_address"],

            row["business_address"]

        )

        scores.append(score)

    candidate_df = candidate_df.copy()

    candidate_df["score"] = scores

    candidate_df = candidate_df.sort_values(

        "score",

        ascending=False

    )

    return candidate_df


# ============================================================
# TOP K
# ============================================================

def top_k_candidates(

    ranked_df,

    k=30,

):

    return ranked_df.head(k)


# ============================================================
# MERGE CANDIDATES
# ============================================================

def merge_candidate_sets(

    ranked_df,

):

    return ranked_df["entity_id"].tolist()