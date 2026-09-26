"""
blocking.py

Production Blocking Engine
Amazon ML Challenge
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Set

import pandas as pd

from similarity import (
    normalize_business_name,
    normalize_address,
    batch_normalize_business_name,
    batch_normalize_address,
    composite_similarity,
)

from block_keys import (
    generate_name_keys,
    generate_address_keys,
)


class CandidateGenerator:

    def __init__(self):
        self.source2 = None
        self.source3 = None
        self.combined = None

        self.name_index = defaultdict(list)
        self.address_index = defaultdict(list)

    # =====================================================
    # PREPROCESS
    # =====================================================

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize large datasets using vectorized pandas string operations.
        This avoids Python-level .map() calls over millions of rows.
        """
        df = df.copy()

        print("Normalizing business names...")
        start = time.time()
        df["norm_name"] = batch_normalize_business_name(
            df["business_name"]
        )
        print(
            f"Business names normalized in "
            f"{time.time() - start:.2f} sec"
        )

        print("Normalizing addresses...")
        start = time.time()
        df["norm_address"] = batch_normalize_address(
            df["business_address"]
        )
        print(
            f"Addresses normalized in "
            f"{time.time() - start:.2f} sec"
        )

        return df

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_sources(self, source2: pd.DataFrame, source3: pd.DataFrame):
        print("Preprocessing Source2...")
        self.source2 = self.preprocess(source2)

        print("Preprocessing Source3...")
        self.source3 = self.preprocess(source3)

        self.combined = pd.concat(
            [self.source2, self.source3],
            ignore_index=True,
        )

    # =====================================================
    # BUILD INDEXES
    # =====================================================

    def build_indexes(self):
        print("Building Name and Address Indexes...")
        start = time.time()

        rows = self.combined[
            ["country", "norm_name", "norm_address"]
        ].itertuples(index=False, name=None)

        for idx, (country, norm_name, norm_address) in enumerate(rows):
            country = str(country)
            for key in generate_name_keys(norm_name):
                self.name_index[(country, key)].append(idx)
            for key in generate_address_keys(norm_address):
                self.address_index[(country, key)].append(idx)

        print(f"Indexes built in {time.time() - start:.2f} sec")
        print("Name Blocks :", len(self.name_index))
        print("Address Blocks :", len(self.address_index))

    # =====================================================
    # RETRIEVE CANDIDATES
    # =====================================================

    def retrieve_candidates(self, row: pd.Series) -> Set[int]:
        country = str(row["country"])
        candidate_ids = set()

        # -----------------------------
        # Name Index
        # -----------------------------
        for key in generate_name_keys(row["norm_name"]):
            candidate_ids.update(
                self.name_index.get((country, key), [])
            )

        # -----------------------------
        # Address Index
        # -----------------------------
        for key in generate_address_keys(row["norm_address"]):
            candidate_ids.update(
                self.address_index.get((country, key), [])
            )

        return candidate_ids

    # =====================================================
    # FAST FILTER
    # =====================================================

    def fast_filter(
        self,
        source_row: pd.Series,
        candidate_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidate_df.empty:
            return candidate_df

        source_len = len(source_row["norm_name"])
        min_len = max(3, source_len // 2)
        max_len = source_len * 2

        filtered = candidate_df[
            candidate_df["norm_name"].str.len().between(min_len, max_len)
        ]

        return filtered

    # =====================================================
    # RANK CANDIDATES
    # =====================================================

    def rank_candidates(
        self,
        source_row: pd.Series,
        candidate_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidate_df.empty:
            return candidate_df

        scores = []
        for _, row in candidate_df.iterrows():
            score = composite_similarity(
                source_row["business_name"],
                row["business_name"],
                source_row["business_address"],
                row["business_address"],
            )
            scores.append(score)

        candidate_df = candidate_df.copy()
        candidate_df["score"] = scores
        candidate_df = candidate_df.sort_values(by="score", ascending=False)

        return candidate_df

    # =====================================================
    # TOP K
    # =====================================================

    def select_top_k(self, ranked_df: pd.DataFrame, k: int = 30) -> pd.DataFrame:
        if ranked_df.empty:
            return ranked_df
        return ranked_df.head(k)

    # =====================================================
    # GENERATE CANDIDATES
    # =====================================================

    def generate(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
        top_k: int = 30,
    ) -> Dict[str, List[str]]:
        print("Loading Sources...")
        self.load_sources(source2, source3)

        print("Building Indexes...")
        self.build_indexes()

        print("Preprocessing Source1...")
        source1 = self.preprocess(source1)

        results = {}
        total_candidates = 0
        max_candidates = 0
        min_candidates = 999999

        print("Generating Candidate Pairs...")
        for idx, row in source1.iterrows():
            candidate_ids = self.retrieve_candidates(row)

            if len(candidate_ids) == 0:
                results[row["entity_id"]] = []
                continue

            candidate_df = self.combined.iloc[list(candidate_ids)]
            candidate_df = self.fast_filter(row, candidate_df)
            candidate_df = self.rank_candidates(row, candidate_df)
            candidate_df = self.select_top_k(candidate_df, top_k)

            ids = candidate_df["entity_id"].tolist()
            results[row["entity_id"]] = ids

            total_candidates += len(ids)
            max_candidates = max(max_candidates, len(ids))
            min_candidates = min(min_candidates, len(ids))

            if idx % 1000 == 0:
                print(f"{idx} Source1 Records Processed")

        avg_candidates = total_candidates / len(source1)

        print("\n==============================")
        print("Blocking Statistics")
        print("==============================")
        print("Average Candidates :", round(avg_candidates, 2))
        print("Maximum Candidates :", max_candidates)
        print("Minimum Candidates :", min_candidates)
        print("==============================\n")

        return results


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":
    print("Loading Training Data...")

    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")

    print("Source1 :", len(s1))
    print("Source2 :", len(s2))
    print("Source3 :", len(s3))

    generator = CandidateGenerator()
    candidates = generator.generate(s1, s2, s3, top_k=30)

    first_key = next(iter(candidates))
    print("\nExample Source1:", first_key)
    print("Candidates:", candidates[first_key][:10])
