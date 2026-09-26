"""
candidate_generation.py

Generate candidate_pairs.tsv
"""

print("candidate_generation.py started")

import pandas as pd

from blocking import CandidateGenerator
from utils import save_candidate_file


def main():

    print("Loading Test Dataset...")

    s1 = pd.read_csv(
        "dataset/test/test_source1.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=10000,
    )

    s2 = pd.read_csv(
        "dataset/test/test_source2.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=10000,
    )

    s3 = pd.read_csv(
        "dataset/test/test_source3.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=10000,
    )

    print("Source1 :", len(s1))
    print("Source2 :", len(s2))
    print("Source3 :", len(s3))

    print("Generating Candidate Pairs...")

    generator = CandidateGenerator()

    candidates = generator.generate(
        s1,
        s2,
        s3,
        top_k=30
    )

    rows_processed = len(candidates)
    rows_with_candidates = sum(1 for ids in candidates.values() if len(ids) > 0)
    total_candidate_ids = sum(len(ids) for ids in candidates.values())

    print("Source1 rows processed :", rows_processed)
    print("Source1 rows with >=1 candidate :", rows_with_candidates)
    print("Source1 rows with 0 candidates (singletons) :", rows_processed - rows_with_candidates)
    print("Total candidate IDs retained :", total_candidate_ids)

    output_path = "output/candidate_pairs.tsv"

    save_candidate_file(
        candidates,
        output_path
    )

    output_row_count = len(candidates)
    print("Final output path :", output_path)
    print("Final output row count :", output_row_count)
    print("candidate_pairs.tsv generated successfully.")


if __name__ == "__main__":
    main()
