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
        sep="\t"
    )

    s2 = pd.read_csv(
        "dataset/test/test_source2.tsv",
        sep="\t"
    )

    s3 = pd.read_csv(
        "dataset/test/test_source3.tsv",
        sep="\t"
    )

    print("Generating Candidate Pairs...")

    generator = CandidateGenerator()

    candidates = generator.generate(

        s1,

        s2,

        s3,

        top_k=30

    )

    save_candidate_file(

        candidates,

        "output/candidate_pairs.tsv"

    )

    print("candidate_pairs.tsv generated successfully.")


if __name__ == "__main__":

    main()