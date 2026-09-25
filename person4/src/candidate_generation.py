"""
candidate_generation.py

Generate candidate_pairs.tsv
"""

import pandas as pd

from blocking import generate_candidates

from utils import save_candidate_file


def main():

    print("Loading datasets...")

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

    print("Generating candidates...")

    candidates = generate_candidates(

        s1,

        s2,

        s3

    )

    save_candidate_file(

        candidates,

        "output/candidate_pairs.tsv"

    )


if __name__ == "__main__":

    main()