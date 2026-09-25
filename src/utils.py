"""
utils.py

Utility functions for the blocking pipeline.
"""

import time
import pandas as pd


def timer():

    return time.time()


def elapsed(start):

    return round(time.time() - start, 2)


def save_candidate_file(candidate_dict, output_path):

    rows = []

    for entity_id, candidates in candidate_dict.items():

        rows.append({

            "source1_entity_id": entity_id,

            "candidate_entity_ids": ",".join(candidates)

        })

    df = pd.DataFrame(rows)

    df.to_csv(

        output_path,

        sep="\t",

        index=False

    )

    print(f"Saved {len(df)} records to {output_path}")