# Team Member 2 — Data Analysis and EDA

## A. Dataset structure

All files are tab-separated TSV files. The observed schema is:

| File | Rows | Columns |
|---|---:|---|
| `train_source1.tsv` | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_source2.tsv` | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_source3.tsv` | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | 2,206,821 | `source1_entity_id`, `matched_entity_ids` |

All observed columns were read as strings. There are no phone, email, URL, or domain columns in the actual dataset, so those requested analyses are not applicable.

Training-country distribution:

| File | US | India |
|---|---:|---:|
| Source 1 | 1,323,633 | 883,188 |
| Source 2 | 3,016,817 | 2,017,799 |
| Source 3 | 3,170,056 | 2,115,547 |

The challenge instructions state that France appears in test data, so country handling must remain open-set and must not be hard-coded to only US and India.

## B. Data quality findings

- Source 1 has no observed missing values in the four record fields.
- Source 2 has missing/empty addresses in **3.356%** of records.
- Source 3 has missing/empty addresses in **3.328%** of records.
- Business names are non-empty in the observed files.
- Ground truth has empty `matched_entity_ids` for **5.585%** of Source 1 rows, indicating a substantial singleton population.
- Exact duplicate rows were not observed in the chunk-level duplicate checks. This should be treated as an operational check, not a proof of globally unique full rows.
- The unique-value counts in the generated raw report are fixed-memory estimates, not exact counts. They should not be presented as exact; some estimates exceed the row count because of estimator error.

Text-length summaries:

| File | Name median | Name p95 | Address median | Address p95 |
|---|---:|---:|---:|---:|
| Source 1 | 24 | 37 | 41 | 103 |
| Source 2 | 24 | 40 | 37 | 96 |
| Source 3 | 25 | 41 | 42 | 91 |

Observed formatting signals include punctuation and digits in both names and addresses. Accented characters occur in Source 2 and Source 3 names and addresses; Source 1 contains very few accented address values and no accented names in the measured scan.

## C. Positive-pair patterns

The following results are measured on a reproducible sample of **50,000 positive pairs** from the ground truth:

| Feature | Positive-pair result |
|---|---:|
| Exact normalized name | 21.428% |
| Name token reordering | 4.942% |
| Name punctuation-only difference | 1.122% |
| Name accent/transliteration signal | 3.464% |
| Name capitalization-only difference | 6.232% |
| Exact normalized address | 8.102% |
| Address token reordering | 3.460% |
| Address punctuation-only difference | 0.052% |
| Address accent/transliteration signal | 0.000% |
| Missing address in either record | 4.384% |
| Same country | 100.000% |
| Mean name token Jaccard | 0.6135 |
| Mean address token Jaccard | 0.5979 |

Interpretation:

- Exact normalized name agreement is more common than exact normalized address agreement.
- Name token reordering, case differences, and accent/transliteration signals are material and should be represented explicitly.
- Address still provides substantial complementary evidence through token overlap, even though exact address agreement is uncommon.
- Country consistency is necessary for blocking and scoring, but it is not sufficient for matching.

## D. Diagnostic non-match patterns

A bounded same-country diagnostic sample produced **13,686 non-match pairs**. These are sampled non-matches, not independently verified labels, and should not be treated as a leaderboard metric.

| Feature | Positive pairs | Diagnostic non-matches |
|---|---:|---:|
| Mean name token Jaccard | 0.6135 | 0.0296 |
| Mean address token Jaccard | 0.5979 | 0.0152 |
| Exact normalized name | 21.428% | 0.000% |
| Exact normalized address | 8.102% | 0.000% |
| Missing address in either record | 4.384% | 3.157% |

The strong separation in token overlap supports name and address Jaccard/token-overlap features. Exact equality features are high-precision signals, while fuzzy/token features are needed for noisy positives.

## E. Recommended normalization

1. Case-fold names and addresses.
2. Normalize Unicode consistently; retain an accent-stripped variant for a secondary comparison rather than discarding the original.
3. Normalize punctuation and whitespace.
4. Keep both token-preserving and compact alphanumeric variants.
5. Normalize common legal suffixes and address abbreviations only through an auditable mapping.
6. Preserve missingness indicators instead of converting missing fields into strong matches.
7. Keep country as an open-set string field; do not restrict it to US and India.

## F. Recommended blocking keys

Use multiple independent blocking passes, measured against the same validation split:

- normalized first name token, excluding empty/very common tokens;
- normalized first two name tokens;
- short name prefix only when the token is sufficiently selective;
- address prefix or rare address token;
- country combined with each key;
- a fallback pass for records with missing names or addresses.

Common tokens should be down-weighted or excluded from candidate generation because they can create large candidate blocks. Every blocking change must be evaluated for candidate count, blocking recall, runtime, and memory.

## G. Recommended matching features

For each candidate pair, consider:

- exact normalized name and address matches;
- name token Jaccard and token-sort similarity;
- address token Jaccard and token-sort similarity;
- token-reordering indicators;
- punctuation-only, case-only, and accent/transliteration indicators;
- compact alphanumeric equality;
- name/address length difference and length ratio;
- missingness indicators for each field;
- country equality and source-pair indicators;
- number of blocking keys shared and rarity of shared tokens.

The positive-vs-diagnostic-negative measurements support these features, but they do not by themselves prove a final model improvement. Any feature or threshold change must be compared with the unchanged baseline on the same validation split.

## Reproducibility and limitations

The analysis was performed with chunked TSV reads. Positive-pair statistics use a bounded 50,000-pair sample. Diagnostic negatives use a deterministic same-country sampling procedure and are not ground-truth labels. Full validation metrics, blocking recall, and model comparisons belong to the pipeline/model roles and were not changed by this EDA work.
