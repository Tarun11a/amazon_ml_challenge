# Person 4 – Blocking & Candidate Generation

## Amazon ML Challenge – Business Entity Resolution

### Role

Responsible for designing and implementing the **Blocking and Candidate Generation** stage of the entity resolution pipeline.

This module reduces the number of pairwise product comparisons by generating a high-quality set of candidate pairs before similarity scoring and classification.

---

## Current Responsibilities

- Product preprocessing for blocking
- Block key generation
- Blocking pipeline
- Candidate generation
- Candidate pair export
- Blocking pipeline optimization (Phase 2)

---

## Project Structure

```
person4/
├── dataset/
├── experiments/
├── notebooks/
├── output/
├── src/
│   ├── similarity.py
│   ├── block_keys.py
│   ├── blocking.py
│   ├── candidate_generation.py
│   ├── utils.py
│   ├── train_at_scale.py
│   └── predict_at_scale.py
├── team_notes/
├── README.md
└── requirements.txt
```

---

# Completed Work

### similarity.py
- String similarity utilities
- Text comparison helper functions
- Similarity metrics used during blocking

Status: **Completed**

---

### block_keys.py
Implemented block key generation using normalized product attributes.

Status: **Completed**

---

### blocking.py
Implemented the blocking pipeline for grouping products into candidate blocks.

Status: **Completed**

---

### candidate_generation.py
Implemented candidate pair generation from generated blocks.

Current capabilities:

- Generate candidate pairs
- Remove duplicate pairs
- Ignore self-pairs
- Export candidate pairs

Status: **Completed (Phase 1)**

---

### utils.py
Contains helper utilities including:

- File handling
- Data preprocessing helpers
- Candidate file saving utilities

Status: **Completed**

---

## Current Progress

### Phase 1

Goal:

Generate

```
candidate_pairs.tsv
```

Current status:

- Similarity module completed
- Block key generation completed
- Blocking completed
- Candidate generation completed
- Candidate pair export implemented

Remaining:

- End-to-end validation
- Verify generated candidate pairs

---

### Phase 2 (Upcoming)

Performance optimization.

Planned improvements:

- Faster preprocessing
- Efficient block indexing
- Reduced memory usage
- Chunked processing
- Parallel blocking
- Large-scale candidate generation

---

## Pipeline

```
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Block Key Generation
    │
    ▼
Blocking
    │
    ▼
Candidate Generation
    │
    ▼
candidate_pairs.tsv
```

---

## Notes

Current focus is correctness of candidate generation.

Performance optimization will begin only after successful generation of `candidate_pairs.tsv`.

---

## Author

**Tarun Botham**

Role: **Person 4 – Blocking & Candidate Generation**
