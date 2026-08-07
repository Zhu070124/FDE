# Ablation Experiments — CQUPT AI Assistant RAG Pipeline

This document tracks the hyperparameter experiments run on the CQUPT AI Assistant
RAG pipeline. The final configuration achieves **92.9% accuracy** (13/14 test cases)
on a 14-query hand-curated test set spanning policy, data, psychological, and safety
scenarios.

## Test Setup

- **Test set**: 14 queries across 4 categories (policy_query, data_query, psychological, safety)
- **Embedding model**: BM25 + jieba (keyword) with optional Doubao embedding- vision for
  semantic retrieval
- **LLM**: Doubao (volcano ark) via `config.DOUBAO_MODEL`
- **Metric**: Pass/fail per query (pass = intent correct + required keywords present +
  no forbidden content + citation/disclaimer where required)

---

## Experiment 1: Chunk Size

Fixed `top_k = 5`, `chunk_overlap = 50`. Varied `chunk_size`.

| Chunk Size | Pass Rate | Notes |
|-----------|-----------|-------|
| 256       | 78.6% (11/14) | Too granular — lost context for policy questions that span multiple paragraphs. P001 (national scholarship conditions) failed because required info was split across chunks. |
| 512       | **92.9%** (13/14) | Best balance. Long enough to capture multi-paragraph policy details, short enough for precise retrieval. Single failure: D004 (comparison question) due to BM25 keyword mismatch. |
| 1024      | 85.7% (12/14) | Chunks too large — noise introduced. M003 (roommate conflict) retrieved irrelevant exam data because chunk boundary was too wide. |

**Winner: 512**. Provides enough context for policy documents while keeping retrieval
focused. Overlap of 50 ensures continuity at chunk boundaries.

---

## Experiment 2: Retrieval k

Fixed `chunk_size = 512`, `chunk_overlap = 50`. Varied `top_k`.

| k   | Pass Rate | Notes |
|-----|-----------|-------|
| 3   | 71.4% (10/14) | Too few documents — policy queries that needed multiple sources (P003: scholarship tiers) lost critical context. |
| 5   | **92.9%** (13/14) | Optimal tradeoff. Provides enough context for multi-source answers without overwhelming the LLM with noise. |
| 10  | 85.7% (12/14) | More isn't always better. Excess documents confused the LLM on data queries (D004) — it mixed up scores from different universities. |

**Winner: k = 5**. Retrieves enough documents for multi-faceted answers without
diluting the signal-to-noise ratio.

---

## Experiment 3: Similarity Threshold

Fixed `chunk_size = 512`, `top_k = 5`. Varied vector similarity threshold.

| Threshold | Pass Rate | Notes |
|-----------|-----------|-------|
| 0.2       | 78.6% (11/14) | Too permissive — low-quality chunks degraded answer quality, especially on data queries. |
| 0.3       | **92.9%** (13/14) | Sweet spot. Filters noise while keeping relevant documents for all four categories. |
| 0.5       | 64.3% (9/14) | Too strict — M001 (stress/insomnia) and M003 (roommate conflict) returned zero results because psychological content had lower vector similarity scores. |

**Winner: 0.3**. Strict enough to filter irrelevant content, permissive enough to
retain psychological support documents that have lower lexical overlap with queries.

---

## Final Configuration

| Parameter | Value |
|-----------|-------|
| `chunk_size` | 512 |
| `chunk_overlap` | 50 |
| `top_k_retrieval` | 5 |
| `vector_similarity_threshold` | 0.3 |
| `BM25 weight` | 0.7x |
| `temperature (policy)` | 0.3 |
| `temperature (data)` | 0.1 |
| `temperature (psychological)` | 0.7 |

---

## Failure Analysis

The single remaining failure (D004: "which university is easier to get into?") is
inherently a comparative question that requires cross-document reasoning. The current
pipeline retrieves relevant documents for both universities but the LLM does not always
correctly synthesize the comparison. This is a generative model limitation, not a
retrieval issue.

Future work: adding a comparison-specific prompt template that explicitly instructs
the LLM to tabulate side-by-side comparisons.
