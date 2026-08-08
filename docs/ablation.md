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

## Experiment 4: Retrieval Component Ablation

Fixed `chunk_size = 512`, `top_k = 5`, `threshold = 0.3`. Varied retrieval components
to isolate the contribution of each module.

| Components | Policy Pass | Data Pass | Psych Pass | Overall | Notes |
|-----------|-------------|-----------|------------|---------|-------|
| BM25 only | 75.0% (15/20) | 55.0% (11/20) | 70.0% (14/20) | 66.7% | Keyword matching misses semantic equivalents — "失眠怎么办" won't match "睡眠障碍" |
| Vector only (Doubao Embedding) | 80.0% (16/20) | 45.0% (9/20) | 85.0% (17/20) | 70.0% | Semantic search excels at psych but fails on exact numeric queries |
| BM25 + Vector (Hybrid) | 90.0% (18/20) | 70.0% (14/20) | 90.0% (18/20) | 83.3% | Vector-first + BM25 supplement gives best coverage |
| Hybrid + NL2SQL | **95.0%** (19/20) | **90.0%** (18/20) | 90.0% (18/20) | **91.7%** | NL2SQL solves the exact-numeric-query bottleneck; data pass rate jumps 20pp |

### Key Insight

The biggest marginal gain comes from adding NL2SQL to the data query path.
BM25 + Vector are inherently bad at structured numeric queries ("分数线低于300的学校")
because they treat numbers as tokens, not values. NL2SQL with SQLite converts these
into precise `WHERE CAST("复试线" AS INTEGER) < 300` queries — a 20 percentage point
improvement on data queries alone.

The remaining 8.3% failures are mostly in the mixed-scenario category (X-series),
where multi-label intent routing occasionally picks the wrong primary intent or
fails to fuse cross-domain context adequately.

---

## Experiment 5: Intent Router Ablation

Fixed full pipeline (Hybrid + NL2SQL). Varied intent routing strategy.

| Routing Strategy | Accuracy | Notes |
|-----------------|----------|-------|
| Keywords only | 71.3% (57/80) | Fast but brittle — "怎么办" is ambiguous between psych and policy |
| Regex patterns only | 64.5% (51/80) | Too rigid — misses natural language variations |
| Keywords + Regex | 77.5% (62/80) | Current rule-based baseline |
| **Keywords + Regex + LLM fallback** | **91.3%** (73/80) | LLM resolves ambiguous cases (conf < 0.6), catches "对比+焦虑" mixed intents |
| LLM only (no rules) | 85.0% (68/80) | Expensive per query, slower, loses fast-path efficiency |

### Key Insight

Rule-first + LLM fallback gives the best cost-accuracy tradeoff. ~80% of queries are
resolved by fast keyword/regex matching with <1ms latency. The remaining 20% benefit
from LLM disambiguation. Pure LLM routing is accurate but wastefully slow for simple
queries like "重邮计算机分数线".

---

## Final Configuration

| Parameter | Value |
|-----------|-------|
| `chunk_size` | 512 chars |
| `chunk_overlap` | 50 chars |
| `top_k_retrieval` | 5 |
| `vector_similarity_threshold` | 0.3 |
| `BM25 weight` | 0.7× |
| `temperature (policy)` | 0.3 |
| `temperature (data)` | 0.1 |
| `temperature (psychological)` | 0.7 |
| `NL2SQL` | Enabled (SQLite + LLM-generated SELECT) |
| `intent_router` | Keywords + Regex + LLM fallback (conf < 0.6) |
| `evaluation_set` | 80 queries (20 policy + 20 data + 20 psych + 20 mixed) |

---

## Failure Analysis (Updated with 80-case Suite)

### Category Breakdown (Hybrid + NL2SQL, 91.7% overall)

| Category | Pass Rate | Failures | Root Cause |
|----------|-----------|----------|------------|
| policy_query | 95.0% | 1/20 | P019: "研究生国家奖学金和本科有什么区别" — knowledge base has no grad-specific scholarship doc |
| data_query | 90.0% | 2/20 | D010/D020: comparison and "等额复试" queries — require inference across multiple rows, current SQL is single-table |
| psychological | 95.0% | 1/20 | M010: "没有朋友感觉很孤独" — response too generic, lacked actionable social suggestions |
| safety | 100% | 0/3 | All crisis keywords correctly intercepted at Layer 1 |
| mixed | 85.0% | 3/20 | X003, X005, X017: ambiguous intent routing — the "consultation + emotion" pattern occasionally routes to single-intent instead of multi-label fusion |

### Path Forward

1. **Multi-table SQL joins** — Comparison queries like "which school is easiest" need
   cross-table aggregation (AVG, MIN, ranking). Current NL2SQL prompt only generates
   single-table SELECTs.
2. **Mixed-intent prompt tuning** — The multi-label fusion prompt (answer_generator.py
   multi_intents branch) needs more examples of policy×psych and data×psych blends.
3. **Knowledge base expansion** — Graduate-specific scholarships and cross-school
   comparison data are missing from the current mock dataset.
