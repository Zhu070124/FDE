# Changelog

All notable changes to the CQUPT AI Assistant project.

## [0.2.0] — 2026-08

### Added

- **Incremental FAISS indexing**: `VectorStore.add_document()` supports manifest-based change detection. A file hash (MD5) is stored in `data/store/manifest.json`; re-indexing is skipped when the file has not changed. New endpoints: `POST /api/admin/documents/incremental` for single-file incremental ingest and `GET /api/admin/manifest` for manifest inspection.
- **Configurable safety YAML**: All four safety layers (Crisis Detection, Medical Boundary, Citation Gate, LLM Guard) are now driven by `config/safety.yaml`. Sensitivity (`low`/`medium`/`high`), keyword blocklists, block patterns, and hotline numbers are configurable without code changes.
- **Query logging + analytics**: `feedback.py` modules — `log_query()` persists every chat request to `query_log` with intent, latency, and response previews. `get_query_analytics()` surfaces top queries, intent distribution, and average latency. Exposed via `GET /api/analytics/queries` and `GET /api/analytics/summary`.
- **Ablation documentation**: `docs/ablation.md` documents the design rationale for hybrid search (BM25 + Doubao embeddings), three-tier crisis detection, multi-label intent routing, and the psychological temperature layer.
- **Docker support**: Multi-stage `Dockerfile` (builder + production) with a non-root `appuser`, health check, and `docker-compose.yml` for one-command deployment.
- **Graceful shutdown**: `lifespan` async context manager validates configuration at startup. The application exits cleanly on config errors rather than failing mid-request.
- **MIT License**: Project is now open-source under the MIT license (`LICENSE` file).

### Fixed

- **httpx connection pool leak**: Replaced ad-hoc `httpx.AsyncClient` instantiation inside `DoubaoEmbedder.embed()` with a unified `httpx.Client` using proper context-manager scoping. Previously, each embedding batch leaked client connections.
- **Logger root fix**: `logger_config.py` now configures the root logger (`logging.getLogger()`) instead of a named logger. This ensures all module-level loggers (e.g., `guardrails`, `vector_store`, `main`) propagate their output to the rotating file handlers.
- **JWT_SECRET mandatory**: The startup check in `auth.py` now raises `RuntimeError` if `JWT_SECRET` is unset, instead of silently falling back to a dev-only default.

### Changed

- **Safety guardrails**: The `Guardrails` class now loads from `safety.yaml` and falls back to hardcoded defaults on parse errors. All four layers (crisis, medical, disclaimer, citation) are individually toggleable and sensitivity-tunable.
- **BM25 + embedding hybrid search**: The retrieval pipeline now runs both vector similarity (Doubao) and BM25 keyword search, merging results with a deduplication pass. BM25 results are weighted at 0.7x relative to vector results.

## [0.1.0] — 2026-07

### Added

- Initial release: RAG-powered student assistant for Chongqing University of Posts and Telecommunications.
- Multi-scene intent routing (policy, exam, psychology).
- Doubao LLM integration for answer generation.
- Four-layer safety guardrails: Crisis Detection, Medical Boundary, LLM Guard, Citation Enforcement.
- JWT-based user authentication (register, login, token refresh).
- Document ingestion pipeline with chunking and BM25 indexing.
- Streaming and non-streaming chat endpoints.
- Multi-step reasoning with query decomposition.
- Self-training reflection loop (hallucination checker, answer grader).
- Eval framework with 14 test cases, 92.9% pass rate.
- Humanistic temperature layer for psychological responses.
- Admin dashboard (users, documents, feedback stats).
- Rate limiting middleware.
