# Contributing to CQUPT AI Assistant

Thanks for your interest in contributing. This document outlines the workflow
and standards for the project.

## Getting Started

1. Fork the repository and clone it locally.
2. Create a branch: `git checkout -b feat/your-feature-name`
3. Install dev dependencies: `pip install -r backend/requirements.txt`
4. Copy `.env.example` to `.env` and set your `DOUBAO_API_KEY`.

## Development Workflow

### Before You Code

- Open an issue describing the bug or feature before writing code.
- For features, include a brief design sketch in the issue.
- Check that the same idea is not already in progress.

### Code Standards

- Python 3.10+. No single-letter variable names except in loops.
- Type hints on all public function signatures.
- Docstrings for modules, classes, and public methods.
- 120-character line limit.
- Use `pathlib.Path` for filesystem operations, not `os.path`.

### What to Work On

The best first contributions:
- Adding CQUPT knowledge documents to `data/documents/`
- Writing new test cases in `backend/eval_framework.py`
- Improving keyword sets for the Chinese psychological safety layer
- Translating error messages to more user-friendly Chinese

### Testing

The pipeline test exercises the full stack in an isolated temporary directory
(does not touch persisted data):

```bash
cd backend
python test_pipeline.py
```

The evaluation framework requires a running server:

```bash
# Terminal 1
cd backend && python main.py

# Terminal 2
cd backend && python eval_framework.py
```

### Commit Messages

Use conventional commits:

```
feat: add 2024 course catalog to knowledge base
fix: handle empty BM25 results gracefully
docs: update safety.yaml comments
```

### Pull Requests

- PR title should match the conventional commit format.
- Link the related issue.
- Confirm `python test_pipeline.py` passes.
- If adding a new feature, include a test case in the eval framework.

## Project Structure Quick Reference

| Directory | Purpose |
|-----------|---------|
| `backend/` | FastAPI server, RAG pipeline, safety guardrails |
| `backend/rag/` | Document ingestion, chunking, FAISS indexing |
| `backend/router/` | Intent classifier and scenario dispatcher |
| `backend/safety/` | Four-layer guardrail modules |
| `config/` | YAML configuration (safety, server) |
| `data/documents/` | CQUPT knowledge source documents |
| `data/store/` | Persisted embeddings, manifests, cache |
| `frontend/` | Chat interface (static HTML/CSS/JS) |

## Questions?

Open a Discussion thread or comment on the relevant issue.
