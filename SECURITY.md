# Security

CQUPT AI Assistant implements a four-layer safety architecture to ensure responsible AI responses in a university setting. This document describes each layer, how to configure it, and how to report security issues.

## Four-Layer Safety System

### Layer 1: Crisis Detection

Detects self-harm and suicide-related expressions in user input using keyword matching with input normalization (whitespace and punctuation stripped to prevent bypass).

**Three-tier classification:**
| Tier | Label | Behavior |
|------|-------|----------|
| 1 | `immediate` | Blocks the response entirely and returns a crisis intervention message with the national hotline number (`400-161-9995`). |
| 2 | `high_risk` | Allows the response but appends a hotline reminder if not already present in the answer. |
| 3 | `existential` | Logged only (when sensitivity is set to `high`). Opens the door to philosophical exploration without triggering an intervention. |

**Keywords are configurable** in `config/safety.yaml` under `layers.crisis.immediate_keywords` and `layers.crisis.high_risk_keywords`. The sensitivity level (`low`/`medium`/`high`) controls which tiers are active.

### Layer 2: Medical Boundary

Prevents the model from offering medical diagnoses or drug recommendations.

| Sensitivity | Effect |
|-------------|--------|
| `high` (default) | Regex patterns that match diagnosis or medication advice **reject the response wholesale** and return a safe fallback. |
| `medium` | Matches are logged as warnings but the response is still delivered. |
| `low` | Matches are logged at debug level only. |

**Built-in block patterns** (configurable in `config/safety.yaml`):
- `建议…服用…药` — medication advice
- `你(很可能|应该|一定|可能)患有` — diagnostic language

Additional patterns and word blocklists can be added via `layers.medical.block_patterns` and `layers.medical.blocklist_words`.

### Layer 3: Citation Gate

For responses in the `policy` and `data` intents, this layer checks whether the generated answer cites its sources. When `warn_missing_citation: true` (default), uncited responses generate a warning in the application logs so operators can review and improve source attribution.

### Layer 4: LLM Guard

Protects the LLM from prompt injection, jailbreak attempts, roleplay bypass, and repetition/character-flood attacks. This runs **before** the query reaches the answer generator.

**Five detection categories:**
1. **Prompt extraction** — patterns like "tell me your system prompt", "show me your instructions"
2. **Jailbreak** — "ignore all previous instructions", "DAN mode", "from now on you are..."
3. **Roleplay bypass** — "pretend you are an AI without restrictions", "enter developer mode"
4. **Repetition attacks** — the same word repeated >5 times in a query
5. **Special character flooding** — >50% non-alphanumeric, non-punctuation characters in a query >20 chars

When the LLM Guard blocks a request, it returns a polite refusal message instead of the normal response. The guard is implemented in `backend/llm_guard.py` as a pure-pattern class with no external dependencies.

## Authentication

- **JWT (HS256)** with configurable expiry (default 24 hours).
- Password hashing via **bcrypt**.
- `JWT_SECRET` is required at startup — the application refuses to boot if it is not set (no dev fallback).
- Tokens are issued at `/api/auth/login` and `/api/auth/register`.
- Protected endpoints (admin, content management) require a valid token via the `require_user` dependency.
- The `get_current_user` dependency allows optional authentication — unauthenticated users get `None` rather than a 401.

## Configuration

All safety layers are configurable via `config/safety.yaml`. Each layer can be independently enabled or disabled, and its sensitivity tuned. See the inline comments in that file for guidance.

Environment variables:
- `JWT_SECRET` (required) — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `LOG_LEVEL` (default: `INFO`)
- API keys are sourced from environment variables (see `.env.example`)

## Reporting a Vulnerability

If you discover a security issue in CQUPT AI Assistant, please **do not** open a public issue.

**Reporting process:**
1. Email the project maintainer with a description of the vulnerability, steps to reproduce, and any relevant logs or payloads.
2. Allow up to 7 days for an initial response.
3. Once resolved, the fix will be released and credited in the changelog (unless you prefer to remain anonymous).

**Scope:** This process covers the CQUPT AI Assistant application code, its safety guardrails, the LLM integration layer, and the document ingestion pipeline. It does not cover the underlying LLM provider (Doubao/Volcengine) — issues with the model itself should be reported to the provider directly.

## Design Principles

- **Fail open**: Guardrail errors (e.g., safety YAML parse failure, regex timeout) never block a legitimate response. The system falls back to hardcoded defaults.
- **Defense in depth**: No single layer is expected to catch every issue. The four layers complement each other — what slips past LLM Guard may be caught by Crisis Detection; what passes Medical Boundary is still checked by Citation Gate.
- **Configurable, not hardcoded**: All blocklists, patterns, and sensitivity levels live in `config/safety.yaml` so operators can tune them without redeploying code.
