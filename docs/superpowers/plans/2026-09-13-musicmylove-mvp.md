# MusicMyLove MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a one-screen app that turns five real recording selections into deterministic, diverse, honestly evaluated recommendations.

**Architecture:** A Next.js application proxies current official music APIs and shares pure TypeScript retrieval/ranking modules with tests and live-smoke tooling. Small Python scripts collect cached user statistics, persist user-level splits, train/evaluate a tiny BPR MLP, and export JSON weights consumed by TypeScript.

**Tech Stack:** Next.js 15+, React 19, TypeScript 5, Vitest, Testing Library, Playwright, Python 3.11+, PyTorch, pytest

**Spec:** `docs/superpowers/specs/2026-09-13-musicmylove-mvp-design.md`

## Global Constraints

- Exactly five selected seeds automatically trigger recommendations; there is no Generate button.
- Production ranking is pure TypeScript and uses no LLM, Spotify ML input, audio analysis, database, account, or server-side Python.
- Candidate features and ranking are invariant to every permutation of the five seeds.
- Results exclude seeds and duplicate MBIDs and contain at most two tracks per normalized artist.
- External calls have timeouts, bounded retry for 429/5xx/network errors, schema validation, descriptive User-Agent, caching, and stale fallback.
- User-level split seed and manifest are persisted before tuning; test is frozen and baselines remain honest.
- Downloaded datasets and API caches are ignored; weights, fixtures, evaluation, and verification reports are committed.

---

### Task 1: Project shell and ranking domain

**Files:**
- Create: `package.json`, `tsconfig.json`, `next.config.ts`, `vitest.config.ts`, `.gitignore`, `AGENTS.md`
- Create: `src/lib/types.ts`, `src/lib/features.ts`, `src/lib/ranking.ts`
- Test: `tests/unit/ranking.test.ts`, `tests/fixtures/similar.json`

**Interfaces:**
- Produces: `mergeCandidates(seeds, lists): CandidateEvidence[]`, `buildFeatures(candidate, seeds): number[]`, `rankCandidates(seeds, lists, ranker): RankedTrack[]`, `diversify(ranked, limit): RankedTrack[]`.

- [ ] Write tests covering merge/dedup, seed exclusion, independent score/rank normalization, exact 17-feature construction, deterministic ties, and two-per-artist diversity.
- [ ] Run `npm test -- tests/unit/ranking.test.ts` and confirm failure because ranking modules do not exist.
- [ ] Implement immutable types, per-list max-score and reciprocal-rank normalization, sorted order-independent 17-value feature construction (including total RRF evidence), max/RRF scorers, stable sorting, explanations, and greedy diversity.
- [ ] Run the focused test and `npm run typecheck`; confirm both pass.
- [ ] Commit with `feat: add deterministic recommendation core`.

### Task 2: JSON MLP and cross-language parity

**Files:**
- Create: `src/lib/mlp.ts`, `ml/model.json`, `ml/model.py`, `scripts/check_parity.py`, `scripts/ts_parity.ts`
- Test: `tests/unit/mlp.test.ts`, `tests/python/test_model.py`

**Interfaces:**
- Consumes: 17-element vectors from `buildFeatures`.
- Produces: `loadModel(): ModelArtifact`, `forward(features, artifact): number`; Python `TinyRanker.forward` and JSON export with `layers`, `activation`, `feature_names`, and `production_ranker`.

- [ ] Write TypeScript and Python tests for dimensions, ReLU forward pass, and malformed artifacts.
- [ ] Run `npm test -- tests/unit/mlp.test.ts` and `python -m pytest tests/python/test_model.py`; observe missing-module failures.
- [ ] Implement identical float64-compatible dense/ReLU/dense/ReLU/dense inference and a deterministic starter artifact.
- [ ] Implement parity scripts that generate at least 32 seeded vectors, obtain TypeScript scores, and fail above `1e-9` maximum absolute error.
- [ ] Run unit tests and `python scripts/check_parity.py`; confirm pass and record the observed maximum error.
- [ ] Commit with `feat: add portable neural reranker`.

### Task 3: Resilient official API clients and caches

**Files:**
- Create: `src/lib/http.ts`, `src/lib/listenbrainz.ts`, `src/lib/cache.ts`
- Create: `app/api/search/route.ts`, `app/api/similar/[mbid]/route.ts`, `app/api/spotify/[mbid]/route.ts`
- Test: `tests/unit/http.test.ts`, `tests/unit/cache.test.ts`, `tests/unit/api-contracts.test.ts`

**Interfaces:**
- Produces: `fetchWithRetry(url, options)`, `searchRecordings(query, signal)`, `fetchSimilar(mbid, signal)`, `lookupSpotify(mbid, signal)`, `VersionedCache<T>`.

- [ ] Write mocked tests for success schemas, malformed responses, AbortSignal timeout, Retry-After/429, 5xx bounded retry, non-retryable 4xx, cache versioning, TTL, and stale reads.
- [ ] Run the API test files and observe failures for absent modules.
- [ ] Implement strict hand-written parsers and a three-attempt exponential-backoff client with jitter injection, rate-limit delay support, and five-second timeout.
- [ ] Implement server memory cache and browser localStorage cache with fresh/stale states; routes return stale data with an `x-cache-stale` marker when refresh fails.
- [ ] Run focused tests, typecheck, and lint; confirm pass.
- [ ] Commit with `feat: integrate resilient music data APIs`.

### Task 4: Automatic recommendation state and one-screen UI

**Files:**
- Create: `app/layout.tsx`, `app/page.tsx`, `app/globals.css`, `src/components/MusicRecommender.tsx`, `src/hooks/useMusicRecommender.ts`
- Test: `tests/unit/recommender-state.test.tsx`, `tests/e2e/happy-path.spec.ts`, `playwright.config.ts`

**Interfaces:**
- Consumes: API routes, `rankCandidates`, MLP artifact, cache.
- Produces: `useMusicRecommender()` actions/state and accessible search/seed/result UI.

- [ ] Write component tests proving 300 ms debounce, stale search cancellation, immediate similarity prefetch per seed, duplicate rejection, `n / 5`, automatic generation exactly at five, local feedback, dislike promotion, and Spotify fallback.
- [ ] Run the focused tests and observe missing-component failures.
- [ ] Implement the reducer/hook with an injectable API and storage boundary, allowing partial seed-list failure and background retry.
- [ ] Implement the responsive single page and recommendation cards with the exact required copy and no Generate button.
- [ ] Add an intercepted Playwright happy path that selects five results and verifies twenty valid cards.
- [ ] Run component tests and Playwright; confirm pass.
- [ ] Commit with `feat: build five-song recommendation experience`.

### Task 5: Honest data collection and split pipeline

**Files:**
- Create: `requirements.txt`, `ml/config.json`, `ml/data.py`, `scripts/collect.py`, `scripts/make_fixture_data.py`
- Create: `ml/splits.json`, `tests/fixtures/users.json`
- Test: `tests/python/test_data.py`

**Interfaces:**
- Produces: cached `UserProfile` JSON, `create_user_splits(usernames, seed)`, and `make_examples(profile, similarity_cache, seed)`.

- [ ] Write pytest cases proving user-disjoint deterministic splits, target/seed separation, repeated-listen filtering, log-count weighting, known-positive exclusion from negatives, and no post-creation split rewriting.
- [ ] Run `python -m pytest tests/python/test_data.py` and observe missing-module failures.
- [ ] Implement a conservative sequential collector for a supplied username file, honoring ListenBrainz rate headers, retrying safely, and atomically caching useful responses in ignored `data/cache/`.
- [ ] Implement fixed-seed split and example generation, plus a compact deterministic fixture generator that exercises the exact pipeline without masquerading as target-scale evaluation.
- [ ] Run the Python tests and fixture generator; confirm pass.
- [ ] Commit with `feat: add reproducible listening-data pipeline`.

### Task 6: Train, validate, freeze, and evaluate

**Files:**
- Create: `ml/train.py`, `ml/evaluate.py`, `scripts/train.py`, `scripts/evaluate.py`
- Create/Modify: `ml/model.json`, `reports/evaluation.json`
- Test: `tests/python/test_evaluation.py`, `tests/python/test_training.py`

**Interfaces:**
- Consumes: split manifest, cached profiles/similarity, BPR pairs.
- Produces: selected JSON artifact and metrics for `max_similarity`, `rrf`, `neural`, and any validation-approved ensemble.

- [ ] Write tests for BPR pair orientation, convergence on a separable fixture, retrieval-vs-ranking metric separation, Recall/NDCG/HitRate formulas, artist diversity, candidate-pool mean, and deterministic paired bootstrap intervals.
- [ ] Run focused pytest and observe missing-module failures.
- [ ] Implement 17→16→8→1 PyTorch training with fixed seeds, Adam, pairwise softplus loss, and validation-only selection/early stopping.
- [ ] Implement frozen-test evaluation for all rankers, select production from validation metrics, compute required test metrics and paired bootstrap delta, and serialize actual results.
- [ ] Run fixture training once, then frozen fixture evaluation once; inspect output for finite values and honest fixture labeling.
- [ ] Re-run parity after exported weights and confirm the TypeScript scorer still matches.
- [ ] Commit with `feat: train and evaluate tiny reranker`.

### Task 7: Live production-path smoke and failure hardening

**Files:**
- Create: `scripts/live-smoke.ts`
- Test: `tests/unit/live-path.test.ts`

**Interfaces:**
- Consumes: five fixed real MBIDs supplied only as smoke inputs, public API clients, ranking, diversity, Spotify fallback.
- Produces: JSON smoke summary with candidate/result counts and invariant checks.

- [ ] Write a fixture-backed production-path test for one failed similarity seed, fewer-than-twenty behavior, stale fallback, exact-or-search Spotify action, exclusions, dedup, artist cap, and reordered-seed identity.
- [ ] Run the test and observe missing live-path orchestration failure.
- [ ] Implement the shared orchestration used by both runtime and smoke, resolving Spotify only for final candidates and expanding through the officially listed alternate similarity algorithm only when necessary.
- [ ] Run all deterministic tests, then `npm run smoke:live` against current APIs; debug any schema or outage failures without weakening assertions.
- [ ] Commit with `test: verify live recommendation path`.

### Task 8: Documentation and final verification

**Files:**
- Create: `README.md`, `reports/verification.md`
- Modify: `AGENTS.md`, `reports/evaluation.json`

**Interfaces:**
- Produces: exact install/collect/train/evaluate/dev/test/smoke/build commands and observed evidence.

- [ ] Document architecture, cache/data policy, commands, current official endpoint assumptions, evaluation interpretation, and external-retrieval leakage limitation.
- [ ] Run `npm test`, `python -m pytest`, `python scripts/check_parity.py`, `npm run test:e2e`, `npm run smoke:live`, and `npm run build`.
- [ ] Record exact observed counts, metrics, parity error, test totals, build result, live-smoke result, download size, usable/split users, selected ranker, and limitations in the two report files.
- [ ] Run `git diff --check`, inspect `git status`, and verify no cache/download artifacts are tracked.
- [ ] Commit with `docs: record verified MVP results`.
