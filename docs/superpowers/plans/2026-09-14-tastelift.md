# TasteLift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MusicMyLove so 5–200+ JSON likes produce an inspected 40-track TasteLift discovery slate.

**Architecture:** Add robust seed resolution and large evidence-preserving retrieval around the existing ListenBrainz adapters, train a four-head permutation-invariant TasteLift encoder on cached histories, and feed its lift/head/support outputs into the existing residual ranker and deterministic slate builder. Export weights/data dictionaries to JSON for pure TypeScript inference.

**Tech Stack:** Next.js 16, TypeScript, PyTorch, Vitest, pytest, Playwright, ListenBrainz, optional Last.fm.

**Spec:** `docs/superpowers/specs/2026-09-14-tastelift-design.md`

## Global Constraints

- Preserve existing UI/API/cache/model-serving paths; add rather than rewrite.
- Spotify content is never a training or ranking input.
- No LLM recommendations, hardcoded genre maps, audio, giant downloads, database, or runtime PyTorch.
- Every >2-minute job runs detached in Kitty with persistent logs/checkpoints/resume.
- Validation/test remain user-disjoint; retrieval Recall@500 gates ranker tuning.

---

### Task 1: JSON contract and robust resolver

**Files:** Create `src/lib/tastelift/input.ts`, `src/lib/tastelift/resolver.ts`, `app/api/tastelift/route.ts`; modify `src/lib/listenbrainz.ts`; test `tests/unit/tastelift-input.test.ts`, `tests/unit/tastelift-resolver.test.ts`.

**Interfaces:** Produce `parseTasteInput(unknown): TasteSeedInput[]` and `resolveTasteSeeds(inputs, adapters): Promise<ResolvedTasteSeed[]>` with structured per-seed failures.

- [ ] Write failing tests for 5/200/201 inputs, Unicode, Spotify URL parsing, no silent losses, and Project Dumb regression.
- [ ] Run focused Vitest and confirm contract/resolver failures.
- [ ] Implement validation, deterministic dedupe, ListenBrainz resolver, optional Last.fm adapter, and batch API route.
- [ ] Run focused tests and commit `feat: add robust TasteLift seed resolution`.

### Task 2: Large candidate pool and identity filtering

**Files:** Create `src/lib/tastelift/retrieval.ts`, `src/lib/tastelift/identity.ts`; modify `src/lib/listenbrainz.ts`; test `tests/unit/tastelift-retrieval.test.ts`.

**Interfaces:** Produce `retrieveTasteCandidates(seeds, adapters, limit=500)` returning candidates with all source/seed evidence.

- [ ] Write failing tests for union/support, 500 cap, all-one-artist input, seed exclusion, alternate MBID/remaster/cover dedupe.
- [ ] Run tests to confirm failures.
- [ ] Implement cached batched retrieval and deterministic identity/version filtering.
- [ ] Run tests and commit `feat: build large TasteLift candidate pools`.

### Task 3: Episodic cached-history dataset

**Files:** Create `ml/tastelift_data.py`, `scripts/build_tastelift_data.py`; test `tests/python/test_tastelift_data.py`.

**Interfaces:** Produce deterministic `TasteEpisode`, popularity percentiles/bands, vocabulary and hashed subword IDs from ignored cached profiles/retrieval.

- [ ] Write failing tests for 5–30 unique unordered seeds, repeated masks, held-out positives, user isolation, and popularity-matched negatives.
- [ ] Run pytest to confirm failures.
- [ ] Implement the dataset builder with atomic checkpoint/resume and no network downloads.
- [ ] Build the local dataset, inspect aggregate counts, run tests, and commit `feat: build TasteLift episodes from cached histories`.

### Task 4: Four-interest TasteLift model

**Files:** Create `ml/tastelift_model.py`, `scripts/train_tastelift.py`; test `tests/python/test_tastelift_model.py`.

**Interfaces:** Produce `TasteLift.encode_set(track_ids, mask) -> [batch,4,dim]`, `score_candidates(...)`, BPR+lift loss, checkpoints, and JSON export.

- [ ] Write failing tests for four distinct heads, permutation invariance, variable 5–200 sets, OOV hashing, BPR direction, and scratch initialization.
- [ ] Run pytest to confirm failures.
- [ ] Implement track tower, four-query attention, max/logsumexp head scoring, popularity prior, and resumable trainer.
- [ ] Run the training job detached in Kitty and commit `feat: train four-head TasteLift model` after inspecting validation history.

### Task 5: TypeScript inference and residual features

**Files:** Create `src/lib/tastelift/model.ts`, `src/lib/tastelift/features.ts`; modify `src/lib/mlp.ts`, `src/lib/ranking.ts`, `src/lib/recommend.ts`; test `tests/unit/tastelift-model.test.ts`, `tests/python/test_tastelift_parity.py`.

**Interfaces:** Produce pure TypeScript TasteLift inference and candidate fields `tasteHead`, `seedSupport`, `popularityPercentile`, `liftScore`.

- [ ] Write failing PyTorch↔TypeScript parity and 120-permutation tests.
- [ ] Run tests to confirm failures.
- [ ] Implement JSON loader/inference and extend residual feature assembly without changing existing calls.
- [ ] Run parity/typecheck/tests and commit `feat: serve TasteLift in TypeScript`.

### Task 6: Forty-track discovery slate

**Files:** Create `src/lib/tastelift/slate.ts`; modify `src/lib/types.ts`, `src/lib/recommend.ts`; test `tests/unit/tastelift-slate.test.ts`.

**Interfaces:** Produce `buildTasteSlate(ranked, 40)` with relevance/lift/serendipity/head coverage and max-two-per-artist constraints.

- [ ] Write failing tests for 40 results, long-tail share, head coverage, deterministic output, duplicate/version removal, and mixed/all-one-artist inputs.
- [ ] Run tests to confirm failures.
- [ ] Implement greedy marginal slate scoring and explanations.
- [ ] Run tests and commit `feat: assemble diverse TasteLift discovery slates`.

### Task 7: Product-shaped evaluation

**Files:** Create `scripts/evaluate_tastelift.py`, `reports/tastelift-evaluation.json`; test `tests/python/test_tastelift_evaluation.py`.

**Interfaces:** Compare popularity, current residual recommender, and TasteLift for Recall@500, NDCG@20, Recall@20, popularity, artist diversity, and head coverage.

- [ ] Write failing metric/baseline/user-split tests.
- [ ] Run tests to confirm failures.
- [ ] Implement frozen evaluation with retrieval gate and user-clustered bootstrap.
- [ ] Run detached evaluation, verify TasteLift beats both baselines, and commit `eval: prove TasteLift personalized lift`.

### Task 8: Real JSON, playlist inspection, and completion

**Files:** Create `scripts/tastelift-playlist.ts`, `reports/my-playlist.json`; modify `README.md`, `reports/verification.md`; test `tests/e2e/tastelift.spec.ts`.

**Interfaces:** CLI reads the supplied JSON and writes exactly 40 enriched recommendations through the real runtime pipeline.

- [ ] Write failing CLI/API/E2E tests including Project Dumb and 200-input behavior.
- [ ] Implement the CLI and preserve the existing UI flow.
- [ ] Run the supplied JSON, inspect mainstream share/repetition/head coverage, diagnose and iterate if constraints fail.
- [ ] Run pytest, Vitest, typecheck, lint, build, Playwright, parity, live regression, privacy audit, and commit `feat: complete TasteLift v2`.
