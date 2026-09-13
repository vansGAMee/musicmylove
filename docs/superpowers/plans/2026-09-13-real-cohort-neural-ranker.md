# Real-Cohort Neural Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, select, export, deploy, and verify a MusicMyLove neural reranker on real public ListenBrainz users.

**Architecture:** One resumable Python pipeline downloads one bounded incremental archive for usernames, collects official cached top-recording profiles and batched Labs candidates, freezes user splits, trains/evaluates rankers, and exports JSON. Existing TypeScript ranking loads the selected artifact and reproduces the validation-selected neural or neural/RRF ensemble.

**Tech Stack:** Python 3.14, PyTorch 2.13, standard-library HTTP/tar/zstd, Next.js 16, TypeScript 5, Vitest, pytest

**Spec:** `docs/superpowers/specs/2026-09-13-real-cohort-neural-ranker-design.md`

## Global Constraints

- Download at most one dataset and reject it above 500 MB compressed.
- Keep API throughput at or below one request per second and cache every useful response.
- Persist a fixed-seed user split before candidate retrieval; never move users across partitions.
- Tune only on train/validation and open the frozen test once after selection.
- Known top recordings are never negatives; hidden targets never appear among seeds.
- Runtime features remain seed-order invariant and production inference remains pure TypeScript.
- Reports must distinguish external Labs retrieval from MusicMyLove's learned ranking.

---

### Task 1: Active-user manifest from one official incremental dump

**Files:**
- Create: `ml/cohort.py`, `scripts/sample_users.py`
- Modify: `.gitignore`, `ml/config.json`
- Test: `tests/python/test_cohort.py`

**Interfaces:**
- Produces: `extract_usernames(lines: Iterable[bytes]) -> list[str]`, `choose_users(names, count, seed) -> list[str]`, and `data/manifests/usernames.json`.

- [ ] Write a failing test with duplicate/malformed listen rows asserting deterministic unique username extraction and selection.
- [ ] Run `.venv/bin/python -m pytest tests/python/test_cohort.py -q`; confirm the missing-module failure.
- [ ] Implement JSON-line extraction and seeded selection; implement archive discovery, Content-Length limit, SHA-256 verification, zstd/tar streaming, manifest write, and archive deletion.
- [ ] Run the focused test, then sample 500 usernames from the latest official incremental archive and record observed compressed bytes/checksum.
- [ ] Commit `feat: sample active ListenBrainz users`.

### Task 2: Real profiles, frozen split, and batched candidate cache

**Files:**
- Create: `ml/api.py`, `ml/examples.py`, `scripts/collect_real.py`
- Modify: `ml/data.py`, `ml/splits.json`
- Test: `tests/python/test_real_examples.py`, `tests/python/test_api.py`

**Interfaces:**
- Produces: `collect_profiles(manifest, cache)`, `freeze_splits(profiles, path, seed)`, `build_examples(profile, seed)`, `fetch_similar_batch(mbids)`, and JSONL examples containing seeds, hidden positives, candidates, evidence, and partition.

- [ ] Write failing tests for 429/5xx/timeout recovery, rate-header delay, malformed-row rejection, atomic cache reuse, three deterministic examples, and known-positive negative exclusion.
- [ ] Run the focused pytest files and verify failures arise from missing interfaces.
- [ ] Implement one-request-per-second resilient clients for user statistics and Labs POST batches, plus fixture-injectable clocks/transports.
- [ ] Implement suitability filtering (20 valid MBIDs, 12 repeated tracks), log-count sampling, immediate immutable split creation, and cached example retrieval grouped by `reference_mbid`.
- [ ] Run collection until 300–500 usable profiles and all example caches exist; never rewrite the split after it is created.
- [ ] Commit `feat: collect real cold-start examples`.

### Task 3: Feature audit and validation-only training loop

**Files:**
- Create: `ml/features.py`, `ml/train_real.py`, `scripts/train_real.py`
- Modify: `ml/model.py`, `ml/model.json`
- Test: `tests/python/test_real_training.py`, `tests/python/test_feature_parity.py`

**Interfaces:**
- Produces: Python `build_features` identical to TypeScript, weighted BPR batches, validation per-ranker metrics, chosen ensemble coefficient, and exported artifact metadata.

- [ ] Write failing tests comparing Python features to recorded TypeScript features, hard/random negative balance, weighted BPR orientation, deterministic convergence, and validation-only early stopping.
- [ ] Run focused tests and observe missing-interface failures.
- [ ] Implement the exact 17 features, distribution audit, weighted pair sampler, gradient clipping, checkpointing, and validation evaluation for max/RRF/neural plus coefficients `0.0..1.0` in steps of `0.1`.
- [ ] Train and diagnose until neural evidence beats the strongest baseline on validation NDCG@20 without touching test; persist every trial configuration and validation result.
- [ ] Freeze the chosen architecture/coefficient and export weights plus normalization/selection metadata.
- [ ] Commit `feat: train owned neural reranker on real users`.

### Task 4: One-shot frozen test and production integration

**Files:**
- Modify: `ml/evaluate.py`, `scripts/evaluate.py`, `src/lib/mlp.ts`, `src/lib/ranking.ts`, `src/lib/recommend.ts`, `src/components/MusicRecommender.tsx`
- Test: `tests/python/test_evaluation.py`, `tests/unit/mlp.test.ts`, `tests/unit/ranking.test.ts`, `tests/unit/recommender-state.test.tsx`

**Interfaces:**
- Consumes: frozen model artifact and untouched test JSONL.
- Produces: test metrics/paired bootstrap data and `scoreWithArtifact(features, model)` used by the browser.

- [ ] Write failing TypeScript tests proving the artifact-selected neural/ensemble score controls ordering and all 120 permutations remain identical.
- [ ] Run focused tests and confirm ordering still uses hardcoded RRF.
- [ ] Implement artifact schema validation, optional feature normalization, neural/RRF ensemble scoring, and automatic production selection.
- [ ] Evaluate the frozen test exactly once; write per-ranker retrieval recall, Recall@20, NDCG@20, HitRate@20, pool size, diversity, and paired-bootstrap delta.
- [ ] Run parity, TypeScript/Python tests, build, and live smoke; commit `feat: deploy validated neural ranking`.

### Task 5: Proof and reproducibility handoff

**Files:**
- Modify: `README.md`, `AGENTS.md`, `reports/evaluation.json`, `reports/verification.md`
- Create: `reports/real-cohort-training.json`

**Interfaces:**
- Produces: exact observed provenance, cohort counts, metrics, winner rationale, parity/build/test/live evidence, and reproducibility commands.

- [ ] Re-run the complete deterministic test suite, Python/TypeScript parity, Playwright, live five-song smoke, lint, and production build.
- [ ] Record archive bytes/checksum, sampled/usable/split users, real examples/pairs, retrieval recall, every ranker metric, bootstrap interval, selected production scorer, and known external-retrieval limitation.
- [ ] Confirm `git ls-files` contains no archive, profile cache, username values, or raw example data; anonymize any per-example report rows.
- [ ] Run `git diff --check` and commit `docs: prove real-cohort neural recommendation`.
