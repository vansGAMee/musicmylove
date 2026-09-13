# Real-Cohort Neural Ranker Design

## Objective

Replace fixture-only evidence and the production RRF fallback with a genuinely trained MusicMyLove neural reranker. ListenBrainz Labs remains candidate retrieval only. MusicMyLove owns the preference labels, order-independent features, pair sampling, learned weights, validation selection, TypeScript inference, diversity policy, and playlist assembly.

Success requires a frozen user-disjoint real-public-user test, complete baseline/model metrics, and a neural or validation-proven neural/RRF ensemble that improves NDCG@20 over the strongest non-learned baseline. Test remains untouched until the final design is selected. A qualitative five-seed playlist is reported separately from quantitative held-out relevance; it is not proof by anecdote.

## User sampling and limits

Use exactly one newest official incremental ListenBrainz listens archive solely to extract active public usernames. On 2026-09-13 the newest archive is `listenbrainz-listens-dump-2659-20260912-000003-incremental.tar.zst`, listed at 314 MB compressed. Verify its published SHA-256, stream-extract usernames, save a deterministic shuffled manifest, and delete the archive immediately. Never commit the archive or API cache.

Fetch all-time top 100 recordings for candidates from the first 500 sampled users, at no more than one ListenBrainz request per second. Retain 300–500 users with at least 20 valid recording MBIDs and at least 12 tracks having two or more listens. Persist the fixed-seed 70/15/15 split before retrieval.

## Real examples and retrieval

For each user, sort strong positives by `log1p(listen_count)` and generate up to three deterministic examples. Each example has five distinct seeds and three to five distinct hidden positives. Batch all seed MBIDs for that user into one official Labs POST request; separate returned rows by `reference_mbid` and cache the response. A candidate absent from all 100 fetched positives may be an implicit negative. Known positives are never negatives.

Measure retrieval recall before training. If it is inadequate, compare the currently listed long-history similarity algorithm with the official `mlhd-similar-recordings` dataset on validation only. Do not increase model capacity to conceal missing targets.

## Model and selection

Use the existing 17 order-independent features. Add user-preference strength only as a training-example weight, never as a runtime input. Train 17→16→8→1 with weighted BPR, balanced hard/random implicit-negative sampling, dropout-free deterministic inference, fixed seeds, gradient clipping, and early stopping on validation NDCG@20.

Compare max similarity, RRF, neural, and a small convex neural/RRF ensemble. Tune the ensemble coefficient on validation only. Freeze the winning architecture/coefficient, evaluate test exactly once, export model/scaler/selection metadata to `ml/model.json`, and make production honor the artifact instead of hardcoding RRF.

If neural ranking loses validation, inspect retrieval recall, feature distributions, labels, negative hardness, convergence, and parity. Correct real defects and retrain. Do not inspect test during this loop. If a legitimate model cannot beat RRF, the report must say so; however the objective remains active because the requested deliverable is a proven owned model.

## Evidence

Commit anonymized split hashes/user counts, collection/training configuration, model weights, per-ranker aggregate metrics, per-example test metrics without usernames, paired bootstrap confidence interval, parity result, and live smoke output. Reports distinguish external retrieval from our learned ranking and state the externally precomputed retrieval leakage limitation.

## API and operational behavior

All requests retain descriptive User-Agent, timeout, schema validation, cache, 429/5xx retry, rate-header awareness, and stale fallback. Training data never enters Vercel. The web app keeps the same one-screen five-song flow and pure TypeScript inference.
