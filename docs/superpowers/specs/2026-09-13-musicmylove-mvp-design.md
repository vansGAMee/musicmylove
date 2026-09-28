# MusicMyLove MVP Design

## Product

MusicMyLove is a single-screen cold-start music recommender. A visitor searches for and selects exactly five MusicBrainz recordings. Each selection immediately starts a cached ListenBrainz Labs similarity request. Selection five automatically ranks the merged pool and shows up to twenty real recordings with seed-evidence explanations, deterministic artist diversity, local like/dislike feedback, and exact-or-search Spotify links.

## Architecture

- One Next.js App Router TypeScript application, deployable to Vercel.
- Small route handlers proxy the official Labs recording-search, similar-recordings, and Spotify-ID endpoints. They provide validation, timeouts, bounded retry, descriptive User-Agent, a small memory cache, and stale-cache response support.
- The browser keeps versioned search/similarity responses and feedback in localStorage. Similarity fetches are started at seed selection rather than delayed until seed five.
- Shared pure TypeScript modules own merge, normalization, feature construction, rankers, diversity, explanations, and JSON-weight MLP inference. Production has no Python dependency.
- Python scripts collect public user statistics into ignored caches, persist deterministic user splits, create training pairs, fit a 17→16→8→1 PyTorch MLP with BPR loss, export JSON weights, and evaluate baselines and the model.

## External contracts

Current official endpoints verified on 2026-09-13:

- `GET https://labs.api.listenbrainz.org/recording-search/json?query=...` returns recording and release names, recording/release MBIDs, and artist-credit name/id.
- `GET https://labs.api.listenbrainz.org/similar-recordings/json?recording_mbids=...&algorithm=...` returns recording metadata, raw numeric score, and reference MBID. The selected algorithm is the listed long-history, top-listeners-limited session model.
- `GET https://labs.api.listenbrainz.org/spotify-id-from-mbid/json?recording_mbid=...` returns an array whose row may include `spotify_track_ids`.
- `GET https://api.listenbrainz.org/1/stats/user/{name}/recordings?range=all_time&count=...` returns top recordings with `listen_count` and exposes rate-limit headers.

All parsers reject malformed rows. Network calls use abortable timeouts and retry only 429/5xx/network failures with capped exponential backoff and jitter. One failed seed leaves a usable partial pool; stale browser data is accepted while refresh continues. Spotify lookup failure always falls back to a URL-encoded Spotify search.

## Ranking and invariance

Each seed list is independently normalized with rank percentile and max-score scaling. Candidate evidence retains seed MBID only for explanations. The model vector is independent of input order:

1. five normalized similarities sorted descending;
2. five reciprocal-rank contributions sorted descending;
3. total reciprocal-rank evidence;
4. support count divided by five;
5. maximum and second-highest normalized similarity;
6. mean and population standard deviation of positive similarities;
7. same-artist seed count divided by five.

Candidates are deduplicated by recording MBID, and seeds are removed before scoring. Baselines are max normalized similarity and RRF. The neural scorer exactly reproduces the exported ReLU MLP. A validation-selected ranker name in the model artifact controls production honestly; ties fall back to recording MBID for deterministic output. Final selection greedily keeps ranked tracks while allowing at most two per normalized artist name.

## Training and evaluation

Collection targets 300–500 public users supplied by a reproducible username manifest. It requests all-time top recordings conservatively and caches every successful response. Suitable users require repeated listening and enough valid recording MBIDs. A fixed seed creates and persists train/validation/test user partitions before examples are generated.

Examples sample five strong positives and distinct hidden strong positives. Candidates absent from all fetched user positives are implicit negatives; known positives are never negatives. Listen counts are log-scaled for weighted sampling. The MLP uses BPR pairwise logistic loss and early model selection on validation NDCG, never test. Evaluation separately reports retrieval recall, Recall@20, NDCG@20, HitRate@20, candidate-pool size, artist diversity, and paired-bootstrap uncertainty versus the strongest baseline. The frozen test set is evaluated only after validation selects the production ranker.

For a runnable repository without committing a large dataset, a small deterministic fixture pipeline produces an observed development evaluation and model/parity artifact. The same commands accept collected live data for the target-scale run. Reports distinguish fixture-scale evidence from claims requiring a larger collected cohort.

## UI and state

The page has the prescribed headline, one debounced autocomplete, selected seed chips and `n / 5` progress. Requests carry an AbortSignal so stale search results cannot overwrite new ones. At five seeds, the input closes and recommendations appear automatically when available. Cards show artist, track, the two strongest contributing seed names, Spotify, Like, and Dislike. Dislike removes the MBID and promotes the next eligible ranked track; feedback persists locally. No percentage, causal claim, login, onboarding, settings, or generation button is present.

## Verification

Vitest covers ranking, all 120 seed permutations, caching/retry behavior, auto-generation state, and mocked end-to-end behavior. Playwright covers the browser happy path with intercepted official API responses. Pytest covers split and negative-sampling integrity. A cross-language parity command compares generated vectors between Python and TypeScript. A separate live smoke command uses five real recordings through the production modules and validates count, exclusions, deduplication, artist cap, Spotify action, and reordered-seed stability. `reports/verification.md` and `reports/evaluation.json` record only observed command results and limitations.

## Known limitation

ListenBrainz Labs similarity is an externally precomputed aggregate system and may include activity from held-out users. Metrics evaluate this repository's reranker on top of that retrieval system; they do not establish that the complete retrieval stack is free of test-user influence.
