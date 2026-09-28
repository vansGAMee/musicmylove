# TasteLift Design

## Objective

Extend MusicMyLove in place so one JSON document containing 5–200+ genuinely liked songs produces 40 specific, long-tail recommendations. Preserve the existing UI, API routes, ListenBrainz caches, residual ranker, pure-TypeScript serving, and test/evaluation machinery. Spotify remains output/link resolution only.

## Input and resolution

The canonical input is `{ "songs": [{ "artist": string, "title": string, "spotify_url"?: string }] }`. Validate 5–500 entries, normalize Unicode NFKC and whitespace without silently merging distinct songs, and retain an error per unresolved seed. Resolution queries ListenBrainz recording search, then optional Last.fm enrichment when `LASTFM_API_KEY` exists. Spotify URLs provide IDs for output lookup only and never ML features. A batch is rejected rather than silently degraded when any valid seed remains unresolved. The regression `Project Dumb — An Italian Magician Be Like` must resolve through a recorded real API-shaped fixture and the live resolver when sources expose it.

## Candidate retrieval

Reuse cached ListenBrainz long-history similar-recording rows and fetch missing seeds through the existing bounded client. Union per-seed results into a Recall@500 pool. Last.fm may add candidates only when resolvable back to MusicBrainz identities. Every candidate retains per-seed rank/score/source evidence. Remove seeds, normalized artist-title duplicates, alternate MBIDs, obvious remaster suffix duplicates, and exact cover/version spam. If many inputs share one artist, retrieval still uses every distinct seed and the final cap is applied only after scoring.

## TasteLift

Train from scratch on the existing 790 suitable public-user caches. A compact track tower combines learned vocabulary embeddings with stable hashed artist/title subword buckets so obscure/OOV seeds remain representable. Four learned query vectors independently attend over the entire unordered seed set; no positional features or mean-only collapse are allowed. Candidate affinity is max/logsumexp evidence across four taste heads plus explicit all-seed support.

Episodic training creates many deterministic masks per train user: expose 5–30 unique listened tracks and hold out other repeated-listen tracks. BPR pairs use positives against popularity-band-matched hard negatives. The optimized score is affinity minus a learned/calibrated global-popularity prior, so popular tracks survive only with high personal affinity. User-disjoint validation selects hyperparameters.

## Final ranking and slate

Extend the existing residual ranker with TasteLift affinity, strongest-head score/index, seed support, empirical popularity percentile, and novelty/lift. The 40-track slate greedily optimizes relevance + lift + serendipity + head coverage + artist diversity, with maximum two tracks per artist and no duplicate work/version. Output includes score, strongest taste head, seed support, popularity percentile, novelty/lift, and Spotify link when resolvable.

## Evaluation and completion

Evaluation uses the product-shaped path: visible seed subset → candidate retrieval → other user-liked tracks hidden. Report Recall@500, NDCG@20, Recall@20, popularity percentile, artist diversity, and head coverage on a fully user-disjoint validation/test split. TasteLift must beat both the current residual recommender and global-popularity baseline. Retrieval is fixed before ranker tuning. Inspect the actual saved 40-track playlist for mainstream dominance, repetition, empty clusters, and generic output.

Training/evaluation jobs expected above two minutes run detached in Kitty with persistent log, checkpoint, deterministic resume, and atomic artifacts. No new giant corpus is downloaded. Reports explicitly distinguish external candidate retrieval from the owned TasteLift encoder/ranker.

## Failure behavior and privacy

Malformed JSON, out-of-range seed counts, duplicate-only inputs, and unresolved songs return structured errors. External 429/5xx/timeouts use existing bounded retry/cache behavior. Public usernames and raw histories remain ignored; only aggregate metrics and model artifacts are committed.
