# TasteLift Task 2 report

## Files

- Added `src/lib/tastelift/retrieval.ts`: a cached, batch-first Recall@500 retrieval pool that retains every resolver outcome and all per-seed evidence.
- Added `src/lib/tastelift/identity.ts`: deterministic normalized artist/title identity plus conservative remaster/edit/cover suffix folding.
- Updated `src/lib/listenbrainz.ts`: bounded-retry POST batch similarity adapter and response parser that preserves source ordering per reference MBID.
- Added `tests/unit/tastelift-retrieval.test.ts`: union/support, final 500 cap, all-one-artist, seed alternate exclusion, and alternate/remaster/cover identity coverage.

## Choices

- Only resolved MBID seeds enter the external batch; text/OOV and unresolved seeds remain unchanged in `TasteCandidatePool.seeds`, so no seed is silently discarded or falsely retrieved.
- Cache entries are per reference MBID. Missing IDs share one ListenBrainz batch request; cached IDs are reused independently.
- Evidence records source, input seed index/MBID, candidate MBID, upstream rank, and raw score. Identity folding merges evidence from alternate MBIDs before deterministic support/RRF ordering and the final cap.
- Identity comparison is artist-scoped. It removes only recognizable version qualifiers, so same-title recordings by different artists are retained.

## TDD evidence

- Red: `npm test -- tests/unit/tastelift-retrieval.test.ts` failed as expected with `Cannot find module '../../src/lib/tastelift/retrieval'` before implementation.
- Green: `npm test -- tests/unit/tastelift-retrieval.test.ts` — 5 passed.
- Regression: `npm test -- tests/unit/tastelift-retrieval.test.ts tests/unit/tastelift-resolver.test.ts tests/unit/api-contracts.test.ts` — 17 passed.
- Typecheck: `npm run typecheck` — passed.
- Diff hygiene: `git diff --check` — passed.

## Self-review

- Candidate collection runs for every distinct resolved MBID, including all-one-artist inputs; final capping never occurs per seed.
- Direct seed MBIDs and normalized alternate/version identities are excluded before output.
- Candidate and evidence order use explicit support/RRF/raw-score/MBID and seed/rank/MBID tie-breakers, so output does not depend on upstream seed ordering.
- No recommendation logic is artist/title hardcoded; version matching is only an identity-deduplication rule.

## Concern

No live ListenBrainz call was made in this scoped task. The POST shape matches the repository's existing cached collector contract (`[{ recording_mbids, algorithm }]`).

## Fix round 1: anchored version qualifiers

### Root cause

`VERSION_WORD.test(qualifier)` matched `cover` inside a larger parenthetical phrase, while the dash expression accepted arbitrary trailing text after a matched version word. Consequently, `Song (Not a Cover)` and `Song - Cover Me` both normalized to `Song`.

### Change and self-review

- Replaced substring recognition with one fully anchored `VERSION_QUALIFIER` built from an explicit version-label grammar.
- The same grammar is used for parenthetical and dash suffixes; only an entire recognized label folds away.
- Positive `A Song (2011 Remaster)` and `A Song - Cover` coverage remains in the existing identity regression; new near-miss coverage retains both reported titles as distinct recordings.
- The grammar remains identity-only and artist-scoped; it does not add artist/title recommendation rules.

### Verification

- Red: `npm test -- tests/unit/tastelift-retrieval.test.ts`
  - Output: `1 failed | 5 passed`; the near-miss test received `['original']` instead of `['original', 'not-a-cover', 'cover-me']`.
- Green: `npm test -- tests/unit/tastelift-retrieval.test.ts`
  - Output: `1 file passed`, `6 passed`.
- `npm run typecheck`
  - Output: exited `0` with `tsc --noEmit`.
- `git diff --check`
  - Output: exited `0` with no whitespace errors.
