# TasteLift Task 1 report

## Files changed

- Added `src/lib/tastelift/input.ts`: canonical JSON validation, NFKC/whitespace normalization, Spotify track URL parsing, first-seen deterministic artist/title deduplication, and structured validation errors.
- Added `src/lib/tastelift/resolver.ts`: ordered per-seed resolution, exact ListenBrainz and optional Last.fm MBID resolution, text/OOV fallback diagnostics, and bounded-retry Last.fm access through the shared HTTP client.
- Added `app/api/tastelift/route.ts`: `POST` batch parsing plus structural `400`, `422`, and `502` responses.
- Updated `src/lib/listenbrainz.ts`: exposed the TasteLift seed-search adapter while retaining all existing APIs.
- Added focused contract and resolver tests in `tests/unit/tastelift-input.test.ts` and `tests/unit/tastelift-resolver.test.ts`.

## Design decisions

- The canonical batch range is 5–500 as specified. Tests exercise 5, 200, and 201 accepted entries, and reject 501.
- Input normalization is NFKC plus collapsed/trimmed Unicode whitespace. Deduplication only uses the normalized, case-insensitive artist/title identity; `Jóga` and `Jóga (Live)` remain distinct.
- A syntactically valid song lacking an exact external MBID remains a resolved `source: "text"` seed with `diagnostic.status: "retrieval_unavailable"`. It is not fuzzy-matched, silently dropped, or rejected from a batch. Actual upstream failures remain `status: "unresolved"`, causing the route to return its structured `422` batch rejection.
- Exact ListenBrainz and Last.fm matches retain the recording MBID. Last.fm matches must also have an exact normalized artist/title match, preventing a false identity.
- The Project Dumb fixture mirrors a recorded ListenBrainz recording-search row (`recording_mbid`, `recording_name`, `artist_credit_name`, `release_name`) whose different title proves no false fuzzy match occurs.

## TDD evidence

Red runs:

```text
$ npm test -- tests/unit/tastelift-input.test.ts tests/unit/tastelift-resolver.test.ts
Failed Suites 2: Cannot find module '../../src/lib/tastelift/input' and '../../src/lib/tastelift/resolver'

$ npm test -- tests/unit/tastelift-input.test.ts
1 failed: duplicate row raised TasteInputError instead of deterministic first-seen deduplication

$ npm test -- tests/unit/tastelift-resolver.test.ts
3 failed: Project Dumb without an exact recording was unresolved instead of a text/OOV resolved seed

$ npm test -- tests/unit/tastelift-resolver.test.ts
1 failed: a mismatched Last.fm result was incorrectly accepted as an MBID identity

$ npm test -- tests/unit/tastelift-input.test.ts
1 failed: 201 entries were rejected by the former 200-entry limit
```

Green verification (fresh final run):

```text
$ npm test -- tests/unit/tastelift-input.test.ts tests/unit/tastelift-resolver.test.ts
Test Files  2 passed (2)
Tests  12 passed (12)

$ npm run typecheck
tsc --noEmit (exit 0)

$ git diff --check
exit 0
```

The live resolver was also run once through the shared bounded-retry client. `Project Dumb — An Italian Magician Be Like` returned the intended text/OOV resolved seed and `no_exact_mbid` diagnostic.

## Self-review

- Confirmed output ordering uses `Promise.all(inputs.map(...))`, preserving every supplied resolved or unresolved seed deterministically.
- Confirmed no best/fuzzy fallback can assign an unrelated ListenBrainz or Last.fm MBID.
- Confirmed Spotify IDs are parsed only for later output linking and never used by resolver matching.
- Confirmed the route rejects only true unresolved upstream seeds, retains valid OOV text seeds, and exposes structured input and resolution response bodies.
- Confirmed staged scope will contain only the Task 1 implementation, tests, and this report; pre-existing dirty files remain unstaged.
