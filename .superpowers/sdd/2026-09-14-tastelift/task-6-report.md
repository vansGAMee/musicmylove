# Task 6 — TasteLift discovery slate

## Objective and rules

`buildTasteSlate(ranked, 40)` consumes the enriched `RankedTrack[]` from
`rankTasteCandidatePool` and applies a deterministic greedy marginal score:

| Component | Weight | Input |
| --- | ---: | --- |
| relevance | 0.36 | min-max normalized residual score (or ranked score when residual is absent) |
| personalized lift | 0.24 | min-max normalized TasteLift lift |
| serendipity | 0.20 | `1 - popularityPercentile` |
| artist diversity | 0.12 | inverse selected count for the normalized artist |
| head coverage | 0.08 | one-time reward for a previously unseen TasteLift head |

Hard rules run before and during selection: preserve a single representative
for each MBID and normalized recording/version identity; select at most two
tracks per normalized artist; reject popularity percentile `>= 0.85` unless
normalized relevance or lift is at least `0.90`; reserve a feasible assignment
for every available TasteLift head; and reserve `ceil(limit * 0.35)` long-tail
tracks when that capacity exists. No candidates are fabricated, so a constrained
pool returns its real shortfall.

`createTasteLiftSlate(pool, limit)` is the production wrapper: it first uses
the existing evidence-preserving pool ranking path, then applies the selector.
It preserves existing recommendation APIs and all ranking fields; selected rows
add optional `slateScore` and `slateComponents` explanation details.

## Red → green

1. Added `tests/unit/tastelift-slate.test.ts` before the implementation.
2. Confirmed red: Vitest failed because `src/lib/tastelift/slate` did not exist.
3. Added the pure TypeScript slate implementation, typed explanation fields,
   and the real-pool wrapper.
4. Confirmed focused green: 6 slate tests passed; `npm run typecheck` passed.

## Self-review

- Canonical sorting and MBID final ties make selection insensitive to input order.
- The selector uses only ranking/evidence/model fields; it neither uses genres
  nor invents artist/track recommendations.
- Artist and recording identity limits remain hard even when the requested
  length cannot be reached.
- Broader repository validation is recorded with the final task handoff.

## Round 1 reviewer fixes

The reviewer report was replaced with this original implementation report and
the following correction evidence is appended.

- Long-tail reservation now measures feasible capacity per normalized artist
  (maximum two) rather than raw row count. After head reservations it rechecks
  the remaining capacity and only keeps choices that preserve the achievable
  quota; it falls back to ordinary eligible rows rather than manufacturing a
  shortfall.
- A deterministic four-head, artist-capacity b-matching is reserved before
  greedy selection. This prevents early high-scoring choices from consuming an
  artist twice and stranding a head that has a feasible alternate assignment.
- Equal or non-finite relevance/lift scales normalize to `0`, so they cannot
  qualify an otherwise mainstream candidate as exceptional.
- Duplicate rows use a total, stable serialization tie-breaker over all source
  fields retained in the output (excluding recomputed slate fields), making the
  selected object itself permutation-invariant.

Red tests were added for the concentrated long-tail, stranded-head,
equal/non-finite-mainstream, duplicate-object permutation, and case/whitespace
artist-cap fixtures. The initial focused run failed in all four reviewer
failure modes; after the changes, all 11 slate tests and TypeScript typecheck
pass. Broader quick-suite evidence is recorded with the round-1 handoff.
