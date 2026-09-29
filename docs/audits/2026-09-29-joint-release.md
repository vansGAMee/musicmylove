# Joint model release audit — 2026-09-29

**Decision: no validated discovery upgrade; keep ranking unchanged. Website
integration is an experimental preview, not a quality-certified release.**

Audited the current `joint-v1` trained export, its candidate preparation, ranking,
user partitions, targeted collection, and the separate legacy discovery path.
The machine-readable measurements are in `2026-09-29-joint-release.json`.

## What the evidence says

- Existing checkpoint-selection DEV: joint NDCG@50 0.0801822 vs graph-only
  0.0801654; paired 95% interval for the difference [-0.0046023, 0.0045017].
  Audio participates in learned inference, but its added value is not established.
- One fixed generic policy was probed: reduce familiar-artist cap from two to one,
  leave new-artist cap at two. No artist names, genres, profile-specific constants,
  manual track insertions, or parameter sweep. The candidate was **rejected**.
- On 1,981 eligible SHADOW users, regular NDCG@50 fell from 0.12355 to 0.09807
  (paired 95% delta [-0.02903, -0.02227]); Recall@50 also fell. New-artist share
  rose from 0.60972 to 0.72331. This is not a free improvement.
- Held-out-artist NDCG rose from 0.01315 to 0.01585 on 1,882 eligible users,
  but that does not cancel the regular-query regression or shorter playlists.
- Mean candidate family recall over both episode types was 0.20513. Retrieval
  misses remain in the denominator; this is conditional on mapped catalog
  positives and eligible profiles, not coverage of all music or all listeners.
- SHADOW users are disjoint from graph representation, ranker training and DEV.
  The probe did not use FINAL. However, SHADOW was available to earlier project
  evaluations; it is **not** a pristine project-wide release holdout. Existing
  `final_consumed.lock` also shows historical FINAL evaluation. Neither may be
  relabeled as fresh independent evidence by merely reshuffling users.

## Bias and implementation findings

1. `targeted_data.select_users` prioritizes representation users connected to
   missing tracks/artists from the supplied library. This is personalized data
   acquisition and selection bias, not neutral population sampling. It uses
   existing representation users, not held-out labels. No artist-specific ranking
   blacklist or whitelist was found in the audited joint recommendation path.
2. Joint retrieval keeps only 24 neighbors per catalog track (16 other-artist,
   eight same-artist), then takes up to 64 balanced input seeds. Neither retrieval
   coverage nor these capacity choices have a demonstrated optimality guarantee.
3. The CLI labeled joint inference “Discovery” while applying one shared policy.
   Corrected the label; did not invent an unvalidated novelty boost.
4. Legacy discovery's familiarity/popularity quotas and optional sound percentile
   filter are separate from joint inference. Their unit tests prove constraints,
   not independent gains in relevance. They have not been certified by this audit.
5. Joint negatives protect known positive families; missing observations remain
   implicit comparisons, not user dislikes. Listener history predicts listening,
   not subjective quality. Joint training does not yet consume the CLI's personal
   ratings, and existing adapters are incompatible. The site states this.

## Website integration

`npm run build:joint` loads the frozen browser export and validates the five
allowlisted files against its checksummed manifest. It builds the existing UI
with joint inference in a Web Worker. No fallback to the old site's model, raw
audio, CLAP weights, or listener histories is included in the prebuilt deployment.
The exact current payload is 23,988,202 bytes (22.9 MiB), before HTTP compression.

The UI reports unmatched tracks, excludes disliked song families, preserves model
order, displays up to 50 results, and does not fill an empty result with demo
recommendations. Likes are saved but do not silently invoke the incompatible old
adapter. Search omits ambiguous recording names it cannot resolve. Names,
aliases and mastering labels are metadata matches, not audio fingerprint proof.

Real artifact check with the existing 417-track library: Python and TypeScript
resolve the same 139 seeds and return identical 50-track order. This is an
integration/parity check, **not** a quality test or a tuning target. Synthetic
cross-language tests cover the same inference path. Phone speed is unmeasured.

Validation: 149 TypeScript tests, 205 Python tests, TypeScript typecheck, existing
Python/TS parity check, live smoke and both ordinary/joint builds passed. A real
headless Chromium run imported the same file, loaded six joint asset files and
zero legacy catalog files, displayed all 50 recommendations and had no page
errors. Existing live smoke covers the older path; it is not joint quality proof.

## Next evidence required for an actual quality upgrade

Improve candidate recall on training/DEV users first; then compare the fixed
candidate generator and trained ranker against the current pipeline. Reserve
new, unexposed users before selecting checkpoints or policy constants. Report
regular accuracy, held-out-artist accuracy, novelty, coverage, and playlist
length separately, using paired user-level intervals. A non-significant loss is
not proof of no loss. Reject the change if the predeclared acceptance criteria
fail; do not tune on that acceptance cohort. This audit does not authorize a
claim that discovery or audio effectiveness has been solved.

## Preview commands

No retraining or audio downloads are needed to publish this exact existing model:

```bash
cd /home/ivan/musicmylove
npm run build:joint
npx vercel deploy --prebuilt
```

The second command uploads `.vercel/output` using Vercel's Build Output API; it
does not deploy Python. Preview deliberately does not replace the production
domain. To choose another saved export: `npm run build:joint -- path/to/browser`.
Do not use the normal remote Git build to publish this ignored local model.
Reference: https://vercel.com/docs/cli/deploy and
https://vercel.com/docs/build-output-api .
