# TasteLift Task 5 Report

## Serving contract

`src/lib/tastelift/model.ts` loads only the committed `tastelift-v1` JSON and
implements the exported contract in pure TypeScript. Text is NFKC-normalized,
lowercased and whitespace-collapsed; padded 3--5 Unicode-code-point ngrams are
UTF-8 FNV-1a-32 hashed, deduplicated and numerically sorted. Artist/title exact
vocabulary lookups share the exported subword embedding fallback.

For a track, the serving equation is:

```
e(track) = L2(tanh(W [artist; title; mean(artist-grams); mean(title-grams)] + b))
h_k = L2(sum_i softmax(L2(q_k) dot e_i / 0.2)_i * e_i)
p_k(c) = softplus(log_score_scale) * h_k dot e(c)
affinity(c) = 0.1 * logsumexp_k(p_k(c) / 0.1)
lift(c) = affinity(c) - softplus(log_popularity_weight) * popularityPercentile(c)
```

The complete 5--200 seed set is canonical-sorted before track encoding and
attention reduction, so no supplied seed order affects head vectors or scores.

## Residual ranking mapping

Legacy `buildFeatures` remains exactly 17 retrieval-evidence features, and
`legacyResidualScore` is the unchanged MLP forward pass. The optional TasteLift
path then attaches and returns:

- `residualScore` (the old ranking score)
- `tasteHead`, `tasteHeadIndex`, `perHeadScores`
- `seedSupport` plus stable `seedSupportEvidence`
- `popularityPercentile`, `affinityScore`, and `liftScore`

Its deterministic additive ranking score is:

```
finalScore = residualScore + liftScore + seedSupport / seedCount
```

This keeps every existing rank/recommend call and legacy output path unchanged;
`createTasteLiftRankedCandidates` exposes only the enriched candidate pool. It
does not create a 40-track slate. Spotify data is not read by the inference or
feature code.

## Parity and tests

`tests/python/test_tastelift_parity.py` runs the real exported PyTorch model and
the executable `scripts/tastelift-parity.ts` on exact-vocabulary metadata,
Unicode/NFKC text, and OOV metadata. It compares four head vectors, per-head
scores, affinity, popularity prior, and lift. Maximum observed absolute error:
`2.95480617307e-7` (asserted below `2e-6`).

`tests/unit/tastelift-model.test.ts` verifies all 120 permutations of a five-
seed Unicode/OOV set produce exactly the same TypeScript result. Ranking tests
verify the legacy score is retained and TasteLift additions are explicit.

Fresh verification performed:

- `npm test` — 48 passed
- `.venv/bin/python -m pytest` — 58 passed
- `npm run typecheck` — passed
- `npm run lint` — passed
- `PYTHONPATH=. .venv/bin/python scripts/check_parity.py` — 32 vectors, max
  error `1.776e-15`
- `npm run smoke:live` — passed
- `npm run build` — passed

## Self-review and concern

The old calls have no TasteLift argument, so they return only their original
fields and scores. All external serving data is the committed model artifact;
no database, LLM, Spotify feature, audio analysis, or network call is involved.

The repository's literal `python scripts/check_parity.py` fails before testing
because this environment does not put the repository root on `sys.path` (and
the system Python lacks pytest); the equivalent virtualenv invocation with
`PYTHONPATH=.` passes. The pre-existing deletion of `reports/evaluation.json`
was preserved and is intentionally not part of this task's commit.
