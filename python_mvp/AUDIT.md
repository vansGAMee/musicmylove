# Discovery audit, 2026-09-25

No dataset training or downloads were run for this update. The existing ranker,
graph vectors and taste weights remain unchanged and usable by the CLI.

## Immediate input fix

The supplied library contains 417 unique artist/title lines. Strict matching
uses 62 catalog records. The CLI now also uses 31 explicit metadata-only records
whose canonical artist/title matches exactly, despite another MBID record having
the same name. It reports these as metadata matches, not recording identity.
Total: 93 profile seeds. IDs and embeddings are not merged. This is better input
coverage, not measured recommendation improvement. `--strict-matching` reproduces
the prior matching policy. Remaster-label matching is separately marked; live,
remix and acoustic distinctions are preserved during profile resolution.

## Proven methodology limitations

1. `discovery.fit` discards the candidate IDs returned by retrieval. Pair sampling
   uses all hidden targets and negatives across the catalog. Thus training can
   include positives that cannot reach serving. This is a real train/serve
   mismatch; its effect on quality is not isolated by the current experiments.
2. Own-user exclusion only removes that user's contribution to listener scores.
   Popularity denominators, graph edges, graph diffusion, learned embeddings and
   taste features still derive from the original training representation. This
   is not a complete leave-one-user-out representation. It is a possible source
   of overfitting, not evidence that DEV/final users leaked into graph fitting.
3. Original evaluation compares neural candidates with full-catalog baselines,
   and only evaluates the constrained playlist for the network. The new
   `audit_discovery` compares all methods on the SAME candidate set and SAME
   selection policy, reporting both ranking and actual playlist metrics.
4. DEV includes users used for epoch selection. Neither full DEV nor its named
   outside-selection subset is a fresh frozen test after iterative inspection.
   Bootstrap intervals on DEV are diagnostic, not independent release evidence.
5. Selecting the best epoch does not ensure that the network beats a baseline.
   Existing `DEV_EVALUATED` labels mean evaluated, not promoted or release-ready.
   The new audit does not silently switch methods or declare a release PASS.

## Scope of changes

Signed training/inference source files were deliberately kept unchanged so the
current CLI remains compatible with trained weights. New matching and fair audit
are separate modules. Fixing the training procedure requires a separately
versioned training run; changing its source alone cannot improve existing weights.
The fair audit reads DEV queries only and never consumes frozen final.

Run the CLI as before: `python -m python_mvp.cli`.
Run a diagnostic comparison: `python -m python_mvp.audit_discovery`.
Private profiles, playlists, checkpoints and metric reports remain under ignored
`python_mvp/data/` and are not committed.

## Measured fair comparison

On 352 DEV profiles, with the same candidate pool and discovery policy for all
methods, neural playlist NDCG@50 is 0.152763 versus listener baseline 0.167222.
For 283 profiles with cross-artist targets, actual-playlist cross-artist NDCG@50
is 0.021111 versus 0.044070. The paired DEV bootstrap interval for playlist
NDCG difference (neural minus listeners) is [-0.029271, -0.000976]. These are
DEV diagnostics, not an independent test. Equal comparison conditions do not
remove the observed neural deficit. Full local evidence: ignored
`data/discovery-v3/fair_audit_mode2.json`.
