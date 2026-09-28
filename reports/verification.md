# Verification — 2026-09-13

## Outcome

The committed production ranker is MusicMyLove's trained residual neural network, not an RRF fallback. On the second, untouched, user-disjoint frozen test:

- 60 users / 180 cold-start examples; five seeds and five hidden favorites per example.
- Neural NDCG@20: **0.0165177**; strongest baseline (max similarity): **0.0097521**; relative improvement: **69.38%**.
- Neural Recall@20: **0.0188889**; strongest baseline: **0.0144444**; relative improvement: **30.77%**.
- Neural HitRate@20: **0.0944444**; strongest baseline: **0.0611111**; relative improvement: **54.55%**.
- User-clustered paired-bootstrap NDCG delta 95% CI: **[0.0011920, 0.0133082]**. The interval is entirely above zero.
- RRF NDCG@20 was **0.0096817**. Retrieval recall was **0.0488889** for every ranker and is reported separately.

The full aggregate and anonymous per-example values are in `evaluation.json`.

## Cohorts and training

- One unique official incremental ListenBrainz archive: dump 2660, **239,571,172 bytes**, SHA-256 **1220457d75b5ab661bb7ef5374f0ce3b0e47d11b9c69eabb04cac8279ee7c2e8**.
- Archive population: 22,144 unique usernames; 12,149 had at least 20 listens in the increment.
- First sampled 500: 394 suitable public users. Second sampled 500: 396 suitable users, with all first-sample users excluded.
- Final split: **671 train / 59 validation / 60 frozen test**, all user-disjoint.
- Training: **2,013 examples / 20,831 weighted BPR pairs**.
- Fixed architecture: 17→16→8→1 residual MLP. Bounded validation grid selected seed 41, learning rate 0.01, epoch 15, residual scale 0.75, pure neural ranker.
- Validation NDCG@20: neural **0.0195874**, RRF **0.0188501**, max similarity **0.0093710**.
- MLHD retrieval was rejected on first-cohort validation: retrieval recall **0.0131827**, versus **0.0508475** for the selected long-history ListenBrainz index.

The first frozen test was not hidden: neural NDCG@20 was 0.0089473 versus RRF 0.0091947. That failure is preserved in `evaluation-first-frozen.json`; its users were promoted to development data, and the final claim uses a separately sampled, previously unseen second test cohort.

## Code and live verification

- Python: **33 passed**.
- TypeScript/Vitest: **21 passed**.
- PyTorch↔TypeScript inference parity: **32 vectors, maximum absolute error 0.000e+00**.
- Typecheck, ESLint, Next.js production build, and Playwright happy path: passed.
- Live five-seed smoke: 0 similarity failures, 314 unique candidates, 20 results, no seeds, no duplicate tracks, at most two tracks per artist, 20 Spotify actions, identical output after reversing seeds.

Live seeds: Portishead — Roads; Massive Attack — Teardrop; Radiohead — Everything in Its Right Place; Björk — Hyper-Ballad; M83 — Midnight City.

Live playlist: Massive Attack — Girl I Love You; Massive Attack — Sly; Björk — Army of Me; Portishead — Elysium; Portishead — Cowboys; Radiohead — Idioteque; Björk — All Is Full of Love; Radiohead — How to Disappear Completely; Muse — Time Is Running Out; Pixies — Debaser; Yeah Yeah Yeahs — Maps; Tori Amos — Cornflake Girl; FKA twigs — home with you; Grimes feat. Janelle Monáe — Venus Fly; Morcheeba — Trigger Hippie; The Strokes — Is This It; The Verve — Bitter Sweet Symphony; Muse — Plug In Baby; FKA twigs — In Time; Weyes Blood — Andromeda.

## Boundary of the claim

ListenBrainz Labs candidate retrieval is externally precomputed and may include held-out-user activity. These results isolate and prove the MusicMyLove reranking layer, identity handling, diversity policy, and playlist assembly on that fixed retrieval layer. They do not claim that the external candidate generator was trained without test-user activity.
