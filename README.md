# MusicMyLove

Give MusicMyLove five real songs and it returns twenty deterministic recommendations. ListenBrainz Labs supplies only the candidate pool; this repository owns the labels, 17 order-independent features, weighted BPR training, residual neural scoring, validation selection, diversity policy, and playlist assembly.

The committed production artifact is a real 17→16→8→1 residual MLP trained on public listening preferences. On the final untouched 60-user test it improved NDCG@20 by 69.4% over the strongest non-learned baseline, with a user-clustered 95% bootstrap interval entirely above zero. Full evidence is in [reports/verification.md](reports/verification.md); the earlier failed frozen attempt remains in [reports/evaluation-first-frozen.json](reports/evaluation-first-frozen.json).

## Run the app

```bash
npm install
npm run dev
```

## Verify it

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
npm test
PYTHONPATH=. .venv/bin/pytest tests/python -q
PYTHONPATH=. .venv/bin/python scripts/check_parity.py
npm run typecheck
npm run lint
npm run test:e2e
npm run smoke:live
npm run build
```

## Reproduce the real-data pipeline

Public usernames, profiles, and retrieval responses are deliberately ignored by Git. `sample_users.py` downloads one official incremental archive, verifies its published SHA-256, extracts active usernames, and deletes the archive. `collect_real.py` is sequential, rate-limited, resumable, schema-validated, and atomically cached. The checked-in aggregate reports contain no usernames.

```bash
SOURCE_URL=https://ftp.musicbrainz.org/pub/musicbrainz/listenbrainz/incremental/listenbrainz-dump-2660-20260913-000002-incremental/listenbrainz-listens-dump-2660-20260913-000002-incremental.tar.zst
PYTHONPATH=. .venv/bin/python scripts/sample_users.py --source-url "$SOURCE_URL" --seed 41 --output data/manifests/usernames.json
PYTHONPATH=. .venv/bin/python scripts/collect_real.py --stage all --manifest data/manifests/usernames.json --split data/manifests/real-splits.json

PYTHONPATH=. .venv/bin/python scripts/sample_users.py --source-url "$SOURCE_URL" --seed 73 --exclude data/manifests/usernames.json --output data/manifests/usernames-second.json
PYTHONPATH=. .venv/bin/python scripts/collect_real.py --stage all --manifest data/manifests/usernames-second.json --split data/manifests/real-splits-second.json
PYTHONPATH=. .venv/bin/python scripts/combine_splits.py --first data/manifests/real-splits.json --second data/manifests/real-splits-second.json --output data/manifests/real-splits-final.json

PYTHONPATH=. .venv/bin/python scripts/train_real.py --split data/manifests/real-splits-final.json
PYTHONPATH=. .venv/bin/python scripts/evaluate_real.py --split data/manifests/real-splits-final.json --marker data/manifests/final-frozen-test-evaluated.json --report reports/evaluation.json
```

The final study used a documented sequential protocol after the first frozen test failed: all 394 first-cohort users became development data; a disjoint second cohort contributed 277 train, 59 validation, and 60 new frozen-test users. The model and residual scale were selected only on validation before the second test was opened once.

## Model

For every candidate, MusicMyLove builds five normalized similarities, five reciprocal ranks, total RRF, multi-seed support, maximum/second/mean/stddev similarity, and same-artist evidence. The neural network learns a residual correction on top of RRF and starts exactly at the baseline. The exported JSON contains all weights, the RRF residual index, scale, and selected production ranker; `src/lib/mlp.ts` runs it without Python.

Artist+title identity normalization prevents alternate MusicBrainz recording MBIDs from becoming false negatives or duplicate seed recommendations. The final list is capped at two tracks per artist and is stable under all 120 permutations of five seeds.

ListenBrainz similarity is externally precomputed and may include held-out-user activity. The experiment therefore proves the owned MusicMyLove reranker on top of a fixed external retrieval layer, not an independently trained end-to-end retrieval system.
