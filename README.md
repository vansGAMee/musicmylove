# MusicMyLove

Give MusicMyLove five real songs and it immediately returns twenty deterministic recommendations from ListenBrainz Labs. There is no login, database, Spotify API dependency, audio processing, or production Python runtime.

## Commands

```bash
npm install
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
npm run dev
npm test
.venv/bin/python -m pytest tests/python
PYTHONPATH=. .venv/bin/python scripts/train.py
PYTHONPATH=. .venv/bin/python scripts/evaluate.py
PYTHONPATH=. .venv/bin/python scripts/check_parity.py
npm run test:e2e
npm run smoke:live
npm run build
```

To collect public all-time top-recording statistics, put one public ListenBrainz username per line in an ignored file and run `.venv/bin/python scripts/collect.py usernames.txt --limit 400`. Responses go to ignored `data/cache/` and the collector is deliberately sequential.

## How it works

Each selected recording immediately starts a cached similarity request. Candidate scores are normalized independently per seed. Five sorted similarities, five sorted reciprocal ranks, total RRF, support, summary statistics, and same-artist evidence form a 17-value order-independent vector. The committed model is a 17→16→8→1 PyTorch MLP exported as JSON; `src/lib/mlp.ts` reproduces inference without Python. Validation selects the production ranker. The current honest fixture result selects RRF, not the neural experiment.

The Labs similarity data is externally precomputed and may include held-out users. Evaluation therefore measures our reranking layer on top of that system, not a fully test-user-isolated retrieval stack. Current evaluation is fixture-scale; collect a 300–500 user cohort before interpreting it as a population result.
