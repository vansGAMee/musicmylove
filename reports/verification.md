# Verification — 2026-09-13

## Observed data and evaluation

- Downloaded training data: **0 bytes**. No remote cohort was downloaded for this run.
- Usable users: **30 deterministic fixture users**; frozen user split: **21 train / 4 validation / 5 test**.
- Training examples: **400 synthetic BPR pairs**, 80 epochs; final observed BPR loss **0.0000630566**.
- Evaluated users/examples: **5 / 5**; average candidate pool: **50**; artist diversity at 20: **18**.
- Retrieval Recall: **1.0** for all rankers on the deterministic fixture.
- Max-similarity: Recall@20 **0.5**, NDCG@20 **0.386853**, HitRate@20 **1.0**.
- RRF: Recall@20 **1.0**, NDCG@20 **0.817530**, HitRate@20 **1.0**.
- Neural: Recall@20 **1.0**, NDCG@20 **0.483813**, HitRate@20 **1.0**.
- RRF-minus-max paired bootstrap NDCG 95% interval: **[0.430677, 0.430677]**. The zero-width interval is an artifact of identical deterministic fixture examples and is not population uncertainty.
- Selected production ranker: **RRF**, because it wins the validation-style fixture comparison and the fixture-trained MLP does not improve it.

These are pipeline-verification metrics, not a claim about recommendation quality across real users. A target-scale 300–500 public-user collection must be run before making that claim.

## Observed verification commands

- `npm test`: **15 tests passed across 7 files**.
- `.venv/bin/python -m pytest tests/python -q`: **4 passed**.
- `PYTHONPATH=. .venv/bin/python scripts/check_parity.py`: **32 vectors**, maximum absolute Python/TypeScript error **3.553e-15**.
- `npm run lint`: **passed with no findings** after fixing the effect and combobox issues it identified.
- `npm run test:e2e`: **1 Playwright happy-path test passed**; five selections automatically produced 20 cards without a Generate button.
- `npm run build`: **passed** on Next.js 16.3.5; `/` static plus three dynamic API routes.

## Live external API smoke

`npm run smoke:live` succeeded against current official Labs APIs with five live search-resolved recordings: Portishead — Roads, Massive Attack — Teardrop, Radiohead — Everything in Its Right Place, Björk — Hyper-Ballad, and M83 — Midnight City.

- Similarity failures: **0**
- Unique retrieved candidates: **314**
- Final results: **20**
- Seeds excluded: **yes**
- Duplicate recording MBIDs: **none**
- Maximum tracks per artist: **2**
- Usable exact-or-search Spotify actions: **20 / 20**
- Reversed-seed ordering identical: **yes**

## Known limitations

- ListenBrainz Labs similarity is an externally precomputed aggregate system and may itself include activity from held-out users. The metrics evaluate this repository's reranker on top of that retrieval system and do not show that the complete retrieval stack is free from test-user influence.
- The committed evaluation is intentionally fixture-scale because no arbitrary username dump was fabricated or downloaded. Use `scripts/collect.py` with a legitimate public-username manifest, then retrain and evaluate, for a meaningful cohort result.
- Browser localStorage is best-effort cache/feedback storage. Clearing site data removes it, as expected for an account-free MVP.
