# Repository instructions

- Keep the app database-free and production inference pure TypeScript.
- Never use LLMs, Spotify data, or audio analysis for recommendation logic.
- Keep ranking permutation-invariant and final output deterministic.
- Split evaluation by user; do not tune on the frozen test split.
- Never treat a known user-positive track as an implicit negative.
- Cache external API responses under ignored `data/cache/`; never commit downloads.
- Use a descriptive User-Agent and bounded retries for ListenBrainz/MusicBrainz.
- Run `npm test`, `npm run typecheck`, `python -m pytest`, `python scripts/check_parity.py`, `npm run smoke:live`, and `npm run build` before claiming completion.
