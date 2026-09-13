# Repository instructions

- Keep the app database-free and production inference pure TypeScript.
- Never use LLMs, Spotify data, or audio analysis for recommendation logic.
- Keep ranking permutation-invariant and final output deterministic.
- Split evaluation by user; do not tune on the frozen test split.
- Never treat a known user-positive track as an implicit negative.
- Cache external API responses under ignored `data/cache/`; never commit downloads.
- Use a descriptive User-Agent and bounded retries for ListenBrainz/MusicBrainz.
- Run `npm test`, `npm run typecheck`, `python -m pytest`, `python scripts/check_parity.py`, `npm run smoke:live`, and `npm run build` before claiming completion.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
