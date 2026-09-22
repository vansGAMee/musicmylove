# Implementation status — 2026-09-22

## Implemented and exercised

- Existing UI and CSS retained; static Next.js export, no application API routes.
- Real human-history projection (45,081 tracks / 671 stored train histories).
  Legacy metadata embeddings are ignored. This is a bootstrap, not a fresh dump.
- Serializable deterministic HNSW with connectivity backbone, separate optional
  audio HNSW, 1–12 heads depending on available distinct vectors, candidate
  union capped to 2,000 for scoring, deterministic top 40 and artist limit.
- Browser Web Worker, local track search, full-playlist exclusion without the
  old 500-track truncation, local feedback, batch context client/cache.
- Separate Node batch gateway, CORS, request limits, disk cache, upstream host
  and redirect checks, conservative rejection of incomplete playlist results.
- Offline dump JSONL importer with user-based split; audio-only CLAP extraction
  with provenance; verified tracklist-to-context index builder.
- Next build/dev use webpack: Turbopack emitted a worker TS asset that caused
  repeat-build type errors. `out/` is excluded from TypeScript source scanning.

## Evidence

- npm test: 136 passed.
- npm run typecheck: passed after static route migration.
- python -m pytest: 66 passed.
- python scripts/check_parity.py: 32 vectors, max error 1.776e-15.
- npm run smoke:live: passed (legacy integrations; NOT new Yandex E2E).
- npm run build: passed; routes /, /_not-found, /icon.svg are static.
- npm run smoke:offline: 40 results; reordered seed input identical.
- Playwright against exported site: imported 20 real catalog tracks, received
  40 recommendations, zero /api/ requests, zero network requests on feedback,
  zero page errors. Screenshot: /tmp/musicmylove-offline.png (local, not committed).
- Local gateway /health: OK. Actual public Yandex playlist request: geo-blocked
  from this server. No successful public Yandex end-to-end claim is justified.

## Required next work — do not claim the requested MVP is finished

1. Deploy gateway on infrastructure with working Yandex access, set
   NEXT_PUBLIC_GATEWAY_URL at static build time, validate a full public playlist
   from a Russian user network. User currently has only Vercel, no server.
2. Acquire actual permitted audio and checkpoint, execute embed_audio.py, rebuild
   with AUDIO_VECTORS. Current audio coverage is ZERO.
3. Acquire real web tracklists, run build_context.ts; implement/connect actual
   batch search provider at CONTEXT_BATCH_URL. Current provider is UNCONFIGURED;
   gateway adapter alone is not a working web search fallback.
4. Import a real ListenBrainz dump via import_dump.py and set GRAPH_INPUT.
   Current small cached histories do not establish broad recommendation quality.
5. Measure relevance and coverage on validation users; never tune on frozen test.

No downloads or generated catalog are committed. Build recreates public/data
from the tracked bootstrap source unless GRAPH_INPUT is supplied. No deployment
or new external service account was created.
