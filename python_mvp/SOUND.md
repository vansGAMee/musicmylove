# Real sound in the Python MVP

Graph-supported candidates → optional CLAP audio gate → existing neural ranker →
existing diversity rules. Only discovery mode 2 uses the gate. No old weights,
graph, train/dev/final partitions or model signatures are changed. No retraining.

The official **laion/larger_clap_music** pretrained audio branch is used through
Transformers 4.57.6. Source: https://huggingface.co/laion/larger_clap_music
Revision and checkpoint/config checksums are pinned. Audio branch loading is
strict; missing parameters cannot silently stay random. Text is not an audio input.

A centered 10-second window, mono 48 kHz, official preprocessing, batch one,
float32 inference and a normalized 512D vector. CUDA is automatic when available;
`--device cuda` fails rather than silently running a long job on CPU. No full-song
library is downloaded. Previews are bounded to 12 MiB/60 seconds and deleted even
on failure. The current provider normally supplies short previews. A short window
cannot describe every section of a song.

## Commands

From the repository root, use a separate environment that reuses existing torch:

```bash
python -m venv --system-site-packages .venv-sound
.venv-sound/bin/python -m pip install -r python_mvp/requirements-sound.txt
.venv-sound/bin/python -m python_mvp.sound_import download
.venv-sound/bin/python -m python_mvp.sound_collect --limit 100 --encode --device cuda
```

The model download is about 776 MB plus small configs. Installation can additionally
fetch Python dependencies. Existing CUDA-enabled torch is reused. The last command
is the first real pretrained integration/performance check; inspect `encoded`,
`errors`, `available_fraction`, elapsed time and the rough full-catalog ETA in
`python_mvp/data/cache/sound-clap-v1/collection-report.json`.

If it succeeds, process the current whole 56,875-track catalog, favorites first:

```bash
.venv-sound/bin/python -m python_mvp.sound_collect --limit 56875 --encode --device cuda --library ~/Downloads/liked.json
python -m python_mvp.cli --run python_mvp/data/targeted-v1/model --mode 2
```

Repeat the same collection command after interruption: completed vectors are
skipped. `--limit` is the TOTAL target prefix, not a number of additional tracks;
increase it to expand coverage. Without `--encode`, only metadata availability
is measured; the checkpoint and preview audio are not downloaded.

Inference does not require transformers, a GPU or the audio checkpoint. The default
CLI reads available cached vectors and reports coverage. Use `--no-sound` for an
exact selection comparison with the old graph/neural policy.

The 512 float32 numbers themselves are 2 KiB per track. Record headers, provenance,
filesystem blocks and metadata caches add overhead: expect several hundred MB for
this catalog, not a 2 KiB total file. Music transfer across tens of thousands of
previews can still total tens of GB, despite deleting each temporary file. Full
catalog time is not guaranteed: benchmark your GPU/network first. No weeks-long
training is required.

## Identity and coverage

Automatic lookup uses https://api.deezer.com/search and only exact normalized
artist + full title matches with a single nonempty provider ISRC among returned
results. Different versions, ambiguous results or missing previews are skipped.
Provider IDs, ISRC and source identity are saved. Preview URL caches respect signed
expiry with a safety margin and at most ten minutes; failed downloads invalidate
the lookup so a restart obtains a fresh URL. This is **metadata matching, not
fingerprint verification against the MusicBrainz recording**. No chart rank,
provider recommendation, Spotify data or LLM output enters ranking. Provider
availability/access limits can change; network requests have a descriptive
User-Agent, bounded retries and a failure stop. Responses are cached under ignored
`data/cache/`. No negative response is silently turned into a synthetic vector.

A live metadata-only probe on 40 deterministically sampled catalog tracks found
29 preview matches (72.5%). This is not proof of majority coverage across the full
catalog, successful decoding of every preview, or recommendation improvement.

Alternative local import (no network): files named exactly `Artist - Title.ext`
or `catalog-ID.ext`; ambiguous matches are skipped. Explicit JSON mapping is a
list of `{"track_id":"actual catalog ID","path":"relative-to-map/file.mp3"}`
or `{"track_id":"actual catalog ID","url":"https://...preview..."}`.

```bash
.venv-sound/bin/python -m python_mvp.sound_import build --audio-dir ~/Music --device cuda
# Or: ... sound_import build --map /path/to/map.json --device cuda
```

## Selection policy and limits

For each seed-artist direction, compare candidates against every available seed
vector in that direction, taking the strongest cosine match. If at least ten
candidate comparisons exist, remove the bottom fifth (strictly below its 20th
percentile). Small groups are only annotated. This is an explicit experimental
product rule, not a calibrated genre/quality classifier. Neural scores themselves
are unchanged. No artist blacklist, global average taste vector or audio-to-graph
coordinate mixing. Missing audio never rejects a candidate. Cache contract/version
mismatches fail visibly. CLI `[звук ...]` means measured cosine similarity, not
verified genre or a guarantee of liking the song.

Tests cover mechanics, bounded real WAV decoding, cache integrity, identity,
resume, missing-audio parity and CLI integration. A tiny randomly initialized
Transformers model was used only to verify adapter API compatibility. The official
776 MB checkpoint and mass processing are deliberately left to the commands above;
this does not constitute a pretrained end-to-end or listening-quality validation.
