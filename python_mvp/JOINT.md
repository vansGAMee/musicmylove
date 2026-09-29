# Learned graph + sound ranking, compact browser deployment

The old optional CLAP percentile filter is NOT joint training. This separate path
uses already collected genuine CLAP vectors as frozen audio inputs and learns from
listener histories. It does not download music or run CLAP again.

## Run after sufficient collection

The collection already running can finish. It may also be stopped with Ctrl+C:
existing completed vectors are usable, and missing audio is explicitly masked.
No need to restart collection from zero. For the cleanest fixed training snapshot,
stop collection or let it finish before preparing a joint run.

```bash
cd /home/ivan/musicmylove
.venv-sound/bin/python -m python_mvp.joint_training --device cuda
```

This command streams existing ranking-user data, reads available cached audio once,
prepares training pairs, trains two small models, evaluates DEV, and exports compact
inference files. No source graph/weights are overwritten. Default output:
`python_mvp/data/joint-v1`. Repeat the command after interruption: completed
preparation and completed model fits are reused. An interrupted model fit restarts
its small optimization phase; it does not redo collection. To include audio gathered
AFTER the snapshot, use a new `--run python_mvp/data/joint-v2`.

Do not run multiple GPU jobs together on 16 GB RAM. Training uses CUDA when requested;
it fails visibly if CUDA is unavailable. Neither GPU speed nor quality gain is
promised before a measured run. This implementation does not retrain CLAP or the
large graph. Default optimization: at most 12 epochs each, early stopping after
three non-improving DEV epochs, batch 512, CPU threads capped at four.

## What is learned

Per track: 96 frozen graph coordinates + 512 frozen music-audio coordinates + audio
availability + representation-only listener popularity. A trainable 610→64→32 MLP
produces a shared normalized embedding. Candidate-to-profile maximum/mean affinity,
artist familiarity, popularity and audio coverage enter a trainable 6→16→1 ranker.
Pairwise preference loss propagates through BOTH networks. Thus audio influence
and its interaction with graph information are learned; they are not a fixed
cosine cutoff or an artist blacklist. Missing audio is a mask, never a pretend
measured vector. Popularity is a learnable signal, not a hand-coded penalty.

Training users are disjoint from graph representation users and DEV. Complete known
positive families/aliases are excluded from negative sampling. Unknown tracks are
implicit comparison items, not explicit dislikes. Regular and held-out-artist
profile episodes are included. Frozen FINAL is not used. No claims of perfect
personal taste or arbitrary songs being objectively good/bad are justified.

The same architecture is trained without audio as an ablation. `evaluation.json`
reports both DEV NDCG@50 values and a paired user-level bootstrap interval. DEV also
selects checkpoints, so this is exploratory evidence, not a final independent
release test. `JOINT_NO_PROVEN_DEV_GAIN` must not be described as proven improvement.
Even `JOINT_DEV_GAIN` needs separate final/listening validation before such a claim.

## Try with the existing CLI

```bash
python -m python_mvp.cli --joint python_mvp/data/joint-v1/browser --mode 2
```

The joint path uses the trained model directly and bypasses the old audio percentile
filter, familiar-share quota and popularity quota. Only seed/known-song exclusions,
family deduplication, two-per-artist and no adjacent same artist remain. Graph
retrieval reserves 16 cross-artist and 8 same-artist neighbors per track; up to 64
seed tracks are selected deterministically across artists. These bounds are shared
by training, evaluation, CLI and browser. This retrieval is a deliberate compact
alternative, not a claim to match the old full-history recommender exactly.

Existing personal-adapter weights are incompatible with this new feature space;
they are disabled explicitly. CLI feedback remains saved, but is not yet included
in the joint training labels. Menu 8 shows compact catalog/audio coverage. Joint
mode ignores the old mixed/neural/discovery ordering distinctions: one trained
policy is used. `--no-sound` is for the OLD recommender, not a joint-model ablation.

## Browser deployment artifacts

`browser/` contains float32 **learned 32D vectors**, small ranker weights, top-24
neighbor indices, minimal per-track info, catalog and checksummed manifest. No
raw audio, 512D CLAP cache, CLAP checkpoint, listener histories or Python runtime.
Estimated current catalog payload is about 24 MB uncompressed; exact bytes are
reported in `manifest.json`. Export fails above 32 MiB. Runtime working memory and
mobile speed still require real device testing; this is not a promise for every
phone or unlimited free hosting traffic.

`src/lib/joint/runtime.ts` implements matching math without Node or model-runtime
dependencies. `src/lib/joint/worker.ts` keeps inference off the UI thread:

```ts
const worker = new Worker(new URL('./path/to/joint/worker.ts', import.meta.url), {type:'module'});
worker.postMessage({id:1,type:'load',base:'/joint-model'});
// After ready, pass catalog indices, not guessed titles:
worker.postMessage({id:2,type:'recommend',seeds:[12,98],known:[]});
```

Copy only `browser/` contents to an independently published static asset folder.
The website now has an explicit joint-model build: `npm run build:joint`.
It verifies and copies this export, enables joint inference in the existing UI,
and prepares `.vercel/output` for `npx vercel deploy --prebuilt` (preview).
The normal `npm run build` still uses the older recommender. See
`docs/audits/2026-09-29-joint-release.md` for the measured quality limitations;
integration is not proof of an improved discovery model.

Tests verify gradients reaching audio inputs, protected negatives, deterministic
retrieval, compact export, CLI integration, and Python↔TypeScript scores/order on
synthetic fixtures. Small synthetic optimization tests check the training loop.
The current trained run and its measured DEV/SHADOW results are documented in
`docs/audits/2026-09-29-joint-release.md`. Phone benchmarks are still unmeasured;
passing unit tests is not evidence of recommendation quality.
