# Non-targeted population training

This removes **personal-library-directed data enrichment** from a new training
run. It is not a promise of good recommendations for everybody. ListenBrainz
users, three cached submission archives, catalog support thresholds and preview
availability still introduce coverage and population biases.

```bash
cd /home/ivan/musicmylove
.venv-sound/bin/python -m python_mvp.population
```

No new download is needed. The command verifies the cached `expanded-v1` data
lineage and its original graph weights, reuses measured audio by track identity,
and trains new joint and graph-only models on CUDA. It never loads your library
or uses `targeted-v1`/`joint-v1` embeddings or ranker weights. The neutral graph
representation is reused; the obsolete neutral ranker is checked for artifact
provenance but its weights are not used by this pipeline.

The local preflight verified 5,507 potential training users and 1,363 reserved
audit users before episode eligibility filtering. Reservation is a fixed hash of
user identity (20% of neutral ranker users); it does not inspect artists, tracks,
genres or likes. Graph representation users are disjoint from all ranker users.
Audit users are excluded from training pairs and checkpoint selection. DEV alone
selects checkpoints; the independent report runs after both checkpoints are fixed.
It does not select a winning model or tune a policy from audit results.

Results: `python_mvp/data/population-v1/population-evaluation.json`. It separates
ordinary held-out tracks from held-out artists, includes unreturned targets,
and reports candidate recall, ranking accuracy, novelty and playlist length.
Additional strata use input popularity and representation-only catalog thresholds,
not handpicked artists. Each task has one episode per eligible user; confidence
intervals use users as the resampling unit. Small or missing strata are not proof
of broad quality. This compares two freshly trained models, not an independent
improvement estimate against the older personalized model (which could already
have trained on these users).

These users may have appeared in prior project experiments. They are held out
from this new joint training, not newly collected untouched humans. FINAL is not
used. `release_certified` remains false; this is an experimental model to assess.

The run is separate and resumable: completed preparation and completed model
fits are reused. Interrupted optimization restarts only the unfinished model.
The training/audit snapshot is checksummed. Input/code changes require a new
`--run`; old weights and personal data remain unchanged. This change extends
`joint_training.py`, so old joint training contracts deliberately reject a code
mismatch; the existing exported models still work for inference.

Try the trained model:

```bash
python -m python_mvp.cli --joint python_mvp/data/population-v1/browser
```

After inspecting the report, publish a **preview**, without replacing production:

```bash
npm run build:joint -- python_mvp/data/population-v1/browser
npx vercel deploy --prebuilt
```

The `-- .../population-v1/browser` argument matters: without it the website build
uses the older `joint-v1` export. The preview uses the same compact browser format;
no audio files, CLAP checkpoint, Python runtime or listener histories are shipped.
Use `python -m python_mvp.population --check-only` for provenance validation without
training. Typical training time is not yet measured for this new pipeline.
