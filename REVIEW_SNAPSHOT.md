# Code review snapshot

Source commit: 2ad5d392b5de8dad2690cc3e9883bc78383ec4b8

This branch is an independent source-only snapshot for review. It deliberately
omits trained weights, embeddings, catalog artifacts and local caches. It does
not contain the development history. Training/inference requiring those artifacts
will not run until the owner supplies or regenerates them. No quality gain is
claimed by this snapshot.

Start with `python_mvp/JOINT.md`. Current joint model entry points:
- `python_mvp/joint_training.py`: disjoint-user graph/audio training and DEV ablation.
- `python_mvp/joint.py`: learned model and compact Python inference.
- `python_mvp/joint_export.py`: browser artifacts (32 MiB maximum).
- `src/lib/joint/runtime.ts` and `worker.ts`: matching browser inference.
- `python_mvp/cli.py --joint`: local use of the compact exported model.
- `python_mvp/sound_collect.py`, `sound_encoder.py`, `sound_prefetch.py`: resumable
  preview collection and real pretrained CLAP vectors.
- `python_mvp/tests/test_joint.py`, `test_sound.py`, `test_sound_prefetch.py`: tests.

Old audio experiments and the website's old engine are also present for context;
they are not the newly integrated joint model. The website has not yet been
switched to the new browser runtime. Full joint training and real-device mobile
benchmarks remain to be run by the owner. Do not initiate large downloads/training
or claim synthetic tests demonstrate musical quality.

Omitted tracked artifacts:
- `ml/model.json`
- `ml/splits.json`
- `ml/tastelift-catalog.json`
- `ml/tastelift-model.json`
- `models/catalog_embeddings.int8.bin`
- `models/catalog_indices.json`
- `models/embeddings_export_manifest.json`
- `models/embeddings_report.json`
- `models/ranker_report.json`
- `models/shuffled_track_embeddings_report.json`
- `models/tasteliftnet_weights.json`
- `models/track_embeddings.int8.bin`
