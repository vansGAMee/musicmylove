# CURRENT STATUS

Код продолжен в рамках текущего python_mvp. Готовы загрузчик официальных daily
ListenBrainz dumps, проверка SHA-256/HTTP Range, извлечение без tar path traversal,
сохранение user_name/timestamp, stable user sampling, изолированные run directories,
artifact contracts, отдельный negative sampler, blockwise co-occurrence,
DEV-only range calibration, автоматический pipeline и реальный engineering smoke.
MultiInterest использует differentiable routing и residual projections, сохраняя
пространство GraphEncoder вместо случайного поворота перед retrieval.

Обычные модели загружаются только после DEV_PASS. Диагностические smoke-веса
НЕ принимаются recommend.py и НЕ обозначают доказанного качества.
Завершённые проверки и real-data результаты записаны в `reports/continuation.json`.

# NEXT COMMAND

Из /home/ivan/musicmylove/python_mvp:

```bash
python run_pipeline.py all --days 14 --user-fraction 0.25
```

Не меняйте конфигурацию незавершённого run. Для другого датасета/кода/config:
добавить `--run data/experiment2`. Для проверки компонентов на небольшой выборке:

```bash
python run_pipeline.py smoke --run data/smoke-real --days 1 --user-fraction 0.05
```

# EXPECTED RUNTIME

Скачивание 14 архивов порядка 3–4 GB; зависит от сети. В этой сессии один архив
около 240 MB скачивается за несколько минут. На той же скорости 14 архивов —
ориентировочно 1–3 часа. Построение и обучение на CPU — дополнительно часы,
возможно дольше суток; большой запуск ещё не измерен. Логи обновляются во время
работы; runner каждые 30 секунд сообщает, что subprocess продолжает работать.

# EXPECTED FILES

- data/main/data/cache/listenbrainz/download_manifest.json: фиксированные URL.
- Там же .tar.zst, .part при обрыве, .jsonl и .report.json.
- data/main/data/dataset.json, split_manifest.json, graph.json и *.npz.
- data/main/models/{graph,shuffled,taste,ranker}_{42,43,44}.pt.
- data/main/reports: audits, epoch curves, baselines, calibrations, comparisons,
  shadow/final reports, hashes, experiments.jsonl и subprocess logs.
- recommend: reports/last_recommendation.json внутри соответствующего run.

# SUCCESS CONDITION

Каждый этап exit 0; release_final.json status PASS для трёх training seeds.
После этого `python run_pipeline.py recommend tracks.txt` печатает до 50 треков.
Если human-supported candidates недостаточно, не заполнять выдачу шумом.
Просто ENGINEERING_PASS или обновление весов НЕ означает качественную модель.

# IF FAIL → EXACT COMMAND / REPORT TO INSPECT

- Network interruption: повторить ту же `run_pipeline.py all ...`. .part сохраняется.
- CHECKSUM_FAIL: проверить названный файл; удалить только повреждённый .part
  (или указанный corrupt archive), затем повторить ту же команду.
- DATA_FAIL/IDENTITY_FAIL: `data/main/reports/data_audit.json`, collection.json.
  MSID никогда не считать MBID; user IDs не реконструировать из old sessions.
- GRAPH_FAIL: graph_audit.json, baselines_dev.json; проверить независимую поддержку,
  покрытие и распределения. Не трогать артистов и не снижать gates для PASS.
- GRAPH_ENCODER_FAIL: graph_curves_SEED.json, graph_comparison_SEED.json.
- TASTE_FAIL: taste_dev_SEED.json и taste_curves_SEED.json.
- RANGE_FAIL: range_calibration_SEED.json / range_dev_SEED.json, recall по профилям.
  При недостатке кандидатов ranker не обучать.
- RANKER_FAIL: ranker_dev_SEED.json и ranker_curves_SEED.json.
- ARTIFACT_MISMATCH: не редактировать hashes/status; создать новый run.
- Большой запуск провалил качество: это ML research failure, а не permission error.
  Данные/архитектуру нельзя объявлять хорошими по прошедшим unit tests.

Отдельные этапы (те же --run и --seeds):

```bash
python run_pipeline.py data --days 14 --user-fraction 0.25
python run_pipeline.py graph
python run_pipeline.py train-graph
python run_pipeline.py train-taste
python run_pipeline.py train-ranker
python run_pipeline.py evaluate
python run_pipeline.py release
```

# DO NOT MODIFY

Frozen split_manifest.json и final_consumed.lock; downloaded originals;
artifact hashes/status; quality thresholds ради PASS; порядок user splits;
единый candidate_range для train/eval/inference. Не подгонять артистов/жанры,
не добавлять score bonuses, pretrained веса, LLM или fake audio.
Не пытаться выдать diagnostic/smoke.pt за graph_SEED.pt.

Known boundaries: in-RAM preprocessing; no mid-epoch optimizer resume; explicit
likes недоступны; confidence/radius direct-neighbor range может иметь низкий recall.
Только данные и DEV evidence определяют, достаточно ли текущей архитектуры.
