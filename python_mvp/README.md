# Свой нейронный рекомендатель: запуск из терминала

Это отдельный Python-проект. Production-сайт не меняется.
GraphEncoder, multi-interest routing и neural ranker обучаются с нуля.
Нет pretrained embeddings, LLM, жанровых правил, Spotify recommendations или аудио.
**Качество TOP50 пока не доказано. Команда обучения может честно закончиться FAIL.**

На текущем компьютере Python, PyTorch, NumPy, SciPy и `zstd` уже доступны.

```bash
cd /home/ivan/musicmylove/python_mvp
python run_pipeline.py all --days 14 --user-fraction 0.25
python run_pipeline.py recommend tracks.txt
```

`tracks.txt`: до 2000 строк `Artist - Title` или точных Recording MBID.
Последнюю команду запускать после успешного обучения и release-проверки.
Никакого ручного скачивания или редактирования путей не нужно.

Pipeline сам найдёт официальные ListenBrainz daily archives, скачает с продолжением
после обрыва, проверит официальную SHA-256, выделит устойчивую выборку пользователей,
подготовит данные, построит граф, обучит три сети с seeds 42/43/44,
проверит shadow и однократно frozen final. Новые модели и данные лежат в
`data/main/`, логи и метрики — в `data/main/reports/`.
Повторить ту же команду можно после обрыва: успешные совместимые этапы пропускаются.
Незавершённый epoch начинается заново; продолжение optimizer state пока не реализовано.

После FAIL смотреть указанный лог. **FAIL качества не обходить, тест не переиспользовать.**
После использования final run заморожен. Новый эксперимент: `--run data/experiment2`.
Архивы, пользовательские listens и веса игнорируются Git.

Для нового компьютера:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
# zstd устанавливается системным пакетным менеджером
.venv/bin/python run_pipeline.py all --days 14 --user-fraction 0.25
```

Подробное состояние и точные команды диагностики: [HANDOFF.md](HANDOFF.md).

## Что защищено кодом

- User splits до graph fit; разные user/session сигналы; отдельные support/confidence.
- MBID не подменяется MSID. Разные recordings сохраняются; ambiguous input unresolved.
- Нет random/cold embeddings в recommendable catalog и heuristic fallback.
- Checkpoints привязаны к dataset, split, graph, ordered vocabulary, model config,
  implementation hashes, seed, epoch и DEV metrics. Несовместимость — ошибка.
- Negatives исключают все известные positives и сильную человеческую поддержку.
- Перестановки/дубли seeds не меняют taste inference. Несколько routing heads,
  проверка collapse, устойчивости и смешанных реальных профилей.
- Радиус и confidence калибруются только по DEV. Ranker обучается на том же range,
  что использует выдача. Итоговый score исключительно neural output.
- Full-catalog metrics, graph/popularity/PPR/random/trained-shuffled controls,
  candidate recall, разные размеры профилей, cross-artist test, paired uncertainty.

## Практические ограничения

Выборка из incremental dumps — **не весь ListenBrainz и не полная история каждого**.
14 дней обозначают окно поступления данных, не обязательно время прослушивания.
Это implicit listening positives, а не явные лайки. Данных может не хватить для
устойчивого качества; downloader не гарантирует прохождения ML quality gates.
Большие библиотеки >2000 уникальных треков пока исключены из graph fitting,
но остаются в датасете; это ограничение покрытия, а не музыкальное правило.
Graph co-occurrences считаются блоками, однако итоговые sparse matrices и
preprocessed dataset всё ещё должны помещаться в RAM. Начните с указанной выборки.

Официальный формат и происхождение:
https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html
https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/

Исходное задание — REQUIREMENTS.txt. Более позднее указание пользователя имеет
приоритет: сначала сложный код, сбор/обучение автоматизировать, PLAN.md не создавать.
