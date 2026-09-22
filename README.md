# MusicMyLove

Текущий UI сохранён. Новый frontend — статический Next.js export; поиск,
GRAPH/AUDIO HNSW, taste heads и детерминированное ранжирование выполняются
в браузерном Web Worker на TypeScript. Like/dislike не вызывает сервер
рекомендаций. Старые API routes оставлены в `legacy/api` для регрессионных
тестов и не попадают в приложение.

## Запуск

```bash
npm ci
npm run build
npm start
```

Открыть http://localhost:3000. Можно импортировать TXT/CSV/JSON/M3U или
найти треки в локальном каталоге. `npm run dev` запускает разработку;
перед первым запуском выполнить `npm run build:catalog`.

По умолчанию сборка использует 671 сохранённую train-историю ListenBrainz
и 45 081 трек из прежнего артефакта. Старые metadata-векторы игнорируются:
новые векторы — детерминированная проекция взвешенной матрицы track/listener.
Названия используются только для поиска идентичности и отображения.
Это ограниченный начальный каталог, а не полный ListenBrainz dump.

## Яндекс: отдельный batch gateway

Vercel публикует только `out/`, без Functions. Для импорта Яндекса нужен
отдельно размещённый gateway с доступом к Яндексу:

```bash
FRONTEND_ORIGIN=http://localhost:3000 npm run gateway
NEXT_PUBLIC_GATEWAY_URL=http://localhost:8787 npm run build
```

Для размещения подготовлен `services/Dockerfile` (build context — корень
репозитория). Подключить HTTPS и постоянный volume `/app/data/cache`.
Переменные перечислены в `.env.example`. Gateway принимает один POST
`/playlist` на весь плейлист, ограничивает запросы и кеширует ответы на диск.
Неполные ответы Яндекса не выдаются за успешный импорт всего плейлиста.

**Gateway пока не размещён.** При пустом `NEXT_PUBLIC_GATEWAY_URL` импорт
показывает явную ошибку настройки. Доступность из РФ и бесплатный хостинг
не подтверждены; локальная сборка не доказывает работу публичного сервиса.

## Офлайн данные

Извлечённый из официального ListenBrainz dump файл listens JSONL:

```bash
python scripts/offline/import_dump.py listens.jsonl
npx tsx scripts/offline/build.ts data/cache/offline/dump-histories.json
```

Импортёр потоковый, имеет ограничения памяти и стабильный split по пользователю.
Validation/test пользователи не входят в граф. Для deployment с этим
источником задайте `GRAPH_INPUT=data/cache/offline/dump-histories.json`.

Для **настоящего аудио** нужны доступные аудиофайлы и manifest
`[{"mbid":"...","path":"data/cache/audio/file.wav","source":"https://...","license":"..."}]`:

```bash
# Отдельное офлайн-окружение; Python не нужен в production.
pip install laion-clap
python scripts/offline/embed_audio.py audio-manifest.json checkpoint.pt data/cache/audio-vectors.json
AUDIO_VECTORS=data/cache/audio-vectors.json npm run build
```

Используется только аудиоветвь CLAP, checksum checkpoint и каждого файла
записываются в provenance. **Аудиофайлы и checkpoint ещё не загружены;
в текущей сборке audio coverage = 0.** Для таких треков звуковой score отсутствует.

Для web context: подготовить извлечённые из реальных страниц tracklists
`[{"url":"https://...","tracks":[{"artist":"...","title":"..."}]}]`, затем:

```bash
npx tsx scripts/offline/build_context.ts tracklists.json data/cache/web-context.json
WEB_CONTEXT=data/cache/web-context.json npm run build
```

Один `/context` batch проверяет постоянный кеш и передаёт оставшиеся пробелы
на `CONTEXT_BATCH_URL`. Контракт провайдера: POST `{tracks:[{artist,title}]}`,
ответ `{context:{"нормализованный artist\\u001fнормализованный title":{neighbors:["recording MBID"],sources:["https://страница"]}}}`.
**Поисковый провайдер и исходный web corpus ещё не подключены.** Без них
неизвестные треки остаются неизвестными, никакие связи не выдумываются.

## Проверки

```bash
npm test
npm run typecheck
python -m pytest
python scripts/check_parity.py
npm run smoke:live
npm run smoke:offline
npm run build
```

`smoke:live` пока проверяет прежние внешние интеграции; новый локальный
пайплайн проверяется `smoke:offline`. Полная новая live-проверка требует
размещённого gateway, реального плейлиста и подключения web/audio данных.
Все скачанные данные и артефакты сборки игнорируются Git.
