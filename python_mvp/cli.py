"""Interactive, inference-only music playground. Run: python -m python_mvp.cli"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import torch
from .config import ROOT, write_json, fingerprint
from .discovery import signature
from .discovery_engine import DiscoveryEngine, DiscoveryRanker, read_library, select_playlist, input_keys
from .recommend import resolve, match_key
from .profile_input import resolve_profile

MODES = {'1': ('Смешанный', 2, 1.), '2': ('Открытия: больше новых исполнителей', 2, .30),
         '3': ('Чистый порядок нейросети', 50, 1.)}


def validate_lines(lines):
    lines = sorted({line.strip() for line in lines if line.strip()})
    if not 1 <= len(lines) <= 2000:
        raise ValueError('Нужно от 1 до 2000 уникальных треков.')
    return lines


def read_path(value):
    path = Path(value.strip().strip('"').strip("'")).expanduser()
    return validate_lines(read_library(path))


def load_ranker(engine, run):
    path = Path(run) / 'ranker.pt'
    if not path.exists():
        raise FileNotFoundError(f'Нет готовой модели: {path}. Обучение из CLI не запускается.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('signature') != signature(engine):
        raise RuntimeError('Веса не соответствуют коду или данным. Укажи совместимую папку через --run.')
    model = DiscoveryRanker(state['dimensions'])
    model.load_state_dict(state['weights']); model.eval()
    return model


def export_playlist(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import tempfile
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y-%m-%d_%H-%M-%S_'), dir=output))
    write_json(folder / 'result.json', result)
    (folder / 'playlist.txt').write_text('\n'.join(
        f"{r['rank']}. {r['artist']} — {r['title']}" for r in result['top50']) + '\n', encoding='utf-8')
    (folder / 'unresolved.txt').write_text('\n'.join(result.get('unresolved', [])) + '\n', encoding='utf-8')
    return folder


def generate(engine, model, lines, known, mode, strict=False):
    profile = resolve_profile(lines, engine.tracks, strict=strict)
    seeds, unresolved = profile['seeds'], profile['unresolved']
    if not seeds:
        raise ValueError('Ни один трек не распознан однозначно. Попробуй другой список или точные Artist - Title.')
    features, candidates, _ = engine.features(seeds)
    scores = engine.score(model, features)
    title, cap, share = MODES[mode]
    excluded = sorted(set(lines) | set(known))
    if hasattr(engine, 'select_playlist'):
        ids = engine.select_playlist(scores, candidates, excluded, artist_cap=cap, familiar_share=share)
    else:
        ids = select_playlist(engine.tracks, scores, candidates, excluded, artist_cap=cap, familiar_share=share)
    familiar = {a for a, _ in input_keys(excluded)}
    rows = [{'rank': rank + 1, 'artist': engine.tracks[i]['artist'], 'title': engine.tracks[i]['title'],
             'id': engine.tracks[i]['id'], 'neural_score': float(scores[i]),
             'artist_not_in_library': match_key(engine.tracks[i]['artist']) not in familiar}
            for rank, i in enumerate(ids)]
    return {'status': 'EXPERIMENTAL', 'mode': title, 'input': lines, 'known_exclusions': known,
            'resolved': len(seeds), 'matching': profile['matches'], 'strict_matching': strict, 'unresolved': unresolved, 'candidates': len(candidates), 'top50': rows}


def show(result, folder):
    print(f"\nРаспознано: {result['resolved']}/{len(result['input'])} · кандидатов: {result['candidates']}")
    approximate = sum(m['kind'] != 'exact_name_or_id' for m in result.get('matching', []))
    if approximate:
        print(f'Из них совпадений по названию/ремастеру: {approximate}; это не точное определение аудиозаписи.')
    if result['unresolved']:
        print('Нераспознанные треки не формируют вкус; список сохранён в unresolved.txt.')
    for row in result['top50']:
        mark = '+' if row['artist_not_in_library'] else ' '
        print(f"{row['rank']:2}. {mark} {row['artist']} — {row['title']}")
    print('+ исполнитель отсутствует в загруженных списках; это не гарантия незнакомой музыки.')
    if len(result['top50']) < 50:
        print(f"Под текущие ограничения подошло {len(result['top50'])} треков. Можно сменить режим.")
    print(f'Сохранено: {folder / "playlist.txt"}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', nargs='?', help='JSON, CSV или TXT; необязательно')
    parser.add_argument('--known', help='Полная библиотека для исключения уже знакомого')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/discovery-v3')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/main')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/cli-playlists')
    parser.add_argument('--mode', choices=MODES, default='1')
    parser.add_argument('--strict-matching', action='store_true', help='Только прежнее однозначное сопоставление')
    parser.add_argument('--once', action='store_true', help='Создать плейлист и выйти')
    args = parser.parse_args()
    if args.once and not args.library:
        parser.error('--once требует путь к библиотеке')
    try:
        # Check before loading large artifacts, and never invoke training/evaluation.
        if not (args.run / 'ranker.pt').exists():
            load_ranker(None, args.run)
        lines = read_path(args.library) if args.library else []
        known = read_path(args.known) if args.known else []
        mode, last = args.mode, None
        print('Музыкальная лаборатория · готовая нейросеть · без обучения и скачиваний')
        print('Загружаю модель один раз…', flush=True)
        if (args.run / 'partition.json').exists():
            from .honest_training import load_run
            engine, model = load_run(args.run)
        else:
            engine = DiscoveryEngine(args.source)
            model = load_ranker(engine, args.run)
        print('Готово. Это экспериментальная модель; проверяй на своём вкусе.')
        if lines:
            last = generate(engine, model, lines, known, mode, strict=args.strict_matching)
            last['checkpoint_sha256'] = fingerprint(args.run / 'ranker.pt')
            show(last, export_playlist(last, args.output))
        if args.once:
            return 0
        while True:
            print(f'Профиль: {len(lines)} треков · уже знакомое: {len(known)} · {MODES[mode][0]}')
            print('1 Файл  2 Вставить треки  3 Получить плейлист  4 Режим\n'
                  '5 Загрузить уже знакомое  6 Отметить знакомые из выдачи  7 Нераспознанные  0 Выход')
            action = input('> ').strip()
            try:
                if action == '0':
                    return 0
                if action == '1':
                    value = input('Путь к JSON/CSV/TXT [~/Downloads/liked.json]: ').strip() or '~/Downloads/liked.json'
                    lines = read_path(value); last = None
                    print(f'Загружено {len(lines)} треков. Нажми 3 для выдачи.')
                elif action == '2':
                    print('Вставь Artist - Title, по одному на строку. Пустая строка завершает ввод.')
                    pasted = []
                    while True:
                        line = input()
                        if not line.strip():
                            break
                        pasted.append(line)
                    lines = validate_lines(pasted); last = None
                elif action == '3':
                    last = generate(engine, model, validate_lines(lines), known, mode, strict=args.strict_matching)
                    last['checkpoint_sha256'] = fingerprint(args.run / 'ranker.pt')
                    show(last, export_playlist(last, args.output))
                elif action == '4':
                    print('\n'.join(f'{key} {value[0]}' for key, value in MODES.items()))
                    chosen = input('Режим [1/2/3]: ').strip()
                    if chosen not in MODES:
                        raise ValueError('Выбери 1, 2 или 3.')
                    mode = chosen
                elif action == '5':
                    known = read_path(input('Файл уже знакомых треков: '))
                    print('Будут исключены из выдачи, но не добавлены к профилю вкуса.')
                elif action == '6':
                    if not last:
                        raise ValueError('Сначала получи плейлист.')
                    numbers = {int(n) for n in input('Номера знакомых треков через пробел: ').replace(',', ' ').split()}
                    if not numbers or not numbers <= set(range(1, len(last['top50']) + 1)):
                        raise ValueError('Укажи номера из последней выдачи.')
                    known = sorted(set(known) | {f"{r['artist']} - {r['title']}" for r in last['top50'] if r['rank'] in numbers})
                    print('Запомнил для этой сессии. Нажми 3, чтобы обновить плейлист.')
                elif action == '7':
                    unresolved = resolve_profile(validate_lines(lines), engine.tracks, strict=args.strict_matching)['unresolved']
                    print('\n'.join(unresolved) or 'Все треки распознаны.')
                else:
                    print('Выбери пункт меню.')
            except (OSError, ValueError, TypeError, KeyError) as error:
                print(f'Не получилось: {error}')
    except (EOFError, KeyboardInterrupt):
        print('\nДо встречи!')
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
