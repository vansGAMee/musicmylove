"""Interactive, inference-only music playground. Run: python -m python_mvp.cli"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import torch
from .config import ROOT, write_json, fingerprint
from .discovery import signature
from .discovery_engine import DiscoveryEngine, DiscoveryRanker, read_library, select_playlist, input_keys
from .recommend import resolve, match_key
from .profile_input import resolve_profile
from . import feedback

MODES = {'1': ('Смешанный', 2, 1.), '2': ('Открытия: больше новых исполнителей', 2, .30),
         '3': ('Чистый порядок нейросети', 50, 1.)}


def default_run():
    for name in ('expanded-v1/model', 'refined-v1', 'honest-v1', 'discovery-v3'):
        run = ROOT / 'data' / name
        if (run / 'ranker.pt').exists():
            return run
    return ROOT / 'data/honest-v1'


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


def open_engine(run, source, *, inference_only=False):
    run = Path(run)
    if (run / 'refinement.json').exists():
        from .refine_ranker import load_refined
        return load_refined(run)
    if (run / 'partition.json').exists():
        if (run / 'recovery.json').exists():
            from .resume_training import verify_recovery, load_recovered
            if inference_only:
                return load_recovered(run)
            verify_recovery(run)
        from .honest_training import load_run
        return load_run(run)
    engine = DiscoveryEngine(source)
    return engine, load_ranker(engine, run)


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


def focus_from_result(result, selection):
    if not result or not result.get('top50'):
        raise ValueError('Сначала получи плейлист.')
    chosen = feedback.numbers(selection, len(result['top50']))
    tracks = [row for row in result['top50'] if row['rank'] in chosen]
    if len(tracks) != len(chosen) or any(not row.get('id') for row in tracks):
        raise ValueError('В выдаче отсутствуют идентификаторы выбранных треков')
    return sorted({row['id'] for row in tracks})


def generate(engine, model, lines, known, mode, strict=False, feedback_dir=feedback.DEFAULT,
             checkpoint=None, personal=True, focus=None):
    library_profile = resolve_profile(lines, engine.tracks, strict=strict)
    inputs = sorted(set(focus)) if focus is not None else lines
    profile = resolve_profile(inputs, engine.tracks, strict=True) if focus is not None else library_profile
    seeds, unresolved = profile['seeds'], profile['unresolved']
    if focus is not None and unresolved:
        raise ValueError('Выбранные треки отсутствуют в текущем каталоге. Получи новый плейлист.')
    if not seeds:
        raise ValueError('Ни один трек не распознан однозначно. Попробуй другой список или точные Artist - Title.')
    features, candidates, _ = engine.features(seeds)
    title, cap, share = MODES[mode]
    seed_lines = [engine.tracks[i]['artist'] + ' - ' + engine.tracks[i]['title']
                  for i in sorted(set(seeds) | set(library_profile['seeds']))]
    library_ids = {engine.tracks[i]['id'] for i in library_profile['seeds']}
    excluded = sorted(set(lines) | library_ids | set(inputs) | set(seed_lines) | set(known)
                      | set(feedback.exclusions(feedback_dir, discovery=mode == '2')))
    discovery_pool = None
    if mode == '2' and hasattr(engine, 'graph'):
        from .discovery_policy import prepare_discovery
        discovery_pool = prepare_discovery(engine, seeds, excluded)
        candidates = discovery_pool.candidates
    scores = engine.score(model, features)
    personal_features = np.column_stack((features, engine.embeddings.detach().cpu().numpy()))
    base_scores = scores.copy()
    personal_scale = max(.05, float(np.std(base_scores[sorted(candidates)]))) if candidates else .05
    personal_status = 'Личный слой выключен'
    if personal and checkpoint:
        from .personal import adjust
        scores, personal_status = adjust(feedback_dir, checkpoint, personal_features, scores, candidates)
    evidence, policy = {}, {}
    if discovery_pool is not None:
        from .discovery_policy import select_discovery
        ids, evidence, policy = select_discovery(discovery_pool, scores)
    elif hasattr(engine, 'select_playlist'):
        ids = engine.select_playlist(scores, candidates, excluded, artist_cap=cap, familiar_share=share)
    else:
        ids = select_playlist(engine.tracks, scores, candidates, excluded, artist_cap=cap, familiar_share=share)
    familiar = {a for a, _ in input_keys(excluded)}
    rows = [{'rank': rank + 1, 'artist': engine.tracks[i]['artist'], 'title': engine.tracks[i]['title'],
             'id': engine.tracks[i]['id'], 'neural_score': float(scores[i]),
             'base_score': float(base_scores[i]), 'personal_features': personal_features[i].tolist(),
             'personal_scale': personal_scale,
             **({'discovery_evidence': evidence[i]} if i in evidence else {}),
             'artist_not_in_library': match_key(engine.tracks[i]['artist']) not in familiar}
            for rank, i in enumerate(ids)]
    return {'status': getattr(engine, 'refinement_status', 'EXPERIMENTAL'), 'mode': title, 'input': inputs,
            'library_input': lines, 'focus': [dict(id=engine.tracks[i]['id'], artist=engine.tracks[i]['artist'],
                                                 title=engine.tracks[i]['title']) for i in seeds] if focus is not None else [],
            'known_exclusions': excluded, 'selection_policy': policy,
            'checkpoint_sha256': checkpoint, 'exposure_id': uuid.uuid4().hex, 'personal_status': personal_status,
            'resolved': len(seeds), 'matching': profile['matches'], 'strict_matching': strict, 'unresolved': unresolved, 'candidates': len(candidates), 'top50': rows}


def show(result, folder):
    if result.get('focus'):
        print('Направление: ' + '; '.join(t['artist'] + ' — ' + t['title'] for t in result['focus']))
    if result.get('personal_status'):
        print(result['personal_status'])
    if result['status'] == 'LISTENER_BASELINE_NO_ACCEPTED_NEURAL_GAIN':
        print('Используется базовый метод по слушателям: нейронная поправка не прошла DEV-отбор.')
    print(f"\nРаспознано: {result['resolved']}/{len(result['input'])} · кандидатов: {result['candidates']}")
    approximate = sum(m['kind'] != 'exact_name_or_id' for m in result.get('matching', []))
    if approximate:
        print(f'Из них совпадений по названию/ремастеру: {approximate}; это не точное определение аудиозаписи.')
    if result['unresolved']:
        print('Нераспознанные треки не формируют вкус; список сохранён в unresolved.txt.')
    for row in result['top50']:
        mark = '+' if row['artist_not_in_library'] else ' '
        direction = ' ← ' + row['discovery_evidence']['seed_artist'] if row.get('discovery_evidence') else ''
        print(f"{row['rank']:2}. {mark} {row['artist']} — {row['title']}{direction}")
    policy = result.get('selection_policy', {})
    if policy:
        print('Стрелка — связь с исходным треком по слушателям и обученным векторам, не анализ звука.')
        if policy['unrepresented_directions']:
            print('Без продолжения при текущих ограничениях: ' + ', '.join(policy['unrepresented_directions']))
    print('+ исполнитель отсутствует в загруженных списках; это не гарантия незнакомой музыки.')
    if len(result['top50']) < 50:
        print(f"Под текущие ограничения подошло {len(result['top50'])} треков. Можно сменить режим.")
    print(f'Сохранено: {folder / "playlist.txt"}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', nargs='?', help='JSON, CSV или TXT; необязательно')
    parser.add_argument('--known', help='Полная библиотека для исключения уже знакомого')
    parser.add_argument('--run', type=Path, default=default_run())
    parser.add_argument('--source', type=Path, default=ROOT / 'data/main')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/cli-playlists')
    parser.add_argument('--mode', choices=MODES, default='1')
    parser.add_argument('--strict-matching', action='store_true', help='Только прежнее однозначное сопоставление')
    parser.add_argument('--once', action='store_true', help='Создать плейлист и выйти')
    parser.add_argument('--feedback', type=Path, default=feedback.DEFAULT, help='Папка личных оценок и модели')
    parser.add_argument('--no-personal', action='store_true', help='Сравнить выдачу без личной поправки')
    args = parser.parse_args()
    if args.once and not args.library:
        parser.error('--once требует путь к библиотеке')
    try:
        # Check before loading large artifacts, and never invoke training/evaluation.
        if not (args.run / 'ranker.pt').exists():
            load_ranker(None, args.run)
        lines = read_path(args.library) if args.library else []
        known = feedback.import_known(args.feedback, args.known) if args.known else feedback.known_lines(args.feedback)
        mode, last, focus = args.mode, None, None
        explored = set()
        print('Музыкальная лаборатория · готовый рекомендатель · без обучения и скачиваний')
        print(f'Эксперимент: {args.run}')
        print('Загружаю модель один раз…', flush=True)
        checkpoint = fingerprint(args.run / 'ranker.pt')
        engine, model = open_engine(args.run, args.source, inference_only=True)
        if checkpoint != fingerprint(args.run / 'ranker.pt'):
            raise RuntimeError('Общая модель изменилась во время загрузки. Перезапусти CLI.')
        def make_playlist(selected_focus=None):
            return generate(engine, model, validate_lines(lines), sorted(set(known) | explored), mode,
                            strict=args.strict_matching, feedback_dir=args.feedback,
                            checkpoint=checkpoint, personal=not args.no_personal, focus=selected_focus)
        print('Готово. Это экспериментальная модель; проверяй на своём вкусе.')
        if lines:
            last = make_playlist()
            show(last, export_playlist(last, args.output))
        if args.once:
            return 0
        while True:
            print(f'Профиль: {len(lines)} треков · уже знакомое: {len(known)} · {MODES[mode][0]}')
            if focus is not None:
                print(f'Поиск по выбранным трекам: {len(focus)}. Вернуться: весь профиль')
            print('1 Файл  2 Вставить треки  3 Получить плейлист  4 Режим\n'
                  '5 Загрузить уже знакомое  6 Отметить знакомые из выдачи  7 Нераспознанные\n'
                  '8 Покрытие графа  9 Мои оценки  0 Выход\n'
                  'Оценки: лайк 1 3-5 · дизлайк 8 · знаю 2 · новое 3 · сброс 8 · отмена\n'
                  'Направить поиск: ещё как 3 7 12 · Вернуться: весь профиль')
            action = input('> ').strip()
            try:
                words = action.split(maxsplit=1)
                actions = {'лайк': 'like', 'дизлайк': 'dislike', 'знаю': 'known', 'новое': 'new', 'сброс': 'clear'}
                direction = action.lower().replace('ё', 'е').split()
                if direction[:2] == ['еще', 'как'] or direction == ['весь', 'профиль']:
                    chosen = focus_from_result(last, ' '.join(direction[2:])) if direction[:2] == ['еще', 'как'] else None
                    result = make_playlist(chosen)
                    folder = export_playlist(result, args.output)
                    focus, last = chosen, result
                    explored.update(chosen or [])
                    explored.update(t['artist'] + ' - ' + t['title'] for t in result['focus'])
                    show(last, folder)
                    continue
                elif words and words[0] in actions:
                    if len(words) != 2:
                        raise ValueError('Добавь номера треков из последней выдачи')
                    count = feedback.record(args.feedback, last, actions[words[0]], words[1])
                    print(f'Сохранено: {count}. Нажми 3 для следующей выдачи.')
                elif action == 'отмена':
                    feedback.undo(args.feedback)
                    known = feedback.known_lines(args.feedback)
                    print('Последнее действие с оценками или импортом знакомого отменено.')
                elif action == '0':
                    return 0
                elif action == '1':
                    value = input('Путь к JSON/CSV/TXT [~/Downloads/liked.json]: ').strip() or '~/Downloads/liked.json'
                    lines = read_path(value); last = focus = None
                    print(f'Загружено {len(lines)} треков. Нажми 3 для выдачи.')
                elif action == '2':
                    print('Вставь Artist - Title, по одному на строку. Пустая строка завершает ввод.')
                    pasted = []
                    while True:
                        line = input()
                        if not line.strip():
                            break
                        pasted.append(line)
                    lines = validate_lines(pasted); last = focus = None
                elif action == '3':
                    last = make_playlist(focus)
                    show(last, export_playlist(last, args.output))
                elif action == '4':
                    print('\n'.join(f'{key} {value[0]}' for key, value in MODES.items()))
                    chosen = input('Режим [1/2/3]: ').strip()
                    if chosen not in MODES:
                        raise ValueError('Выбери 1, 2 или 3.')
                    mode = chosen
                elif action == '5':
                    known = feedback.import_known(args.feedback, input('Файл уже знакомых треков: '))
                    print('Сохранены между запусками и исключены из выдачи; к профилю вкуса не добавлены.')
                elif action == '6':
                    if not last:
                        raise ValueError('Сначала получи плейлист.')
                    feedback.record(args.feedback, last, 'known', input('Номера знакомых треков: '))
                    print('Сохранено между запусками. В режиме открытий будут исключены.')
                elif action == '7':
                    unresolved = resolve_profile(validate_lines(lines), engine.tracks, strict=args.strict_matching)['unresolved']
                    print('\n'.join(unresolved) or 'Все треки распознаны.')
                elif action == '8':
                    from .coverage import diagnose
                    coverage_lines = validate_lines(lines)
                    coverage_engine = engine
                    if hasattr(engine, 'raw_data_path'):
                        from types import SimpleNamespace
                        from .ranker_data import load_coverage_data
                        print('Проверяю исходные данные потоком, без загрузки всей истории в память…', flush=True)
                        coverage_engine = SimpleNamespace(tracks=engine.tracks, meta=engine.meta,
                            data=load_coverage_data(engine.raw_data_path, engine.meta, coverage_lines))
                    report = diagnose(coverage_engine, coverage_lines)
                    target = args.feedback/'coverage.json'
                    write_json(target, report)
                    print(report['counts']); print(f'Подробности: {target}')
                elif action == '9':
                    import shlex
                    print(json.dumps(feedback.summary(args.feedback), ensure_ascii=False))
                    print('Обновить личную модель отдельной командой:')
                    print('python -m python_mvp.personal train --run ' + shlex.quote(str(args.run))
                          + ' --feedback ' + shlex.quote(str(args.feedback)))
                    print('Откат: python -m python_mvp.personal rollback --feedback ' + shlex.quote(str(args.feedback)))
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
