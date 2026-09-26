"""Local append-only event semantics, stored atomically as JSON; no database."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import csv
import json
from pathlib import Path
import uuid
from .config import ROOT, write_json

DEFAULT = ROOT / 'data/personal'
ACTIONS = {'like', 'dislike', 'known', 'new', 'clear'}


def numbers(value, maximum):
    selected = set()
    for token in value.replace(',', ' ').split():
        ends = token.split('-')
        if len(ends) == 1:
            selected.add(int(token))
        elif len(ends) == 2:
            a, b = map(int, ends)
            if not 1 <= a <= b <= maximum:
                raise ValueError('Диапазон вне последней выдачи')
            selected.update(range(a, b+1))
        else:
            raise ValueError('Нужны номера, например: 1 3 8-12')
    if not selected or not selected <= set(range(1, maximum+1)):
        raise ValueError('Укажи номера из последней выдачи')
    return selected


def journal(directory=DEFAULT):
    path = Path(directory)/'feedback.json'
    if not path.exists():
        return {'schema': 1, 'events': []}
    result = json.loads(path.read_text())
    if result.get('schema') != 1:
        raise ValueError('Unsupported feedback schema')
    return result


@contextmanager
def locked(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'feedback.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def active_events(data):
    undone = {e['target'] for e in data['events'] if e['action'] == 'undo'}
    return [e for e in data['events'] if e['action'] != 'undo' and e['id'] not in undone]


def record(directory, result, action, selection):
    if action not in ACTIONS:
        raise ValueError('Неизвестная оценка')
    if not result or not result.get('checkpoint_sha256'):
        raise ValueError('Сначала получи плейлист с версией модели')
    chosen = numbers(selection, len(result['top50']))
    tracks = [r for r in result['top50'] if r['rank'] in chosen]
    if len(tracks) != len(chosen) or any(not t.get('id') for t in tracks):
        raise ValueError('В выдаче отсутствуют идентификаторы треков')
    with locked(directory):
        data = journal(directory)
        data['events'].append({'id': uuid.uuid4().hex, 'sequence': len(data['events']),
            'time': datetime.now(timezone.utc).isoformat(), 'action': action,
            'exposure_id': result.get('exposure_id'), 'checkpoint': result['checkpoint_sha256'],
            'profile': result.get('input', []), 'mode': result.get('mode'), 'tracks': tracks})
        write_json(Path(directory)/'feedback.json', data)
    return len(tracks)


def undo(directory=DEFAULT):
    with locked(directory):
        data = journal(directory); events = active_events(data)
        if not events:
            raise ValueError('Нет оценок для отмены')
        data['events'].append({'id': uuid.uuid4().hex, 'action': 'undo', 'target': events[-1]['id'],
                              'time': datetime.now(timezone.utc).isoformat(), 'sequence': len(data['events'])})
        write_json(Path(directory)/'feedback.json', data)


def current(directory=DEFAULT):
    result = {}
    for event in active_events(journal(directory)):
        if event['action'] == 'import_known':
            continue
        for track in event['tracks']:
            state = result.setdefault(track['id'], {'preference': None, 'familiarity': None, 'track': track})
            action = event['action']
            if action in ('like', 'dislike', 'clear'):
                state['preference'] = None if action == 'clear' else action
                state['preference_event'] = event
                state['track'] = track
            else:
                state['familiarity'] = action
    return result


def exclusions(directory=DEFAULT, discovery=False):
    return sorted({s['track']['artist']+' - '+s['track']['title'] for s in current(directory).values()
                   if s['preference'] == 'dislike' or (discovery and (s['familiarity'] == 'known' or s['preference']))})


def training_rows(directory, checkpoint):
    rows = []
    for tid, state in current(directory).items():
        if state['preference'] not in ('like', 'dislike'):
            continue
        event = state['preference_event']; track = state['track']
        if event['checkpoint'] != checkpoint or not track.get('personal_features'):
            continue
        rows.append({'id': tid, 'features': track['personal_features'],
                     'label': int(state['preference'] == 'like'), 'sequence': event['sequence'],
                     'base_score': track.get('base_score'), 'personal_scale': track.get('personal_scale'),
                     'exposure_id': event.get('exposure_id') or event['id']})
    return sorted(rows, key=lambda r: (r['sequence'], r['id']))


def known_lines(directory=DEFAULT):
    return sorted({line for event in active_events(journal(directory))
                   if event['action'] == 'import_known' for line in event['lines']})


def import_known(directory, path):
    """Import identity fields only; a full known library is not a 2,000-track profile."""
    path = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = path.read_text(encoding='utf-8-sig').splitlines()
    if not isinstance(rows, list):
        raise ValueError('Ожидается список треков')
    lines = set()
    for row in rows:
        if isinstance(row, str):
            if row.strip():
                lines.add(row.strip())
            continue
        if not isinstance(row, dict):
            continue
        title = row.get('Track Name', row.get('name', row.get('title')))
        artist = row.get('Artist Name(s)', row.get('artist'))
        if not artist and row.get('artists'):
            artist = row['artists'][0].get('name')
        if isinstance(title, str) and isinstance(artist, str) and title.strip() and artist.strip():
            lines.add(artist.split(';')[0].strip()+' - '+title.strip())
    if not lines:
        raise ValueError('Нет названий треков для импорта')
    with locked(directory):
        data = journal(directory)
        data['events'].append({'id': uuid.uuid4().hex, 'action': 'import_known', 'lines': sorted(lines),
            'time': datetime.now(timezone.utc).isoformat(), 'sequence': len(data['events'])})
        write_json(Path(directory)/'feedback.json', data)
    return known_lines(directory)


def summary(directory=DEFAULT):
    values = list(current(directory).values())
    new = [s for s in values if s['familiarity'] == 'new' and s['preference'] in ('like', 'dislike')]
    return {'likes': sum(s['preference'] == 'like' for s in values),
            'dislikes': sum(s['preference'] == 'dislike' for s in values),
            'known': sum(s['familiarity'] == 'known' for s in values),
            'rated_new': len(new),
            'new_like_share': sum(s['preference'] == 'like' for s in new)/len(new) if new else None}
