"""Train a small personal preference layer from explicit local ratings only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .config import fingerprint, write_json, deterministic
from .feedback import DEFAULT, training_rows, summary, locked


def label_signature(row):
    return hashlib.sha256(json.dumps([row['label'], row['features'], row.get('base_score'),
                                      row.get('personal_scale')], sort_keys=True).encode()).hexdigest()


def temporal_split(rows):
    if len(rows) < 20 or sum(r['label'] for r in rows) < 5 or sum(not r['label'] for r in rows) < 5:
        raise ValueError('Нужно минимум 20 оценённых треков, среди них 5 лайков и 5 дизлайков. Пропуски не считаются.')
    groups = {}
    for row in rows:
        groups.setdefault(row['exposure_id'], []).append(row)
    if len(groups) < 3:
        raise ValueError('Нужны оценки хотя бы трёх разных выдач для проверки на поздних рекомендациях.')
    ordered = sorted(rows, key=lambda r: (r['sequence'], r['id']))
    cutoff = ordered[-max(4, len(rows)//5)]['sequence']
    validation = [r for r in rows if r['sequence'] >= cutoff]
    held = {r['exposure_id'] for r in validation}
    # Earlier ratings from a held-out exposure are discarded, not trained on.
    # A late correction therefore never moves future labels into training.
    train = [r for r in rows if r['exposure_id'] not in held and r['sequence'] < cutoff]
    if len(train) < 10 or len({r['label'] for r in train}) < 2:
        raise ValueError('Пока мало более ранних разнородных оценок для обучения без утечки.')
    return train, validation


def predict(model, features):
    x = np.asarray(features, dtype=float)
    x = np.clip((x - np.array(model['mean']))/np.array(model['scale']), -5, 5)
    return x @ np.array(model['weights']) + model['bias']


def fit_rows(rows):
    deterministic(42)
    train, validation = temporal_split(rows)
    x = torch.tensor([r['features'] for r in train], dtype=torch.float32)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('Некорректные признаки оценок')
    y = torch.tensor([r['label'] for r in train], dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0).clamp_min(.05)
    x = ((x-mean)/scale).clamp(-5, 5)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.tensor(float(torch.logit(y.mean().clamp(.01, .99))), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.03)
    for _ in range(200):
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y) + .05*weight.square().sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model = {'schema': 1, 'mean': mean.tolist(), 'scale': scale.tolist(),
             'weights': weight.detach().tolist(), 'bias': float(bias.detach()),
             'strength': min(.35, len(train)/200),
             'validation_labels': {r['id']: label_signature(r) for r in validation},
             'training_labels': {r['id']: label_signature(r) for r in train}}
    logits = predict(model, [r['features'] for r in validation])
    labels = np.array([r['label'] for r in validation])
    if not np.isfinite(logits).all():
        raise ValueError('Invalid validation features')
    val_loss = float(np.mean(np.logaddexp(0, logits) - labels*logits))
    prior = float(y.mean()); baseline = float(np.mean(-labels*np.log(prior) - (1-labels)*np.log(1-prior)))
    base_hits, personal_hits = [], []
    for i, positive in enumerate(validation):
        if positive['label'] != 1 or positive.get('base_score') is None or positive.get('personal_scale') is None:
            continue
        for j, negative in enumerate(validation):
            if (negative['label'] or negative['exposure_id'] != positive['exposure_id']
                    or negative.get('base_score') is None or negative.get('personal_scale') is None):
                continue
            b = positive['base_score'] - negative['base_score']
            p = b + model['strength']*(positive['personal_scale']*np.tanh(logits[i]/2)
                                      - negative['personal_scale']*np.tanh(logits[j]/2))
            base_hits.append(float(b > 0) + .5*float(b == 0))
            personal_hits.append(float(p > 0) + .5*float(p == 0))
    ranking_ok = bool(base_hits) and np.mean(personal_hits) >= np.mean(base_hits)
    model['report'] = {'train_count': len(train), 'validation_count': len(validation),
        'validation_loss': val_loss, 'baseline_loss': baseline, 'split': 'chronological_by_exposure',
        'validation_used_for_activation': True, 'independent_test': False,
        'discarded_overlap_count': len(rows)-len(train)-len(validation),
        'ranking_pairs': len(base_hits), 'base_pair_accuracy': float(np.mean(base_hits)) if base_hits else None,
        'personal_pair_accuracy': float(np.mean(personal_hits)) if personal_hits else None,
        'accepted': bool(val_loss < baseline and ranking_ok), 'explicit_ratings_only': True}
    return model


def compatible_labels(model, rows):
    latest = {r['id']: label_signature(r) for r in rows}
    evidence = {**model['training_labels'], **model.get('validation_labels', {})}
    return all(latest.get(tid) == value for tid, value in evidence.items())


def load_active(directory, checkpoint):
    directory = Path(directory); pointer = directory/'active.json'
    if not pointer.exists():
        return None, 'Личный слой ещё не активирован'
    name = json.loads(pointer.read_text())['current']
    if Path(name).name != name:
        raise ValueError('Invalid personal model path')
    model = json.loads((directory/'models'/name).read_text())
    if model.get('schema') != 1 or model['checkpoint'] != checkpoint:
        return None, 'Личный слой относится к другой общей модели; нужно новое обучение'
    if model.get('implementation') != fingerprint(Path(__file__)):
        return None, 'Код личной модели изменился; нужно новое обучение'
    if not compatible_labels(model, training_rows(directory, checkpoint)):
        return None, 'Оценки обучения или проверки изменены; личный слой отключён до обновления'
    return model, 'Активен личный слой: ' + name


def adjust(directory, checkpoint, features, scores, candidates):
    model, status = load_active(directory, checkpoint)
    if model is None:
        return scores, status
    logits = predict(model, features)
    spread = max(.05, float(np.std(scores[sorted(candidates)]))) if candidates else .05
    correction = model['strength'] * spread * np.tanh(logits/2)
    return scores + correction, status


def train(directory, checkpoint):
    directory = Path(directory)
    rows = training_rows(directory, checkpoint)
    model = fit_rows(rows)
    model.update(checkpoint=checkpoint, implementation=fingerprint(Path(__file__)),
                 created=datetime.now(timezone.utc).isoformat())
    encoded = json.dumps(model, sort_keys=True).encode()
    name = hashlib.sha256(encoded).hexdigest()[:20]+'.json'
    with locked(directory):
        if not compatible_labels(model, training_rows(directory, checkpoint)):
            raise ValueError('Оценки изменились во время обучения; повтори команду')
        write_json(directory/'models'/name, model)
        if model['report']['accepted']:
            pointer = directory/'active.json'
            previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
            write_json(pointer, {'current': name, 'previous': previous})
    return model


def rollback(directory):
    directory = Path(directory)
    with locked(directory):
        pointer = directory/'active.json'
        if not pointer.exists():
            raise ValueError('Нет активной личной модели')
        state = json.loads(pointer.read_text())
        if state.get('previous'):
            write_json(pointer, {'current': state['previous'], 'previous': state['current']})
        else:
            pointer.unlink()  # return to unchanged global model; snapshot is retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'status', 'rollback'))
    parser.add_argument('--feedback', type=Path, default=DEFAULT)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'status':
            print(json.dumps(summary(args.feedback), ensure_ascii=False)); return
        if args.action == 'rollback':
            rollback(args.feedback); print('Личная модель откачена; оценки сохранены.'); return
        if args.run is None:
            from .cli import default_run
            args.run = default_run()
        checkpoint = fingerprint(args.run/'ranker.pt')
        model = train(args.feedback, checkpoint)
        print(json.dumps(model['report'], ensure_ascii=False, indent=2))
        print('Личная модель активирована.' if model['report']['accepted'] else
              'Кандидат сохранён, но не активирован: на поздних оценках нет улучшения. Предыдущая модель сохранена.')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()
