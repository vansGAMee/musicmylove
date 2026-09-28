"""Offline listener retrieval and learned ranking. No audio/platform features or APIs."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import numpy as np
from scipy import sparse
import torch
from torch import nn
from .config import fingerprint, deterministic
from .artifacts import contract
from .build_graph import load_graph
from .networks import MultiInterest
from .recommend import match_key, resolve, known_track_indices
from .retrieval import diffusion


def read_library(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    else:
        return sorted({s.strip() for s in path.read_text().splitlines() if s.strip()})
    if not isinstance(rows, list):
        raise ValueError('Library must be a list of tracks')
    lines = set()
    for row in rows:
        title = row.get('Track Name', row.get('name', row.get('title')))
        artist = row.get('Artist Name(s)', row.get('artist'))
        if not artist and row.get('artists'):
            artist = row['artists'][0]['name']
        # Only user-supplied identity fields enter the system. No popularity,
        # audio features, platform IDs, previews, or platform recommendations.
        if isinstance(title, str) and isinstance(artist, str) and title.strip() and artist.strip():
            lines.add(artist.split(';')[0].strip() + ' - ' + title.strip())
    if not lines or len(lines) > 2000:
        raise ValueError('Expected 1–2000 unique artist/title pairs')
    return sorted(lines)


def song_key(artist, title):
    title = match_key(title)
    title = re.sub(r'\s*[\[(][^\])]*(?:remaster|live|version|remix|acoustic|edit)[^\])]*[\])]\s*$', '', title)
    title = re.sub(r'\s+-\s+(?:\d{4}\s+)?(?:remaster|live|version|remix|acoustic|edit).*$', '', title)
    return match_key(artist), title.strip()


def input_keys(lines):
    return {song_key(*match_key(line).split(' - ', 1)) for line in lines if ' - ' in match_key(line)}


def listener_scores(incidence, seeds, exclude_user=None):
    hits = np.asarray(incidence[:, sorted(set(seeds))].sum(1)).ravel()
    if exclude_user is not None:
        hits[exclude_user] = 0
    sizes = np.asarray(incidence.sum(1)).ravel()
    popularity = np.asarray(incidence.sum(0)).ravel()
    scores = np.asarray(incidence.T @ (hits / np.sqrt(np.maximum(sizes, 1)))).ravel()
    scores /= np.sqrt(np.maximum(popularity, 1))
    support = np.asarray(incidence.T @ (hits > 0).astype(float)).ravel()
    return scores, support


def training_pairs(features, targets, known, retrieval, rng):
    positive = np.array(sorted(targets), dtype=int)
    allowed = np.array(sorted(set(range(len(features))) - set(known)), dtype=int)
    if not len(positive) or not len(allowed):
        return np.empty((0, features.shape[1]), np.float32), np.empty((0, features.shape[1]), np.float32)
    positive = rng.choice(positive, min(12, len(positive)), replace=False)
    ranked = allowed[np.argsort(-retrieval[allowed], kind='stable')]
    hard = ranked[:min(500, len(ranked))]
    negatives = np.concatenate((rng.choice(hard, len(positive) * 3), rng.choice(allowed, len(positive))))
    positives = np.tile(positive, 4)
    return features[positives], features[negatives]


def select_playlist(tracks, scores, candidates, lines, size=50, artist_cap=2, familiar_share=.30):
    """Explicit discovery product policy; musical scores remain neural outputs."""
    keys = input_keys(lines)
    known = known_track_indices(lines, tracks)
    keys.update(song_key(tracks[i]['artist'], tracks[i]['title']) for i in known)
    artists = {a for a, _ in keys}
    artists.update(match_key(tracks[i]['artist']) for i in known)
    counts, selected, used_songs = Counter(), [], set()
    familiar_count = 0
    for i in sorted(candidates, key=lambda i: (-float(scores[i]), tracks[i]['id'])):
        track = tracks[i]
        key = song_key(track['artist'], track['title'])
        aliases = {song_key(*pair) for pair in track.get('aliases', [])} | {key}
        artist = key[0]
        familiar = artist in artists
        if (i in known or aliases & keys or key in used_songs or not np.isfinite(scores[i])
                or counts[artist] >= artist_cap
                or (familiar and familiar_count >= int(size * familiar_share))):
            continue
        selected.append(i); used_songs.add(key); counts[artist] += 1
        familiar_count += familiar
        if len(selected) == size:
            break
    # Shortfalls must not silently turn a discovery list into familiar artists.
    while selected and familiar_count > familiar_share * len(selected):
        remove = next(i for i in reversed(selected) if match_key(tracks[i]['artist']) in artists)
        selected.remove(remove)
        familiar_count -= 1
    return selected


def rank_feature(values):
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    # Equal evidence gets equal features, independent of catalog position.
    sorted_values = values[order]
    first = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ranks[order] = np.repeat(first, np.diff(np.r_[first, len(values)]))
    return -np.log1p(ranks)


class DiscoveryRanker(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))
        self.linear = nn.Linear(dimensions, 1)
        self.net = nn.Sequential(nn.Linear(dimensions, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, features):
        x = ((features - self.mean) / self.scale).clamp(-12, 12)
        return (self.linear(x) + self.net(x)).squeeze(-1)


class DiscoveryEngine:
    def __init__(self, source, seed=42):
        deterministic(seed)
        self.source = Path(source).resolve()
        self.data = json.loads((self.source / 'data/dataset.json').read_text())
        self.meta, self.graph = load_graph(self.source / 'data')
        graph_path, taste_path = [self.source / 'models' / f'{s}_{seed}.pt' for s in ('graph', 'taste')]
        graph_state = torch.load(graph_path, map_location='cpu', weights_only=True)
        taste_state = torch.load(taste_path, map_location='cpu', weights_only=True)
        current = contract(self.source / 'data')
        vocabulary = [t['id'] for t in self.meta['tracks']]
        for state in (graph_state, taste_state):
            if (state.get('status') != 'DEV_PASS' or state.get('vocabulary') != vocabulary
                    or state.get('fingerprint') != self.data['fingerprint']
                    or state.get('contract', {}).get('artifacts') != current['artifacts']):
                raise RuntimeError('Frozen representation does not match accepted data/vocabulary')
        if taste_state.get('parent_sha256') != fingerprint(graph_path):
            raise RuntimeError('Taste/graph provenance mismatch')
        # Explicitly import historical frozen representations; their old release
        # verdict is NOT transferred to the new retrieval/ranking system.
        self.provenance = {'graph': fingerprint(graph_path), 'taste': fingerprint(taste_path),
                           'artifacts': current['artifacts'], 'seed': seed}
        self.embeddings = graph_state['embeddings']
        self.taste = MultiInterest().eval()
        self.taste.load_state_dict(taste_state['weights'])
        for param in self.taste.parameters():
            param.requires_grad_(False)
        self.tracks = self.meta['tracks']
        self.ui = self.graph['user_incidence']
        self.user_index = {u: i for i, u in enumerate(self.meta['train_users'])}
        self.pop = np.asarray(self.ui.sum(0)).ravel()
        self.artist_names = sorted({match_key(t['artist']) for t in self.tracks})
        artist_index = {a: i for i, a in enumerate(self.artist_names)}
        self.artist_ids = np.array([artist_index[match_key(t['artist'])] for t in self.tracks])
        mapping = sparse.csr_matrix((np.ones(len(self.tracks)), (np.arange(len(self.tracks)), self.artist_ids)),
                                    shape=(len(self.tracks), len(artist_index)))
        self.artist_ui = (self.ui @ mapping).tocsr(); self.artist_ui.data[:] = 1
        self.artist_pop = np.asarray(self.artist_ui.sum(0)).ravel()

    def features(self, seeds, user=None):
        seeds = sorted(set(seeds))
        if not seeds:
            raise ValueError('No seeds in the trained catalog')
        exclude_user = self.user_index.get(user)
        cf, support = listener_scores(self.ui, seeds, exclude_user)
        artist_cf, artist_support = listener_scores(self.artist_ui, sorted(set(self.artist_ids[seeds])), exclude_user)
        walk = diffusion(seeds, self.graph)
        g = np.asarray(self.graph['global'][seeds].multiply(self.graph['confidence'][seeds]).sum(0)).ravel()
        local = np.asarray(self.graph['local'][seeds].sum(0)).ravel()
        with torch.no_grad():
            heads, masses, assignments = self.taste(self.embeddings, seeds)
            active = sorted(set(assignments.argmax(-1).tolist()))
            similarity = (self.embeddings @ heads[active].T).numpy()
            mean = (self.embeddings @ torch.nn.functional.normalize(self.embeddings[seeds].mean(0), dim=0)).numpy()
        artist_counts = Counter(self.artist_ids[seeds])
        familiar = np.array([artist_counts[a] for a in self.artist_ids], dtype=float)
        columns = [np.log1p(1000 * walk), rank_feature(walk), np.log1p(cf), rank_feature(cf),
                   np.log1p(g), rank_feature(g), np.log1p(local), np.log1p(support),
                   np.log1p(self.pop), np.log1p(artist_cf[self.artist_ids]),
                   rank_feature(artist_cf)[self.artist_ids], np.log1p(artist_support[self.artist_ids]),
                   np.log1p(self.artist_pop[self.artist_ids]), (familiar > 0).astype(float),
                   np.log1p(familiar), familiar / len(seeds), similarity.max(1), similarity.mean(1),
                   mean, rank_feature(similarity.max(1)), np.full(len(cf), np.log1p(len(seeds))),
                   self.pop / np.maximum(self.artist_pop[self.artist_ids], 1)]
        features = np.column_stack(columns).astype(np.float32)
        # Artist bridges are candidates only with independent human listeners.
        supported = (support > 0) | (walk > 0)
        candidates = set()
        for signal in (walk, cf, similarity.max(1)):
            ids = np.flatnonzero(supported)
            candidates.update(ids[np.argsort(-signal[ids], kind='stable')[:2000]].tolist())
        candidates.difference_update(seeds)
        return features, candidates, {'ppr': walk, 'listeners': cf, 'graph': g,
                                       'support': support, 'head': np.asarray(active)[similarity.argmax(1)]}

    def score(self, model, features):
        with torch.no_grad():
            return model(torch.from_numpy(features)).numpy()
