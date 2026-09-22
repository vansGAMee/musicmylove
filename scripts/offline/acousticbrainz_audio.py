"""Real audio-derived embeddings from CC0 AcousticBrainz/Essentia measurements.
No artist/title/genre features, no synthesized missing audio, no API key.
Offline only: bulk requests of 25 MBIDs; cache compact measured features.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import requests

UA = 'MusicMyLove/0.1 (offline acoustic index; https://github.com/vansGAMee/musicmylove)'
SCALARS = ['spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_complexity', 'zerocrossingrate']


def features(doc):
    low = doc.get('lowlevel', {})
    result = []
    for name in ['mfcc', 'gfcc']:
        for stat in ['mean', 'var']:
            values = low.get(name, {}).get(stat)
            if stat == 'var':
                cov = low.get(name, {}).get('cov')
                values = [cov[i][i] for i in range(13)] if isinstance(cov, list) and len(cov) == 13 else None
            if not isinstance(values, list) or len(values) != 13:
                return None
            result.extend(math.sqrt(max(0, x)) if stat == 'var' else x for x in values)
    for name in SCALARS:
        value = low.get(name, {})
        if 'mean' not in value or 'var' not in value:
            return None
        result.extend([math.log1p(max(0, value['mean'])), math.log1p(math.sqrt(max(0, value['var'])))])
    rhythm = doc.get('rhythm', {})
    for name in ['bpm', 'danceability', 'onset_rate']:
        if name not in rhythm:
            return None
        result.append(math.log1p(max(0, rhythm[name])))
    return result if all(math.isfinite(x) for x in result) else None


def collect(ids, cache):
    path = cache / (hashlib.sha256(';'.join(ids).encode()).hexdigest() + '.json')
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            r = requests.get('https://acousticbrainz.org/api/v1/low-level', params={'recording_ids': ';'.join(ids)}, headers={'User-Agent': UA}, timeout=20)
            r.raise_for_status()
            payload = r.json()
            rows = []
            for mbid in ids:
                doc = payload.get(mbid, {}).get('0')
                if not doc:
                    continue
                vector = features(doc)
                if vector:
                    rows.append({'id': mbid, 'features': vector, 'sha256': hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(), 'source': f'https://acousticbrainz.org/api/v1/{mbid}/low-level?n=0'})
            path.write_text(json.dumps(rows, allow_nan=False))
            return rows
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.5 * 2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', default='ml/tastelift-catalog.json')
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='data/cache/audio-vectors.json')
    args = parser.parse_args()
    tracks = json.loads(Path(args.catalog).read_text())['tracks']
    # Coverage-first offline bootstrap; ranking never uses these metadata fields.
    tracks = sorted(tracks, key=lambda t: (-t.get('popularityPercentile', 0), t['mbid']))[:args.limit]
    ids = sorted(t['mbid'] for t in tracks)
    cache = Path('data/cache/acousticbrainz-v2')
    cache.mkdir(parents=True, exist_ok=True)
    batches = [ids[i:i+25] for i in range(0, len(ids), 25)]
    rows, failed = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, args.workers))) as pool:
        futures = [pool.submit(collect, batch, cache) for batch in batches]
        for i, future in enumerate(futures):
            try:
                rows.extend(future.result())
            except Exception as error:
                failed += 1
                print('batch failed:', type(error).__name__, flush=True)
            if (i+1) % 20 == 0:
                print(f'batches {i+1}/{len(batches)}, actual audio records {len(rows)}', flush=True)
    rows.sort(key=lambda row: row['id'])
    if len(rows) < 2:
        raise ValueError('Not enough measured audio; no artifact written')
    values = np.array([r['features'] for r in rows], dtype=np.float64)
    center, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1
    values = np.clip((values-center)/scale, -5, 5)
    norms = np.linalg.norm(values, axis=1)
    keep = norms > 1e-8
    rows = [r for r, valid in zip(rows, keep) if valid]
    values = values[keep] / norms[keep, None]
    result = {'encoder': 'essentia-acousticbrainz-measured-mfcc-gfcc-spectral-rhythm-v1',
              'ids': [r['id'] for r in rows], 'vectors': values.round(7).tolist(),
              'transform': {'center': center.tolist(), 'scale': scale.tolist(), 'clip': 5},
              'provenance': [{k: r[k] for k in ['id', 'sha256', 'source']} | {'license': 'CC0-1.0'} for r in rows],
              'coverage': {'requested': len(ids), 'audio': len(rows), 'failedBatches': failed},
              'note': 'Measured audio descriptor embedding, not CLAP or metadata embedding.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result['coverage']), flush=True)

if __name__ == '__main__':
    main()
