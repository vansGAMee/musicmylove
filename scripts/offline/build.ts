import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { buildHnsw, unit } from '../../src/lib/offline/hnsw';
import { key, type Catalog, type Song } from '../../src/lib/offline/engine';

// Graph is built from ListenBrainz histories + real web tracklists.
const input = process.argv[2] ?? process.env.GRAPH_INPUT ?? 'ml/tastelift-catalog.json';
const raw = await readFile(input, 'utf8');
const source = JSON.parse(raw);
if (!Array.isArray(source.histories) || !source.histories.length) throw new Error('Actual human listening histories required');

// 1. Gather all tracks from ListenBrainz
const allTracks: { mbid: string; artist: string; title: string; release?: string }[] = [...source.tracks];
const trackKeyToIndex = new Map<string, number>();
allTracks.forEach((t, i) => {
  const k = key(t);
  if (!trackKeyToIndex.has(k)) trackKeyToIndex.set(k, i);
});

// 2. Gather real web playlists, DJ tracklists, radio tracklists, and curated lists
interface WebList { url: string; tracks: Song[] }
const webTracklistsPath = process.env.WEB_TRACKLISTS ?? (await readFile('data/cache/web-tracklists.json', 'utf8').then(() => 'data/cache/web-tracklists.json').catch(() => undefined));
const webSessions: number[][] = [];
let webTrackCount = 0;

if (webTracklistsPath) {
  const webLists = JSON.parse(await readFile(webTracklistsPath, 'utf8')) as WebList[];
  for (const list of webLists) {
    if (!Array.isArray(list.tracks) || list.tracks.length < 2) continue;
    const sessionIds: number[] = [];
    for (const song of list.tracks) {
      const k = key(song);
      let idx = trackKeyToIndex.get(k);
      if (idx === undefined) {
        const mbid = `web-${createHash('sha256').update(k).digest('hex').slice(0, 16)}`;
        idx = allTracks.length;
        allTracks.push({ mbid, artist: song.artist.trim(), title: song.title.trim() });
        trackKeyToIndex.set(k, idx);
        webTrackCount++;
      }
      sessionIds.push(idx);
    }
    const uniqueSession = [...new Set(sessionIds)].sort((a, b) => a - b);
    if (uniqueSession.length >= 2) {
      webSessions.push(uniqueSession);
    }
  }
}

// 3. Combine ListenBrainz histories + web playlists / DJ tracklists / radio tracklists
const lbHistories: number[][] = source.histories
  .map((h: number[]) => [...new Set(h)].sort((a, b) => a - b))
  .sort((a: number[], b: number[]) => JSON.stringify(a).localeCompare(JSON.stringify(b)));

const allHistories = [...lbHistories, ...webSessions];

const dimension = 64;

function splitmix32(seed: number) {
  let a = (seed ^ 0xdeadbeef) | 0;
  return function() {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return ((t ^ (t >>> 15)) >>> 0);
  };
}

function trackRandom(id: number) {
  const next = splitmix32(id + 1);
  const v = Array(dimension);
  for (let d = 0; d < dimension; d++) {
    v[d] = (next() & 0x80000000) ? 1 : -1;
  }
  return v;
}

const trackIdentities = Array.from({ length: allTracks.length }, (_, i) => trackRandom(i));
const graph: number[][] = allTracks.map(() => Array(dimension).fill(0));
const listeners: number[] = allTracks.map(() => 0);

for (let s = 0; s < allHistories.length; s++) {
  const session = allHistories[s];
  const weight = 1 / Math.sqrt(session.length);
  const sessionSum = Array(dimension).fill(0);
  for (const id of session) {
    if (!graph[id]) throw new Error('Invalid history recording ID');
    listeners[id]++;
    const ti = trackIdentities[id];
    for (let d = 0; d < dimension; d++) sessionSum[d] += ti[d];
  }
  for (const id of session) {
    const ti = trackIdentities[id];
    for (let d = 0; d < dimension; d++) {
      graph[id][d] += (sessionSum[d] - ti[d]) * weight;
    }
  }
}

if (listeners.some(n => !n)) throw new Error('Catalog contains tracks with no human evidence');

const multiListeners = listeners.filter(n => n > 1);
const multiCounts = new Map<number, number>();
for (const n of multiListeners) multiCounts.set(n, (multiCounts.get(n) ?? 0) + 1);
const multiSortedUnique = [...multiCounts.keys()].sort((a, b) => a - b);
const multiLessThan = new Map<number, number>();
let multiRunning = 0;
for (const n of multiSortedUnique) {
  multiLessThan.set(n, multiRunning);
  multiRunning += multiCounts.get(n)!;
}

const normalized = graph.map(v => unit(v).map(x => Math.round(x * 1e6) / 1e6));

// 4. Load trained TasteLiftNet neural ranker weights
const rankerWeightsPath = 'models/tasteliftnet_weights.json';
const neuralRanker = await readFile(rankerWeightsPath, 'utf8').then(r => JSON.parse(r)).catch(() => undefined);

const catalog: Catalog = {
  format: 'human-graph-v1',
  source: `${source.source ?? 'listenbrainz-histories'}+web-tracklists(${webSessions.length});sha256=${createHash('sha256').update(raw).digest('hex')}`,
  tracks: allTracks.map((t, i) => {
    let popularity = 0;
    if (listeners[i] > 1 && multiListeners.length > 0) {
      const eq = multiCounts.get(listeners[i]) ?? 1;
      const less = multiLessThan.get(listeners[i]) ?? 0;
      const rawP = (less + eq / 2) / multiListeners.length;
      popularity = Math.round((0.15 + 0.84 * rawP) * 1000) / 1000;
    }
    return {
      mbid: t.mbid,
      artist: t.artist,
      title: t.title,
      ...(t.release ? { release: t.release } : {}),
      popularity,
    };
  }),
  graph: normalized,
  graphIndex: buildHnsw(normalized),
  context: {},
  ...(neuralRanker ? { neuralRanker } : {}),
};

// 5. Optional real audio vectors: generated by embed_audio.py or acousticbrainz_audio.py, never synthesized.
const audioPath = process.env.AUDIO_VECTORS ?? (await readFile('data/cache/audio-vectors.json', 'utf8').then(() => 'data/cache/audio-vectors.json').catch(() => undefined));
if (audioPath) {
  const a = JSON.parse(await readFile(audioPath, 'utf8'));
  if (!a.encoder || !Array.isArray(a.ids) || a.ids.length !== a.vectors.length || !a.provenance?.length) throw new Error('Audio provenance required');
  const dims = a.vectors[0]?.length;
  if (!dims || a.vectors.some((v: number[]) => v.length !== dims || v.some(x => !Number.isFinite(x)) || v.every(x => x === 0))) throw new Error('Invalid audio vectors');
  const known = new Set(catalog.tracks.map(t => t.mbid));
  const rows = a.ids.map((id: string, i: number) => ({ id, vector: a.vectors[i] })).filter((r: { id: string }) => known.has(r.id)).sort((a: { id: string }, b: { id: string }) => a.id.localeCompare(b.id));
  const vectors = rows.map((r: { vector: number[] }) => unit(r.vector));
  catalog.audio = { encoder: a.encoder, ids: rows.map((r: { id: string }) => r.id), vectors, index: buildHnsw(vectors) };
}

const contextPath = process.env.WEB_CONTEXT ?? (await readFile('data/cache/web-context.json', 'utf8').then(() => 'data/cache/web-context.json').catch(() => undefined));
if (contextPath) catalog.context = JSON.parse(await readFile(contextPath, 'utf8'));

await mkdir('public/data', { recursive: true });
const serialized = JSON.stringify(catalog);
const sha256 = createHash('sha256').update(serialized).digest('hex');
await writeFile('public/data/catalog.json', serialized);
await writeFile('public/data/manifest.json', JSON.stringify({
  format: 'human-graph-manifest-v1',
  sha256,
  file: 'catalog.json',
  tracks: catalog.tracks.length,
  histories: allHistories.length,
  audio: catalog.audio?.ids.length ?? 0,
  hasNeuralRanker: Boolean(catalog.neuralRanker),
  source: catalog.source
}));

console.log(JSON.stringify({
  tracks: catalog.tracks.length,
  histories: allHistories.length,
  audio: catalog.audio?.ids.length ?? 0,
  hasNeuralRanker: Boolean(catalog.neuralRanker),
  source: catalog.source
}));
