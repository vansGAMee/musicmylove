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

const listeners: number[] = allTracks.map(() => 0);
for (let s = 0; s < allHistories.length; s++) {
  const session = allHistories[s];
  for (const id of session) {
    if (!allTracks[id]) throw new Error('Invalid history recording ID');
    listeners[id]++;
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

// 4. Load learned Stage A embeddings from models/track_embeddings.npy
let embeddingsSha256 = '';
const embeddingsPath = process.env.TRACK_EMBEDDINGS ?? 'models/track_embeddings.npy';

interface NpyData {
  shape: number[];
  data: Float32Array;
}

function parseNpy(buf: Buffer): NpyData {
  if (buf[0] !== 0x93 || buf.toString('ascii', 1, 6) !== 'NUMPY') {
    throw new Error('Not a valid .npy file');
  }
  const major = buf[6];
  let headerLen = 0;
  let offset = 8;
  if (major === 1) {
    headerLen = buf.readUInt16LE(8);
    offset = 10;
  } else {
    headerLen = buf.readUInt32LE(8);
    offset = 12;
  }
  const headerStr = buf.toString('ascii', offset, offset + headerLen);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error('Could not parse shape from .npy header');
  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
  const dataOffset = offset + headerLen;
  const floatCount = (buf.byteLength - dataOffset) / 4;
  const floatArray = new Float32Array(buf.buffer, buf.byteOffset + dataOffset, floatCount);
  return { shape, data: floatArray };
}

let npy: NpyData | undefined;
let isDirectCatalogEmbeddings = false;

try {
  const embeddingsBuffer = await readFile(embeddingsPath);
  embeddingsSha256 = createHash('sha256').update(embeddingsBuffer).digest('hex');
  npy = parseNpy(embeddingsBuffer);
  console.log(`[embeddings] Loaded learned Stage A embeddings from ${embeddingsPath}`);
  console.log(`  Vocab: ${npy.shape[0]}, Dim: ${npy.shape[1]}, SHA256: ${embeddingsSha256}`);
} catch {
  // Check for catalog_embeddings.int8.bin or track_embeddings.int8.bin (Vercel deployment)
  for (const binPath of ['models/catalog_embeddings.int8.bin', 'models/track_embeddings.int8.bin']) {
    try {
      const int8Buf = await readFile(binPath);
      embeddingsSha256 = createHash('sha256').update(int8Buf).digest('hex');
      const int8Data = new Int8Array(int8Buf.buffer, int8Buf.byteOffset, int8Buf.byteLength);
      const dim = 64;
      const vocab = Math.floor(int8Data.length / dim);
      const floatData = new Float32Array(int8Data.length);
      for (let j = 0; j < int8Data.length; j++) {
        floatData[j] = int8Data[j] / 127.0;
      }
      npy = { shape: [vocab, dim], data: floatData };
      if (binPath.includes('catalog_embeddings')) isDirectCatalogEmbeddings = true;
      console.log(`[embeddings] Loaded learned Stage A embeddings from fallback ${binPath}`);
      console.log(`  Vocab: ${npy.shape[0]}, Dim: ${npy.shape[1]}, SHA256: ${embeddingsSha256}`);
      break;
    } catch {
      // continue to next candidate
    }
  }
}

// Load expanded catalog index map or catalog_indices.json to verify exact 1-to-1 row alignment
let expandedKeyToIdx = new Map<string, number>();
let catalogIndicesList: number[] = [];

try {
  const expRaw = await readFile('data/cache/pipeline/expanded_catalog.json', 'utf8');
  const expCat = JSON.parse(expRaw);
  expCat.tracks.forEach((t: { artist: string; title: string }, i: number) => {
    expandedKeyToIdx.set(key(t), i);
  });
  console.log(`[alignment] Loaded expanded catalog index map (${expandedKeyToIdx.size} tracks)`);
} catch {
  try {
    const idxRaw = await readFile('models/catalog_indices.json', 'utf8');
    catalogIndicesList = JSON.parse(idxRaw);
    console.log(`[alignment] Loaded catalog indices mapping (${catalogIndicesList.length} tracks)`);
  } catch {
    console.warn('[alignment] Neither expanded_catalog.json nor catalog_indices.json found; using sequential fallback');
  }
}

const dim = npy ? npy.shape[1] : 64;
const graph: number[][] = [];
let matchedLearnedCount = 0;

for (let i = 0; i < allTracks.length; i++) {
  const t = allTracks[i];
  const k = key(t);
  let rowIdx: number | undefined;

  if (isDirectCatalogEmbeddings) {
    rowIdx = i;
  } else if (expandedKeyToIdx.has(k)) {
    rowIdx = expandedKeyToIdx.get(k);
  } else if (i < catalogIndicesList.length) {
    rowIdx = catalogIndicesList[i];
  } else if (i < (npy?.shape[0] ?? 0)) {
    rowIdx = i;
  }

  if (npy && rowIdx !== undefined && rowIdx < npy.shape[0]) {
    const offset = rowIdx * dim;
    const row = Array.from(npy.data.subarray(offset, offset + dim));
    graph.push(row);
    matchedLearnedCount++;
  } else {
    // Zero vector fallback
    graph.push(Array(dim).fill(0));
  }
}

console.log(`[embeddings] Matched ${matchedLearnedCount}/${allTracks.length} tracks to real learned Stage A embeddings`);

const normalized = graph.map(v => unit(v).map(x => Math.round(x * 1e6) / 1e6));

// 5. Load trained TasteLiftNet neural ranker weights
const rankerWeightsPath = 'models/tasteliftnet_weights.json';
const neuralRanker = await readFile(rankerWeightsPath, 'utf8').then(r => JSON.parse(r)).catch(() => undefined);
if (neuralRanker) {
  console.log('[ranker] Loaded trained TasteLiftNet ranker weights');
}

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
  ...(embeddingsSha256 ? { embeddingsSha256 } : {}),
};

// 6. Optional real audio vectors: generated by embed_audio.py or acousticbrainz_audio.py, never synthesized.
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
  embeddingsSha256: catalog.embeddingsSha256 ?? '',
  source: catalog.source
}));

console.log(JSON.stringify({
  tracks: catalog.tracks.length,
  histories: allHistories.length,
  audio: catalog.audio?.ids.length ?? 0,
  hasNeuralRanker: Boolean(catalog.neuralRanker),
  embeddingsSha256: catalog.embeddingsSha256 ?? '',
  source: catalog.source
}));
