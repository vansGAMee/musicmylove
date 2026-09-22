import { dot, unit, searchHnsw, type Hnsw } from './hnsw';
import type { Track } from '../types';

export type Song = Pick<Track, 'artist' | 'title'>;
export type Feedback = Record<string, 'like' | 'dislike'>;
export interface Context { neighbors: string[]; sources: string[] }

export interface NeuralRankerWeights {
  queries: number[][]; // [4][64]
  audio_proj_0_w: number[][]; // [32][65]
  audio_proj_0_b: number[]; // [32]
  audio_proj_2_w: number[][]; // [16][32]
  audio_proj_2_b: number[]; // [16]
  mlp_0_w: number[][]; // [64][91]
  mlp_0_b: number[]; // [64]
  mlp_2_w: number[][]; // [32][64]
  mlp_2_b: number[]; // [32]
  mlp_4_w: number[][]; // [1][32]
  mlp_4_b: number[]; // [1]
}

export interface Catalog {
  format: 'human-graph-v1';
  source: string;
  tracks: (Track & { popularity: number })[];
  graph: number[][];
  graphIndex: Hnsw;
  audio?: { ids: string[]; vectors: number[][]; index: Hnsw; encoder: string };
  context: Record<string, Context>;
  neuralRanker?: NeuralRankerWeights;
}

export const key = (s: Song) => [s.artist, s.title].map(x => x.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim()).join('\u001f');

interface Seed { track: Song; id?: number; graph?: number[]; audio?: number[] }

const cache = new WeakMap<Catalog, { ids: Map<string, number>; keys: Map<string, number>; audio: Map<string, number> }>();
function maps(c: Catalog) {
  let m = cache.get(c);
  if (!m) {
    m = {
      ids: new Map(c.tracks.map((t, i) => [t.mbid, i])),
      keys: new Map(),
      audio: new Map(c.audio?.ids.map((id, i) => [id, i]) ?? [])
    };
    c.tracks.forEach((t, i) => {
      if (!m!.keys.has(key(t))) m!.keys.set(key(t), i);
    });
    cache.set(c, m);
  }
  return m;
}

export function missing(c: Catalog, songs: Song[]): Song[] {
  const m = maps(c);
  return [...new Map(songs.filter(s => !m.keys.has(key(s)) && !c.context[key(s)]).map(s => [key(s), s])).values()].sort((a, b) => key(a) < key(b) ? -1 : 1);
}

function mean(vs: number[][]) {
  return unit(vs[0].map((_, i) => vs.reduce((n, v) => n + v[i], 0) / vs.length));
}

function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function projectAudio(feat: number[], w: NeuralRankerWeights): number[] {
  const h1 = new Array(32);
  for (let i = 0; i < 32; i++) {
    let s = w.audio_proj_0_b[i];
    const wi = w.audio_proj_0_w[i];
    for (let j = 0; j < 65; j++) s += wi[j] * (feat[j] ?? 0);
    h1[i] = gelu(s);
  }
  const h2 = new Array(16);
  for (let i = 0; i < 16; i++) {
    let s = w.audio_proj_2_b[i];
    const wi = w.audio_proj_2_w[i];
    for (let j = 0; j < 32; j++) s += wi[j] * h1[j];
    h2[i] = s;
  }
  return unit(h2);
}

function evaluateNeuralMlp(input: number[], w: NeuralRankerWeights): number {
  const h1 = new Array(64);
  for (let i = 0; i < 64; i++) {
    let s = w.mlp_0_b[i];
    const wi = w.mlp_0_w[i];
    for (let j = 0; j < input.length; j++) s += wi[j] * input[j];
    h1[i] = gelu(s);
  }
  const h2 = new Array(32);
  for (let i = 0; i < 32; i++) {
    let s = w.mlp_2_b[i];
    const wi = w.mlp_2_w[i];
    for (let j = 0; j < 64; j++) s += wi[j] * h1[j];
    h2[i] = gelu(s);
  }
  let logit = w.mlp_4_b[0];
  const w4 = w.mlp_4_w[0];
  for (let j = 0; j < 32; j++) logit += w4[j] * h2[j];
  return logit;
}

function heads(vectors: number[][]): number[][] {
  if (!vectors.length) return [];
  const count = Math.min(vectors.length, 12, Math.max(2, Math.ceil(Math.sqrt(vectors.length))));
  let centers = [vectors[0]];
  while (centers.length < count) {
    let best = -1, idx = -1;
    vectors.forEach((v, i) => {
      const d = 1 - Math.max(...centers.map(c => dot(c, v)));
      if (d > best) { best = d; idx = i; }
    });
    if (best < 1e-7) break;
    centers.push(vectors[idx]);
  }
  for (let iteration = 0; iteration < 4; iteration++) {
    const groups: number[][][] = centers.map(() => []);
    for (const v of vectors) {
      let best = 0;
      for (let i = 1; i < centers.length; i++) if (dot(v, centers[i]) > dot(v, centers[best])) best = i;
      groups[best].push(v);
    }
    centers = groups.filter(g => g.length).map(mean);
  }
  return centers;
}

export function recommend(c: Catalog, songs: Song[], feedback: Feedback, context: Record<string, Context> = {}) {
  const m = maps(c);
  const unique = [...new Map(songs.map(s => [key(s), s])).entries()].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([, s]) => s);
  const seeds: Seed[] = unique.map(track => {
    const id = m.keys.get(key(track));
    const evidence = c.context[key(track)] ?? context[key(track)];
    const neighbors = evidence?.sources.length ? [...new Set(evidence.neighbors)].sort().flatMap(nid => { const i = m.ids.get(nid); return i === undefined ? [] : [c.graph[i]]; }) : [];
    const ai = id === undefined ? undefined : m.audio.get(c.tracks[id].mbid);
    return { track, id, graph: id === undefined ? (neighbors.length ? mean(neighbors) : undefined) : c.graph[id], audio: ai === undefined ? undefined : c.audio!.vectors[ai] };
  });

  const validSeedVectors = seeds.flatMap(s => s.graph ? [s.graph] : []);
  if (!validSeedVectors.length) {
    return {
      recommendations: [],
      seeds: seeds.map(s => ({ input: s.track, status: 'unresolved' as const, track: undefined })),
      coverage: { total: seeds.length, graph: 0, audio: 0 },
      candidateCount: 0
    };
  }

  const positive = Object.keys(feedback).filter(id => feedback[id] === 'like').sort().flatMap(id => { const i = m.ids.get(id); return i === undefined ? [] : [i]; });
  const excluded = new Set(unique.map(key));
  positive.forEach(i => excluded.add(key(c.tracks[i])));

  const candidates = new Map<number, number>();

  // TASTELIFTNET NEURAL RECOMMENDER PATH
  if (c.neuralRanker) {
    const nr = c.neuralRanker;
    const numHeads = nr.queries.length;
    const seedVectorsWithPositives = [...validSeedVectors, ...positive.map(i => c.graph[i])];

    // 1. Multi-Head Taste Attention over seeds
    const tasteHeads: number[][] = [];
    for (let k = 0; k < numHeads; k++) {
      const q = nr.queries[k];
      const scores = seedVectorsWithPositives.map(v => dot(q, v) / 8);
      const maxScore = Math.max(...scores);
      const exps = scores.map(s => Math.exp(s - maxScore));
      const sumExp = Math.max(1e-9, exps.reduce((a, b) => a + b, 0));
      const weights = exps.map(e => e / sumExp);
      const head = new Array(64).fill(0);
      for (let mIdx = 0; mIdx < seedVectorsWithPositives.length; mIdx++) {
        const vec = seedVectorsWithPositives[mIdx];
        const wt = weights[mIdx];
        for (let d = 0; d < 64; d++) head[d] += wt * vec[d];
      }
      tasteHeads.push(unit(head));
    }

    // 2. MULTI-HEAD ANN RETRIEVAL (Separate queries, no centroid collapse)
    for (const h of tasteHeads) {
      for (const hit of searchHnsw(c.graphIndex, c.graph, h, 1000)) {
        candidates.set(hit.id, Math.max(candidates.get(hit.id) ?? -1, hit.score));
      }
    }
    if (candidates.size < 2000 && tasteHeads.length) {
      const meanHead = unit(new Array(64).fill(0).map((_, d) => tasteHeads.reduce((acc, h) => acc + h[d], 0) / tasteHeads.length));
      for (const hit of searchHnsw(c.graphIndex, c.graph, meanHead, 2000)) {
        candidates.set(hit.id, Math.max(candidates.get(hit.id) ?? -1, hit.score));
      }
    }

    // 3. Audio ANN Retrieval if audio features and index exist
    const seedAudioVectors = seeds.flatMap(s => s.audio ? [s.audio] : []);
    let seedAudioProj = new Array(16).fill(0);
    if (c.audio && seedAudioVectors.length) {
      const meanAudio = seedAudioVectors[0].map((_, idx) => seedAudioVectors.reduce((acc, v) => acc + v[idx], 0) / seedAudioVectors.length);
      seedAudioProj = projectAudio(meanAudio, nr);
      for (const hit of searchHnsw(c.audio.index, c.audio.vectors, meanAudio, 500)) {
        const id = m.ids.get(c.audio.ids[hit.id]);
        if (id !== undefined) candidates.set(id, Math.max(candidates.get(id) ?? -1, hit.score));
      }
    }

    // 4. Neural Candidate Scoring
    const scoredCandidates = [...candidates.keys()].flatMap(id => {
      const t = c.tracks[id];
      if (excluded.has(key(t)) || feedback[t.mbid] === 'dislike') return [];
      const candVec = c.graph[id];

      const dots = tasteHeads.map(h => dot(h, candVec));
      const maxDot = Math.max(...dots);

      let candAudioProj = new Array(16).fill(0);
      let acousticSim = 0.0;
      let audioMask = 0.0;
      const ai = m.audio.get(t.mbid);
      if (c.audio && ai !== undefined) {
        const rawAudio = c.audio.vectors[ai];
        candAudioProj = projectAudio(rawAudio, nr);
        acousticSim = dot(candAudioProj, seedAudioProj);
        audioMask = 1.0;
      }

      const pop = t.popularity ?? 0.1;
      const sessionFeat = Math.min(1.0, Math.log1p(pop * 10) / 5.0);

      const fused = [
        ...candVec,
        ...dots,
        maxDot,
        acousticSim,
        ...candAudioProj,
        audioMask,
        pop,
        sessionFeat
      ];

      const neuralLogit = evaluateNeuralMlp(fused, nr);

      const g = Math.max(0, ...dots);
      const strongestTasteHead = Math.max(0, dots.indexOf(maxDot));
      const seedSims = seeds.flatMap(s => s.graph ? [{ track: s.track, sim: dot(s.graph, candVec) }] : []);
      const topInHead = seedSims.sort((a, b) => b.sim - a.sim).slice(0, 3);
      const inHeadRelevance = topInHead.length ? topInHead.reduce((acc, x) => acc + Math.max(0, x.sim), 0) / topInHead.length : g;
      const rawAffinity = 0.50 * g + 0.50 * inHeadRelevance;
      const calibratedAffinity = 0.60 * rawAffinity + 0.40 * Math.min(1.0, rawAffinity / 0.8);

      const inHeadSupport = seedSims.filter(x => x.sim > 0.15).length;
      const supportBonus = 0.06 * (inHeadSupport / Math.max(1, seeds.length));

      const sound = (c.audio && ai !== undefined) ? Math.max(0, acousticSim) : undefined;
      const tasteScore = sound === undefined ? calibratedAffinity : 0.70 * calibratedAffinity + 0.30 * sound;
      const rare = Math.max(0, tasteScore * (1 - pop));

      let userShift = 0.0;
      for (const posId of positive) {
        userShift += 0.05 * Math.max(0, dot(c.graph[posId], candVec));
      }

      const neuralBoost = Math.max(-0.15, Math.min(0.15, neuralLogit * 0.08));
      const finalScore = tasteScore + neuralBoost + supportBonus + 0.03 * rare - 0.02 * pop + userShift;

      return [{
        ...t,
        score: finalScore,
        strongestTasteHead,
        seedSupport: inHeadSupport,
        supportingSeeds: seeds.slice(0, 3).map(s => ({ artist: s.track.artist, title: s.track.title })),
        popularityPercentile: pop,
        noveltyLiftScore: rare,
        spotifyLink: `https://open.spotify.com/search/${encodeURIComponent(t.artist + ' ' + t.title)}`
      }];
    }).sort((a, b) => b.score - a.score || (a.mbid < b.mbid ? -1 : 1));

    const seen = new Set<string>();
    const artists = new Map<string, number>();
    const recommendations = scoredCandidates.filter(t => {
      const a = t.artist.toLowerCase();
      const k = key(t);
      if (seen.has(k) || (artists.get(a) ?? 0) >= 2) return false;
      seen.add(k);
      artists.set(a, (artists.get(a) ?? 0) + 1);
      return true;
    }).slice(0, 40);

    return {
      recommendations,
      seeds: seeds.map(s => ({
        input: s.track,
        status: (s.graph || s.audio ? 'resolved' : 'unresolved') as 'resolved' | 'unresolved',
        track: s.id === undefined ? undefined : c.tracks[s.id]
      })),
      coverage: {
        total: seeds.length,
        graph: seeds.filter(s => s.graph).length,
        audio: seeds.filter(s => s.audio).length
      },
      candidateCount: candidates.size
    };
  }

  // FALLBACK HEURISTIC PATH (Used when neuralRanker is not embedded)
  const gh = heads([...validSeedVectors, ...positive.map(i => c.graph[i])]);
  const ah = heads([...seeds.flatMap(s => s.audio ? [s.audio] : []), ...positive.flatMap(i => { const a = m.audio.get(c.tracks[i].mbid); return a === undefined ? [] : [c.audio!.vectors[a]]; })]);

  for (const h of gh) for (const hit of searchHnsw(c.graphIndex, c.graph, h, Math.ceil(2000 / Math.max(1, gh.length)))) candidates.set(hit.id, Math.max(candidates.get(hit.id) ?? -1, hit.score));
  if (c.audio) for (const h of ah) for (const hit of searchHnsw(c.audio.index, c.audio.vectors, h, Math.ceil(2000 / Math.max(1, ah.length)))) { const id = m.ids.get(c.audio.ids[hit.id]); if (id !== undefined) candidates.set(id, Math.max(candidates.get(id) ?? -1, hit.score)); }
  if (candidates.size < 2000 && gh.length) for (const hit of searchHnsw(c.graphIndex, c.graph, mean(gh), 2000)) candidates.set(hit.id, Math.max(candidates.get(hit.id) ?? -1, hit.score));
  const negatives = Object.keys(feedback).filter(id => feedback[id] === 'dislike').sort().flatMap(id => { const i = m.ids.get(id); return i === undefined ? [] : [c.graph[i]]; });
  const seedHeadMap = seeds.map(s => { if (!s.graph || !gh.length) return 0; const sims = gh.map(h => dot(h, s.graph!)); return Math.max(0, sims.indexOf(Math.max(...sims))); });
  const headSeedCounts = gh.map((_, h) => seedHeadMap.filter(x => x === h).length);
  const maxAffinityPerHead = gh.map(() => 0.01);
  const preCandidates = [...candidates].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 2000).flatMap(([id]) => {
    const t = c.tracks[id]; if (excluded.has(key(t)) || feedback[t.mbid] === 'dislike') return [];
    const candVec = c.graph[id];
    const gs = gh.map(h => Math.max(0, dot(h, candVec))), g = Math.max(0, ...gs), strongestTasteHead = Math.max(0, gs.indexOf(g));
    const seedSims = seeds.flatMap((s, idx) => s.graph ? [{ track: s.track, sim: dot(s.graph, candVec), head: seedHeadMap[idx] }] : []);
    const inHeadSims = seedSims.filter(x => x.head === strongestTasteHead).sort((a, b) => b.sim - a.sim);
    const topInHead = inHeadSims.slice(0, Math.min(3, inHeadSims.length));
    const inHeadRelevance = topInHead.length ? topInHead.reduce((acc, x) => acc + Math.max(0, x.sim), 0) / topInHead.length : g;
    const rawAffinity = 0.50 * g + 0.50 * inHeadRelevance;
    if (rawAffinity > maxAffinityPerHead[strongestTasteHead]) maxAffinityPerHead[strongestTasteHead] = rawAffinity;
    return [{ id, t, candVec, gs, g, strongestTasteHead, seedSims, inHeadSims, rawAffinity }];
  });
  const pool = preCandidates.map(item => {
    const { t, candVec, strongestTasteHead, seedSims, inHeadSims, rawAffinity } = item;
    const maxH = maxAffinityPerHead[strongestTasteHead] || 1;
    const calibratedAffinity = 0.60 * rawAffinity + 0.40 * Math.min(1.0, rawAffinity / Math.max(0.25, maxH));
    const inHeadSupport = inHeadSims.filter(x => x.sim > 0.15).length;
    const supportRatio = inHeadSupport / Math.max(1, headSeedCounts[strongestTasteHead]);
    const supportBonus = 0.06 * supportRatio;
    const ai = m.audio.get(t.mbid);
    const sound = ai === undefined || !ah.length ? undefined : Math.max(0, ...ah.map(h => dot(h, c.audio!.vectors[ai])));
    const tasteScore = sound === undefined ? calibratedAffinity : 0.70 * calibratedAffinity + 0.30 * sound;
    const rare = Math.max(0, tasteScore * (1 - t.popularity));
    const negative = negatives.length ? Math.max(0, ...negatives.map(v => dot(v, candVec))) : 0;
    const score = tasteScore + supportBonus + 0.03 * rare - 0.02 * t.popularity - 0.35 * negative;
    const posSeeds = seedSims.filter(x => x.sim > 0.05).sort((a, b) => b.sim - a.sim);
    const bestSeeds = posSeeds.length >= 2 ? posSeeds : seedSims.sort((a, b) => b.sim - a.sim);
    const supportingSeeds = bestSeeds.slice(0, 4).map(x => ({ artist: x.track.artist, title: x.track.title }));
    return { ...t, score, strongestTasteHead, seedSupport: inHeadSupport, supportingSeeds, popularityPercentile: t.popularity, noveltyLiftScore: rare, spotifyLink: `https://open.spotify.com/search/${encodeURIComponent(t.artist + ' ' + t.title)}` };
  }).sort((a, b) => b.score - a.score || (a.mbid < b.mbid ? -1 : 1));
  const seen = new Set<string>(), artists = new Map<string, number>();
  const recommendations = pool.filter(t => { const a = t.artist.toLowerCase(), k = key(t); if (seen.has(k) || (artists.get(a) ?? 0) >= 2) return false; seen.add(k); artists.set(a, (artists.get(a) ?? 0) + 1); return true; }).slice(0, 40);
  return { recommendations, seeds: seeds.map(s => ({ input: s.track, status: s.graph || s.audio ? 'resolved' : 'unresolved', track: s.id === undefined ? undefined : c.tracks[s.id] })), coverage: { total: seeds.length, graph: seeds.filter(s => s.graph).length, audio: seeds.filter(s => s.audio).length }, candidateCount: candidates.size };
}
