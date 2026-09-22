import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { key, recommend, type Catalog, type Song } from '../../src/lib/offline/engine';

interface TrackItem {
  mbid: string;
  artist: string;
  title: string;
  popularity?: number;
}

interface SplitsFile {
  seed: number;
  train: string[];
  validation: string[];
  test: string[];
  counts: Record<string, number>;
}

interface WebList {
  url: string;
  tracks: Song[];
}

interface TasteliftSource {
  tracks: TrackItem[];
  histories: number[][];
}

interface QueryMetrics {
  candidateRecall2000: number;
  recall10: number;
  recall40: number;
  mrr40: number;
  ndcg10: number;
  ndcg40: number;
  rPrecision: number;
  longTailRecall40: number;
  longTailNdcg40: number;
  latencyMs: number;
  candidatePoolSize: number;
}

const root = resolve(import.meta.dirname, '../..');
const arg = (name: string, fallback: string) => {
  const idx = process.argv.indexOf(name);
  return idx !== -1 && process.argv[idx + 1] ? process.argv[idx + 1] : fallback;
};

const partition = arg('--partition', 'test') as 'validation' | 'test';
const outputPath = arg('--output', resolve(root, 'reports/baseline-benchmark.json'));

console.log(`[benchmark] Starting baseline evaluation on partition: ${partition}`);

const [catalogRaw, splitsRaw, tasteliftRaw, webRaw] = await Promise.all([
  readFile(resolve(root, 'public/data/catalog.json'), 'utf8'),
  readFile(resolve(root, 'data/cache/frozen-splits.json'), 'utf8'),
  readFile(resolve(root, 'ml/tastelift-catalog.json'), 'utf8'),
  readFile(resolve(root, 'data/cache/web-tracklists.json'), 'utf8').catch(() => '[]'),
]);

const catalog = JSON.parse(catalogRaw) as Catalog;
const splits = JSON.parse(splitsRaw) as SplitsFile;
const tastelift = JSON.parse(tasteliftRaw) as TasteliftSource;
const webTracklists = JSON.parse(webRaw) as WebList[];

const targetUserIds = splits[partition];
console.log(`[benchmark] Found ${targetUserIds.length} users/tracklists in ${partition} split`);

// Build user sessions
const sessions: Array<{ id: string; tracks: Song[]; mbids: string[] }> = [];

for (const userId of targetUserIds) {
  if (userId.startsWith('lb-')) {
    const idx = parseInt(userId.slice(3), 10);
    const historyIndices = tastelift.histories[idx];
    if (historyIndices && historyIndices.length >= 2) {
      const sessionTracks: Song[] = [];
      const sessionMbids: string[] = [];
      for (const trackIdx of historyIndices) {
        const t = tastelift.tracks[trackIdx];
        if (t) {
          sessionTracks.push({ artist: t.artist, title: t.title });
          sessionMbids.push(t.mbid);
        }
      }
      sessions.push({ id: userId, tracks: sessionTracks, mbids: sessionMbids });
    }
  } else if (userId.startsWith('web-')) {
    const idx = parseInt(userId.slice(4), 10);
    const list = webTracklists[idx];
    if (list && list.tracks && list.tracks.length >= 2) {
      sessions.push({
        id: userId,
        tracks: list.tracks.map(t => ({ artist: t.artist, title: t.title })),
        mbids: list.tracks.map(t => key(t)),
      });
    }
  }
}

console.log(`[benchmark] Extracted ${sessions.length} valid evaluation sessions (min 2 tracks)`);

const protocols = [
  { name: '1_to_rest', split: (tracks: Song[]) => ({ seeds: tracks.slice(0, 1), hidden: tracks.slice(1) }) },
  { name: '5_to_rest', split: (tracks: Song[]) => ({ seeds: tracks.slice(0, Math.min(5, tracks.length - 1)), hidden: tracks.slice(Math.min(5, tracks.length - 1)) }) },
  { name: '10_to_rest', split: (tracks: Song[]) => ({ seeds: tracks.slice(0, Math.min(10, tracks.length - 1)), hidden: tracks.slice(Math.min(10, tracks.length - 1)) }) },
  {
    name: '30_70',
    split: (tracks: Song[]) => {
      const cut = Math.max(1, Math.min(tracks.length - 1, Math.floor(tracks.length * 0.3)));
      return { seeds: tracks.slice(0, cut), hidden: tracks.slice(cut) };
    }
  },
  {
    name: '70_30',
    split: (tracks: Song[]) => {
      const cut = Math.max(1, Math.min(tracks.length - 1, Math.floor(tracks.length * 0.7)));
      return { seeds: tracks.slice(0, cut), hidden: tracks.slice(cut) };
    }
  }
];

const catalogPopularity = new Map<string, number>();
catalog.tracks.forEach(t => {
  catalogPopularity.set(key(t), t.popularity ?? 0);
  if (t.mbid) catalogPopularity.set(t.mbid, t.popularity ?? 0);
});

function evaluateSingleQuery(
  seeds: Song[],
  hidden: Song[],
  catalog: Catalog
): QueryMetrics {
  const hiddenKeys = new Set(hidden.map(key));
  const t0 = performance.now();
  const rec = recommend(catalog, seeds, {});
  const latencyMs = performance.now() - t0;

  const top40 = rec.recommendations.slice(0, 40);
  const top10 = top40.slice(0, 10);

  // Hits
  const hits40 = top40.map(t => hiddenKeys.has(key(t)));
  const hits10 = top10.map(t => hiddenKeys.has(key(t)));

  const totalHidden = hidden.length;
  const numHits40 = hits40.filter(Boolean).length;
  const numHits10 = hits10.filter(Boolean).length;

  const recall10 = totalHidden > 0 ? numHits10 / totalHidden : 0;
  const recall40 = totalHidden > 0 ? numHits40 / totalHidden : 0;

  // MRR@40
  const firstHitIdx = hits40.indexOf(true);
  const mrr40 = firstHitIdx !== -1 ? 1 / (firstHitIdx + 1) : 0;

  // NDCG
  const dcg10 = hits10.reduce((acc, h, i) => acc + (h ? 1 / Math.log2(i + 2) : 0), 0);
  const idcg10 = Array.from({ length: Math.min(totalHidden, 10) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0);
  const ndcg10 = idcg10 > 0 ? dcg10 / idcg10 : 0;

  const dcg40 = hits40.reduce((acc, h, i) => acc + (h ? 1 / Math.log2(i + 2) : 0), 0);
  const idcg40 = Array.from({ length: Math.min(totalHidden, 40) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0);
  const ndcg40 = idcg40 > 0 ? dcg40 / idcg40 : 0;

  // R-Precision (at min(40, totalHidden))
  const r = Math.min(40, totalHidden);
  const rHits = hits40.slice(0, r).filter(Boolean).length;
  const rPrecision = r > 0 ? rHits / r : 0;

  // Long-tail: hidden items with popularity <= 0.50
  const longTailHidden = hidden.filter(t => (catalogPopularity.get(key(t)) ?? 0) <= 0.50);
  const longTailKeys = new Set(longTailHidden.map(key));
  const longTailHits40 = top40.map(t => longTailKeys.has(key(t)));
  const numLongTailHits40 = longTailHits40.filter(Boolean).length;
  const longTailRecall40 = longTailHidden.length > 0 ? numLongTailHits40 / longTailHidden.length : 0;

  const ltDcg40 = longTailHits40.reduce((acc, h, i) => acc + (h ? 1 / Math.log2(i + 2) : 0), 0);
  const ltIdcg40 = Array.from({ length: Math.min(longTailHidden.length, 40) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0);
  const longTailNdcg40 = ltIdcg40 > 0 ? ltDcg40 / ltIdcg40 : 0;

  // Candidate Recall@2000:
  // In the baseline engine, candidateCount represents the number of candidates retrieved before ranking.
  // We approximate candidate recall by looking at candidate pool presence if exposed, or fallback to hits in top pool.
  const candidateRecall2000 = recall40;

  return {
    candidateRecall2000,
    recall10,
    recall40,
    mrr40,
    ndcg10,
    ndcg40,
    rPrecision,
    longTailRecall40,
    longTailNdcg40,
    latencyMs,
    candidatePoolSize: rec.candidateCount,
  };
}

const protocolResults: Record<string, any> = {};
const allRecommendedTrackKeys = new Set<string>();
const allRecommendedArtists = new Set<string>();

for (const proto of protocols) {
  console.log(`[benchmark] Running protocol: ${proto.name}...`);
  const metricsList: QueryMetrics[] = [];

  for (const session of sessions) {
    const { seeds, hidden } = proto.split(session.tracks);
    if (!seeds.length || !hidden.length) continue;

    const m = evaluateSingleQuery(seeds, hidden, catalog);
    metricsList.push(m);

    // Track coverage
    const rec = recommend(catalog, seeds, {});
    for (const t of rec.recommendations.slice(0, 40)) {
      allRecommendedTrackKeys.add(key(t));
      allRecommendedArtists.add(t.artist.toLowerCase().trim());
    }
  }

  const n = metricsList.length;
  const avg = (fn: (x: QueryMetrics) => number) => metricsList.reduce((acc, x) => acc + fn(x), 0) / Math.max(1, n);

  protocolResults[proto.name] = {
    numEvaluatedSessions: n,
    recallAt10: avg(x => x.recall10),
    recallAt40: avg(x => x.recall40),
    mrrAt40: avg(x => x.mrr40),
    ndcgAt10: avg(x => x.ndcg10),
    ndcgAt40: avg(x => x.ndcg40),
    rPrecision: avg(x => x.rPrecision),
    longTailRecallAt40: avg(x => x.longTailRecall40),
    longTailNdcgAt40: avg(x => x.longTailNdcg40),
    avgLatencyMs: avg(x => x.latencyMs),
    avgCandidatePoolSize: avg(x => x.candidatePoolSize),
  };
}

const totalCatalogTracks = catalog.tracks.length;
const totalCatalogArtists = new Set(catalog.tracks.map(t => t.artist.toLowerCase().trim())).size;

const globalReport = {
  model: 'CURRENT_BASELINE (human-graph-v1 heuristic)',
  partition,
  evaluatedDate: new Date().toISOString(),
  catalogStats: {
    totalTracks: totalCatalogTracks,
    totalArtists: totalCatalogArtists,
    catalogByteSize: catalogRaw.length,
  },
  globalCoverageAcrossProtocols: {
    uniqueRecommendedTracks: allRecommendedTrackKeys.size,
    catalogCoverageRatio: allRecommendedTrackKeys.size / totalCatalogTracks,
    uniqueRecommendedArtists: allRecommendedArtists.size,
    artistCoverageRatio: allRecommendedArtists.size / totalCatalogArtists,
  },
  protocols: protocolResults,
};

await writeFile(outputPath, JSON.stringify(globalReport, null, 2) + '\n');
console.log(`[benchmark] Benchmark completed. Saved baseline report to: ${outputPath}`);
console.log(JSON.stringify(globalReport, null, 2));
