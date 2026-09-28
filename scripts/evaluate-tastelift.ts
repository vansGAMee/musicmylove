import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import catalogJson from "../ml/tastelift-catalog.json";
import modelJson from "../ml/tastelift-model.json";
import rankerJson from "../ml/model.json";
import { VersionedCache } from "../src/lib/cache";
import { createRecommendations } from "../src/lib/recommend";
import { rankTasteCandidatePool } from "../src/lib/ranking";
import { buildTasteSlate } from "../src/lib/tastelift/slate";
import { mergeTasteCandidateSources, retrieveTasteCatalogCandidates, retrieveTasteHistoryCandidates, type TasteLiftCatalogArtifact } from "../src/lib/tastelift/catalog";
import { TasteLiftModel, type TasteLiftArtifact } from "../src/lib/tastelift/model";
import { retrieveTasteCandidates } from "../src/lib/tastelift/retrieval";
import type { ResolvedTasteSeed } from "../src/lib/tastelift/resolver";
import type { ModelArtifact } from "../src/lib/mlp";
import type { RankedTrack, SimilarTrack } from "../src/lib/types";

type Partition = "validation" | "test";
type DatasetTrack = { id: string; artist: string; title: string; popularity: { percentile: number } };
type CacheRow = { reference_mbid: string; recording_mbid: string; recording_name: string; artist_credit_name: string; release_name?: string; score: number };
type Example = { seeds: string[]; hidden: string[] };

const root = resolve(import.meta.dirname, "..");
const arg = (name: string, fallback: string) => process.argv[process.argv.indexOf(name) + 1] ?? fallback;
const partition = arg("--partition", "validation") as Partition;
if (partition !== "validation" && partition !== "test") throw new Error("--partition must be validation or test");

const mean = (values: readonly number[]) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const normalized = (value: string) => value.normalize("NFKC").trim().toLowerCase().replace(/\s+/gu, " ");
function rankingMetrics(hidden: readonly string[], ranked: readonly RankedTrack[]) {
  const wanted = new Set(hidden);
  const top = ranked.slice(0, 20);
  const hits = top.map((track) => Number(wanted.has(track.mbid)));
  const recall = hits.reduce((sum, value) => sum + value, 0) / hidden.length;
  const dcg = hits.reduce((sum, value, index) => sum + value / Math.log2(index + 2), 0);
  const ideal = Array.from({ length: Math.min(hidden.length, 20) }, (_, index) => 1 / Math.log2(index + 2)).reduce((a, b) => a + b, 0);
  return {
    recallAt20: recall,
    ndcgAt20: dcg / ideal,
    popularityAt20: mean(top.map((track) => track.popularityPercentile ?? 0)),
    artistDiversityAt20: new Set(top.map((track) => normalized(track.artist))).size / Math.max(1, top.length),
    headCoverageAt20: new Set(top.flatMap((track) => track.tasteHeadIndex === undefined ? [] : [track.tasteHeadIndex])).size,
  };
}

const dataset = JSON.parse(await readFile(resolve(root, "data/cache/tastelift/dataset.json"), "utf8")) as { tracks: DatasetTrack[] };
const tracks = new Map(dataset.tracks.map((track) => [track.id, track]));
const manifest = JSON.parse(await readFile(resolve(root, "data/manifests/real-splits-final.json"), "utf8")) as Record<Partition, string[]>;
const model = TasteLiftModel.fromArtifact(modelJson as TasteLiftArtifact);
const catalog = catalogJson as TasteLiftCatalogArtifact;
const popularity = new Map(catalog.tracks.map((track) => [track.mbid, track.popularityPercentile]));
const catalogIds = new Set(catalog.tracks.map((track) => track.mbid));
const rows: Array<Record<string, number | string>> = [];

for (const [userIndex, username] of manifest[partition].entries()) {
  const userKey = createHash("sha256").update(username).digest("hex");
  const cache = JSON.parse(await readFile(resolve(root, `data/cache/real-retrieval/${userKey}.json`), "utf8")) as { examples: Example[]; rows: CacheRow[] };
  for (const [exampleIndex, example] of cache.examples.entries()) {
    const seedTracks = example.seeds.map((id) => tracks.get(id)).filter((track): track is DatasetTrack => Boolean(track));
    if (seedTracks.length !== example.seeds.length || seedTracks.length < 5) continue;
    const seeds: ResolvedTasteSeed[] = seedTracks.map((track) => ({ input: { artist: track.artist, title: track.title }, status: "resolved", source: "listenbrainz", track: { mbid: track.id, artist: track.artist, title: track.title } }));
    const seedSet = new Set(example.seeds);
    const lists: Record<string, SimilarTrack[]> = {};
    for (const row of cache.rows) if (seedSet.has(row.reference_mbid)) (lists[row.reference_mbid] ??= []).push({ mbid: row.recording_mbid, artist: row.artist_credit_name, title: row.recording_name, ...(row.release_name ? { release: row.release_name } : {}), score: row.score });
    const storage = new Map<string, string>();
    const versioned = new VersionedCache<readonly SimilarTrack[]>(`eval-${userKey}-${exampleIndex}`, { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) });
    const external = await retrieveTasteCandidates(seeds, { fetchSimilarBatch: async () => lists, cache: versioned }, 500, { cache: versioned });
    const neuralPool = retrieveTasteCatalogCandidates(seeds, catalog, model, 500);
    const historyPool = retrieveTasteHistoryCandidates(seeds, catalog, 500);
    const expanded = mergeTasteCandidateSources(external, neuralPool, 500, historyPool);
    const oldRanked = createRecommendations(seedTracks.map((track) => ({ mbid: track.id, artist: track.artist, title: track.title })), lists, 20).map((track) => ({ ...track, popularityPercentile: popularity.get(track.mbid) ?? 0 }));
    const ranked = rankTasteCandidatePool(expanded, rankerJson as ModelArtifact, model);
    const v2 = buildTasteSlate(ranked, 40);
    const popular = [...ranked].sort((left, right) => (right.popularityPercentile ?? 0) - (left.popularityPercentile ?? 0) || left.mbid.localeCompare(right.mbid));
    const retrievalBefore = example.hidden.filter((id) => external.candidates.some((candidate) => candidate.mbid === id)).length / example.hidden.length;
    const catalogCoverage = example.hidden.filter((id) => catalogIds.has(id)).length / example.hidden.length;
    const neuralRecall500 = example.hidden.filter((id) => neuralPool.some((candidate) => candidate.mbid === id)).length / example.hidden.length;
    const historyRecall500 = example.hidden.filter((id) => historyPool.some((candidate) => candidate.mbid === id)).length / example.hidden.length;
    const retrievalAfter = example.hidden.filter((id) => expanded.candidates.some((candidate) => candidate.mbid === id)).length / example.hidden.length;
    const old = rankingMetrics(example.hidden, oldRanked);
    const neural = rankingMetrics(example.hidden, v2);
    const baseline = rankingMetrics(example.hidden, popular);
    rows.push({ user: userIndex, example: exampleIndex, retrievalBefore, catalogCoverage, neuralRecall500, historyRecall500, retrievalAfter, externalPoolSize: external.candidates.length, ...Object.fromEntries(Object.entries(old).map(([key, value]) => [`old_${key}`, value])), ...Object.fromEntries(Object.entries(neural).map(([key, value]) => [`v2_${key}`, value])), ...Object.fromEntries(Object.entries(baseline).map(([key, value]) => [`popularity_${key}`, value])) });
  }
  console.log(JSON.stringify({ event: "progress", partition, users: userIndex + 1, examples: rows.length }));
}

const numericKeys = Object.keys(rows[0] ?? {}).filter((key) => !["user", "example"].includes(key));
const metrics = Object.fromEntries(numericKeys.map((key) => [key, mean(rows.map((row) => Number(row[key])))]));
const report = { partition, users: manifest[partition].length, examples: rows.length, catalogTracks: catalog.tracks.length, candidateLimit: 500, metrics };
await mkdir(resolve(root, "reports"), { recursive: true });
await writeFile(resolve(root, `reports/tastelift-${partition}.json`), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ event: "complete", ...report }));
