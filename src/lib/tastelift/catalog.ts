import type { TasteCandidate, TasteCandidateEvidence, TasteCandidatePool } from "./retrieval";
import type { ResolvedTasteSeed } from "./resolver";
import { recordingIdentity } from "./identity";
import { TasteLiftModel, type TasteLiftTrack } from "./model";

export interface TasteLiftCatalogTrack {
  mbid: string;
  artist: string;
  title: string;
  release?: string;
  popularityPercentile: number;
}

export interface TasteLiftCatalogArtifact {
  format: "tastelift-catalog-v1";
  dimensions: number;
  source: "listenbrainz-train-histories";
  quantization: { type: "symmetric-int8"; scale: number; encoding: "base64-row-major"; maximumError?: number };
  tracks: readonly TasteLiftCatalogTrack[];
  histories?: readonly (readonly number[])[];
  vectors: string;
  modelSha256?: string;
  manifestSha256?: string;
}

function dot(left: readonly number[], right: readonly number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) sum += left[index]! * (right[index] ?? 0);
  return sum;
}

function strongestSeedMatches(encodedSeeds: readonly (readonly number[])[], vector: readonly number[]): { index: number; score: number }[] {
  const strongest: { index: number; score: number }[] = [];
  for (let index = 0; index < encodedSeeds.length; index += 1) {
    const match = { index, score: dot(encodedSeeds[index]!, vector) };
    let position = 0;
    while (position < strongest.length && (strongest[position]!.score > match.score || (strongest[position]!.score === match.score && strongest[position]!.index < match.index))) position += 1;
    if (position < 8) strongest.splice(position, 0, match);
    if (strongest.length > 8) strongest.pop();
  }
  const best = strongest[0]?.score ?? -1;
  return strongest.filter((item, index) => index === 0 || (item.score >= 0 && item.score >= best - 0.08));
}

function tasteSeeds(seeds: readonly ResolvedTasteSeed[]): TasteLiftTrack[] {
  return seeds.flatMap((seed): TasteLiftTrack[] => {
    if (seed.status === "unresolved") return [];
    return seed.source === "text" ? [{ artist: seed.input.artist, title: seed.input.title }] : [seed.track];
  });
}

function seedMbid(seed: ResolvedTasteSeed, index: number): string {
  return seed.status === "resolved" && seed.source !== "text" ? seed.track.mbid : `tastelift-text-${index}`;
}

const decodedVectorCache = new WeakMap<object, Int8Array>();

function decodeVectors(artifact: TasteLiftCatalogArtifact): Int8Array {
  if (artifact.format !== "tastelift-catalog-v1" || artifact.dimensions !== 24 || artifact.quantization.type !== "symmetric-int8" || artifact.quantization.scale <= 0) {
    throw new Error("Unsupported TasteLift catalog");
  }
  const cached = decodedVectorCache.get(artifact);
  if (cached) return cached;
  const binary = atob(artifact.vectors);
  if (binary.length !== artifact.tracks.length * artifact.dimensions) throw new Error("TasteLift catalog vector dimensions are invalid");
  const bytes = new Int8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index) > 127 ? binary.charCodeAt(index) - 256 : binary.charCodeAt(index);
  decodedVectorCache.set(artifact, bytes);
  return bytes;
}

interface CatalogShortlistRow {
  track: TasteLiftCatalogTrack;
  vector: number[];
  lift: number;
}

/** Approximate full-catalog retrieval; final ranking re-encodes selected metadata exactly. */
export function retrieveTasteCatalogCandidates(seeds: readonly ResolvedTasteSeed[], artifact: TasteLiftCatalogArtifact, model: TasteLiftModel, limit = 500): TasteCandidate[] {
  const activeSeeds = tasteSeeds(seeds);
  if (activeSeeds.length !== seeds.length) throw new Error("TasteLift catalog retrieval requires resolved seeds");
  const heads = model.encodeSet(activeSeeds);
  const encodedSeeds = activeSeeds.map((seed) => model.encodeTrack(seed));
  const bytes = decodeVectors(artifact);
  const scale = artifact.quantization.scale;
  const seedTracks = activeSeeds.map((seed, index) => ({ mbid: seed.mbid ?? seedMbid(seeds[index]!, index), artist: seed.artist, title: seed.title }));
  const seedMbids = new Set(seedTracks.map((seed) => seed.mbid));
  const seedIdentities = new Set(seedTracks.map(recordingIdentity));
  const byIdentity = new Map<string, CatalogShortlistRow>();

  artifact.tracks.forEach((track, catalogIndex) => {
    const identity = recordingIdentity(track);
    if (seedMbids.has(track.mbid) || seedIdentities.has(identity)) return;
    const vector = Array.from({ length: artifact.dimensions }, (_, dimension) => bytes[catalogIndex * artifact.dimensions + dimension]! / scale);
    const scored = model.scoreEncodedCandidate(heads, vector, track.popularityPercentile);
    const row = { track, vector, lift: scored.lift };
    const previous = byIdentity.get(identity);
    if (!previous || row.lift > previous.lift || (row.lift === previous.lift && row.track.mbid.localeCompare(previous.track.mbid) < 0)) byIdentity.set(identity, row);
  });

  // Full-set heads cheaply screen the catalog. Every plausible result is then
  // scored against every seed; this keeps 500-seed requests bounded without
  // truncating or sampling the user's input.
  const shortlistSize = Math.max(Math.max(0, Math.floor(limit)) * 4, 2_000);
  const shortlist = [...byIdentity.values()]
    .sort((left, right) => right.lift - left.lift || left.track.mbid.localeCompare(right.track.mbid))
    .slice(0, shortlistSize);
  const candidates = shortlist.map(({ track, vector }): TasteCandidate => {
    const supported = strongestSeedMatches(encodedSeeds, vector);
    const supportBonus = 0.12 * Math.log1p(supported.length) / Math.log1p(Math.max(1, activeSeeds.length));
    const retrievalScore = model.scoreTargetCandidate(encodedSeeds, vector).relevance + supportBonus;
    const evidence: TasteCandidateEvidence[] = supported.map((item) => ({
      source: "tastelift-catalog",
      seedIndex: item.index,
      seedMbid: seedMbid(seeds[item.index]!, item.index),
      recordingMbid: track.mbid,
      rank: 1,
      rawScore: (item.score + 1) / 2,
    }));
    return { ...track, alternateMbids: [track.mbid], support: supported.length, retrievalScore, evidence };
  });

  return candidates.sort((left, right) => right.retrievalScore - left.retrievalScore || right.support - left.support || left.mbid.localeCompare(right.mbid)).slice(0, Math.max(0, Math.floor(limit)));
}

/** Sparse train-user co-listen retrieval stored without usernames or listen counts. */
export function retrieveTasteHistoryCandidates(seeds: readonly ResolvedTasteSeed[], artifact: TasteLiftCatalogArtifact, limit = 500): TasteCandidate[] {
  if (!artifact.histories?.length) return [];
  const seedByTrackIndex = new Map<number, number>();
  const catalogIndex = new Map(artifact.tracks.map((track, index) => [track.mbid, index]));
  seeds.forEach((seed, seedIndex) => {
    if (seed.status === "resolved" && seed.source !== "text") {
      const index = catalogIndex.get(seed.track.mbid);
      if (index !== undefined) seedByTrackIndex.set(index, seedIndex);
    }
  });
  if (seedByTrackIndex.size === 0) return [];
  const ownerCounts = new Map<number, number>();
  for (const history of artifact.histories) for (const trackIndex of history) if (seedByTrackIndex.has(trackIndex)) ownerCounts.set(trackIndex, (ownerCounts.get(trackIndex) ?? 0) + 1);
  const scores = new Map<number, number>();
  const supporters = new Map<number, Set<number>>();
  for (const history of artifact.histories) {
    const matched = history.filter((trackIndex) => seedByTrackIndex.has(trackIndex));
    if (!matched.length) continue;
    for (const trackIndex of history) {
      if (seedByTrackIndex.has(trackIndex)) continue;
      for (const seedTrackIndex of matched) {
        const weight = 1 / Math.sqrt(ownerCounts.get(seedTrackIndex) ?? 1);
        scores.set(trackIndex, (scores.get(trackIndex) ?? 0) + weight);
        const set = supporters.get(trackIndex) ?? new Set<number>();
        set.add(seedByTrackIndex.get(seedTrackIndex)!);
        supporters.set(trackIndex, set);
      }
    }
  }
  const byIdentity = new Map<string, TasteCandidate>();
  for (const [trackIndex, retrievalScore] of scores) {
    const track = artifact.tracks[trackIndex];
    if (!track) continue;
    const seedIndexes = [...(supporters.get(trackIndex) ?? [])].sort((left, right) => left - right);
    const evidence: TasteCandidateEvidence[] = seedIndexes.map((seedIndex) => ({
      source: "listenbrainz-history",
      seedIndex,
      seedMbid: seedMbid(seeds[seedIndex]!, seedIndex),
      recordingMbid: track.mbid,
      rank: 1,
      rawScore: retrievalScore,
    }));
    const candidate: TasteCandidate = { ...track, alternateMbids: [track.mbid], support: seedIndexes.length, retrievalScore, evidence };
    const identity = recordingIdentity(track);
    const previous = byIdentity.get(identity);
    if (!previous || candidate.retrievalScore > previous.retrievalScore || (candidate.retrievalScore === previous.retrievalScore && candidate.mbid.localeCompare(previous.mbid) < 0)) byIdentity.set(identity, candidate);
  }
  return [...byIdentity.values()].sort((left, right) => right.retrievalScore - left.retrievalScore || right.support - left.support || left.mbid.localeCompare(right.mbid)).slice(0, Math.max(0, Math.floor(limit)));
}

/** Preserve both external and neural retrieval coverage before the shared ranker. */
export function expandTasteCandidatePool(pool: TasteCandidatePool, artifact: TasteLiftCatalogArtifact, model: TasteLiftModel, limit = 500): TasteCandidatePool {
  const target = Math.max(0, Math.floor(limit));
  const neural = retrieveTasteCatalogCandidates(pool.seeds, artifact, model, target);
  const history = retrieveTasteHistoryCandidates(pool.seeds, artifact, target);
  return mergeTasteCandidateSources(pool, neural, target, history);
}

export function mergeTasteCandidateSources(pool: TasteCandidatePool, neural: readonly TasteCandidate[], limit = 500, history: readonly TasteCandidate[] = []): TasteCandidatePool {
  const target = Math.max(0, Math.floor(limit));
  const combined = new Map<string, { candidate: TasteCandidate; fusion: number; bestRaw: number }>();
  const append = (candidate: TasteCandidate, fusion: number) => {
    const identity = recordingIdentity(candidate);
    const existing = combined.get(identity);
    if (existing) {
      const previous = existing.candidate;
      const evidence = [...previous.evidence, ...candidate.evidence].filter((item, index, rows) => rows.findIndex((other) => other.source === item.source && other.seedIndex === item.seedIndex && other.recordingMbid === item.recordingMbid) === index);
      const bestRaw = Math.max(existing.bestRaw, candidate.retrievalScore);
      combined.set(identity, { fusion: existing.fusion + fusion, bestRaw, candidate: {
        ...(candidate.retrievalScore > previous.retrievalScore ? candidate : previous),
        alternateMbids: [...new Set([...previous.alternateMbids, ...candidate.alternateMbids])].sort(), evidence,
        support: new Set(evidence.map((item) => item.seedIndex)).size, retrievalScore: existing.fusion + fusion,
        popularityPercentile: previous.popularityPercentile ?? candidate.popularityPercentile,
      } });
      return;
    }
    combined.set(identity, { candidate: { ...candidate, retrievalScore: fusion }, fusion, bestRaw: candidate.retrievalScore });
  };
  const sources = [
    { rows: [...pool.candidates], weight: 1 },
    { rows: [...history], weight: 1.2 },
    { rows: [...neural], weight: 1 },
  ];
  for (const { rows, weight } of sources) rows.forEach((candidate, index) => append(candidate, weight / (60 + index + 1)));
  const candidates = [...combined.values()]
    .sort((left, right) => right.fusion - left.fusion || right.bestRaw - left.bestRaw || left.candidate.mbid.localeCompare(right.candidate.mbid))
    .slice(0, target)
    .map((row) => ({ ...row.candidate, retrievalScore: row.fusion }));
  return { seeds: [...pool.seeds], candidates };
}
