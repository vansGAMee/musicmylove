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
  const candidates = shortlist.map(({ track, vector, lift }): TasteCandidate => {
    const supported = strongestSeedMatches(encodedSeeds, vector);
    const supportBonus = 0.12 * Math.log1p(supported.length) / Math.log1p(Math.max(1, activeSeeds.length));
    const retrievalScore = lift + supportBonus;
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

/** Preserve both external and neural retrieval coverage before the shared ranker. */
export function expandTasteCandidatePool(pool: TasteCandidatePool, artifact: TasteLiftCatalogArtifact, model: TasteLiftModel, limit = 500): TasteCandidatePool {
  const target = Math.max(0, Math.floor(limit));
  const neural = retrieveTasteCatalogCandidates(pool.seeds, artifact, model, target);
  return mergeTasteCandidateSources(pool, neural, target);
}

export function mergeTasteCandidateSources(pool: TasteCandidatePool, neural: readonly TasteCandidate[], limit = 500): TasteCandidatePool {
  const target = Math.max(0, Math.floor(limit));
  const external = [...pool.candidates];
  const selected: TasteCandidate[] = [];
  const identities = new Set<string>();
  const append = (candidate: TasteCandidate) => {
    const identity = recordingIdentity(candidate);
    if (selected.length >= target || identities.has(identity)) return;
    identities.add(identity);
    selected.push(candidate);
  };
  const sourceQuota = Math.floor(target / 2);
  external.slice(0, sourceQuota).forEach(append);
  neural.slice(0, sourceQuota).forEach(append);
  let externalIndex = Math.min(sourceQuota, external.length);
  let neuralIndex = Math.min(sourceQuota, neural.length);
  while (selected.length < target && (externalIndex < external.length || neuralIndex < neural.length)) {
    if (externalIndex < external.length) append(external[externalIndex++]!);
    if (neuralIndex < neural.length) append(neural[neuralIndex++]!);
  }
  return { seeds: [...pool.seeds], candidates: selected };
}
