import { VersionedCache } from "../cache";
import { fetchSimilarBatch } from "../listenbrainz";
import type { SimilarTrack, Track } from "../types";
import type { ResolvedTasteSeed } from "./resolver";
import { hasSameRecordingIdentity, recordingIdentity } from "./identity";

export type TasteCandidateSource = "listenbrainz" | "listenbrainz-history" | "tastelift-catalog";

export interface TasteCandidateEvidence {
  source: TasteCandidateSource;
  seedIndex: number;
  seedMbid: string;
  recordingMbid: string;
  rank: number;
  rawScore: number;
}

export interface TasteCandidate extends Track {
  /** All MusicBrainz recording IDs collapsed into this artist/title/version identity. */
  alternateMbids: readonly string[];
  /** Number of distinct input seeds that supplied evidence for this candidate. */
  support: number;
  /** Stable pre-ranker retrieval score used solely to cap a fully merged pool. */
  retrievalScore: number;
  evidence: readonly TasteCandidateEvidence[];
  popularityPercentile?: number;
}

/** Retains every seed outcome, including text/OOV and resolver failures, alongside candidates. */
export interface TasteCandidatePool {
  seeds: readonly ResolvedTasteSeed[];
  candidates: readonly TasteCandidate[];
}

export interface TasteRetrievalAdapters {
  fetchSimilarBatch?: (mbids: readonly string[]) => Promise<Readonly<Record<string, readonly SimilarTrack[]>>>;
  fetchSimilar?: (mbid: string) => Promise<readonly SimilarTrack[]>;
  cache?: VersionedCache<readonly SimilarTrack[]>;
}

export interface TasteRetrievalOptions {
  cache?: VersionedCache<readonly SimilarTrack[]>;
  cacheTtlMs?: number;
}

interface MbidSeed {
  index: number;
  mbid: string;
  track: Track;
}

interface CandidateVariant {
  track: SimilarTrack;
  evidence: TasteCandidateEvidence[];
}

const retrievalStorage = new Map<string, string>();
const retrievalCache = new VersionedCache<readonly SimilarTrack[]>("tastelift-retrieval-v1", {
  getItem: (key) => retrievalStorage.get(key) ?? null,
  setItem: (key, value) => retrievalStorage.set(key, value),
});
const DEFAULT_CACHE_TTL_MS = 60 * 60 * 1000;
const RRF_K = 60;

export function createTasteRetrievalAdapters(): TasteRetrievalAdapters {
  return { fetchSimilarBatch, cache: retrievalCache };
}

function mbidSeeds(seeds: readonly ResolvedTasteSeed[]): MbidSeed[] {
  return seeds.flatMap((seed, index): MbidSeed[] => seed.status === "resolved" && seed.source !== "text"
    ? [{ index, mbid: seed.track.mbid, track: seed.track }]
    : []);
}

function compareEvidence(left: TasteCandidateEvidence, right: TasteCandidateEvidence): number {
  return left.seedMbid.localeCompare(right.seedMbid)
    || left.seedIndex - right.seedIndex
    || left.rank - right.rank
    || left.recordingMbid.localeCompare(right.recordingMbid)
    || right.rawScore - left.rawScore;
}

function variantMetrics(variant: CandidateVariant): { support: number; retrievalScore: number; bestScore: number; bestRank: number } {
  const seedIndexes = new Set(variant.evidence.map((item) => item.seedIndex));
  return {
    support: seedIndexes.size,
    retrievalScore: variant.evidence.reduce((sum, item) => sum + 1 / (RRF_K + item.rank), 0),
    bestScore: Math.max(...variant.evidence.map((item) => item.rawScore)),
    bestRank: Math.min(...variant.evidence.map((item) => item.rank)),
  };
}

function compareVariants(left: CandidateVariant, right: CandidateVariant): number {
  const leftMetrics = variantMetrics(left);
  const rightMetrics = variantMetrics(right);
  return rightMetrics.support - leftMetrics.support
    || rightMetrics.retrievalScore - leftMetrics.retrievalScore
    || rightMetrics.bestScore - leftMetrics.bestScore
    || leftMetrics.bestRank - rightMetrics.bestRank
    || left.track.mbid.localeCompare(right.track.mbid);
}

function compareCandidates(left: TasteCandidate, right: TasteCandidate): number {
  return right.support - left.support
    || right.retrievalScore - left.retrievalScore
    || right.evidence.reduce((best, item) => Math.max(best, item.rawScore), -Infinity) - left.evidence.reduce((best, item) => Math.max(best, item.rawScore), -Infinity)
    || left.mbid.localeCompare(right.mbid);
}

async function retrieveLists(mbids: readonly string[], adapters: TasteRetrievalAdapters, options: TasteRetrievalOptions): Promise<Readonly<Record<string, readonly SimilarTrack[]>>> {
  const cache = options.cache ?? adapters.cache ?? retrievalCache;
  const ttlMs = options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
  const lists: Record<string, readonly SimilarTrack[]> = {};
  const missing: string[] = [];
  for (const mbid of mbids) {
    const cached = cache.get(mbid);
    if (cached && !cached.stale) lists[mbid] = cached.value;
    else missing.push(mbid);
  }
  if (missing.length === 0) return lists;

  let fetched: Readonly<Record<string, readonly SimilarTrack[]>>;
  if (adapters.fetchSimilarBatch) {
    fetched = await adapters.fetchSimilarBatch(missing);
  } else if (adapters.fetchSimilar) {
    fetched = Object.fromEntries(await Promise.all(missing.map(async (mbid) => [mbid, await adapters.fetchSimilar!(mbid)] as const)));
  } else {
    throw new Error("Taste retrieval requires a similarity adapter");
  }
  for (const mbid of missing) {
    const rows = fetched[mbid] ?? [];
    lists[mbid] = rows;
    cache.set(mbid, rows, ttlMs);
  }
  return lists;
}

/**
 * Fetches every MBID seed as one cached batch, preserves all evidence, then caps the merged pool.
 * Text/OOV and unresolved seed outcomes remain in the returned pool but never trigger fake lookups.
 */
export async function retrieveTasteCandidates(seeds: readonly ResolvedTasteSeed[], adapters: TasteRetrievalAdapters, limit = 500, options: TasteRetrievalOptions = {}): Promise<TasteCandidatePool> {
  const resolved = mbidSeeds(seeds);
  const uniqueMbids = [...new Set(resolved.map((seed) => seed.mbid))].sort((left, right) => left.localeCompare(right));
  const lists = await retrieveLists(uniqueMbids, adapters, options);
  const seedMbids = new Set(resolved.map((seed) => seed.mbid));
  const seedIdentities = resolved.map((seed) => seed.track);
  const variantsByIdentity = new Map<string, Map<string, CandidateVariant>>();

  for (const seed of [...resolved].sort((left, right) => left.mbid.localeCompare(right.mbid) || left.index - right.index)) {
    for (const [index, track] of (lists[seed.mbid] ?? []).entries()) {
      if (seedMbids.has(track.mbid) || seedIdentities.some((seedTrack) => hasSameRecordingIdentity(seedTrack, track))) continue;
      const identity = recordingIdentity(track);
      const variants = variantsByIdentity.get(identity) ?? new Map<string, CandidateVariant>();
      const variant = variants.get(track.mbid) ?? { track, evidence: [] };
      variant.evidence.push({ source: "listenbrainz", seedIndex: seed.index, seedMbid: seed.mbid, recordingMbid: track.mbid, rank: index + 1, rawScore: track.score });
      variants.set(track.mbid, variant);
      variantsByIdentity.set(identity, variants);
    }
  }

  const candidates = [...variantsByIdentity.values()].map((variants): TasteCandidate => {
    const orderedVariants = [...variants.values()].sort(compareVariants);
    const representative = orderedVariants[0]!;
    const evidence = orderedVariants.flatMap((variant) => variant.evidence).sort(compareEvidence);
    const metrics = variantMetrics({ track: representative.track, evidence });
    return {
      mbid: representative.track.mbid,
      artist: representative.track.artist,
      title: representative.track.title,
      ...(representative.track.release ? { release: representative.track.release } : {}),
      alternateMbids: orderedVariants.map((variant) => variant.track.mbid).sort((left, right) => left.localeCompare(right)),
      support: metrics.support,
      retrievalScore: metrics.retrievalScore,
      evidence,
    };
  }).sort(compareCandidates).slice(0, Math.max(0, Math.floor(limit)));

  return { seeds: [...seeds], candidates };
}
