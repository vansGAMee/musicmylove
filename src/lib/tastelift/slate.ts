import type { RankedTrack, SlateComponents } from "../types";
import { normalizeIdentityText, recordingIdentity } from "./identity";

const MAINSTREAM_PERCENTILE = 0.85;
const LONG_TAIL_SHARE = 0.35;
const MAX_PER_ARTIST = 2;

/**
 * Marginal weights for a discovery slate. Relevance preserves the residual
 * retrieval evidence; lift and novelty are personalized TasteLift signals.
 * The remaining terms prevent a single artist or taste head monopolising a slate.
 */
const WEIGHTS = {
  relevance: 0.36,
  lift: 0.24,
  serendipity: 0.20,
  artistDiversity: 0.12,
  headCoverage: 0.08,
} as const;

interface SlateCandidate {
  track: RankedTrack;
  artist: string;
  identity: string;
  head?: number;
  popularity: number;
  relevance: number;
  lift: number;
  serendipity: number;
}

const finite = (value: number | undefined, fallback = 0): number => Number.isFinite(value) ? value! : Number.isFinite(fallback) ? fallback : 0;
const clamp = (value: number): number => Math.max(0, Math.min(1, value));

function stableValue(value: unknown): string {
  if (value === undefined) return "undefined";
  if (value === null) return "null";
  if (typeof value === "number") return Number.isFinite(value) ? `number:${value}` : `number:${String(value)}`;
  if (typeof value === "string") return `string:${JSON.stringify(value)}`;
  if (typeof value === "boolean") return `boolean:${value}`;
  if (Array.isArray(value)) return `array:[${value.map(stableValue).join(",")}]`;
  if (typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `object:{${Object.keys(object).sort((left, right) => left.localeCompare(right)).map((key) => `${JSON.stringify(key)}:${stableValue(object[key])}`).join(",")}}`;
  }
  return `${typeof value}:${String(value)}`;
}

/** Total ordering for duplicate rows, excluding slate fields that this selector recomputes. */
function duplicateKey(track: RankedTrack): string {
  const { slateScore: _slateScore, slateComponents: _slateComponents, ...source } = track;
  return stableValue(source);
}

function compareRanked(left: RankedTrack, right: RankedTrack): number {
  return finite(right.residualScore, right.score) - finite(left.residualScore, left.score)
    || finite(right.liftScore) - finite(left.liftScore)
    || finite(right.score) - finite(left.score)
    || left.mbid.localeCompare(right.mbid)
    || normalizeIdentityText(left.artist).localeCompare(normalizeIdentityText(right.artist))
    || normalizeIdentityText(left.title).localeCompare(normalizeIdentityText(right.title))
    || duplicateKey(left).localeCompare(duplicateKey(right));
}

function normalise(values: readonly number[]): number[] {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum === minimum) return values.map(() => 0);
  return values.map((value) => clamp((value - minimum) / (maximum - minimum)));
}

function canonicalCandidates(ranked: readonly RankedTrack[]): SlateCandidate[] {
  const uniqueMbids = new Set<string>();
  const uniqueRecordings = new Set<string>();
  const unique = [...ranked].sort(compareRanked).filter((track) => {
    const identity = recordingIdentity(track);
    if (uniqueMbids.has(track.mbid) || uniqueRecordings.has(identity)) return false;
    uniqueMbids.add(track.mbid);
    uniqueRecordings.add(identity);
    return true;
  });
  const relevance = normalise(unique.map((track) => finite(track.residualScore, track.score)));
  const lift = normalise(unique.map((track) => finite(track.liftScore)));
  return unique.map((track, index) => {
    const popularity = clamp(finite(track.popularityPercentile, 0.5));
    return {
      track,
      artist: normalizeIdentityText(track.artist),
      identity: recordingIdentity(track),
      head: track.tasteHeadIndex !== undefined && track.tasteHeadIndex >= 0 && track.tasteHeadIndex < 4 ? track.tasteHeadIndex : undefined,
      popularity,
      relevance: relevance[index]!,
      lift: lift[index]!,
      serendipity: 1 - popularity,
    };
  });
}

function isMainstream(candidate: SlateCandidate): boolean {
  return candidate.popularity >= MAINSTREAM_PERCENTILE;
}

function isExceptional(candidate: SlateCandidate): boolean {
  return candidate.relevance >= 0.9 || candidate.lift >= 0.9;
}

function details(candidate: SlateCandidate, artistCount: number, seenHeads: ReadonlySet<number>): SlateComponents {
  const artistDiversity = 1 / (artistCount + 1);
  const headCoverage = candidate.head !== undefined && !seenHeads.has(candidate.head) ? 1 : 0;
  const marginalScore = WEIGHTS.relevance * candidate.relevance
    + WEIGHTS.lift * candidate.lift
    + WEIGHTS.serendipity * candidate.serendipity
    + WEIGHTS.artistDiversity * artistDiversity
    + WEIGHTS.headCoverage * headCoverage;
  return { relevance: candidate.relevance, lift: candidate.lift, serendipity: candidate.serendipity, artistDiversity, headCoverage, marginalScore };
}

function compareMarginal(left: SlateCandidate, right: SlateCandidate, artistCounts: ReadonlyMap<string, number>, seenHeads: ReadonlySet<number>): number {
  const leftDetails = details(left, artistCounts.get(left.artist) ?? 0, seenHeads);
  const rightDetails = details(right, artistCounts.get(right.artist) ?? 0, seenHeads);
  return rightDetails.marginalScore - leftDetails.marginalScore
    || right.relevance - left.relevance
    || right.lift - left.lift
    || left.track.mbid.localeCompare(right.track.mbid);
}

function compareReservation(left: SlateCandidate, right: SlateCandidate): number {
  const leftScore = details(left, 0, new Set()).marginalScore;
  const rightScore = details(right, 0, new Set()).marginalScore;
  return rightScore - leftScore || compareRanked(left.track, right.track);
}

/**
 * Find a deterministic capacity-two artist assignment for all available heads.
 * Every candidate belongs to exactly one strongest head, so the small (<=4)
 * DFS is a bipartite head-to-artist b-matching rather than a track permutation.
 */
function reserveHeadCandidates(candidates: readonly SlateCandidate[], target: number): SlateCandidate[] {
  const byHead = new Map<number, Map<string, SlateCandidate>>();
  for (const candidate of candidates) {
    if (candidate.head === undefined) continue;
    const byArtist = byHead.get(candidate.head) ?? new Map<string, SlateCandidate>();
    const previous = byArtist.get(candidate.artist);
    if (!previous || compareReservation(candidate, previous) < 0) byArtist.set(candidate.artist, candidate);
    byHead.set(candidate.head, byArtist);
  }
  const heads = [...byHead.keys()].sort((left, right) => {
    const count = (byHead.get(left)?.size ?? 0) - (byHead.get(right)?.size ?? 0);
    return count || left - right;
  });
  if (heads.length === 0 || heads.length > target) return [];
  const search = (index: number, artistCounts: ReadonlyMap<string, number>): SlateCandidate[] | undefined => {
    if (index === heads.length) return [];
    const head = heads[index]!;
    const choices = [...byHead.get(head)!.values()].sort(compareReservation);
    for (const choice of choices) {
      if ((artistCounts.get(choice.artist) ?? 0) >= MAX_PER_ARTIST) continue;
      const nextCounts = new Map(artistCounts);
      nextCounts.set(choice.artist, (nextCounts.get(choice.artist) ?? 0) + 1);
      const rest = search(index + 1, nextCounts);
      if (rest) return [choice, ...rest];
    }
    return undefined;
  };
  return search(0, new Map()) ?? [];
}

function remainingLongTailCapacity(candidates: readonly SlateCandidate[], selectedMbids: ReadonlySet<string>, selectedIdentities: ReadonlySet<string>, artistCounts: ReadonlyMap<string, number>): number {
  const supplyByArtist = new Map<string, number>();
  for (const candidate of candidates) {
    if (isMainstream(candidate) || selectedMbids.has(candidate.track.mbid) || selectedIdentities.has(candidate.identity)) continue;
    if ((artistCounts.get(candidate.artist) ?? 0) >= MAX_PER_ARTIST) continue;
    supplyByArtist.set(candidate.artist, (supplyByArtist.get(candidate.artist) ?? 0) + 1);
  }
  return [...supplyByArtist].reduce((total, [artist, supply]) => total + Math.min(supply, MAX_PER_ARTIST - (artistCounts.get(artist) ?? 0)), 0);
}

/**
 * Builds a deterministic 40-track discovery slate from a fully ranked TasteLift pool.
 * It never invents a candidate: shortages are returned as shorter slates for the caller to diagnose.
 */
export function buildTasteSlate(ranked: readonly RankedTrack[], limit = 40): RankedTrack[] {
  const target = Math.max(0, Math.floor(limit));
  const candidates = canonicalCandidates(ranked).filter((candidate) => !isMainstream(candidate) || isExceptional(candidate));
  const initialLongTailCapacity = remainingLongTailCapacity(candidates, new Set(), new Set(), new Map());
  const longTailTarget = Math.min(Math.ceil(target * LONG_TAIL_SHARE), initialLongTailCapacity);
  const selected: RankedTrack[] = [];
  const selectedMbids = new Set<string>();
  const selectedIdentities = new Set<string>();
  const artistCounts = new Map<string, number>();
  const seenHeads = new Set<number>();
  let longTailCount = 0;

  const select = (chosen: SlateCandidate): void => {
    const components = details(chosen, artistCounts.get(chosen.artist) ?? 0, seenHeads);
    selected.push({ ...chosen.track, slateScore: components.marginalScore, slateComponents: components });
    selectedMbids.add(chosen.track.mbid);
    selectedIdentities.add(chosen.identity);
    artistCounts.set(chosen.artist, (artistCounts.get(chosen.artist) ?? 0) + 1);
    if (chosen.head !== undefined) seenHeads.add(chosen.head);
    if (!isMainstream(chosen)) longTailCount += 1;
  };

  // Reserve a capacity-feasible head representative before normal greedy choice.
  for (const candidate of reserveHeadCandidates(candidates, target)) select(candidate);
  const feasibleLongTailTarget = Math.min(longTailTarget, longTailCount + remainingLongTailCapacity(candidates, selectedMbids, selectedIdentities, artistCounts));

  while (selected.length < target) {
    let eligible = candidates.filter((candidate) => !selectedMbids.has(candidate.track.mbid)
      && !selectedIdentities.has(candidate.identity)
      && (artistCounts.get(candidate.artist) ?? 0) < MAX_PER_ARTIST);
    // Keep the feasible long-tail reserve when possible, but never turn it into
    // a false inventory shortfall if other valid artist-capped candidates remain.
    const reserveSafe = eligible.filter((candidate) => {
      const nextMbids = new Set(selectedMbids).add(candidate.track.mbid);
      const nextIdentities = new Set(selectedIdentities).add(candidate.identity);
      const nextArtists = new Map(artistCounts);
      nextArtists.set(candidate.artist, (nextArtists.get(candidate.artist) ?? 0) + 1);
      return longTailCount + (isMainstream(candidate) ? 0 : 1) + remainingLongTailCapacity(candidates, nextMbids, nextIdentities, nextArtists) >= feasibleLongTailTarget;
    });
    if (reserveSafe.length > 0) eligible = reserveSafe;
    if (eligible.length === 0) break;
    eligible.sort((left, right) => compareMarginal(left, right, artistCounts, seenHeads));
    select(eligible[0]!);
  }
  return selected;
}
