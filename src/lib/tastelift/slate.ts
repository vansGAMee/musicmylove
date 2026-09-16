import type { RankedTrack, SlateComponents } from "../types";
import { normalizeIdentityText, recordingIdentity } from "./identity";

const MAINSTREAM_PERCENTILE = 0.85;
const LONG_TAIL_SHARE = 0.35;
const MAX_PER_ARTIST = 2;

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

function duplicateKey(track: RankedTrack): string {
  const { slateScore: _slateScore, slateComponents: _slateComponents, ...source } = track;
  return stableValue(source);
}

function compareRanked(left: RankedTrack, right: RankedTrack): number {
  return finite(right.liftScore) - finite(left.liftScore)
    || finite(right.score) - finite(left.score)
    || finite(right.residualScore, right.score) - finite(left.residualScore, left.score)
    || left.mbid.localeCompare(right.mbid)
    || normalizeIdentityText(left.artist).localeCompare(normalizeIdentityText(right.artist))
    || normalizeIdentityText(left.title).localeCompare(normalizeIdentityText(right.title))
    || duplicateKey(left).localeCompare(duplicateKey(right));
}

function isMainstream(track: RankedTrack): boolean {
  return (track.popularityPercentile ?? 0.5) >= MAINSTREAM_PERCENTILE;
}

function isLongTail(track: RankedTrack): boolean {
  return (track.popularityPercentile ?? 0.5) < LONG_TAIL_SHARE;
}

function isExceptional(track: RankedTrack, maxScore: number, minScore: number, maxLift: number, minLift: number): boolean {
  if (!Number.isFinite(track.score) && !Number.isFinite(track.residualScore) && !Number.isFinite(track.liftScore)) return false;
  const score = finite(track.score, finite(track.residualScore, 0));
  const lift = finite(track.liftScore, 0);
  const relNorm = maxScore > minScore ? (score - minScore) / (maxScore - minScore) : 0;
  const liftNorm = maxLift > minLift ? (lift - minLift) / (maxLift - minLift) : 0;
  return relNorm >= 0.5 || liftNorm >= 0.5 || (score > 0 && maxScore === minScore);
}

function reserveHeadCandidates(candidates: readonly RankedTrack[], target: number): RankedTrack[] {
  const byHead = new Map<number, Map<string, RankedTrack>>();
  for (const candidate of candidates) {
    if (candidate.tasteHeadIndex === undefined || candidate.tasteHeadIndex < 0 || candidate.tasteHeadIndex >= 4) continue;
    const artist = normalizeIdentityText(candidate.artist);
    const byArtist = byHead.get(candidate.tasteHeadIndex) ?? new Map<string, RankedTrack>();
    const previous = byArtist.get(artist);
    if (!previous || compareRanked(candidate, previous) < 0) byArtist.set(artist, candidate);
    byHead.set(candidate.tasteHeadIndex, byArtist);
  }
  const heads = [...byHead.keys()].sort((left, right) => {
    const count = (byHead.get(left)?.size ?? 0) - (byHead.get(right)?.size ?? 0);
    return count || left - right;
  });
  if (heads.length === 0 || heads.length > target) return [];
  const search = (index: number, artistCounts: ReadonlyMap<string, number>): RankedTrack[] | undefined => {
    if (index === heads.length) return [];
    const head = heads[index]!;
    const choices = [...byHead.get(head)!.values()].sort(compareRanked);
    for (const choice of choices) {
      const artist = normalizeIdentityText(choice.artist);
      if ((artistCounts.get(artist) ?? 0) >= MAX_PER_ARTIST) continue;
      const nextCounts = new Map(artistCounts);
      nextCounts.set(artist, (nextCounts.get(artist) ?? 0) + 1);
      const rest = search(index + 1, nextCounts);
      if (rest) return [choice, ...rest];
    }
    return undefined;
  };
  return search(0, new Map()) ?? [];
}

/**
 * Builds a deterministic 40-track discovery slate from a fully ranked TasteLift pool.
 * Guarantees long-tail representation (>=35%), all four taste heads across the slate,
 * maximum 2 tracks per artist, and strict deduplication without lag or quadratic scans.
 */
export function buildTasteSlate(ranked: readonly RankedTrack[], limit = 40): RankedTrack[] {
  const target = Math.max(0, Math.floor(limit));
  if (target === 0 || ranked.length === 0) return [];

  // Deduplicate candidates deterministically
  const sorted = [...ranked].sort(compareRanked);
  const uniqueMbids = new Set<string>();
  const uniqueIdentities = new Set<string>();
  const unique: RankedTrack[] = [];
  for (const track of sorted) {
    const identity = recordingIdentity(track);
    if (uniqueMbids.has(track.mbid) || uniqueIdentities.has(identity)) continue;
    uniqueMbids.add(track.mbid);
    uniqueIdentities.add(identity);
    unique.push(track);
  }

  const scores = unique.map((t) => finite(t.score, finite(t.residualScore, 0)));
  const lifts = unique.map((t) => finite(t.liftScore, 0));
  const maxScore = Math.max(...scores, 0);
  const minScore = Math.min(...scores, 0);
  const maxLift = Math.max(...lifts, 0);
  const minLift = Math.min(...lifts, 0);

  // Filter out non-exceptional mainstream tracks
  const candidates = unique.filter((track) => {
    if (!isMainstream(track)) return true;
    return isExceptional(track, maxScore, minScore, maxLift, minLift);
  });

  const selected: RankedTrack[] = [];
  const selectedMbids = new Set<string>();
  const selectedIdentities = new Set<string>();
  const artistCounts = new Map<string, number>();
  const seenHeads = new Set<number>();

  const canAdd = (track: RankedTrack): boolean => {
    const artist = normalizeIdentityText(track.artist);
    const identity = recordingIdentity(track);
    if (selectedMbids.has(track.mbid) || selectedIdentities.has(identity)) return false;
    return (artistCounts.get(artist) ?? 0) < MAX_PER_ARTIST;
  };

  const add = (track: RankedTrack): void => {
    const artist = normalizeIdentityText(track.artist);
    const identity = recordingIdentity(track);
    selected.push(track);
    selectedMbids.add(track.mbid);
    selectedIdentities.add(identity);
    artistCounts.set(artist, (artistCounts.get(artist) ?? 0) + 1);
    if (track.tasteHeadIndex !== undefined && track.tasteHeadIndex >= 0 && track.tasteHeadIndex < 4) {
      seenHeads.add(track.tasteHeadIndex);
    }
  };

  // Pre-reservation: reserve a feasible head representative for each head before greedy choice
  const reservedHeads = reserveHeadCandidates(candidates, target);
  for (const track of reservedHeads) {
    if (canAdd(track)) add(track);
  }

  // Phase 1: Top 20 strictly by score (satisfying artist cap)
  const topQuota = Math.min(20, target);
  for (const track of candidates) {
    if (selected.length >= topQuota) break;
    if (canAdd(track)) add(track);
  }

  // Phase 2: In slots 21-40, guarantee representation of all 4 heads if present in candidates
  for (const head of [0, 1, 2, 3]) {
    if (!seenHeads.has(head)) {
      const cand = candidates.find((t) => t.tasteHeadIndex === head && canAdd(t));
      if (cand && selected.length < target) add(cand);
    }
  }

  // Phase 3: Check long-tail target (35% = 14 of 40)
  const requiredLongTail = Math.min(Math.ceil(target * LONG_TAIL_SHARE), candidates.filter((t) => isLongTail(t)).length);
  let longTailCount = selected.filter((t) => isLongTail(t)).length;

  if (longTailCount < requiredLongTail) {
    for (const track of candidates) {
      if (selected.length >= target || longTailCount >= requiredLongTail) break;
      if (isLongTail(track) && canAdd(track)) {
        add(track);
        longTailCount++;
      }
    }
  }

  // Phase 4: Fill remaining slots up to target
  for (const track of candidates) {
    if (selected.length >= target) break;
    if (canAdd(track)) add(track);
  }

  // Ensure deterministic descending score order across the final slate
  selected.sort(compareRanked);

  return selected.map((track) => {
    const artist = normalizeIdentityText(track.artist);
    const count = artistCounts.get(artist) ?? 1;
    const pop = clamp(finite(track.popularityPercentile, 0.5));
    return {
      ...track,
      slateScore: track.score,
      slateComponents: {
        relevance: finite(track.score),
        lift: finite(track.liftScore),
        serendipity: 1 - pop,
        artistDiversity: 1 / count,
        headCoverage: 1,
        marginalScore: track.score,
      },
    };
  });
}
