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

const finite = (value: number | undefined, fallback = 0): number => Number.isFinite(value) ? value! : fallback;
const clamp = (value: number): number => Math.max(0, Math.min(1, value));

function compareRanked(left: RankedTrack, right: RankedTrack): number {
  return finite(right.residualScore, right.score) - finite(left.residualScore, left.score)
    || finite(right.liftScore) - finite(left.liftScore)
    || finite(right.score) - finite(left.score)
    || left.mbid.localeCompare(right.mbid)
    || normalizeIdentityText(left.artist).localeCompare(normalizeIdentityText(right.artist))
    || normalizeIdentityText(left.title).localeCompare(normalizeIdentityText(right.title));
}

function normalise(values: readonly number[]): number[] {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (!Number.isFinite(minimum) || maximum === minimum) return values.map(() => 1);
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

/**
 * Builds a deterministic 40-track discovery slate from a fully ranked TasteLift pool.
 * It never invents a candidate: shortages are returned as shorter slates for the caller to diagnose.
 */
export function buildTasteSlate(ranked: readonly RankedTrack[], limit = 40): RankedTrack[] {
  const target = Math.max(0, Math.floor(limit));
  const candidates = canonicalCandidates(ranked).filter((candidate) => !isMainstream(candidate) || isExceptional(candidate));
  const longTailTarget = Math.min(Math.ceil(target * LONG_TAIL_SHARE), candidates.filter((candidate) => !isMainstream(candidate)).length);
  const availableHeads = new Set(candidates.flatMap((candidate) => candidate.head === undefined ? [] : [candidate.head]));
  const headsToCover = Math.min(target, availableHeads.size);
  const selected: RankedTrack[] = [];
  const selectedMbids = new Set<string>();
  const selectedIdentities = new Set<string>();
  const artistCounts = new Map<string, number>();
  const seenHeads = new Set<number>();
  let longTailCount = 0;

  while (selected.length < target) {
    const slotsRemaining = target - selected.length;
    const longTailNeeded = longTailTarget - longTailCount;
    const headsNeeded = headsToCover - seenHeads.size;
    let eligible = candidates.filter((candidate) => !selectedMbids.has(candidate.track.mbid)
      && !selectedIdentities.has(candidate.identity)
      && (artistCounts.get(candidate.artist) ?? 0) < MAX_PER_ARTIST);
    if (longTailNeeded > 0 && slotsRemaining <= longTailNeeded) eligible = eligible.filter((candidate) => !isMainstream(candidate));
    // Cover each available TasteLift head before allowing repeated-head choices.
    // This also reserves artist capacity for a head before it can be spent elsewhere.
    if (headsNeeded > 0) eligible = eligible.filter((candidate) => candidate.head !== undefined && !seenHeads.has(candidate.head));
    if (eligible.length === 0) break;
    eligible.sort((left, right) => compareMarginal(left, right, artistCounts, seenHeads));
    const chosen = eligible[0]!;
    const components = details(chosen, artistCounts.get(chosen.artist) ?? 0, seenHeads);
    selected.push({ ...chosen.track, slateScore: components.marginalScore, slateComponents: components });
    selectedMbids.add(chosen.track.mbid);
    selectedIdentities.add(chosen.identity);
    artistCounts.set(chosen.artist, (artistCounts.get(chosen.artist) ?? 0) + 1);
    if (chosen.head !== undefined) seenHeads.add(chosen.head);
    if (!isMainstream(chosen)) longTailCount += 1;
  }
  return selected;
}
