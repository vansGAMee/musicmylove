import type { RankedTrack, SlateComponents } from "../types";
import { normalizeIdentityText, recordingIdentity } from "./identity";

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

/**
 * Builds a deterministic 40-track discovery slate from a fully ranked TasteLift pool.
 * Keeps learned relevance in order, with only recording dedupe and a weak two-per-artist cap.
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

  const candidates = unique;

  const selected: RankedTrack[] = [];
  const selectedMbids = new Set<string>();
  const selectedIdentities = new Set<string>();
  const artistCounts = new Map<string, number>();

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
  };

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
