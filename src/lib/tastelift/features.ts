import type { CandidateEvidence, SeedEvidence } from "../types";
import { TasteLiftModel, type TasteLiftTrack } from "./model";

export interface TasteLiftRankingFields {
  tasteHead: number;
  tasteHeadIndex: number;
  perHeadScores: number[];
  seedSupport: number;
  seedSupportEvidence: SeedEvidence[];
  popularityPercentile: number;
  affinityScore: number;
  liftScore: number;
}

const asTasteTrack = (track: Pick<TasteLiftTrack, "mbid" | "artist" | "title"> & { popularityPercentile?: number }): TasteLiftTrack => ({
  mbid: track.mbid,
  artist: track.artist,
  title: track.title,
  popularityPercentile: track.popularityPercentile,
});

/**
 * Keeps retrieval support as explicit evidence and makes the learned lift additive
 * to the legacy residual score. Spotify identifiers are deliberately absent.
 */
export function buildTasteLiftRankingFields(model: TasteLiftModel, seeds: readonly TasteLiftTrack[], candidates: readonly CandidateEvidence[]): Map<string, TasteLiftRankingFields> {
  const scored = model.scoreCandidates(seeds.map(asTasteTrack), candidates.map(asTasteTrack));
  return new Map(candidates.map((candidate, index) => {
    const score = scored.candidates[index]!;
    const seedSupportEvidence = [...candidate.evidence].sort((left, right) => left.seedMbid.localeCompare(right.seedMbid) || (left.seedIndex ?? 0) - (right.seedIndex ?? 0) || left.rank - right.rank || (left.recordingMbid ?? "").localeCompare(right.recordingMbid ?? "") || right.rawScore - left.rawScore);
    const seedSupport = new Set(seedSupportEvidence.map((evidence) => evidence.seedIndex ?? evidence.seedMbid)).size;
    return [candidate.mbid, {
      tasteHead: score.strongestHead,
      tasteHeadIndex: score.strongestHeadIndex,
      perHeadScores: score.perHeadScores,
      seedSupport,
      seedSupportEvidence,
      popularityPercentile: candidate.popularityPercentile ?? score.popularityPercentile,
      affinityScore: score.affinity,
      liftScore: score.lift,
    }];
  }));
}

export function tasteLiftContribution(fields: Pick<TasteLiftRankingFields, "liftScore" | "seedSupport">, seedCount: number): number {
  return fields.liftScore + fields.seedSupport / seedCount;
}
