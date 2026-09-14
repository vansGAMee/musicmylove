import type { CandidateEvidence, SeedEvidence, SeedTrack } from "../types";
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

const asTasteTrack = (track: Pick<TasteLiftTrack, "mbid" | "artist" | "title">): TasteLiftTrack => ({ mbid: track.mbid, artist: track.artist, title: track.title });

/**
 * Keeps retrieval support as explicit evidence and makes the learned lift additive
 * to the legacy residual score. Spotify identifiers are deliberately absent.
 */
export function buildTasteLiftRankingFields(model: TasteLiftModel, seeds: readonly SeedTrack[], candidates: readonly CandidateEvidence[]): Map<string, TasteLiftRankingFields> {
  const scored = model.scoreCandidates(seeds.map(asTasteTrack), candidates.map(asTasteTrack));
  return new Map(candidates.map((candidate, index) => {
    const score = scored.candidates[index]!;
    const seedSupportEvidence = [...candidate.evidence].sort((left, right) => left.seedMbid.localeCompare(right.seedMbid) || left.rank - right.rank);
    const seedSupport = new Set(seedSupportEvidence.map((evidence) => evidence.seedMbid)).size;
    return [candidate.mbid, {
      tasteHead: score.strongestHead,
      tasteHeadIndex: score.strongestHeadIndex,
      perHeadScores: score.perHeadScores,
      seedSupport,
      seedSupportEvidence,
      popularityPercentile: score.popularityPercentile,
      affinityScore: score.affinity,
      liftScore: score.lift,
    }];
  }));
}

export function tasteLiftContribution(fields: Pick<TasteLiftRankingFields, "liftScore" | "seedSupport">, seedCount: number): number {
  return fields.liftScore + fields.seedSupport / seedCount;
}
