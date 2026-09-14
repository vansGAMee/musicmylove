export interface Track {
  mbid: string;
  title: string;
  artist: string;
  release?: string;
}

export type SeedTrack = Track;

export interface SimilarTrack extends Track {
  score: number;
}

export interface SeedEvidence {
  seedMbid: string;
  rank: number;
  rawScore: number;
  normalizedScore: number;
  reciprocalRank: number;
  /** Retained when this evidence originated in a TasteCandidatePool. */
  seedIndex?: number;
  recordingMbid?: string;
  source?: "listenbrainz";
}

export interface CandidateEvidence extends Track {
  evidence: SeedEvidence[];
}

export interface RankedTrack extends Track {
  score: number;
  features: number[];
  pickedFrom: SeedTrack[];
  /** Present only on the additive TasteLift ranking path. */
  residualScore?: number;
  tasteHead?: number;
  tasteHeadIndex?: number;
  perHeadScores?: number[];
  seedSupport?: number;
  seedSupportEvidence?: SeedEvidence[];
  popularityPercentile?: number;
  affinityScore?: number;
  liftScore?: number;
}

export type SimilarityLists = Readonly<Record<string, readonly SimilarTrack[]>>;
export type RankerName = "max" | "rrf";
