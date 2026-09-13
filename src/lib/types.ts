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
}

export interface CandidateEvidence extends Track {
  evidence: SeedEvidence[];
}

export interface RankedTrack extends Track {
  score: number;
  features: number[];
  pickedFrom: SeedTrack[];
}

export type SimilarityLists = Readonly<Record<string, readonly SimilarTrack[]>>;
export type RankerName = "max" | "rrf";
