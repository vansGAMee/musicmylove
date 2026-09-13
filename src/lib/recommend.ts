import { diversify, rankCandidates } from "./ranking";
import type { RankedTrack, SeedTrack, SimilarityLists } from "./types";

export function createRecommendations(seeds: readonly SeedTrack[], lists: SimilarityLists, limit = 20): RankedTrack[] {
  return diversify(rankCandidates(seeds, lists, "rrf"), limit);
}
