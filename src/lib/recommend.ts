import { diversify, rankCandidates } from "./ranking";
import modelJson from "../../ml/model.json";
import type { ModelArtifact } from "./mlp";
import type { RankedTrack, SeedTrack, SimilarityLists } from "./types";

export function createRecommendations(seeds: readonly SeedTrack[], lists: SimilarityLists, limit = 20): RankedTrack[] {
  return diversify(rankCandidates(seeds, lists, modelJson as ModelArtifact), limit);
}
