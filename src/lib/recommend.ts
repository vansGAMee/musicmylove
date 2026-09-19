import { diversify, rankCandidates, rankTasteCandidatePool } from "./ranking";
import modelJson from "../../ml/model.json";
import tasteLiftJson from "../../ml/tastelift-model.json";
import type { ModelArtifact } from "./mlp";
import type { RankedTrack, SeedTrack, SimilarityLists } from "./types";
import { TasteLiftModel, type TasteLiftArtifact } from "./tastelift/model";
import type { TasteLiftFeedback } from "./tastelift/model";
import type { TasteCandidatePool } from "./tastelift/retrieval";
import { buildTasteSlate } from "./tastelift/slate";

export function createRecommendations(seeds: readonly SeedTrack[], lists: SimilarityLists, limit = 20): RankedTrack[] {
  return diversify(rankCandidates(seeds, lists, modelJson as ModelArtifact), limit);
}

/**
 * Serving integration only: returns an enriched, deterministically ranked pool.
 * Slate construction remains a later concern; this deliberately does not select 40 tracks.
 */
export function createTasteLiftRankedCandidates(seeds: readonly SeedTrack[], lists: SimilarityLists): RankedTrack[] {
  return rankCandidates(seeds, lists, modelJson as ModelArtifact, TasteLiftModel.fromArtifact(tasteLiftJson as TasteLiftArtifact));
}

/** Production path for Task 2's evidence-preserving pool; it performs no retrieval. */
export function createTasteLiftRankedPool(pool: TasteCandidatePool, feedback: readonly TasteLiftFeedback[] = []): RankedTrack[] {
  return rankTasteCandidatePool(pool, modelJson as ModelArtifact, TasteLiftModel.fromArtifact(tasteLiftJson as TasteLiftArtifact), feedback);
}

/** Production TasteLift path: rank the evidence-preserving pool then assemble its discovery slate. */
export function createTasteLiftSlate(pool: TasteCandidatePool, limit = 40, feedback: readonly TasteLiftFeedback[] = []): RankedTrack[] {
  return buildTasteSlate(createTasteLiftRankedPool(pool, feedback), limit);
}
