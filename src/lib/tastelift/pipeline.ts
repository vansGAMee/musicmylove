import catalogJson from "../../../ml/tastelift-catalog.json";
import modelJson from "../../../ml/tastelift-model.json";
import { createTasteLiftSlate } from "../recommend";
import type { RankedTrack } from "../types";
import { expandTasteCandidatePool, type TasteLiftCatalogArtifact } from "./catalog";
import { TasteLiftModel, type TasteLiftArtifact } from "./model";
import { createTasteRetrievalAdapters, retrieveTasteCandidates, type TasteRetrievalAdapters } from "./retrieval";
import type { ResolvedTasteSeed } from "./resolver";

export interface TasteLiftPipelineResult {
  candidateCount: number;
  recommendations: RankedTrack[];
}

export interface TasteLiftPipelineOptions {
  retrievalAdapters?: TasteRetrievalAdapters;
  catalog?: TasteLiftCatalogArtifact;
  model?: TasteLiftModel;
  candidateLimit?: number;
  slateLimit?: number;
}

/** Real application path: external evidence + train-only neural scan + rank + slate. */
export async function recommendTasteSeeds(seeds: readonly ResolvedTasteSeed[], options: TasteLiftPipelineOptions = {}): Promise<TasteLiftPipelineResult> {
  const model = options.model ?? TasteLiftModel.fromArtifact(modelJson as TasteLiftArtifact);
  const candidateLimit = options.candidateLimit ?? 500;
  const external = await retrieveTasteCandidates(seeds, options.retrievalAdapters ?? createTasteRetrievalAdapters(), candidateLimit);
  const expanded = expandTasteCandidatePool(external, options.catalog ?? catalogJson as TasteLiftCatalogArtifact, model, candidateLimit);
  return { candidateCount: expanded.candidates.length, recommendations: createTasteLiftSlate(expanded, options.slateLimit ?? 40) };
}
