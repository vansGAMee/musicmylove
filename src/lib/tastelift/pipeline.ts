import catalogJson from "../../../ml/tastelift-catalog.json";
import modelJson from "../../../ml/tastelift-model.json";
import { createTasteLiftSlate } from "../recommend";
import type { RankedTrack } from "../types";
import { expandTasteCandidatePool, type TasteLiftCatalogArtifact } from "./catalog";
import { recordingIdentity } from "./identity";
import { TasteLiftModel, type TasteLiftArtifact, type TasteLiftFeedback } from "./model";
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
  feedback?: readonly TasteLiftFeedback[];
}

/** Real application path: external evidence + train-only neural scan + rank + slate. */
export async function recommendTasteSeeds(seeds: readonly ResolvedTasteSeed[], options: TasteLiftPipelineOptions = {}): Promise<TasteLiftPipelineResult> {
  const model = options.model ?? TasteLiftModel.fromArtifact(modelJson as TasteLiftArtifact);
  const candidateLimit = options.candidateLimit ?? 2_000;
  const external = await retrieveTasteCandidates(seeds, options.retrievalAdapters ?? createTasteRetrievalAdapters(), candidateLimit);
  const expanded = expandTasteCandidatePool(external, options.catalog ?? catalogJson as TasteLiftCatalogArtifact, model, candidateLimit);

  // Guarantee that no song already in the user's playlist/seeds is ever recommended
  const excludedMbids = new Set<string>();
  const excludedIdentities = new Set<string>();
  for (const s of seeds) {
    if ("track" in s && s.track?.mbid) {
      excludedMbids.add(s.track.mbid);
      excludedIdentities.add(recordingIdentity(s.track));
    }
    if (s.input?.artist && s.input?.title) {
      excludedIdentities.add(recordingIdentity({ artist: s.input.artist, title: s.input.title }));
    }
  }

  const filteredExpanded = {
    ...expanded,
    candidates: expanded.candidates.filter((c) => {
      if (excludedMbids.has(c.mbid)) return false;
      if (c.alternateMbids?.some((id) => excludedMbids.has(id))) return false;
      if (excludedIdentities.has(recordingIdentity(c))) return false;
      return true;
    }),
  };

  const rawSlate = createTasteLiftSlate(filteredExpanded, options.slateLimit ?? 40, options.feedback);
  const recommendations = rawSlate.filter((r) => !excludedMbids.has(r.mbid) && !excludedIdentities.has(recordingIdentity(r)));

  return { candidateCount: filteredExpanded.candidates.length, recommendations };
}
