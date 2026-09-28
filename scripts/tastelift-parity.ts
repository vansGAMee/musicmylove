import { readFileSync } from "node:fs";
import artifact from "../ml/tastelift-model.json";
import { TasteLiftModel, type TasteLiftArtifact, type TasteLiftTrack } from "../src/lib/tastelift/model";

const input = JSON.parse(readFileSync(process.argv[2]!, "utf8")) as { seeds: TasteLiftTrack[]; candidates: TasteLiftTrack[] };
const scored = TasteLiftModel.fromArtifact(artifact as TasteLiftArtifact).scoreCandidates(input.seeds, input.candidates);
process.stdout.write(JSON.stringify({
  heads: scored.heads,
  candidates: scored.candidates.map(({ perHeadScores, affinity, popularityPrior, lift }) => ({
    per_head: perHeadScores,
    affinity,
    popularity_prior: popularityPrior,
    lift,
  })),
}));
