import { readFileSync } from "node:fs";
import { buildFeatures, mergeCandidates } from "../src/lib/ranking";
import type { SeedTrack, SimilarTrack } from "../src/lib/types";

const input = JSON.parse(readFileSync(process.argv[2], "utf8")) as { seeds: SeedTrack[]; lists: Record<string, SimilarTrack[]> };
const candidates = mergeCandidates(input.seeds, input.lists);
console.log(JSON.stringify({ mbids: candidates.map((item) => item.mbid).sort(), features: Object.fromEntries(candidates.map((item) => [item.mbid, buildFeatures(item, input.seeds)])) }));
