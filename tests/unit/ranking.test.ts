import { describe, expect, test } from "vitest";
import {
  buildFeatures,
  diversify,
  mergeCandidates,
  rankCandidates,
} from "../../src/lib/ranking";
import type { SeedTrack, SimilarTrack } from "../../src/lib/types";
import type { ModelArtifact } from "../../src/lib/mlp";
import tasteLiftArtifact from "../../ml/tastelift-model.json";
import { TasteLiftModel } from "../../src/lib/tastelift/model";

const seeds: SeedTrack[] = [
  { mbid: "seed-a", title: "A", artist: "Artist A" },
  { mbid: "seed-b", title: "B", artist: "Artist B" },
  { mbid: "seed-c", title: "C", artist: "Artist C" },
  { mbid: "seed-d", title: "D", artist: "Artist D" },
  { mbid: "seed-e", title: "E", artist: "Artist E" },
];

const track = (
  mbid: string,
  artist: string,
  score: number,
  title = mbid,
): SimilarTrack => ({ mbid, title, artist, score });

const lists: Record<string, SimilarTrack[]> = {
  "seed-a": [track("x", "Artist X", 100), track("seed-b", "Artist B", 90), track("y", "Artist Y", 50)],
  "seed-b": [track("y", "Artist Y", 20), track("x", "Artist X", 10)],
  "seed-c": [track("z", "Artist A", 7)],
  "seed-d": [],
  "seed-e": [track("x", "Artist X", 3)],
};

describe("candidate construction", () => {
  test("deduplicates candidates and excludes every seed", () => {
    const merged = mergeCandidates(seeds, lists);
    expect(merged.map((item) => item.mbid).sort()).toEqual(["x", "y", "z"]);
    expect(merged.find((item) => item.mbid === "x")?.evidence).toHaveLength(3);
  });

  test("excludes alternate MBIDs of a seed with the same artist and title", () => {
    const duplicate = track("alternate-mbid", " artist a ", 100, " A ");
    expect(mergeCandidates(seeds, { ...lists, "seed-a": [duplicate, ...lists["seed-a"]] }).some((item) => item.mbid === duplicate.mbid)).toBe(false);
  });

  test("normalizes each list independently and constructs exactly 17 ordered features", () => {
    const candidate = mergeCandidates(seeds, lists).find((item) => item.mbid === "x")!;
    const features = buildFeatures(candidate, seeds);
    expect(features).toHaveLength(17);
    expect(features.slice(0, 5)).toEqual([1, 1, 0.5, 0, 0]);
    expect(features.slice(5, 10)).toEqual([1 / 61, 1 / 61, 1 / 62, 0, 0]);
    expect(features[10]).toBeCloseTo(1 / 61 + 1 / 61 + 1 / 62);
    expect(features[11]).toBeCloseTo(3 / 5);
    expect(features[12]).toBe(1);
    expect(features[13]).toBe(1);
    expect(features[14]).toBeCloseTo(5 / 6);
    expect(features[15]).toBeCloseTo(Math.sqrt(1 / 18));
    expect(features[16]).toBe(0);
  });
});

describe("ranking", () => {
  test("adds TasteLift lift and retrieval support to the legacy residual score", () => {
    const residual = rankCandidates(seeds, lists, "rrf");
    const ranked = rankCandidates(seeds, lists, "rrf", TasteLiftModel.fromArtifact(tasteLiftArtifact));
    for (const item of ranked) {
      const previous = residual.find((candidate) => candidate.mbid === item.mbid)!;
      expect(item.residualScore).toBe(previous.score);
      expect(item.tasteHeadIndex).toBeGreaterThanOrEqual(0);
      expect(item.tasteHeadIndex).toBeLessThan(4);
      expect(item.perHeadScores).toHaveLength(4);
      expect(item.seedSupportEvidence?.length).toBeGreaterThanOrEqual(item.seedSupport!);
      expect(item.popularityPercentile).toBeGreaterThanOrEqual(0);
      expect(item.liftScore).toBeTypeOf("number");
      expect(item.score).toBeCloseTo(item.residualScore! + item.liftScore! + item.seedSupport! / seeds.length);
    }
  });

  test("uses the production neural artifact rather than hardcoded RRF", () => {
    const neuralLists: Record<string, SimilarTrack[]> = {
      "seed-a": [track("x", "Other", 100), track("y", "Artist A", 50)],
      "seed-b": [], "seed-c": [], "seed-d": [], "seed-e": [],
    };
    const artifact: ModelArtifact = {
      version: 1, feature_names: Array.from({ length: 17 }, (_, i) => `f${i}`), activation: "relu", production_ranker: "neural",
      layers: [{ weight: [Array.from({ length: 17 }, (_, i) => i === 16 ? 10 : 0)], bias: [0] }],
    };
    expect(rankCandidates(seeds, neuralLists, artifact).map((item) => item.mbid)).toEqual(["y", "x"]);
  });
  test("is identical for all 120 permutations of five seeds", () => {
    const permutations = <T,>(items: T[]): T[][] =>
      items.length === 0
        ? [[]]
        : items.flatMap((item, index) =>
            permutations(items.filter((_, candidate) => candidate !== index)).map((rest) => [item, ...rest]),
          );
    const expected = rankCandidates(seeds, lists, "rrf").map(({ mbid, score, features }) => ({ mbid, score, features }));
    for (const permutation of permutations(seeds)) {
      expect(rankCandidates(permutation, lists, "rrf").map(({ mbid, score, features }) => ({ mbid, score, features }))).toEqual(expected);
    }
  });

  test("production neural ranking is identical for all 120 seed permutations", () => {
    const permutations = <T,>(items: T[]): T[][] => items.length === 0 ? [[]] : items.flatMap((item, index) => permutations(items.filter((_, candidate) => candidate !== index)).map((rest) => [item, ...rest]));
    const artifact: ModelArtifact = { version: 1, feature_names: Array.from({ length: 17 }, (_, i) => `f${i}`), activation: "relu", production_ranker: "neural", layers: [{ weight: [Array(17).fill(0.2)], bias: [0] }] };
    const expected = rankCandidates(seeds, lists, artifact).map(({ mbid, score, features }) => ({ mbid, score, features }));
    for (const permutation of permutations(seeds)) expect(rankCandidates(permutation, lists, artifact).map(({ mbid, score, features }) => ({ mbid, score, features }))).toEqual(expected);
  });

  test("uses MBID as a stable final tie breaker", () => {
    const tied: Record<string, SimilarTrack[]> = {
      "seed-a": [track("b", "B", 1), track("a", "A", 1)],
      "seed-b": [], "seed-c": [], "seed-d": [], "seed-e": [],
    };
    expect(rankCandidates(seeds, tied, "max").map((item) => item.mbid)).toEqual(["a", "b"]);
  });

  test("keeps at most two final tracks from one artist without shuffling", () => {
    const ranked = ["1", "2", "3", "4"].map((mbid, index) => ({
      mbid,
      title: mbid,
      artist: index < 3 ? "Same Artist" : "Other",
      score: 4 - index,
      features: Array(17).fill(0),
      pickedFrom: [],
    }));
    expect(diversify(ranked, 20).map((item) => item.mbid)).toEqual(["1", "2", "4"]);
  });

  test("removes alternate MBIDs of the same artist and title", () => {
    const ranked = ["one", "alternate", "other"].map((mbid, index) => ({ mbid, title: index < 2 ? "Same Song" : "Other", artist: index < 2 ? "Artist" : "Else", score: 3 - index, features: Array(17).fill(0), pickedFrom: [] }));
    expect(diversify(ranked, 20).map((item) => item.mbid)).toEqual(["one", "other"]);
  });
});
