import { describe, expect, test } from "vitest";
import {
  buildFeatures,
  diversify,
  mergeCandidates,
  rankCandidates,
} from "../../src/lib/ranking";
import type { SeedTrack, SimilarTrack } from "../../src/lib/types";

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
});
