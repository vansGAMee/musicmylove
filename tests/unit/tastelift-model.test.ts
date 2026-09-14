import artifact from "../../ml/tastelift-model.json";
import { describe, expect, test } from "vitest";
import { TasteLiftModel } from "../../src/lib/tastelift/model";

const seeds = [
  { mbid: "seed-c", artist: "Björk", title: "Jóga" },
  { mbid: "seed-a", artist: "  Ｚｅｂｒａ   Ж ", title: "Unknown 🐻" },
  { mbid: "seed-e", artist: "$uicideboy$", title: "23" },
  { mbid: "seed-b", artist: "1991", title: "28" },
  { mbid: "seed-d", artist: "Artist D", title: "Song D" },
] as const;

const candidates = [
  { mbid: "candidate-z", artist: "Björk", title: "Hidden Place" },
  { mbid: "candidate-a", artist: "An OOV Artist", title: "A Very New Song" },
] as const;

const permutations = <T,>(items: readonly T[]): T[][] =>
  items.length === 0
    ? [[]]
    : items.flatMap((item, index) =>
      permutations(items.filter((_, candidate) => candidate !== index)).map((rest) => [item, ...rest]),
    );

describe("TasteLiftModel", () => {
  test("canonically pools all 120 seed permutations including Unicode/OOV metadata", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    const expected = model.scoreCandidates(seeds, candidates);
    for (const permutation of permutations(seeds)) {
      expect(model.scoreCandidates(permutation, candidates)).toEqual(expected);
    }
  });

  test("requires the five-to-five-hundred serving seed set contract", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    expect(() => model.encodeSet(seeds.slice(0, 4))).toThrow(/5 to 500/i);
    const fiveHundredOne = Array.from({ length: 501 }, (_, index) => ({ mbid: `seed-${index}`, artist: `Artist ${index}`, title: `Title ${index}` }));
    expect(() => model.encodeSet(fiveHundredOne)).toThrow(/5 to 500/i);
  });

  test("accepts exactly five hundred seeds in canonical order and scores an empty candidate set", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    const fiveHundred = Array.from({ length: 500 }, (_, index) => ({ mbid: `seed-${String(index).padStart(3, "0")}`, artist: `Artist ${index}`, title: `Title ${index}` }));
    const score = model.scoreCandidates(fiveHundred, []);
    expect(score.heads).toHaveLength(4);
    expect(score.candidates).toEqual([]);
    expect(model.scoreCandidates([...fiveHundred].reverse(), [])).toEqual(score);
  });
});
