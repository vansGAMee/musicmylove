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

  test("known track without popularity metadata receives neutral 50% popularity rather than 0%", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    const candidateWithoutPop = [{ mbid: "mystery-track-no-pop", artist: "Mystery", title: "Song" }];
    const score = model.scoreCandidates(seeds, candidateWithoutPop);
    expect(score.candidates[0]!.popularityPercentile).toBe(0.5);
    expect(score.candidates[0]!.popularityPercentile).not.toBe(0);
  });

  test("selects a bounded candidate-specific context from all seeds", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    const manySeeds = Array.from({ length: 500 }, (_, index) => ({
      mbid: `seed-${String(index).padStart(3, "0")}`,
      artist: `Artist ${index}`,
      title: `Title ${index}`,
    }));
    const candidate = manySeeds[437]!;

    const score = model.scoreCandidates(manySeeds, [candidate]).candidates[0]!;

    expect(score.contextSeedIndexes.length).toBeLessThanOrEqual(8);
    expect(score.contextSeedIndexes).toContain(437);
    expect(model.scoreCandidates(manySeeds.slice(0, 5), [manySeeds[0]!]).candidates[0]!.contextSeedIndexes).toHaveLength(5);
  });

  test("applies signed learned-embedding feedback without changing the seed profile", () => {
    const model = TasteLiftModel.fromArtifact(artifact);
    const candidate = candidates[0]!;
    const neutral = model.scoreCandidates(seeds, [candidate]).candidates[0]!.lift;
    const liked = model.scoreCandidates(seeds, [candidate], [{ track: candidate, value: "like" }]).candidates[0]!.lift;
    const disliked = model.scoreCandidates(seeds, [candidate], [{ track: candidate, value: "dislike" }]).candidates[0]!.lift;

    expect(liked).toBeGreaterThan(neutral);
    expect(disliked).toBeLessThan(neutral);
    expect(model.scoreCandidates(seeds, [candidate], []).candidates[0]!.lift).toBe(neutral);
  });
});
