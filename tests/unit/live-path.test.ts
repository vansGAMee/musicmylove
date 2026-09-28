import { expect, test } from "vitest";
import { createRecommendations } from "../../src/lib/recommend";
import type { SeedTrack, SimilarTrack } from "../../src/lib/types";

test("production path tolerates a failed seed and preserves all output invariants", () => {
  const seeds: SeedTrack[] = Array.from({ length: 5 }, (_, i) => ({ mbid: `s${i}`, title: `S${i}`, artist: `Seed${i}` }));
  const lists: Record<string, SimilarTrack[]> = Object.fromEntries(seeds.map((seed, i) => [seed.mbid, i === 2 ? [] : Array.from({ length: 30 }, (_, j) => ({ mbid: j === 0 ? "s0" : `c${j}`, title: `C${j}`, artist: `Artist${Math.floor(j / 2)}`, score: 100 - j }))]));
  const first = createRecommendations(seeds, lists, 20);
  const reversed = createRecommendations([...seeds].reverse(), lists, 20);
  expect(first.map((item) => item.mbid)).toEqual(reversed.map((item) => item.mbid));
  expect(first).toHaveLength(20);
  expect(new Set(first.map((item) => item.mbid)).size).toBe(20);
  expect(first.some((item) => item.mbid === "s0")).toBe(false);
  const artistCounts = first.reduce<Record<string, number>>((counts, item) => ({ ...counts, [item.artist]: (counts[item.artist] ?? 0) + 1 }), {});
  expect(Math.max(...Object.values(artistCounts))).toBeLessThanOrEqual(2);
});
