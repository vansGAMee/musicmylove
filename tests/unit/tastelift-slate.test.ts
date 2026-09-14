import { describe, expect, test } from "vitest";
import { createTasteLiftSlate } from "../../src/lib/recommend";
import { buildTasteSlate } from "../../src/lib/tastelift/slate";
import type { RankedTrack } from "../../src/lib/types";
import type { TasteCandidatePool } from "../../src/lib/tastelift/retrieval";

const candidate = (index: number, overrides: Partial<RankedTrack> = {}): RankedTrack => ({
  mbid: `track-${String(index).padStart(3, "0")}`,
  title: `Track ${index}`,
  artist: `Artist ${index}`,
  score: 100 - index,
  residualScore: 100 - index,
  liftScore: 20 - index / 10,
  popularityPercentile: index < 18 ? 0.92 : 0.2,
  tasteHeadIndex: index % 4,
  tasteHead: 1,
  seedSupport: 2,
  features: Array(17).fill(0),
  pickedFrom: [],
  ...overrides,
});

const candidates = (count = 60): RankedTrack[] => Array.from({ length: count }, (_, index) => candidate(index));

describe("TasteLift discovery slate", () => {
  test("returns exactly forty eligible tracks with a material long-tail share and all four heads", () => {
    const slate = buildTasteSlate(candidates(), 40);
    expect(slate).toHaveLength(40);
    expect(slate.filter((track) => (track.popularityPercentile ?? 0) < 0.85).length).toBeGreaterThanOrEqual(14);
    expect(new Set(slate.map((track) => track.tasteHeadIndex))).toEqual(new Set([0, 1, 2, 3]));
    expect(slate.every((track) => track.slateComponents?.relevance !== undefined && track.slateComponents.lift !== undefined && track.slateComponents.serendipity !== undefined)).toBe(true);
  });

  test("is deterministic for permutations of a ranked candidate pool", () => {
    const input = candidates(48);
    const expected = buildTasteSlate(input, 40).map((track) => track.mbid);
    expect(buildTasteSlate([...input].reverse(), 40).map((track) => track.mbid)).toEqual(expected);
  });

  test("removes duplicate MBIDs, recording identities, and remaster spam while limiting artists to two", () => {
    const slate = buildTasteSlate([
      candidate(1, { mbid: "same", artist: "Same Artist", title: "Song", score: 99 }),
      candidate(2, { mbid: "same", artist: "Other Artist", title: "Different", score: 98 }),
      candidate(3, { mbid: "alternate", artist: "Same Artist", title: "Song (2011 Remaster)", score: 97 }),
      candidate(4, { artist: "Same Artist", title: "Third", score: 96 }),
      candidate(5, { artist: "Same Artist", title: "Fourth", score: 95 }),
      ...candidates(45).map((track) => ({ ...track, mbid: `unique-${track.mbid}`, artist: `Unique ${track.artist}` })),
    ], 40);
    expect(slate.filter((track) => track.mbid === "same")).toHaveLength(1);
    expect(slate.some((track) => track.mbid === "alternate")).toBe(false);
    expect(slate.filter((track) => track.artist === "Same Artist")).toHaveLength(2);
  });

  test("only admits mainstream tracks with exceptional relevance or personalized lift", () => {
    const slate = buildTasteSlate([
      candidate(1, { mbid: "ordinary-mainstream", popularityPercentile: 0.95, score: 20, residualScore: 20, liftScore: 1 }),
      candidate(2, { mbid: "exceptional-mainstream", popularityPercentile: 0.95, score: 100, residualScore: 100, liftScore: 20 }),
      ...Array.from({ length: 42 }, (_, index) => candidate(index + 10, { popularityPercentile: 0.2, score: 60 - index / 10, residualScore: 60 - index / 10, liftScore: 8 })),
    ], 40);
    expect(slate.map((track) => track.mbid)).not.toContain("ordinary-mainstream");
    expect(slate.map((track) => track.mbid)).toContain("exceptional-mainstream");
  });

  test("returns all eligible rows for mixed metadata and all-one-artist source candidates without fabricating replacements", () => {
    const input = [
      candidate(1, { artist: "Seed Artist", title: "One", tasteHeadIndex: undefined, popularityPercentile: undefined, liftScore: undefined, residualScore: undefined }),
      candidate(2, { artist: "Seed Artist", title: "Two" }),
      candidate(3, { artist: "Seed Artist", title: "Three" }),
      ...Array.from({ length: 8 }, (_, index) => candidate(index + 10, { artist: `Discovery Artist ${index}`, tasteHeadIndex: index % 4 })),
    ];
    const slate = buildTasteSlate(input, 40);
    expect(slate).toHaveLength(10);
    expect(slate.filter((track) => track.artist === "Seed Artist")).toHaveLength(2);
    expect(new Set(slate.map((track) => track.artist)).size).toBeGreaterThan(4);
  });

  test("builds a slate from the real pool path without changing legacy recommendation APIs", () => {
    const pool: TasteCandidatePool = {
      seeds: Array.from({ length: 5 }, (_, index) => ({ input: { artist: `Seed ${index}`, title: `Song ${index}` }, status: "resolved" as const, source: "listenbrainz" as const, track: { mbid: `seed-${index}`, artist: `Seed ${index}`, title: `Song ${index}` } })),
      candidates: Array.from({ length: 42 }, (_, index) => ({
        mbid: `pool-${index}`, artist: `Pool Artist ${index}`, title: `Pool Track ${index}`, alternateMbids: [`pool-${index}`], support: 1, retrievalScore: 1 / (61 + index),
        evidence: [{ source: "listenbrainz" as const, seedIndex: index % 5, seedMbid: `seed-${index % 5}`, recordingMbid: `pool-${index}`, rank: index + 1, rawScore: 100 - index }],
      })),
    };
    expect(createTasteLiftSlate(pool)).toHaveLength(40);
  });

  test("fills forty when raw long-tail rows exceed their feasible artist-capped capacity", () => {
    const longTail = Array.from({ length: 14 }, (_, index) => candidate(index, {
      mbid: `tail-${index}`, artist: "Artist A", title: `Tail ${index}`, popularityPercentile: 0.2, score: 20, residualScore: 20, liftScore: 20, tasteHeadIndex: undefined,
    }));
    const mainstream = Array.from({ length: 38 }, (_, index) => candidate(index + 20, {
      mbid: `main-${index}`, artist: `Mainstream ${index}`, title: `Main ${index}`, popularityPercentile: 0.95, score: 100, residualScore: 100, liftScore: 100, tasteHeadIndex: undefined,
    }));
    const reference = candidate(99, { mbid: "reference", artist: "Artist A", title: "Reference", popularityPercentile: 0.2, score: 0, residualScore: 0, liftScore: 0, tasteHeadIndex: undefined });
    const slate = buildTasteSlate([...longTail, ...mainstream, reference], 40);
    expect(slate).toHaveLength(40);
    expect(slate.filter((track) => track.artist === "Artist A")).toHaveLength(2);
  });

  test("reserves a feasible four-head assignment before greedy selection can strand a head", () => {
    const heads = [
      candidate(1, { mbid: "h0-a", artist: "A", title: "H0 A", score: 100, residualScore: 100, liftScore: 100, popularityPercentile: 0.2, tasteHeadIndex: 0 }),
      candidate(2, { mbid: "h0-b", artist: "B", title: "H0 B", score: 60, residualScore: 60, liftScore: 60, popularityPercentile: 0.2, tasteHeadIndex: 0 }),
      candidate(3, { mbid: "h1-a", artist: "A", title: "H1 A", score: 99, residualScore: 99, liftScore: 99, popularityPercentile: 0.2, tasteHeadIndex: 1 }),
      candidate(4, { mbid: "h1-c", artist: "C", title: "H1 C", score: 59, residualScore: 59, liftScore: 59, popularityPercentile: 0.2, tasteHeadIndex: 1 }),
      candidate(5, { mbid: "h2-a", artist: "A", title: "H2 A", score: 80, residualScore: 80, liftScore: 80, popularityPercentile: 0.2, tasteHeadIndex: 2 }),
      candidate(6, { mbid: "h3-d", artist: "D", title: "H3 D", score: 70, residualScore: 70, liftScore: 70, popularityPercentile: 0.2, tasteHeadIndex: 3 }),
    ];
    const neutral = Array.from({ length: 36 }, (_, index) => candidate(index + 20, { mbid: `neutral-${index}`, artist: `Neutral ${index}`, title: `Neutral ${index}`, score: 10, residualScore: 10, liftScore: 10, popularityPercentile: 0.2, tasteHeadIndex: undefined }));
    const slate = buildTasteSlate([...heads, ...neutral], 40);
    expect(slate).toHaveLength(40);
    expect(new Set(slate.map((track) => track.tasteHeadIndex).filter((head): head is number => head !== undefined))).toEqual(new Set([0, 1, 2, 3]));
  });

  test("does not treat equal or non-finite mainstream scores as exceptional", () => {
    const equalMainstream = Array.from({ length: 40 }, (_, index) => candidate(index, {
      mbid: `equal-${index}`, artist: `Equal ${index}`, popularityPercentile: 0.95, score: Number.NaN, residualScore: Number.NaN, liftScore: Number.NaN, tasteHeadIndex: undefined,
    }));
    expect(buildTasteSlate(equalMainstream, 40)).toEqual([]);
  });

  test("chooses the same complete duplicate record independent of input order", () => {
    const first = candidate(1, { mbid: "duplicate", artist: "Duplicate", title: "Song", popularityPercentile: 0.2, pickedFrom: [{ mbid: "seed-b", artist: "Seed B", title: "B" }] });
    const second = candidate(1, { mbid: "duplicate", artist: "Duplicate", title: "Song", popularityPercentile: 0.3, pickedFrom: [{ mbid: "seed-a", artist: "Seed A", title: "A" }] });
    const rest = Array.from({ length: 39 }, (_, index) => candidate(index + 20, { mbid: `stable-${index}`, artist: `Stable ${index}`, title: `Stable ${index}`, popularityPercentile: 0.2 }));
    expect(buildTasteSlate([first, second, ...rest], 40)).toEqual(buildTasteSlate([...rest, second, first], 40));
  });

  test("applies the artist cap across case and whitespace variants", () => {
    const variants = ["  CASE ARTIST ", "case artist", "Case Artist"].map((artist, index) => candidate(index, { mbid: `case-${index}`, artist, title: `Case ${index}`, popularityPercentile: 0.2 }));
    const rest = Array.from({ length: 38 }, (_, index) => candidate(index + 20, { mbid: `other-${index}`, artist: `Other ${index}`, title: `Other ${index}`, popularityPercentile: 0.2 }));
    const slate = buildTasteSlate([...variants, ...rest], 40);
    expect(slate.filter((track) => track.artist.normalize("NFKC").trim().toLowerCase() === "case artist")).toHaveLength(2);
  });
});
