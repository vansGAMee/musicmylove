import { expect, test } from "vitest";
import { retrieveTasteCandidates, type TasteRetrievalAdapters } from "../../src/lib/tastelift/retrieval";
import type { ResolvedTasteSeed } from "../../src/lib/tastelift/resolver";
import type { SimilarTrack } from "../../src/lib/types";
import { VersionedCache } from "../../src/lib/cache";

const resolved = (mbid: string, artist = `Artist ${mbid}`, title = `Song ${mbid}`): ResolvedTasteSeed => ({
  input: { artist, title },
  status: "resolved",
  source: "listenbrainz",
  track: { mbid, artist, title },
});

const textOnly: ResolvedTasteSeed = {
  input: { artist: "Uncatalogued Artist", title: "Uncatalogued Song" },
  status: "resolved",
  source: "text",
  diagnostic: {
    status: "retrieval_unavailable",
    code: "no_exact_mbid",
    message: "No exact MusicBrainz recording found for Uncatalogued Artist — Uncatalogued Song",
  },
};

const unavailable: ResolvedTasteSeed = {
  input: { artist: "Unavailable Artist", title: "Unavailable Song" },
  status: "unresolved",
  error: { code: "upstream_error", message: "ListenBrainz unavailable" },
};

const similar = (mbid: string, score: number, artist = `Artist ${mbid}`, title = `Song ${mbid}`): SimilarTrack => ({ mbid, score, artist, title });
const cache = () => {
  const storage = new Map<string, string>();
  return new VersionedCache<readonly SimilarTrack[]>("tastelift-retrieval-test", { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) });
};

test("unions every MBID seed while retaining source evidence and non-MBID seed outcomes", async () => {
  const seen: string[][] = [];
  const adapters: TasteRetrievalAdapters = {
    cache: cache(),
    fetchSimilarBatch: async (mbids) => {
      seen.push([...mbids]);
      return {
        "seed-a": [similar("shared", 0.9, "Shared Artist", "Shared Song"), similar("only-a", 0.6)],
        "seed-b": [similar("shared", 0.8, "Shared Artist", "Shared Song"), similar("only-b", 0.7)],
      };
    },
  };

  const pool = await retrieveTasteCandidates([resolved("seed-b"), textOnly, unavailable, resolved("seed-a")], adapters);

  expect(seen).toEqual([["seed-a", "seed-b"]]);
  expect(pool.seeds).toEqual([resolved("seed-b"), textOnly, unavailable, resolved("seed-a")]);
  expect(pool.candidates.map((candidate) => candidate.mbid)).toEqual(["shared", "only-b", "only-a"]);
  expect(pool.candidates[0]).toMatchObject({
    support: 2,
    evidence: [
      { recordingMbid: "shared", seedMbid: "seed-a", source: "listenbrainz", rank: 1, rawScore: 0.9 },
      { recordingMbid: "shared", seedMbid: "seed-b", source: "listenbrainz", rank: 1, rawScore: 0.8 },
    ],
  });
});

test("caps the fully merged candidate pool at the requested limit rather than per-seed", async () => {
  const rows = Array.from({ length: 501 }, (_, index) => similar(`candidate-${String(index).padStart(3, "0")}`, 1000 - index));
  const pool = await retrieveTasteCandidates([resolved("seed")], {
    cache: cache(),
    fetchSimilarBatch: async () => ({ seed: rows }),
  }, 500);

  expect(pool.candidates).toHaveLength(500);
  expect(pool.candidates[0]?.mbid).toBe("candidate-000");
  expect(pool.candidates.at(-1)?.mbid).toBe("candidate-499");
});

test("retrieves every distinct MBID when all inputs share one artist", async () => {
  const calls: string[][] = [];
  const sameArtistSeeds = [
    resolved("seed-1", "One Artist", "First"),
    resolved("seed-2", "One Artist", "Second"),
    resolved("seed-3", "One Artist", "Third"),
  ];
  const pool = await retrieveTasteCandidates(sameArtistSeeds, {
    cache: cache(),
    fetchSimilarBatch: async (mbids) => {
      calls.push([...mbids]);
      return Object.fromEntries(mbids.map((mbid) => [mbid, [similar(`candidate-for-${mbid}`, 1)]]));
    },
  });

  expect(calls).toEqual([["seed-1", "seed-2", "seed-3"]]);
  expect(pool.candidates.map((candidate) => candidate.mbid)).toEqual(["candidate-for-seed-1", "candidate-for-seed-2", "candidate-for-seed-3"]);
});

test("excludes seed recordings even when ListenBrainz returns an alternate MBID", async () => {
  const pool = await retrieveTasteCandidates([resolved("seed", "Seed Artist", "Seed Song")], {
    cache: cache(),
    fetchSimilarBatch: async () => ({
      seed: [
        similar("seed", 1, "Seed Artist", "Seed Song"),
        similar("seed-alternate", 0.9, " seed artist ", " Seed Song "),
        similar("candidate", 0.8),
      ],
    }),
  });

  expect(pool.candidates.map((candidate) => candidate.mbid)).toEqual(["candidate"]);
});

test("collapses alternate MBIDs and obvious remaster or cover versions while preserving their evidence", async () => {
  const pool = await retrieveTasteCandidates([resolved("seed-a"), resolved("seed-b")], {
    cache: cache(),
    fetchSimilarBatch: async () => ({
      "seed-a": [
        similar("z-remaster", 0.7, "Version Artist", "A Song (2011 Remaster)"),
        similar("other", 0.5),
      ],
      "seed-b": [
        similar("a-original", 0.9, "Version Artist", "A Song"),
        similar("cover", 0.8, "Version Artist", "A Song - Cover"),
      ],
    }),
  });

  expect(pool.candidates.map((candidate) => candidate.mbid)).toEqual(["a-original", "other"]);
  expect(pool.candidates[0]).toMatchObject({
    alternateMbids: ["a-original", "cover", "z-remaster"],
    support: 2,
    evidence: [
      { recordingMbid: "z-remaster", seedMbid: "seed-a", rank: 1 },
      { recordingMbid: "a-original", seedMbid: "seed-b", rank: 1 },
      { recordingMbid: "cover", seedMbid: "seed-b", rank: 2 },
    ],
  });
});
