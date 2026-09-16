import { expect, test } from "vitest";
import { parseSearch } from "../../src/lib/listenbrainz";
import { createTasteResolverAdapters, LastFmResponseError, parseLastFmTrack, resolveTasteSeeds, type ResolvedTasteSeed, type TasteResolverAdapters } from "../../src/lib/tastelift/resolver";
import type { TasteSeedInput } from "../../src/lib/tastelift/input";
import { VersionedCache } from "../../src/lib/cache";

const inputs: TasteSeedInput[] = [
  { artist: "Project Dumb", title: "An Italian Magician Be Like" },
  { artist: "Missing Artist", title: "Missing Song" },
];

const projectDumbSearchFixture = [{
  recording_mbid: "06a87e55-6cc4-4c1a-96ad-4e13da26822a",
  recording_name: "Chicken Little",
  artist_credit_name: "Project Dumb",
  release_name: "Chicken Little",
}];

const adapters: TasteResolverAdapters = {
  searchRecordings: async (query) => query.includes("Project Dumb") ? parseSearch(projectDumbSearchFixture) : [],
};

test("retains Project Dumb as a deterministic text seed when the real API-shaped fixture has no exact recording", async () => {
  await expect(resolveTasteSeeds([inputs[0]], adapters)).resolves.toEqual([{
    input: inputs[0],
    status: "resolved",
    source: "text",
    diagnostic: {
      status: "retrieval_unavailable",
      code: "no_exact_mbid",
      message: "No exact MusicBrainz recording found for Project Dumb — An Italian Magician Be Like",
    },
  }]);
});

test("uses an exact ListenBrainz recording identity when available", async () => {
  const exactInput: TasteSeedInput = { artist: "Exact Artist", title: "Exact Song" };
  const exactFixture = [{ recording_mbid: "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890", recording_name: "Exact Song", artist_credit_name: "Exact Artist", release_name: "Exact Album" }];
  await expect(resolveTasteSeeds([exactInput], { searchRecordings: async () => parseSearch(exactFixture) })).resolves.toEqual([{
    input: exactInput,
    status: "resolved",
    track: { mbid: "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890", artist: "Exact Artist", title: "Exact Song", release: "Exact Album" },
    source: "listenbrainz",
  }]);
});

test("chooses the same lowest MBID exact match regardless of upstream row order", async () => {
  const exactInput: TasteSeedInput = { artist: "Exact Artist", title: "Exact Song" };
  const matches = [
    { mbid: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", artist: "Exact Artist", title: "Exact Song" },
    { mbid: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", artist: "Exact Artist", title: "Exact Song" },
  ];
  const forward = await resolveTasteSeeds([exactInput], { searchRecordings: async () => matches });
  const reverse = await resolveTasteSeeds([exactInput], { searchRecordings: async () => [...matches].reverse() });
  expect(forward[0]).toMatchObject({ source: "listenbrainz", track: { mbid: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" } });
  expect(reverse[0]).toEqual(forward[0]);
});

test("returns one deterministic result per input without silently losing text-only seeds", async () => {
  await expect(resolveTasteSeeds(inputs, adapters)).resolves.toEqual([
    {
      input: inputs[0],
      status: "resolved",
      source: "text",
      diagnostic: {
        status: "retrieval_unavailable",
        code: "no_exact_mbid",
        message: "No exact MusicBrainz recording found for Project Dumb — An Italian Magician Be Like",
      },
    },
    {
      input: inputs[1],
      status: "resolved",
      source: "text",
      diagnostic: {
        status: "retrieval_unavailable",
        code: "no_exact_mbid",
        message: "No exact MusicBrainz recording found for Missing Artist — Missing Song",
      },
    },
  ]);
});

test("uses a Last.fm fallback only after ListenBrainz cannot find a recording", async () => {
  const withLastFm: TasteResolverAdapters = {
    ...adapters,
    resolveLastFm: async () => ({
      mbid: "9a8b7c6d-e5f4-3210-9a8b-7c6d5e4f3210",
      artist: "Missing Artist",
      title: "Missing Song",
    }),
  };
  const [result] = await resolveTasteSeeds([inputs[1]], withLastFm);
  expect(result).toEqual({
    input: inputs[1],
    status: "resolved",
    track: { mbid: "9a8b7c6d-e5f4-3210-9a8b-7c6d5e4f3210", artist: "Missing Artist", title: "Missing Song" },
    source: "lastfm",
  });
});

test("does not turn a mismatched Last.fm result into a false MBID identity", async () => {
  const [result] = await resolveTasteSeeds([inputs[1]], {
    searchRecordings: async () => [],
    resolveLastFm: async () => ({ mbid: "9a8b7c6d-e5f4-3210-9a8b-7c6d5e4f3210", artist: "Different Artist", title: "Different Song" }),
  });
  expect(result).toEqual({
    input: inputs[1],
    status: "resolved",
    source: "text",
    diagnostic: {
      status: "retrieval_unavailable",
      code: "no_exact_mbid",
      message: "No exact MusicBrainz recording found for Missing Artist — Missing Song",
    },
  });
});

test("reports upstream failures per seed without losing neighboring seeds", async () => {
  const unavailable: TasteResolverAdapters = {
    searchRecordings: async (query) => {
      if (query.includes("Missing Artist")) throw new Error("upstream unavailable");
      return adapters.searchRecordings(query);
    },
  };
  const results = await resolveTasteSeeds(inputs, unavailable);
  expect(results).toHaveLength(2);
  expect(results[0]?.status).toBe("resolved");
  expect(results[1]).toEqual({
    input: inputs[1],
    status: "unresolved",
    error: { code: "upstream_error", message: "upstream unavailable" },
  });
});

test("treats a Last.fm HTTP-200 error payload as a typed upstream failure", async () => {
  expect(() => parseLastFmTrack({ error: 6, message: "Track not found" })).toThrow(LastFmResponseError);
  expect(() => parseLastFmTrack({ message: "Rate limit exceeded" })).toThrow(LastFmResponseError);
  const [result] = await resolveTasteSeeds([inputs[1]], {
    searchRecordings: async () => [],
    resolveLastFm: async () => { throw new LastFmResponseError(6, "Track not found"); },
  });
  expect(result).toEqual({
    input: inputs[1],
    status: "unresolved",
    error: { code: "upstream_error", message: "Last.fm error 6: Track not found" },
  });
});

test("bounds adapter concurrency and caches resolved seed outcomes", async () => {
  const storage = new Map<string, string>();
  const cache = new VersionedCache<ResolvedTasteSeed>("tastelift-resolver-test", { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) });
  const batch = Array.from({ length: 6 }, (_, index): TasteSeedInput => ({ artist: `Artist ${index}`, title: `Song ${index}` }));
  let active = 0;
  let maximumActive = 0;
  let calls = 0;
  const adapters: TasteResolverAdapters = {
    searchRecordings: async () => {
      calls++;
      active++;
      maximumActive = Math.max(maximumActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active--;
      return [];
    },
    resolveLastFm: async () => {
      calls++;
      active++;
      maximumActive = Math.max(maximumActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active--;
      return null;
    },
  };
  await resolveTasteSeeds(batch, adapters, { cache, maxConcurrency: 2 });
  expect(maximumActive).toBeLessThanOrEqual(2);
  await resolveTasteSeeds(batch, adapters, { cache, maxConcurrency: 2 });
  expect(calls).toBe(12);
});

test("createTasteResolverAdapters resolves catalog tracks locally and bounds live searches", async () => {
  const adapters = createTasteResolverAdapters();
  const knownSeed: TasteSeedInput = { artist: "Radiohead", title: "Karma Police" };
  const unknownSeeds: TasteSeedInput[] = Array.from({ length: 30 }, (_, i) => ({
    artist: `Unknown Artist ${i}`,
    title: `Unknown Song ${i}`,
  }));

  const results = await resolveTasteSeeds([knownSeed, ...unknownSeeds], adapters);
  expect(results).toHaveLength(31);

  // Known track is resolved from local catalog with its MBID
  expect(results[0]).toEqual({
    input: knownSeed,
    status: "resolved",
    track: expect.objectContaining({
      artist: "Radiohead",
      title: "Karma Police",
      mbid: "9e2ad5bc-c6f9-40d2-a36f-3122ee2072a3",
    }),
    source: "listenbrainz",
  });

  // All 31 seeds must be resolved (none unresolved/errored)
  expect(results.every((r) => r.status === "resolved")).toBe(true);
});
