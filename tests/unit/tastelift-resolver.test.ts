import { expect, test } from "vitest";
import { parseSearch } from "../../src/lib/listenbrainz";
import { resolveTasteSeeds, type TasteResolverAdapters } from "../../src/lib/tastelift/resolver";
import type { TasteSeedInput } from "../../src/lib/tastelift/input";

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
