import { expect, test } from "vitest";
import { handleTasteLiftPost } from "../../app/api/tastelift/route";
import { LastFmResponseError, resolveTasteSeeds as resolveSeeds, type ResolvedTasteSeed } from "../../src/lib/tastelift/resolver";

const body = { songs: Array.from({ length: 5 }, (_, index) => ({ artist: `Artist ${index}`, title: `Song ${index}` })) };

test("returns a structured 400 for malformed JSON", async () => {
  const response = await handleTasteLiftPost(new Request("https://example.test/api/tastelift", { method: "POST", body: "{" }));
  expect(response.status).toBe(400);
  await expect(response.json()).resolves.toEqual({ error: "Request body must be valid JSON" });
});

test("returns structured unresolved seed failures as 422", async () => {
  const response = await handleTasteLiftPost(new Request("https://example.test/api/tastelift", { method: "POST", body: JSON.stringify(body) }), {
    resolveTasteSeeds: (inputs): Promise<ResolvedTasteSeed[]> => resolveSeeds(inputs.slice(0, 1), {
      searchRecordings: async () => [],
      resolveLastFm: async () => { throw new LastFmResponseError(6, "Track not found"); },
    }),
  });
  expect(response.status).toBe(422);
  await expect(response.json()).resolves.toEqual({
    error: "Every seed must resolve before recommendations can be generated",
    seeds: [{
      input: body.songs[0],
      status: "unresolved",
      error: { code: "upstream_error", message: "Last.fm error 6: Track not found" },
    }],
  });
});

test("returns 200 while retaining a valid text/OOV seed", async () => {
  const response = await handleTasteLiftPost(new Request("https://example.test/api/tastelift", { method: "POST", body: JSON.stringify(body) }), {
    resolveTasteSeeds: async (inputs): Promise<ResolvedTasteSeed[]> => [{
      input: inputs[0]!,
      status: "resolved",
      source: "text",
      diagnostic: {
        status: "retrieval_unavailable",
        code: "no_exact_mbid",
        message: "No exact MusicBrainz recording found",
      },
    }],
  });
  expect(response.status).toBe(200);
  await expect(response.json()).resolves.toEqual({
    seeds: [{
      input: body.songs[0],
      status: "resolved",
      source: "text",
      diagnostic: {
        status: "retrieval_unavailable",
        code: "no_exact_mbid",
        message: "No exact MusicBrainz recording found",
      },
    }],
  });
});
