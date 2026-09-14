import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { mkdtemp, writeFile } from "node:fs/promises";
import { expect, test } from "vitest";
import { runTasteLiftPlaylist } from "../../scripts/tastelift-playlist";
import type { ResolvedTasteSeed } from "../../src/lib/tastelift/resolver";

test("CLI writes forty enriched recommendations from Spotify-history JSON", async () => {
  const directory = await mkdtemp(join(tmpdir(), "tastelift-cli-"));
  const input = join(directory, "history.json");
  const output = join(directory, "playlist.json");
  await writeFile(input, JSON.stringify(Array.from({ length: 5 }, (_, index) => ({ artistName: `Artist ${index}`, trackName: `Song ${index}`, msPlayed: 10_000 }))));
  const recommendations = Array.from({ length: 40 }, (_, index) => ({
    mbid: `candidate-${index}`, artist: `Candidate Artist ${index}`, title: `Candidate ${index}`,
    score: 1 - index / 100, slateScore: 0.9 - index / 100, features: Array(17).fill(0), pickedFrom: [],
    tasteHeadIndex: index % 4, seedSupport: 3, popularityPercentile: 0.1, liftScore: 0.8,
  }));

  await runTasteLiftPlaylist(input, output, {
    resolveTasteSeeds: async (inputs): Promise<ResolvedTasteSeed[]> => inputs.map((item) => ({ input: item, status: "resolved", source: "text", diagnostic: { status: "retrieval_unavailable", code: "no_exact_mbid", message: "text" } })),
    recommendTasteSeeds: async () => ({ candidateCount: 500, recommendations }),
    lookupSpotify: async (mbid) => mbid === "candidate-0" ? "4uLU6hMCjMI75M1A2tKUQC" : null,
  });

  const payload = JSON.parse(await readFile(output, "utf8"));
  expect(payload.seedCount).toBe(5);
  expect(payload.candidateCount).toBe(500);
  expect(payload.recommendations).toHaveLength(40);
  expect(payload.recommendations[0]).toEqual(expect.objectContaining({
    score: 0.9,
    strongestTasteHead: 0,
    seedSupport: 3,
    popularityPercentile: 0.1,
    noveltyLiftScore: 0.8,
    spotifyLink: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
  }));
  expect(payload.recommendations[1].spotifyLink).toContain("open.spotify.com/search/");
});
