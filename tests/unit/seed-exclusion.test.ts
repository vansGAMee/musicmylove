import { describe, expect, test } from "vitest";
import { recommendTasteSeeds } from "../../src/lib/tastelift/pipeline";
import type { ResolvedTasteSeed } from "../../src/lib/tastelift/resolver";
import type { TasteRetrievalAdapters } from "../../src/lib/tastelift/retrieval";
import type { TasteLiftCatalogArtifact } from "../../src/lib/tastelift/catalog";

describe("Strict zero-seed recommendation guarantee", () => {
  test("never recommends any song present in user seeds by mbid or title/artist", async () => {
    const seedTrack1 = {
      mbid: "seed-mbid-001",
      title: "Seed Track One",
      artist: "Seed Artist",
    };
    const seedTrack2 = {
      mbid: "seed-mbid-002",
      title: "Another Seed",
      artist: "Different Artist",
    };

    const seeds: ResolvedTasteSeed[] = Array.from({ length: 5 }, (_, i) => ({
      status: "resolved" as const,
      track: {
        mbid: `seed-mbid-00${i + 1}`,
        title: i === 0 ? "Seed Track One" : `Seed Track ${i + 1}`,
        artist: i === 0 ? "Seed Artist" : `Artist ${i + 1}`,
      },
      source: "listenbrainz" as const,
      input: {
        artist: i === 0 ? "Seed Artist" : `Artist ${i + 1}`,
        title: i === 0 ? "Seed Track One" : `Seed Track ${i + 1}`,
      },
    }));

    const mockAdapters: TasteRetrievalAdapters = {
      fetchSimilarBatch: async (mbids: readonly string[]) => {
        const simList = [
          // Includes one of the seed tracks in the candidate pool!
          {
            mbid: "seed-mbid-001",
            title: "Seed Track One",
            artist: "Seed Artist",
            score: 0.99,
          },
          // Another track with same title and artist but different MBID (remaster/duplicate)
          {
            mbid: "remaster-mbid-001",
            title: "Seed Track One (Remastered)",
            artist: "Seed Artist",
            score: 0.98,
          },
          // A genuine recommendation
          {
            mbid: "genuine-rec-001",
            title: "Genuine Recommendation",
            artist: "Novel Artist",
            score: 0.85,
          },
        ];
        return Object.fromEntries(mbids.map((id) => [id, simList]));
      },
    };

    const result = await recommendTasteSeeds(seeds, {
      retrievalAdapters: mockAdapters,
      slateLimit: 10,
    });

    // Seed track 1 and its remaster must NOT be in the recommendations
    expect(result.recommendations.some((r) => r.mbid === "seed-mbid-001")).toBe(false);
    expect(result.recommendations.some((r) => r.title.toLowerCase().includes("seed track one"))).toBe(false);
  });
});
