import tasteLiftArtifact from "../../ml/tastelift-model.json";
import { describe, expect, test } from "vitest";
import { rankTasteCandidatePool, TasteLiftPoolInputError } from "../../src/lib/ranking";
import { TasteLiftModel } from "../../src/lib/tastelift/model";
import type { TasteCandidatePool } from "../../src/lib/tastelift/retrieval";

const resolvedSeeds = [
  { input: { artist: "Seed A", title: "One" }, status: "resolved" as const, source: "listenbrainz" as const, track: { mbid: "seed-a", artist: "Seed A", title: "One" } },
  { input: { artist: "Text Artist", title: "OOV 🐻" }, status: "resolved" as const, source: "text" as const, diagnostic: { status: "retrieval_unavailable" as const, code: "no_exact_mbid" as const, message: "no MBID" } },
  ...["b", "c", "d"].map((suffix) => ({ input: { artist: `Seed ${suffix}`, title: suffix }, status: "resolved" as const, source: "listenbrainz" as const, track: { mbid: `seed-${suffix}`, artist: `Seed ${suffix}`, title: suffix } })),
];

const pool: TasteCandidatePool = {
  seeds: resolvedSeeds,
  candidates: [
    {
      mbid: "candidate-a", artist: "Candidate", title: "A", alternateMbids: ["candidate-a", "candidate-a-alt"], support: 2, retrievalScore: 0.1,
      evidence: [
        { source: "listenbrainz", seedIndex: 0, seedMbid: "seed-a", recordingMbid: "candidate-a", rank: 1, rawScore: 100 },
        { source: "listenbrainz", seedIndex: 0, seedMbid: "seed-a", recordingMbid: "candidate-a-alt", rank: 2, rawScore: 90 },
        { source: "listenbrainz", seedIndex: 2, seedMbid: "seed-b", recordingMbid: "candidate-a-alt", rank: 3, rawScore: 50 },
      ],
    },
  ],
};

const model = TasteLiftModel.fromArtifact(tasteLiftArtifact);

describe("TasteCandidatePool ranking", () => {
  test("uses text/OOV seeds and preserves every alternate-MBID retrieval row", () => {
    const [ranked] = rankTasteCandidatePool(pool, "rrf", model);
    expect(ranked!.features).toHaveLength(17);
    expect(ranked!.seedSupport).toBe(2);
    expect(ranked!.seedSupportEvidence).toHaveLength(3);
    expect(ranked!.seedSupportEvidence?.map((item) => item.recordingMbid)).toContain("candidate-a-alt");
    expect(ranked!.score).toBeCloseTo(ranked!.liftScore!);
  });

  test("does not feed neural catalog evidence into residual features trained on external ranks", () => {
    const catalogOnly: TasteCandidatePool = {
      seeds: pool.seeds,
      candidates: [{
        mbid: "catalog-only", artist: "Catalog", title: "Discovery", alternateMbids: ["catalog-only"], support: 2, retrievalScore: 1,
        evidence: [0, 1].map((seedIndex) => ({ source: "tastelift-catalog" as const, seedIndex, seedMbid: `seed-${seedIndex}`, recordingMbid: "catalog-only", rank: 1, rawScore: 0.9 })),
      }],
    };

    const [ranked] = rankTasteCandidatePool(catalogOnly, "rrf", model);

    expect(ranked?.features).toEqual(Array(17).fill(0));
    expect(ranked?.seedSupport).toBe(2);
  });

  test("returns an empty ranked pool without fabricating a candidate", () => {
    expect(rankTasteCandidatePool({ ...pool, candidates: [] }, "rrf", model)).toEqual([]);
  });

  test("serves all five hundred resolved seeds without truncating the model input", () => {
    const fiveHundred = Array.from({ length: 500 }, (_, index) => ({
      input: { artist: `Artist ${index}`, title: `Title ${index}` }, status: "resolved" as const,
      source: "listenbrainz" as const, track: { mbid: `seed-${index}`, artist: `Artist ${index}`, title: `Title ${index}` },
    }));
    expect(rankTasteCandidatePool({ seeds: fiveHundred, candidates: [] }, "rrf", model)).toEqual([]);
  });

  test("reports unresolved seed outcomes structurally", () => {
    const unresolved = { input: { artist: "Broken", title: "Seed" }, status: "unresolved" as const, error: { code: "upstream_error" as const, message: "unavailable" } };
    expect(() => rankTasteCandidatePool({ ...pool, seeds: [...pool.seeds, unresolved] }, "rrf", model)).toThrow(TasteLiftPoolInputError);
    try {
      rankTasteCandidatePool({ ...pool, seeds: [...pool.seeds, unresolved] }, "rrf", model);
    } catch (error) {
      expect((error as TasteLiftPoolInputError).failures).toEqual([unresolved]);
    }
  });
});
