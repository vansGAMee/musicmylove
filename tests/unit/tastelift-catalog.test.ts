import modelArtifact from "../../ml/tastelift-model.json";
import { describe, expect, test } from "vitest";
import { retrieveTasteCatalogCandidates, retrieveTasteHistoryCandidates, type TasteLiftCatalogArtifact } from "../../src/lib/tastelift/catalog";
import { TasteLiftModel, type TasteLiftArtifact, type TasteLiftTrack } from "../../src/lib/tastelift/model";
import { recommendTasteSeeds } from "../../src/lib/tastelift/pipeline";
import type { ResolvedTasteSeed } from "../../src/lib/tastelift/resolver";

const model = TasteLiftModel.fromArtifact(modelArtifact as TasteLiftArtifact);
const seedTracks: TasteLiftTrack[] = Array.from({ length: 5 }, (_, index) => ({
  mbid: `seed-${index}`,
  artist: `Seed Artist ${index}`,
  title: `Seed Song ${index}`,
}));
const seeds: ResolvedTasteSeed[] = seedTracks.map((track) => ({ input: { artist: track.artist, title: track.title }, status: "resolved", source: "listenbrainz", track: track as Required<TasteLiftTrack> }));

function catalog(tracks: TasteLiftCatalogArtifact["tracks"], vectors: readonly number[][]): TasteLiftCatalogArtifact {
  const bytes = Uint8Array.from(vectors.flatMap((row) => row.map((value) => value < 0 ? value + 256 : value)));
  return {
    format: "tastelift-catalog-v1",
    dimensions: 24,
    source: "listenbrainz-train-histories",
    quantization: { type: "symmetric-int8", scale: 127, encoding: "base64-row-major" },
    tracks,
    vectors: Buffer.from(bytes).toString("base64"),
  };
}

describe("TasteLift train-only catalog retrieval", () => {
  test("uses learned vectors while excluding seeds and duplicate recording versions", () => {
    const head = model.encodeSet(seedTracks)[0]!;
    const quantized = (vector: readonly number[]) => vector.map((value) => Math.max(-127, Math.min(127, Math.round(value * 127))));
    const artifact = catalog([
      { mbid: "seed-0", artist: "Seed Artist 0", title: "Seed Song 0", popularityPercentile: 0.1 },
      { mbid: "best", artist: "Deep Artist", title: "Hidden Song", popularityPercentile: 0.05 },
      { mbid: "version-a", artist: "Version Artist", title: "One Song", popularityPercentile: 0.2 },
      { mbid: "version-b", artist: " version artist ", title: "One Song (2011 Remaster)", popularityPercentile: 0.2 },
      { mbid: "opposite", artist: "Other Artist", title: "Other Song", popularityPercentile: 0.1 },
    ], [model.encodeTrack(seedTracks[0]!), head, head, head, head.map((value) => -value)].map(quantized));

    const candidates = retrieveTasteCatalogCandidates(seeds, artifact, model, 10);

    expect(candidates.map((track) => track.mbid)).not.toContain("seed-0");
    expect(candidates.filter((track) => track.artist.trim().toLowerCase() === "version artist")).toHaveLength(1);
    expect(candidates[0]?.mbid).toBe("best");
    expect(candidates[0]?.evidence.some((item) => item.source === "tastelift-catalog")).toBe(true);
    expect(candidates[0]?.support).toBeGreaterThan(0);
  });

  test("passes the complete five-hundred-seed set to the encoder", () => {
    const fiveHundred = Array.from({ length: 500 }, (_, index) => ({ input: { artist: `Artist ${index}`, title: `Song ${index}` }, status: "resolved" as const, source: "text" as const, diagnostic: { status: "retrieval_unavailable" as const, code: "no_exact_mbid" as const, message: "text only" } }));
    const artifact = catalog([{ mbid: "candidate", artist: "Candidate", title: "Track", popularityPercentile: 0.1 }], [Array<number>(24).fill(0)]);

    const [candidate] = retrieveTasteCatalogCandidates(fiveHundred, artifact, model, 10);
    expect(candidate).toBeDefined();
    expect(candidate?.support).toBe(8);
    expect(candidate?.evidence.map((item) => item.seedIndex)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  test("feeds neural-only catalog discoveries through the real rank-and-slate pipeline", async () => {
    const vector = model.encodeSet(seedTracks)[0]!.map((value) => Math.max(-127, Math.min(127, Math.round(value * 127))));
    const tracks = Array.from({ length: 45 }, (_, index) => ({ mbid: `neural-${index}`, artist: `Neural Artist ${index}`, title: `Discovery ${index}`, popularityPercentile: 0.1 }));
    const artifact = catalog(tracks, tracks.map(() => vector));

    const result = await recommendTasteSeeds(seeds, {
      model,
      catalog: artifact,
      retrievalAdapters: { fetchSimilarBatch: async () => ({}) },
    });

    expect(result.candidateCount).toBe(45);
    expect(result.recommendations).toHaveLength(40);
    expect(result.recommendations.every((track) => (track.seedSupport ?? 0) > 0 && track.liftScore !== undefined)).toBe(true);
  });

  test("retrieves co-listened tracks only from anonymous train histories", () => {
    const tracks = [
      { mbid: "seed-0", artist: "Seed Artist 0", title: "Seed Song 0", popularityPercentile: 0.2 },
      { mbid: "co-a", artist: "Co Artist", title: "Hidden A", popularityPercentile: 0.1 },
      { mbid: "co-b", artist: "Other Co Artist", title: "Hidden B", popularityPercentile: 0.1 },
    ];
    const artifact = { ...catalog(tracks, tracks.map(() => Array<number>(24).fill(0))), histories: [[0, 1], [0, 1, 2]] };

    const candidates = retrieveTasteHistoryCandidates(seeds, artifact, 10);

    expect(candidates.map((track) => track.mbid)).toEqual(["co-a", "co-b"]);
    expect(candidates[0]?.support).toBe(1);
    expect(candidates[0]?.evidence[0]?.source).toBe("listenbrainz-history");
  });
});
