import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { lookupSpotify, spotifySearch } from "../src/lib/listenbrainz";
import { parseTasteInput } from "../src/lib/tastelift/input";
import { recommendTasteSeeds } from "../src/lib/tastelift/pipeline";
import { createTasteResolverAdapters, resolveTasteSeeds, type ResolvedTasteSeed } from "../src/lib/tastelift/resolver";

interface PlaylistDependencies {
  resolveTasteSeeds?: (inputs: ReturnType<typeof parseTasteInput>) => Promise<ResolvedTasteSeed[]>;
  recommendTasteSeeds?: typeof recommendTasteSeeds;
  lookupSpotify?: typeof lookupSpotify;
}

async function concurrentMap<T, U>(items: readonly T[], concurrency: number, transform: (item: T) => Promise<U>): Promise<U[]> {
  const output = Array<U>(items.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      output[index] = await transform(items[index]!);
    }
  }));
  return output;
}

export async function runTasteLiftPlaylist(inputPath: string, outputPath: string, dependencies: PlaylistDependencies = {}): Promise<void> {
  const value: unknown = JSON.parse(await readFile(inputPath, "utf8"));
  const inputs = parseTasteInput(value);
  const seeds = await (dependencies.resolveTasteSeeds ?? ((items) => resolveTasteSeeds(items, createTasteResolverAdapters())))(inputs);
  const failures = seeds.filter((seed) => seed.status === "unresolved");
  if (failures.length) throw new Error(`Unable to resolve ${failures.length} seed(s): ${JSON.stringify(failures)}`);
  const pipeline = await (dependencies.recommendTasteSeeds ?? recommendTasteSeeds)(seeds);
  if (pipeline.recommendations.length !== 40) throw new Error(`TasteLift returned ${pipeline.recommendations.length} recommendations instead of 40`);
  const spotify = dependencies.lookupSpotify ?? lookupSpotify;
  const recommendations = await concurrentMap(pipeline.recommendations, 6, async (track) => {
    let spotifyId: string | null = null;
    try { spotifyId = await spotify(track.mbid); } catch { /* stable search fallback */ }
    return {
      mbid: track.mbid,
      artist: track.artist,
      title: track.title,
      ...(track.release ? { release: track.release } : {}),
      score: track.slateScore ?? track.score,
      strongestTasteHead: track.tasteHeadIndex ?? 0,
      seedSupport: track.seedSupport ?? 0,
      popularityPercentile: track.popularityPercentile ?? 0,
      noveltyLiftScore: track.liftScore ?? 0,
      spotifyLink: spotifyId ? `https://open.spotify.com/track/${spotifyId}` : spotifySearch(track),
    };
  });
  const payload = { seedCount: seeds.length, candidateCount: pipeline.candidateCount, recommendations };
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp-${process.pid}`;
  try {
    await writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`);
    await rename(temporary, outputPath);
  } finally {
    await unlink(temporary).catch(() => undefined);
  }
}

async function main() {
  const input = process.argv[2];
  if (!input) throw new Error("Usage: npx tsx scripts/tastelift-playlist.ts INPUT.json [OUTPUT.json]");
  const output = process.argv[3] ?? "reports/my-playlist.json";
  await runTasteLiftPlaylist(resolve(input), resolve(output));
  console.log(`Wrote 40 TasteLift recommendations to ${output}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main();
