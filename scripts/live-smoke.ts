import { fetchSimilar, lookupSpotify, searchRecordings, spotifySearch } from "../src/lib/listenbrainz";
import { createRecommendations } from "../src/lib/recommend";
import type { SeedTrack, SimilarTrack } from "../src/lib/types";

const queries = ["Roads Portishead", "Teardrop Massive Attack", "Everything in Its Right Place Radiohead", "Hyperballad Björk", "Midnight City M83"];

async function main() {
  const seeds: SeedTrack[] = [];
  for (const query of queries) {
    const result = (await searchRecordings(query))[0];
    if (!result) throw new Error(`No live search result for ${query}`);
    seeds.push(result);
  }
  const lists: Record<string, SimilarTrack[]> = {};
  const failures: string[] = [];
  for (const seed of seeds) {
    try { lists[seed.mbid] = await fetchSimilar(seed.mbid); }
    catch { lists[seed.mbid] = []; failures.push(seed.mbid); }
  }
  const recommendations = createRecommendations(seeds, lists);
  const reordered = createRecommendations([...seeds].reverse(), lists);
  const actions: string[] = [];
  for (const track of recommendations) {
    const id = await lookupSpotify(track.mbid).catch(() => null);
    actions.push(id ? `https://open.spotify.com/track/${id}` : spotifySearch(track));
  }
  const artists = new Map<string, number>();
  recommendations.forEach((track) => artists.set(track.artist.toLowerCase(), (artists.get(track.artist.toLowerCase()) ?? 0) + 1));
  const summary = {
    searched: seeds.map(({ mbid, title, artist }) => ({ mbid, title, artist })),
    similarity_failures: failures,
    candidate_pool_size: new Set(Object.values(lists).flat().map((track) => track.mbid)).size,
    result_count: recommendations.length,
    no_seeds: recommendations.every((track) => !seeds.some((seed) => seed.mbid === track.mbid)),
    no_duplicates: new Set(recommendations.map((track) => track.mbid)).size === recommendations.length,
    max_artist_count: Math.max(0, ...artists.values()),
    spotify_actions: actions.length,
    permutation_identical: recommendations.map((track) => track.mbid).join() === reordered.map((track) => track.mbid).join(),
    playlist: recommendations.map((track, index) => ({ position: index + 1, title: track.title, artist: track.artist, mbid: track.mbid, spotify: actions[index] })),
  };
  console.log(JSON.stringify(summary, null, 2));
  if (summary.result_count < 20 || !summary.no_seeds || !summary.no_duplicates || summary.max_artist_count > 2 || summary.spotify_actions !== summary.result_count || !summary.permutation_identical) process.exitCode = 1;
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
