import { fetchWithRetry } from "./http";
import type { SimilarTrack, Track } from "./types";

const LABS = "https://labs.api.listenbrainz.org";
export const SIMILARITY_ALGORITHM = "session_based_days_7500_session_300_contribution_5_threshold_15_limit_50_skip_30_top_n_listeners_1000";
const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

export function parseSearch(value: unknown): Track[] {
  if (!Array.isArray(value)) throw new Error("Malformed search response");
  return value.flatMap((row) => object(row) && typeof row.recording_mbid === "string" && typeof row.recording_name === "string" && typeof row.artist_credit_name === "string" ? [{ mbid: row.recording_mbid, title: row.recording_name, artist: row.artist_credit_name, release: typeof row.release_name === "string" ? row.release_name : undefined }] : []);
}
export function parseSimilar(value: unknown): SimilarTrack[] {
  if (!Array.isArray(value)) throw new Error("Malformed similarity response");
  return value.flatMap((row) => object(row) && typeof row.recording_mbid === "string" && typeof row.recording_name === "string" && typeof row.artist_credit_name === "string" && typeof row.score === "number" ? [{ mbid: row.recording_mbid, title: row.recording_name, artist: row.artist_credit_name, release: typeof row.release_name === "string" ? row.release_name : undefined, score: row.score }] : []);
}
export function parseSpotify(value: unknown): string | null {
  if (!Array.isArray(value)) throw new Error("Malformed Spotify response");
  const ids = object(value[0]) ? value[0].spotify_track_ids : null;
  return Array.isArray(ids) && typeof ids[0] === "string" ? ids[0] : null;
}
async function json(url: string, signal?: AbortSignal) { return (await fetchWithRetry(url, { signal })).json(); }
export async function searchRecordings(query: string, signal?: AbortSignal) { return parseSearch(await json(`${LABS}/recording-search/json?query=${encodeURIComponent(query)}`, signal)); }
export async function fetchSimilar(mbid: string, signal?: AbortSignal) { return parseSimilar(await json(`${LABS}/similar-recordings/json?recording_mbids=${encodeURIComponent(mbid)}&algorithm=${encodeURIComponent(SIMILARITY_ALGORITHM)}`, signal)); }
export async function lookupSpotify(mbid: string, signal?: AbortSignal) { return parseSpotify(await json(`${LABS}/spotify-id-from-mbid/json?recording_mbid=${encodeURIComponent(mbid)}`, signal)); }
export function spotifySearch(track: Track) { return `https://open.spotify.com/search/${encodeURIComponent(`${track.artist} ${track.title}`)}`; }
