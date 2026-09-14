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
export function rankSearchResults(query: string, tracks: Track[]): Track[] {
  const normQ = query.normalize("NFKC").toLowerCase().trim().replace(/\s+/gu, " ");
  if (!normQ) return tracks;
  const qTokens = normQ.split(" ").filter(Boolean);

  return tracks.map((track, originalRank) => {
    const normA = track.artist.normalize("NFKC").toLowerCase().trim().replace(/\s+/gu, " ");
    const normT = track.title.normalize("NFKC").toLowerCase().trim().replace(/\s+/gu, " ");

    let score = 0;

    // Exact full matches
    if (normA === normQ) score += 120;
    if (normT === normQ) score += 90;
    if (`${normA} ${normT}` === normQ || `${normT} ${normA}` === normQ) score += 150;

    // Token coverage in artist
    const qInArtist = qTokens.filter((tok) => normA.includes(tok)).length;
    const artistRatio = qInArtist / qTokens.length;
    if (artistRatio === 1) score += 60;
    else if (artistRatio >= 0.6) score += 35;

    // Token coverage in title
    const qInTitle = qTokens.filter((tok) => normT.includes(tok)).length;
    const titleRatio = qInTitle / qTokens.length;
    if (titleRatio === 1) score += 40;
    else if (titleRatio >= 0.5) score += 20;

    // Prefix match bonus
    if (normA.startsWith(normQ)) score += 50;
    if (normT.startsWith(normQ)) score += 30;

    // Both artist and title match tokens in query
    if (qTokens.some((tok) => normA.includes(tok)) && qTokens.some((tok) => normT.includes(tok))) {
      score += 40;
    }

    // Downrank covers/flips/karaoke unless query asked for them
    const isCoverOrRemix = normT.includes("cover") || normT.includes("tribute") || normT.includes("karaoke") || normT.includes("flip");
    if (!normQ.includes("cover") && !normQ.includes("tribute") && !normQ.includes("flip") && isCoverOrRemix) {
      score -= 40;
    }

    // Upstream rank preservation
    score += (tracks.length - originalRank) * 2;

    return { track, score, originalRank };
  }).sort((a, b) => b.score - a.score || a.originalRank - b.originalRank)
    .map((item) => item.track);
}

async function json(url: string, signal?: AbortSignal) { return (await fetchWithRetry(url, { signal })).json(); }
export async function searchRecordings(query: string, signal?: AbortSignal) {
  const tracks = parseSearch(await json(`${LABS}/recording-search/json?query=${encodeURIComponent(query)}`, signal));
  return rankSearchResults(query, tracks);
}
/** Searches ListenBrainz with the same artist/title query used by TasteLift seed resolution. */
export async function searchRecordingsForSeed(query: string, signal?: AbortSignal) { return searchRecordings(query, signal); }
export async function fetchSimilar(mbid: string, signal?: AbortSignal) { return parseSimilar(await json(`${LABS}/similar-recordings/json?recording_mbids=${encodeURIComponent(mbid)}&algorithm=${encodeURIComponent(SIMILARITY_ALGORITHM)}`, signal)); }
/** Parses the Labs batched similarity response while keeping upstream rank order per reference MBID. */
export function parseSimilarBatch(value: unknown): Record<string, SimilarTrack[]> {
  if (!Array.isArray(value)) throw new Error("Malformed batched similarity response");
  const lists: Record<string, SimilarTrack[]> = {};
  for (const row of value) {
    if (!object(row) || typeof row.reference_mbid !== "string" || typeof row.recording_mbid !== "string" || typeof row.recording_name !== "string" || typeof row.artist_credit_name !== "string" || typeof row.score !== "number") continue;
    (lists[row.reference_mbid] ??= []).push({
      mbid: row.recording_mbid,
      title: row.recording_name,
      artist: row.artist_credit_name,
      ...(typeof row.release_name === "string" ? { release: row.release_name } : {}),
      score: row.score,
    });
  }
  return lists;
}
/** Fetches many similarity lists in one bounded-retry ListenBrainz request. */
export async function fetchSimilarBatch(mbids: readonly string[], signal?: AbortSignal): Promise<Record<string, SimilarTrack[]>> {
  if (mbids.length === 0) return {};
  const response = await fetchWithRetry(`${LABS}/similar-recordings/json`, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([{ recording_mbids: [...new Set(mbids)].sort((left, right) => left.localeCompare(right)), algorithm: SIMILARITY_ALGORITHM }]),
  });
  return parseSimilarBatch(await response.json());
}
export async function lookupSpotify(mbid: string, signal?: AbortSignal) { return parseSpotify(await json(`${LABS}/spotify-id-from-mbid/json?recording_mbid=${encodeURIComponent(mbid)}`, signal)); }
export function spotifySearch(track: Track) { return `https://open.spotify.com/search/${encodeURIComponent(`${track.artist} ${track.title}`)}`; }
