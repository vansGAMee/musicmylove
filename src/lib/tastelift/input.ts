export interface TasteSeedInput {
  artist: string;
  title: string;
  spotifyUrl?: string;
  spotifyId?: string;
}

export interface TasteInputIssue {
  path: string;
  code: "invalid_type" | "invalid_value" | "out_of_range" | "duplicate";
  message: string;
}

export class TasteInputError extends Error {
  readonly issues: readonly TasteInputIssue[];

  constructor(issues: readonly TasteInputIssue[]) {
    super(issues.map((issue) => issue.message).join(" "));
    this.name = "TasteInputError";
    this.issues = issues;
  }
}

const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;
const normalizeText = (value: string) => value.normalize("NFKC").replace(/\s+/gu, " ").trim();

function spotifyHistorySongs(value: readonly unknown[]): Record<string, unknown>[] {
  const aggregated = new Map<string, { row: Record<string, unknown>; playTime: number; plays: number; first: number }>();
  value.forEach((entry, index) => {
    if (!object(entry)) return;
    const artistValue = entry.master_metadata_album_artist_name ?? entry.artistName ?? entry.artist;
    const titleValue = entry.master_metadata_track_name ?? entry.trackName ?? entry.title;
    if (typeof artistValue !== "string" || typeof titleValue !== "string" || !normalizeText(artistValue) || !normalizeText(titleValue)) return;
    const artist = normalizeText(artistValue);
    const title = normalizeText(titleValue);
    const uri = typeof entry.spotify_track_uri === "string" ? entry.spotify_track_uri : "";
    const spotifyId = /^spotify:track:([A-Za-z0-9]{22})$/u.exec(uri)?.[1];
    const spotifyUrl = spotifyId ? `https://open.spotify.com/track/${spotifyId}` : entry.spotify_url;
    const row = { artist, title, ...(typeof spotifyUrl === "string" ? { spotify_url: spotifyUrl } : {}) };
    const key = tasteSeedKey(row);
    const duration = entry.ms_played ?? entry.msPlayed;
    const playTime = typeof duration === "number" && Number.isFinite(duration) ? Math.max(0, duration) : 1;
    const previous = aggregated.get(key);
    if (previous) {
      previous.playTime += playTime;
      previous.plays += 1;
      if (!previous.row.spotify_url && row.spotify_url) previous.row = row;
    } else {
      aggregated.set(key, { row, playTime, plays: 1, first: index });
    }
  });
  return [...aggregated.values()]
    .sort((left, right) => right.playTime - left.playTime || right.plays - left.plays || left.first - right.first)
    .slice(0, 500)
    .map((item) => item.row);
}

function parseSpotifyUrl(value: unknown, path: string): Pick<TasteSeedInput, "spotifyUrl" | "spotifyId"> | TasteInputIssue | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") return { path, code: "invalid_type", message: `${path} must be a Spotify track URL` };
  const spotifyUrl = value.trim();
  try {
    const url = new URL(spotifyUrl);
    const parts = url.pathname.split("/").filter(Boolean);
    const trackIndex = parts.indexOf("track");
    const spotifyId = trackIndex < 0 ? undefined : parts[trackIndex + 1];
    if (!/^https:$/u.test(url.protocol) || !/^(www\.)?open\.spotify\.com$/iu.test(url.hostname) || !spotifyId || !/^[A-Za-z0-9]{22}$/u.test(spotifyId)) throw new Error("invalid");
    return { spotifyUrl, spotifyId };
  } catch {
    return { path, code: "invalid_value", message: `${path} must be a valid Spotify track URL` };
  }
}

export function tasteSeedKey(seed: Pick<TasteSeedInput, "artist" | "title">): string {
  return `${seed.artist.normalize("NFKC").toLocaleLowerCase("en-US")}\u0000${seed.title.normalize("NFKC").toLocaleLowerCase("en-US")}`;
}

/** Validates canonical JSON input and removes only exact normalized artist/title duplicates in input order. */
export function parseTasteInput(value: unknown): TasteSeedInput[] {
  const fromHistory = Array.isArray(value);
  const songs = fromHistory ? spotifyHistorySongs(value) : object(value) && Array.isArray(value.songs) ? value.songs : undefined;
  if (!songs) {
    throw new TasteInputError([{ path: "songs", code: "invalid_type", message: "songs must be an array" }]);
  }
  if (songs.length < 5 || (!fromHistory && songs.length > 500)) {
    throw new TasteInputError([{ path: "songs", code: "out_of_range", message: "songs must contain between 5 and 500 entries" }]);
  }

  const issues: TasteInputIssue[] = [];
  const seeds: TasteSeedInput[] = [];
  const seen = new Set<string>();
  for (const [index, row] of songs.entries()) {
    const path = `songs[${index}]`;
    if (!object(row)) {
      issues.push({ path, code: "invalid_type", message: `${path} must be an object` });
      continue;
    }
    if (typeof row.artist !== "string" || !normalizeText(row.artist)) {
      issues.push({ path: `${path}.artist`, code: "invalid_value", message: `${path}.artist must be a non-empty string` });
      continue;
    }
    if (typeof row.title !== "string" || !normalizeText(row.title)) {
      issues.push({ path: `${path}.title`, code: "invalid_value", message: `${path}.title must be a non-empty string` });
      continue;
    }
    const spotify = parseSpotifyUrl(row.spotify_url, `${path}.spotify_url`);
    if (spotify && "code" in spotify) {
      issues.push(spotify);
      continue;
    }
    const seed: TasteSeedInput = { artist: normalizeText(row.artist), title: normalizeText(row.title), ...spotify };
    const key = tasteSeedKey(seed);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    seeds.push(seed);
  }
  if (issues.length > 0) throw new TasteInputError(issues);
  if (seeds.length < 5) {
    throw new TasteInputError([{ path: "songs", code: "duplicate", message: "at least 5 distinct songs are required after duplicate removal" }]);
  }
  return seeds;
}
