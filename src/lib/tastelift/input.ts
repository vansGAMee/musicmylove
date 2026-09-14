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
  if (!object(value) || !Array.isArray(value.songs)) {
    throw new TasteInputError([{ path: "songs", code: "invalid_type", message: "songs must be an array" }]);
  }
  if (value.songs.length < 5 || value.songs.length > 500) {
    throw new TasteInputError([{ path: "songs", code: "out_of_range", message: "songs must contain between 5 and 500 entries" }]);
  }

  const issues: TasteInputIssue[] = [];
  const seeds: TasteSeedInput[] = [];
  const seen = new Set<string>();
  for (const [index, row] of value.songs.entries()) {
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
