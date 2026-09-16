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

function extractArtistAndTitle(entry: unknown): { artist?: string; title?: string; spotifyUrl?: string; duration?: number } {
  if (typeof entry === "string") {
    const cleanEntry = entry
      .replace(/^\s*(?:\[\d+\]|\d+[\.\)\-:]|\d+\s+[-–—])\s*/u, "")
      .replace(/^\s*[•\*\-]\s+/u, "")
      .trim();
    const match = /^(.+?)\s*(?:[-–—:]|\s+by\s+)\s*(.+)$/iu.exec(cleanEntry);
    if (match) {
      return { artist: normalizeText(match[1]), title: normalizeText(match[2]) };
    }
    return {};
  }
  if (!object(entry)) return {};

  const trackObj = (
    object(entry.attributes)
      ? entry.attributes
      : object(entry.track)
      ? entry.track
      : entry
  ) as Record<string, unknown>;

  let artist: string | undefined;
  if (typeof trackObj.master_metadata_album_artist_name === "string") artist = trackObj.master_metadata_album_artist_name;
  else if (typeof trackObj.artistName === "string") artist = trackObj.artistName;
  else if (typeof trackObj.artist === "string") artist = trackObj.artist;
  else if (object(trackObj.artist) && typeof trackObj.artist.name === "string") artist = trackObj.artist.name;
  else if (object(trackObj.artist) && typeof trackObj.artist["#text"] === "string") artist = trackObj.artist["#text"] as string;
  else if (typeof trackObj["Artist Name(s)"] === "string") artist = trackObj["Artist Name(s)"] as string;
  else if (typeof trackObj["Artist Name"] === "string") artist = trackObj["Artist Name"] as string;
  else if (typeof trackObj.Artist === "string") artist = trackObj.Artist as string;
  else if (typeof trackObj.artist_name === "string") artist = trackObj.artist_name as string;
  else if (Array.isArray(trackObj.artists) && trackObj.artists.length > 0) {
    const first = trackObj.artists[0];
    if (typeof first === "string") artist = first;
    else if (object(first) && typeof first.name === "string") artist = first.name;
    else if (object(first) && object(first.artist) && typeof (first.artist as Record<string, unknown>).name === "string") artist = (first.artist as Record<string, unknown>).name as string;
  } else if (typeof trackObj.author === "string") artist = trackObj.author as string;
  else if (object(trackObj.author) && typeof trackObj.author.name === "string") artist = trackObj.author.name;
  else if (typeof trackObj.performer === "string") artist = trackObj.performer as string;
  else if (object(trackObj.performer) && typeof trackObj.performer.name === "string") artist = trackObj.performer.name;
  else if (typeof trackObj.byArtist === "string") artist = trackObj.byArtist;
  else if (object(trackObj.byArtist) && typeof trackObj.byArtist.name === "string") artist = trackObj.byArtist.name;
  else if (typeof trackObj["Исполнитель"] === "string") artist = trackObj["Исполнитель"] as string;
  else if (typeof trackObj["Артист"] === "string") artist = trackObj["Артист"] as string;
  else if (typeof trackObj["Автор"] === "string") artist = trackObj["Автор"] as string;

  let title: string | undefined;
  if (typeof trackObj.master_metadata_track_name === "string") title = trackObj.master_metadata_track_name;
  else if (typeof trackObj.trackName === "string") title = trackObj.trackName;
  else if (typeof trackObj.title === "string") title = trackObj.title;
  else if (typeof trackObj.track === "string") title = trackObj.track;
  else if (typeof trackObj.Track === "string") title = trackObj.Track as string;
  else if (typeof trackObj["Track Name"] === "string") title = trackObj["Track Name"] as string;
  else if (typeof trackObj.name === "string") title = trackObj.name as string;
  else if (typeof trackObj.Title === "string") title = trackObj.Title as string;
  else if (typeof trackObj.song === "string") title = trackObj.song as string;
  else if (typeof trackObj.Song === "string") title = trackObj.Song as string;
  else if (typeof trackObj.track_name === "string") title = trackObj.track_name as string;
  else if (typeof trackObj["Название"] === "string") title = trackObj["Название"] as string;
  else if (typeof trackObj["Трек"] === "string") title = trackObj["Трек"] as string;
  else if (typeof trackObj["Песня"] === "string") title = trackObj["Песня"] as string;

  let spotifyUrl: string | undefined;
  const uri = typeof trackObj.spotify_track_uri === "string" ? trackObj.spotify_track_uri : typeof trackObj.uri === "string" ? trackObj.uri : "";
  const spotifyIdFromUri = /^spotify:track:([A-Za-z0-9]{22})$/u.exec(uri)?.[1];
  const spotifyIdCol = typeof trackObj["Spotify Track Id"] === "string" ? trackObj["Spotify Track Id"] as string : typeof trackObj.id === "string" ? trackObj.id : undefined;

  if (spotifyIdFromUri) {
    spotifyUrl = `https://open.spotify.com/track/${spotifyIdFromUri}`;
  } else if (spotifyIdCol && /^[A-Za-z0-9]{22}$/u.test(spotifyIdCol)) {
    spotifyUrl = `https://open.spotify.com/track/${spotifyIdCol}`;
  } else if (typeof trackObj.spotify_url === "string") {
    spotifyUrl = trackObj.spotify_url;
  } else if (object(trackObj.external_urls) && typeof trackObj.external_urls.spotify === "string") {
    spotifyUrl = trackObj.external_urls.spotify;
  }

  const durationVal = entry.ms_played ?? entry.msPlayed ?? trackObj["Duration (ms)"] ?? trackObj.duration_ms;
  const duration = typeof durationVal === "number" && Number.isFinite(durationVal) ? Math.max(0, durationVal) : undefined;

  return {
    ...(artist && normalizeText(artist) ? { artist: normalizeText(artist) } : {}),
    ...(title && normalizeText(title) ? { title: normalizeText(title) } : {}),
    ...(spotifyUrl ? { spotifyUrl } : {}),
    ...(duration !== undefined ? { duration } : {}),
  };
}

function spotifyHistorySongs(value: readonly unknown[], options?: { unlimited?: boolean }): Record<string, unknown>[] {
  const aggregated = new Map<string, { row: Record<string, unknown>; playTime: number; plays: number; first: number }>();
  value.forEach((entry, index) => {
    const extracted = extractArtistAndTitle(entry);
    if (!extracted.artist || !extracted.title) return;
    const { artist, title, spotifyUrl, duration } = extracted;
    const row: Record<string, unknown> = { artist, title, ...(spotifyUrl ? { spotify_url: spotifyUrl } : {}) };
    const key = tasteSeedKey({ artist, title });
    const playTime = duration ?? 1;
    const previous = aggregated.get(key);
    if (previous) {
      previous.playTime += playTime;
      previous.plays += 1;
      if (!previous.row.spotify_url && row.spotify_url) previous.row = row;
    } else {
      aggregated.set(key, { row, playTime, plays: 1, first: index });
    }
  });
  const list = [...aggregated.values()]
    .sort((left, right) => right.playTime - left.playTime || right.plays - left.plays || left.first - right.first);
  return (options?.unlimited ? list : list.slice(0, 500))
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

export function parseCsvLine(line: string, delimiter: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

export function detectDelimiter(headerLine: string): string {
  const counts = {
    ",": (headerLine.match(/,/gu) || []).length,
    ";": (headerLine.match(/;/gu) || []).length,
    "\t": (headerLine.match(/\t/gu) || []).length,
  };
  if (counts[";"] > counts[","] && counts[";"] > counts["\t"]) return ";";
  if (counts["\t"] > counts[","] && counts["\t"] > counts[";"]) return "\t";
  return ",";
}

/**
 * Universal parser: extracts tracks from JSON (any format), CSV (Exportify or general),
 * or line-by-line TXT.
 */
export function parseFileContentToSongs(text: string, options?: { allowPartial?: boolean; unlimited?: boolean }): TasteSeedInput[] {
  const content = text.replace(/^\uFEFF/u, "").trim();
  if (!content) return [];

  // 1. Try JSON parsing
  try {
    const parsed = JSON.parse(content);
    return parseTasteInput(parsed, options);
  } catch (error) {
    if (error instanceof TasteInputError) {
      if (error.issues.some((issue) => issue.code === "out_of_range" || issue.code === "duplicate")) {
        throw error;
      }
    }
  }

  // 2. CSV / TSV / TXT / M3U / PLS parsing
  const rawLines = content.split(/\r?\n/u).map((l) => l.trim()).filter(Boolean);
  if (rawLines.length === 0) return [];

  // Check for M3U / M3U8
  const isM3u = rawLines.some((l) => l.startsWith("#EXTM3U") || l.startsWith("#EXTINF:"));
  if (isM3u) {
    const rawRows: Record<string, unknown>[] = [];
    for (const rawLine of rawLines) {
      if (rawLine.startsWith("#EXTINF:")) {
        const commaIdx = rawLine.indexOf(",");
        const trackStr = (commaIdx !== -1 ? rawLine.slice(commaIdx + 1) : rawLine.replace("#EXTINF:", "")).trim();
        const extracted = extractArtistAndTitle(trackStr);
        if (extracted.artist && extracted.title) {
          rawRows.push({ artist: extracted.artist, title: extracted.title });
        }
      } else if (!rawLine.startsWith("#")) {
        const cleanName = rawLine.replace(/\.[a-z0-9]{2,4}$/iu, "").trim();
        const extracted = extractArtistAndTitle(cleanName);
        if (extracted.artist && extracted.title) {
          rawRows.push({ artist: extracted.artist, title: extracted.title });
        }
      }
    }
    if (rawRows.length > 0) {
      return parseTasteInput(rawRows, options);
    }
  }

  // Check for PLS playlist
  const isPls = rawLines.some((l) => /^\[playlist\]/i.test(l) || /^Title\d+=/i.test(l));
  if (isPls) {
    const rawRows: Record<string, unknown>[] = [];
    for (const rawLine of rawLines) {
      const match = /^Title\d+\s*=\s*(.+)$/i.exec(rawLine);
      if (match) {
        const extracted = extractArtistAndTitle(match[1].trim());
        if (extracted.artist && extracted.title) {
          rawRows.push({ artist: extracted.artist, title: extracted.title });
        }
      }
    }
    if (rawRows.length > 0) {
      return parseTasteInput(rawRows, options);
    }
  }

  const lines = rawLines.filter((l) => !l.startsWith("#"));
  if (lines.length === 0) return [];

  const firstLine = lines[0];
  const delimiter = detectDelimiter(firstLine);
  const headerCols = parseCsvLine(firstLine, delimiter).map((col) => col.toLowerCase().replace(/["']/gu, "").trim());

  const isHeader = headerCols.some((col) =>
    col.includes("artist") ||
    col.includes("track") ||
    col.includes("title") ||
    col.includes("song") ||
    col.includes("spotify") ||
    col.includes("исполнитель") ||
    col.includes("артист") ||
    col.includes("автор") ||
    col.includes("название") ||
    col.includes("трек") ||
    col.includes("песня")
  );

  let artistIdx = -1;
  let titleIdx = -1;
  let spotifyIdx = -1;

  if (isHeader) {
    spotifyIdx = headerCols.findIndex((col) => col.includes("spotify") || col === "id" || col.includes("track id"));
    artistIdx = headerCols.findIndex((col) =>
      col.includes("artist") || col.includes("author") || col.includes("performer") ||
      col.includes("исполнитель") || col.includes("артист") || col.includes("автор")
    );
    titleIdx = headerCols.findIndex((col, i) =>
      i !== spotifyIdx && i !== artistIdx && (
        col === "track name" || col === "track" || col === "title" || col === "song" || col === "name" ||
        col === "название" || col === "трек" || col === "песня"
      )
    );
    if (titleIdx === -1) {
      titleIdx = headerCols.findIndex((col, i) =>
        i !== spotifyIdx && i !== artistIdx && (
          col.includes("track name") || col.includes("title") || col.includes("song") ||
          col.includes("название") || col.includes("трек") || col.includes("песня")
        )
      );
    }
    if (titleIdx === -1) {
      titleIdx = headerCols.findIndex((col, i) =>
        i !== spotifyIdx && i !== artistIdx && (
          col.includes("track") || col.includes("name")
        )
      );
    }
  }

  const rawRows: Record<string, unknown>[] = [];
  const startLine = isHeader ? 1 : 0;

  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;

    if (isHeader && artistIdx !== -1 && titleIdx !== -1) {
      const cols = parseCsvLine(line, delimiter);
      const artist = cols[artistIdx]?.replace(/^["']|["']$/gu, "").trim();
      const title = cols[titleIdx]?.replace(/^["']|["']$/gu, "").trim();
      const spotifyVal = spotifyIdx !== -1 ? cols[spotifyIdx]?.replace(/^["']|["']$/gu, "").trim() : undefined;

      let spotifyUrl: string | undefined;
      if (spotifyVal) {
        if (/^[A-Za-z0-9]{22}$/u.test(spotifyVal)) {
          spotifyUrl = `https://open.spotify.com/track/${spotifyVal}`;
        } else if (spotifyVal.startsWith("spotify:track:")) {
          const id = spotifyVal.replace("spotify:track:", "").trim();
          if (/^[A-Za-z0-9]{22}$/u.test(id)) spotifyUrl = `https://open.spotify.com/track/${id}`;
        } else if (/^https?:\/\//u.test(spotifyVal)) {
          spotifyUrl = spotifyVal;
        }
      }

      if (artist && title) {
        rawRows.push({ artist, title, ...(spotifyUrl ? { spotify_url: spotifyUrl } : {}) });
      }
    } else {
      const cleanLine = line
        .replace(/^\s*(?:\[\d+\]|\d+[\.\)\-:]|\d+\s+[-–—])\s*/u, "")
        .replace(/^\s*[•\*\-]\s+/u, "")
        .trim();

      const match = /^(.+?)\s*(?:[-–—:]|\s+by\s+)\s*(.+)$/iu.exec(cleanLine);
      if (match) {
        const artist = match[1].trim().replace(/^["']|["']$/gu, "");
        const title = match[2].trim().replace(/^["']|["']$/gu, "");
        if (artist && title) rawRows.push({ artist, title });
      } else {
        const cols = parseCsvLine(cleanLine, delimiter);
        if (cols.length >= 2) {
          if (/^\d+$/u.test(cols[0]) && cols.length >= 3) {
            const artist = cols[1].replace(/^["']|["']$/gu, "").trim();
            const title = cols[2].replace(/^["']|["']$/gu, "").trim();
            if (artist && title) rawRows.push({ artist, title });
          } else {
            const artist = cols[0].replace(/^["']|["']$/gu, "").trim();
            const title = cols[1].replace(/^["']|["']$/gu, "").trim();
            if (artist && title) rawRows.push({ artist, title });
          }
        }
      }
    }
  }

  return parseTasteInput(rawRows, options);
}

/** Validates input and removes only exact normalized artist/title duplicates in input order. */
export function parseTasteInput(value: unknown, options?: { allowPartial?: boolean; unlimited?: boolean }): TasteSeedInput[] {
  let rawList: unknown[] | undefined;
  let isHistory = false;

  if (typeof value === "string") {
    return parseFileContentToSongs(value, options);
  }

  if (Array.isArray(value)) {
    rawList = value;
    isHistory = true;
  } else if (object(value)) {
    if (Array.isArray(value.songs)) {
      rawList = value.songs;
      isHistory = false;
    } else if (Array.isArray(value.tracks)) {
      rawList = value.tracks;
      isHistory = true;
    } else if (Array.isArray(value.items)) {
      rawList = value.items;
      isHistory = true;
    } else if (Array.isArray(value.data)) {
      rawList = value.data;
      isHistory = true;
    } else if (Array.isArray(value.history)) {
      rawList = value.history;
      isHistory = true;
    } else if (object(value.tracks) && Array.isArray((value.tracks as Record<string, unknown>).items)) {
      rawList = (value.tracks as Record<string, unknown>).items as unknown[];
      isHistory = true;
    } else if (object(value.playlist) && Array.isArray((value.playlist as Record<string, unknown>).tracks)) {
      rawList = (value.playlist as Record<string, unknown>).tracks as unknown[];
      isHistory = true;
    } else if (object(value.playlist) && Array.isArray((value.playlist as Record<string, unknown>).items)) {
      rawList = (value.playlist as Record<string, unknown>).items as unknown[];
      isHistory = true;
    } else if (object(value.result) && Array.isArray((value.result as Record<string, unknown>).tracks)) {
      rawList = (value.result as Record<string, unknown>).tracks as unknown[];
      isHistory = true;
    } else if (object(value.results) && Array.isArray((value.results as Record<string, unknown>).track)) {
      rawList = (value.results as Record<string, unknown>).track as unknown[];
      isHistory = true;
    } else if (object(value.results) && Array.isArray((value.results as Record<string, unknown>).tracks)) {
      rawList = (value.results as Record<string, unknown>).tracks as unknown[];
      isHistory = true;
    } else if (object(value.recenttracks) && Array.isArray((value.recenttracks as Record<string, unknown>).track)) {
      rawList = (value.recenttracks as Record<string, unknown>).track as unknown[];
      isHistory = true;
    } else if (object(value.library) && Array.isArray((value.library as Record<string, unknown>).tracks)) {
      rawList = (value.library as Record<string, unknown>).tracks as unknown[];
      isHistory = true;
    } else {
      // Automatic fallback scan: search for any array in value that contains track-like objects
      for (const val of Object.values(value)) {
        if (Array.isArray(val) && val.length > 0) {
          const sample = extractArtistAndTitle(val[0]);
          if (sample.artist && sample.title) {
            rawList = val;
            isHistory = true;
            break;
          }
        } else if (object(val)) {
          for (const nestedVal of Object.values(val)) {
            if (Array.isArray(nestedVal) && nestedVal.length > 0) {
              const sample = extractArtistAndTitle(nestedVal[0]);
              if (sample.artist && sample.title) {
                rawList = nestedVal;
                isHistory = true;
                break;
              }
            }
          }
          if (rawList) break;
        }
      }
    }
  }

  if (!rawList) {
    throw new TasteInputError([{ path: "songs", code: "invalid_type", message: "songs must be an array" }]);
  }

  const songs = isHistory ? spotifyHistorySongs(rawList, options) : rawList;
  const minRequired = options?.allowPartial ? 1 : 5;
  if (songs.length < minRequired || (!isHistory && !options?.unlimited && songs.length > 500)) {
    throw new TasteInputError([{ path: "songs", code: "out_of_range", message: `songs must contain between ${minRequired} and 500 entries` }]);
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
    const artist = typeof row.artist === "string" ? normalizeText(row.artist) : undefined;
    const title = typeof row.title === "string" ? normalizeText(row.title) : undefined;

    if (!artist) {
      issues.push({ path: `${path}.artist`, code: "invalid_value", message: `${path}.artist must be a non-empty string` });
      continue;
    }
    if (!title) {
      issues.push({ path: `${path}.title`, code: "invalid_value", message: `${path}.title must be a non-empty string` });
      continue;
    }
    const spotify = parseSpotifyUrl(row.spotify_url, `${path}.spotify_url`);
    if (spotify && "code" in spotify) {
      issues.push(spotify);
      continue;
    }
    const seed: TasteSeedInput = { artist, title, ...spotify };
    const key = tasteSeedKey(seed);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    seeds.push(seed);
  }
  if (issues.length > 0) throw new TasteInputError(issues);
  if (seeds.length < minRequired) {
    throw new TasteInputError([{ path: "songs", code: "duplicate", message: `at least ${minRequired} distinct songs are required after duplicate removal` }]);
  }
  return seeds;
}
