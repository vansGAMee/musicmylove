import { fetchWithRetry } from "../http";
import { searchRecordingsForSeed } from "../listenbrainz";
import type { Track } from "../types";
import type { TasteSeedInput } from "./input";

export interface TasteResolverAdapters {
  searchRecordings: (query: string) => Promise<Track[]>;
  resolveLastFm?: (input: TasteSeedInput) => Promise<Track | null>;
}

export interface ResolvedTasteSeed {
  input: TasteSeedInput;
  status: "resolved";
  source: "listenbrainz" | "lastfm" | "text";
  track?: Track;
  diagnostic?: {
    status: "retrieval_unavailable";
    code: "no_exact_mbid";
    message: string;
  };
}

export interface UnresolvedTasteSeed {
  input: TasteSeedInput;
  status: "unresolved";
  error: {
    code: "not_found" | "upstream_error";
    message: string;
  };
}

export type TasteSeedResolution = ResolvedTasteSeed | UnresolvedTasteSeed;

const key = (value: string) => value.normalize("NFKC").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
const queryFor = (input: TasteSeedInput) => `${input.artist} ${input.title}`;

function exactMatch(input: TasteSeedInput, tracks: readonly Track[]): Track | null {
  return tracks.find((track) => key(track.artist) === key(input.artist) && key(track.title) === key(input.title)) ?? null;
}

function upstreamError(input: TasteSeedInput, error: unknown): UnresolvedTasteSeed {
  return {
    input,
    status: "unresolved",
    error: { code: "upstream_error", message: error instanceof Error ? error.message : "Resolver unavailable" },
  };
}

export async function resolveTasteSeeds(inputs: readonly TasteSeedInput[], adapters: TasteResolverAdapters): Promise<TasteSeedResolution[]> {
  return Promise.all(inputs.map(async (input): Promise<TasteSeedResolution> => {
    try {
      const listenBrainzMatch = exactMatch(input, await adapters.searchRecordings(queryFor(input)));
      if (listenBrainzMatch) return { input, status: "resolved", track: listenBrainzMatch, source: "listenbrainz" };
    } catch (error) {
      return upstreamError(input, error);
    }

    if (adapters.resolveLastFm) {
      try {
        const lastFmMatch = await adapters.resolveLastFm(input);
        if (lastFmMatch && exactMatch(input, [lastFmMatch])) return { input, status: "resolved", track: lastFmMatch, source: "lastfm" };
      } catch (error) {
        return upstreamError(input, error);
      }
    }
    return {
      input,
      status: "resolved",
      source: "text",
      diagnostic: {
        status: "retrieval_unavailable",
        code: "no_exact_mbid",
        message: `No exact MusicBrainz recording found for ${input.artist} — ${input.title}`,
      },
    };
  }));
}

function parseLastFmTrack(value: unknown): Track | null {
  if (typeof value !== "object" || value === null) return null;
  const track = (value as Record<string, unknown>).track;
  if (typeof track !== "object" || track === null) return null;
  const row = track as Record<string, unknown>;
  const artist = typeof row.artist === "object" && row.artist !== null && typeof (row.artist as Record<string, unknown>).name === "string" ? (row.artist as Record<string, string>).name : null;
  return typeof row.mbid === "string" && row.mbid && typeof row.name === "string" && artist ? { mbid: row.mbid, title: row.name, artist } : null;
}

export function createTasteResolverAdapters(lastFmApiKey = process.env.LASTFM_API_KEY): TasteResolverAdapters {
  return {
    searchRecordings: (query) => searchRecordingsForSeed(query),
    ...(lastFmApiKey ? {
      resolveLastFm: async (input: TasteSeedInput): Promise<Track | null> => {
        const url = new URL("https://ws.audioscrobbler.com/2.0/");
        url.search = new URLSearchParams({ method: "track.getInfo", api_key: lastFmApiKey, artist: input.artist, track: input.title, format: "json" }).toString();
        return parseLastFmTrack(await (await fetchWithRetry(url.toString())).json());
      },
    } : {}),
  };
}
