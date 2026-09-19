import catalogJson from "../../../ml/tastelift-catalog.json";
import { fetchWithRetry } from "../http";
import { searchRecordingsForSeed } from "../listenbrainz";
import type { Track } from "../types";
import type { TasteLiftCatalogArtifact } from "./catalog";
import { tasteSeedKey, type TasteSeedInput } from "./input";
import { VersionedCache } from "../cache";

export interface TasteResolverAdapters {
  searchRecordings: (query: string) => Promise<Track[]>;
  resolveLastFm?: (input: TasteSeedInput) => Promise<Track | null>;
  resolveLocal?: (input: TasteSeedInput) => Track | null;
  cache?: VersionedCache<ResolvedTasteSeed>;
}

export interface MbidResolvedTasteSeed {
  input: TasteSeedInput;
  status: "resolved";
  source: "listenbrainz" | "lastfm";
  track: Track;
}

export interface TextResolvedTasteSeed {
  input: TasteSeedInput;
  status: "resolved";
  source: "text";
  diagnostic: {
    status: "retrieval_unavailable";
    code: "no_exact_mbid" | "upstream_error";
    message: string;
  };
}

export interface UnresolvedTasteSeed {
  input: TasteSeedInput;
  status: "unresolved";
  error: {
    code: "upstream_error";
    message: string;
  };
}

/** Every requested seed is returned as an MBID, text/OOV, or structured failure outcome. */
export type ResolvedTasteSeed = MbidResolvedTasteSeed | TextResolvedTasteSeed | UnresolvedTasteSeed;
export type TasteSeedResolution = ResolvedTasteSeed;

export interface TasteResolverOptions {
  cache?: VersionedCache<ResolvedTasteSeed>;
  maxConcurrency?: number;
  cacheTtlMs?: number;
}

export class LastFmResponseError extends Error {
  constructor(public readonly code: number | string, message: string) {
    super(`Last.fm error ${code}: ${message}`);
    this.name = "LastFmResponseError";
  }
}

const resolverStorage = new Map<string, string>();
const resolverCache = new VersionedCache<ResolvedTasteSeed>("tastelift-resolver-v1", {
  getItem: (key) => resolverStorage.get(key) ?? null,
  setItem: (key, value) => resolverStorage.set(key, value),
});
const DEFAULT_CONCURRENCY = 8;
const DEFAULT_CACHE_TTL_MS = 60 * 60 * 1000;

const key = (value: string) => value.normalize("NFKC").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
const queryFor = (input: TasteSeedInput) => `${input.artist} ${input.title}`;

const catalogIndex = new Map<string, Track>();
const catalogSearchIndex = new Map<string, Track>();

for (const t of (catalogJson as TasteLiftCatalogArtifact).tracks) {
  const cKey = `${key(t.artist)}\u0000${key(t.title)}`;
  const sKey = `${key(t.artist)} ${key(t.title)}`;
  const existing = catalogIndex.get(cKey);
  if (!existing || t.mbid < existing.mbid) {
    const track: Track = {
      mbid: t.mbid,
      artist: t.artist,
      title: t.title,
      ...(t.release ? { release: t.release } : {}),
    };
    catalogIndex.set(cKey, track);
    catalogSearchIndex.set(sKey, track);
  }
}

function exactMatch(input: TasteSeedInput, tracks: readonly Track[]): Track | null {
  return tracks
    .filter((track) => key(track.artist) === key(input.artist) && key(track.title) === key(input.title))
    .sort((left, right) => left.mbid < right.mbid ? -1 : left.mbid > right.mbid ? 1 : 0)[0] ?? null;
}

function upstreamError(input: TasteSeedInput, error: unknown): TextResolvedTasteSeed {
  return {
    input,
    status: "resolved",
    source: "text",
    diagnostic: { status: "retrieval_unavailable", code: "upstream_error", message: error instanceof Error ? error.message : "Resolver unavailable" },
  };
}

function withInput(input: TasteSeedInput, result: ResolvedTasteSeed): ResolvedTasteSeed {
  return { ...result, input };
}

async function resolveTasteSeed(input: TasteSeedInput, adapters: TasteResolverAdapters): Promise<ResolvedTasteSeed> {
  if (adapters.resolveLocal) {
    try {
      const localMatch = adapters.resolveLocal(input);
      if (localMatch) return { input, status: "resolved", track: localMatch, source: "listenbrainz" };
    } catch {
      // fallback to upstream adapters
    }
  }

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
}

async function mapWithConcurrency<T, R>(values: readonly T[], maxConcurrency: number, mapper: (value: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  const worker = async () => {
    while (next < values.length) {
      const index = next++;
      results[index] = await mapper(values[index]!);
    }
  };
  await Promise.all(Array.from({ length: Math.min(Math.max(1, maxConcurrency), values.length) }, worker));
  return results;
}

export async function resolveTasteSeeds(inputs: readonly TasteSeedInput[], adapters: TasteResolverAdapters, options: TasteResolverOptions = {}): Promise<ResolvedTasteSeed[]> {
  const cache = options.cache ?? adapters.cache;
  const ttlMs = options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
  return mapWithConcurrency(inputs, options.maxConcurrency ?? DEFAULT_CONCURRENCY, async (input) => {
    const cacheKey = tasteSeedKey(input);
    const cached = cache?.get(cacheKey);
    if (cached && !cached.stale) return withInput(input, cached.value);
    const result = await resolveTasteSeed(input, adapters);
    if (result.status === "resolved") cache?.set(cacheKey, result, ttlMs);
    return result;
  });
}

export function parseLastFmTrack(value: unknown): Track | null {
  if (typeof value !== "object" || value === null) return null;
  const response = value as Record<string, unknown>;
  if (typeof response.error === "number" || typeof response.error === "string" || (typeof response.message === "string" && response.track === undefined)) {
    const code = typeof response.error === "number" || typeof response.error === "string" ? response.error : "unknown";
    throw new LastFmResponseError(code, typeof response.message === "string" ? response.message : "Last.fm returned an error response");
  }
  const track = response.track;
  if (typeof track !== "object" || track === null) return null;
  const row = track as Record<string, unknown>;
  const artist = typeof row.artist === "object" && row.artist !== null && typeof (row.artist as Record<string, unknown>).name === "string" ? (row.artist as Record<string, string>).name : null;
  return typeof row.mbid === "string" && row.mbid && typeof row.name === "string" && artist ? { mbid: row.mbid, title: row.name, artist } : null;
}

export function createTasteResolverAdapters(lastFmApiKey = process.env.LASTFM_API_KEY): TasteResolverAdapters {
  let liveSearchCount = 0;
  const MAX_LIVE_SEARCHES = 15;

  return {
    resolveLocal: (input: TasteSeedInput): Track | null => {
      const normKey = `${key(input.artist)}\u0000${key(input.title)}`;
      return catalogIndex.get(normKey) ?? null;
    },
    searchRecordings: async (query: string) => {
      const normQuery = key(query);
      const catalogMatch = catalogSearchIndex.get(normQuery);
      if (catalogMatch) {
        return [catalogMatch];
      }
      if (liveSearchCount >= MAX_LIVE_SEARCHES) {
        return [];
      }
      liveSearchCount++;
      return searchRecordingsForSeed(query);
    },
    cache: resolverCache,
    ...(lastFmApiKey ? {
      resolveLastFm: async (input: TasteSeedInput): Promise<Track | null> => {
        if (liveSearchCount >= MAX_LIVE_SEARCHES) {
          return null;
        }
        liveSearchCount++;
        const url = new URL("https://ws.audioscrobbler.com/2.0/");
        url.search = new URLSearchParams({ method: "track.getInfo", api_key: lastFmApiKey, artist: input.artist, track: input.title, format: "json" }).toString();
        return parseLastFmTrack(await (await fetchWithRetry(url.toString())).json());
      },
    } : {}),
  };
}
