export interface YandexTrack {
  id: string;
  title: string;
  artists: string[];
}

export interface YandexPlaylistResult {
  id: string;
  title: string;
  owner?: string;
  trackCount: number;
  tracks: YandexTrack[];
}

export type YandexErrorCode = "invalid_url" | "not_found" | "private" | "upstream_error";

export class YandexPlaylistError extends Error {
  constructor(public readonly code: YandexErrorCode, message: string) {
    super(message);
    this.name = "YandexPlaylistError";
  }
}

export interface YandexFetchOptions {
  timeoutMs?: number;
  retries?: number;
  fetcher?: typeof fetch;
  skipCache?: boolean;
}

interface CachedPlaylist {
  result: YandexPlaylistResult;
  timestamp: number;
}

const PLAYLIST_CACHE = new Map<string, CachedPlaylist>();
const MAX_PLAYLIST_CACHE = 500;
const PLAYLIST_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

export function clearYandexPlaylistCache(): void {
  PLAYLIST_CACHE.clear();
}

function cacheAndReturn(
  canonicalUrl: string,
  result: YandexPlaylistResult,
  skipCache?: boolean
): YandexPlaylistResult {
  if (!skipCache) {
    if (PLAYLIST_CACHE.size >= MAX_PLAYLIST_CACHE) {
      const oldestKey = PLAYLIST_CACHE.keys().next().value;
      if (oldestKey) PLAYLIST_CACHE.delete(oldestKey);
    }
    PLAYLIST_CACHE.set(canonicalUrl, {
      result,
      timestamp: Date.now(),
    });
  }
  return result;
}

export interface NormalizedYandexUrl {
  canonicalUrl: string;
  type: "user_playlist" | "uuid_playlist" | "lk_redirect" | "unknown";
  owner?: string;
  kind?: string;
  uuid?: string;
}

/**
 * Normalizes any Yandex Music playlist link, stripping tracking queries and UTM params.
 * Supports:
 * - https://music.yandex.ru/playlists/<uuid>
 * - https://music.yandex.com/playlists/<uuid>
 * - https://lk.music.yandex.ru/...
 * - https://music.yandex.ru/users/<owner>/playlists/<kind>
 * - Legacy format: https://music.yandex.ru/?owner=<owner>&kinds=<kind>
 */
export function normalizeYandexPlaylistUrl(inputUrl: string): NormalizedYandexUrl {
  const trimmed = inputUrl.trim();
  if (!trimmed) {
    throw new YandexPlaylistError("invalid_url", "Ссылка не может быть пустой");
  }

  // Ensure protocol
  let urlStr = trimmed;
  if (!/^https?:\/\//i.test(urlStr)) {
    urlStr = `https://${urlStr}`;
  }

  let parsed: URL;
  try {
    parsed = new URL(urlStr);
  } catch {
    throw new YandexPlaylistError("invalid_url", "Некорректный формат URL");
  }

  const hostname = parsed.hostname.toLowerCase();
  if (!hostname.includes("yandex.") && !hostname.includes("yamusic.")) {
    throw new YandexPlaylistError("invalid_url", "Ссылка должна вести на Яндекс Музыку (music.yandex.ru)");
  }

  // Handle lk.* short/redirect links
  if (hostname.startsWith("lk.")) {
    return {
      canonicalUrl: `https://${hostname}${parsed.pathname}`,
      type: "lk_redirect",
    };
  }

  const pathname = parsed.pathname;

  // 1. Classic format: /users/<owner>/playlists/<kind>
  const userPlaylistMatch = /^\/users\/([^/]+)\/playlists\/(\d+)/i.exec(pathname);
  if (userPlaylistMatch) {
    const owner = decodeURIComponent(userPlaylistMatch[1]!);
    const kind = userPlaylistMatch[2]!;
    return {
      canonicalUrl: `https://music.yandex.ru/users/${encodeURIComponent(owner)}/playlists/${kind}`,
      type: "user_playlist",
      owner,
      kind,
    };
  }

  // 2. New format: /playlists/<uuid> or /playlists/ps.<uuid>
  const uuidPlaylistMatch = /^\/playlists\/([A-Za-z0-9._-]+)/i.exec(pathname);
  if (uuidPlaylistMatch) {
    const uuid = uuidPlaylistMatch[1]!;
    return {
      canonicalUrl: `https://music.yandex.ru/playlists/${uuid}`,
      type: "uuid_playlist",
      uuid,
    };
  }

  // 3. Legacy query format: /?owner=<owner>&kinds=<kind>
  const ownerParam = parsed.searchParams.get("owner");
  const kindsParam = parsed.searchParams.get("kinds") ?? parsed.searchParams.get("kind");
  if (ownerParam && kindsParam && /^\d+$/.test(kindsParam)) {
    return {
      canonicalUrl: `https://music.yandex.ru/users/${encodeURIComponent(ownerParam)}/playlists/${kindsParam}`,
      type: "user_playlist",
      owner: ownerParam,
      kind: kindsParam,
    };
  }

  // Fallback for general paths containing playlist
  if (pathname.includes("playlist")) {
    return {
      canonicalUrl: `https://music.yandex.ru${pathname}`,
      type: "unknown",
    };
  }

  throw new YandexPlaylistError(
    "invalid_url",
    "Не удалось распознать плейлист. Поддерживаются ссылки вида music.yandex.ru/users/.../playlists/... или music.yandex.ru/playlists/..."
  );
}

const DESKTOP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

async function fetchWithBackoff(
  url: string,
  init: RequestInit,
  options: { timeoutMs: number; retries: number; fetcher: typeof fetch; deadline?: number }
): Promise<Response> {
  const { timeoutMs, retries, fetcher, deadline } = options;
  let lastError: unknown;

  for (let attempt = 0; attempt < retries; attempt++) {
    if (deadline && Date.now() >= deadline) {
      break;
    }
    const remainingTime = deadline ? Math.max(1000, deadline - Date.now()) : timeoutMs;
    const effectiveTimeout = Math.min(timeoutMs, remainingTime);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), effectiveTimeout);

    try {
      const response = await fetcher(url, {
        ...init,
        signal: controller.signal,
        headers: {
          "User-Agent": DESKTOP_USER_AGENT,
          "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
          ...init.headers,
        },
      });

      if (response.status === 404) {
        throw new YandexPlaylistError("not_found", "Плейлист не найден. Проверьте правильность ссылки.");
      }
      if (response.status === 401 || response.status === 403) {
        throw new YandexPlaylistError("private", "Плейлист приватный или доступ ограничен. Сделайте его публичным в настройках.");
      }

      if (response.ok) {
        return response;
      }

      // Retryable statuses: 429 or 5xx
      const isRetryable = response.status === 429 || response.status >= 500;
      if (!isRetryable || attempt === retries - 1) {
        throw new YandexPlaylistError("upstream_error", `Ошибка сервиса Яндекс Музыки (HTTP ${response.status})`);
      }

      const backoff = 250 * Math.pow(2, attempt) + Math.random() * 100;
      await new Promise((resolve) => setTimeout(resolve, backoff));
    } catch (err) {
      lastError = err;
      if (err instanceof YandexPlaylistError) {
        throw err;
      }
      if (attempt === retries - 1) {
        break;
      }
      const backoff = 250 * Math.pow(2, attempt) + Math.random() * 100;
      await new Promise((resolve) => setTimeout(resolve, backoff));
    } finally {
      clearTimeout(timer);
    }
  }

  if (lastError instanceof YandexPlaylistError) {
    throw lastError;
  }
  throw new YandexPlaylistError(
    "upstream_error",
    lastError instanceof Error ? lastError.message : "Не удалось связаться с сервером Яндекс Музыки"
  );
}

interface RawTrackObj {
  id?: string | number;
  title?: string;
  name?: string;
  artists?: Array<string | { name?: string }>;
  author?: string | { name?: string };
  performer?: string | { name?: string };
  byArtist?: string | { name?: string } | Array<string | { name?: string }>;
  albums?: Array<{ id?: string | number; title?: string }>;
}

function parseRawTrack(raw: unknown): YandexTrack | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;

  // Track can be nested under .track (common in Yandex API: { id: ..., track: { title: ... } })
  const item = (typeof obj.track === "object" && obj.track !== null ? obj.track : obj) as RawTrackObj;

  const id = item.id !== undefined ? String(item.id) : "";
  const title = (item.title ?? item.name ?? "").trim();
  if (!title) return null;

  const artists: string[] = [];
  if (Array.isArray(item.artists)) {
    for (const a of item.artists) {
      if (typeof a === "string" && a.trim()) {
        artists.push(a.trim());
      } else if (typeof a === "object" && a !== null && typeof a.name === "string" && a.name.trim()) {
        artists.push(a.name.trim());
      }
    }
  } else if (item.byArtist) {
    const list = Array.isArray(item.byArtist) ? item.byArtist : [item.byArtist];
    for (const a of list) {
      if (typeof a === "string" && a.trim()) artists.push(a.trim());
      else if (typeof a === "object" && a !== null && typeof a.name === "string" && a.name.trim()) artists.push(a.name.trim());
    }
  } else if (typeof item.author === "string" && item.author.trim()) {
    artists.push(item.author.trim());
  } else if (typeof item.author === "object" && item.author !== null && typeof item.author.name === "string") {
    artists.push(item.author.name.trim());
  } else if (typeof item.performer === "string" && item.performer.trim()) {
    artists.push(item.performer.trim());
  } else if (typeof item.performer === "object" && item.performer !== null && typeof item.performer.name === "string") {
    artists.push(item.performer.name.trim());
  }

  return {
    id: id || `${artists.join(", ")} - ${title}`,
    title,
    artists: artists.length > 0 ? artists : ["Unknown Artist"],
  };
}

function deduplicateTracks(tracks: YandexTrack[]): YandexTrack[] {
  const seen = new Set<string>();
  const result: YandexTrack[] = [];

  for (const track of tracks) {
    const key = `${track.artists.join(", ").toLowerCase()} — ${track.title.toLowerCase()}`.trim();
    if (!seen.has(key)) {
      seen.add(key);
      result.push(track);
    }
  }

  return result;
}

/**
 * Extracts metadata and tracks from Next.js server-rendered HTML or embedded state.
 */
export function extractFromHtmlState(html: string): {
  owner?: string;
  kind?: string;
  title?: string;
  tracks: YandexTrack[];
} {
  if (html.includes("404: This page could not be found") || html.includes("notFound")) {
    throw new YandexPlaylistError("not_found", "Плейлист не найден. Проверьте правильность ссылки.");
  }

  let owner: string | undefined;
  let kind: string | undefined;
  let title: string | undefined;
  const tracks: YandexTrack[] = [];

  // 1. Try extracting owner and kind from Next.js params: {"userId":"...","kind":"..."}
  const paramsMatch = /"params"\s*:\s*\{[^}]*"userId"\s*:\s*"([^"]+)"[^}]*"kind"\s*:\s*"(\d+)"/i.exec(html) ??
    /"userId"\s*:\s*"([^"]+)"\s*,\s*"kind"\s*:\s*"(\d+)"/i.exec(html);
  if (paramsMatch) {
    owner = paramsMatch[1];
    kind = paramsMatch[2];
  }

  // 2. Try extracting title from og:title or HTML title
  const ogTitleMatch = /<meta\s+property=["']og:title["']\s+content=["']([^"']+)["']/i.exec(html);
  const titleTagMatch = /<title>([^<]+)<\/title>/i.exec(html);
  if (ogTitleMatch) {
    title = ogTitleMatch[1]!.replace(/\s*—\s*Яндекс Музыка.*/i, "").trim();
  } else if (titleTagMatch) {
    const clean = titleTagMatch[1]!.replace(/\s*—\s*Яндекс Музыка.*/i, "").trim();
    if (!clean.includes("собираем музыку") && !clean.includes("This page could not be found")) {
      title = clean;
    }
  }

  // 3. Try parsing JSON-LD Schema.org MusicPlaylist
  const jsonLdMatches = html.matchAll(/<script\s+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi);
  for (const match of jsonLdMatches) {
    try {
      const data = JSON.parse(match[1]!);
      if (data["@type"] === "MusicPlaylist") {
        if (data.name && !title) title = data.name;
        if (Array.isArray(data.track)) {
          for (const item of data.track) {
            const parsed = parseRawTrack(item);
            if (parsed) tracks.push(parsed);
          }
        }
      }
    } catch {
      // ignore
    }
  }

  // 4. Try extracting tracks from self.__next_f.push streams or state patches
  const trackMatches = html.matchAll(/"title"\s*:\s*"([^"]+)"\s*,\s*"artists"\s*:\s*\[\{"name"\s*:\s*"([^"]+)"/g);
  for (const m of trackMatches) {
    const trackTitle = m[1]!.replace(/\\"/g, '"');
    const artistName = m[2]!.replace(/\\"/g, '"');
    tracks.push({
      id: `${artistName} - ${trackTitle}`,
      title: trackTitle,
      artists: [artistName],
    });
  }

  return { owner, kind, title, tracks: deduplicateTracks(tracks) };
}

/**
 * Main function to fetch a public Yandex Music playlist.
 * Follows redirects, parses API responses or HTML embedded state,
 * retries on 429/5xx, and returns deduplicated tracks.
 */
export async function fetchYandexPlaylist(
  rawUrl: string,
  options: YandexFetchOptions = {}
): Promise<YandexPlaylistResult> {
  const timeoutMs = options.timeoutMs ?? 3500;
  const retries = options.retries ?? 2;
  const fetcher = options.fetcher ?? fetch;
  const deadline = Date.now() + 7500;

  const normalized = normalizeYandexPlaylistUrl(rawUrl);

  if (!options.skipCache) {
    const cached = PLAYLIST_CACHE.get(normalized.canonicalUrl);
    if (cached && Date.now() - cached.timestamp < PLAYLIST_CACHE_TTL_MS) {
      return cached.result;
    }
  }

  let currentOwner = normalized.owner;
  let currentKind = normalized.kind;
  let currentUuid = normalized.uuid;
  let targetUrl = normalized.canonicalUrl;

  // Step 1: If lk.* or redirect, follow redirects to resolve destination
  if (normalized.type === "lk_redirect") {
    try {
      const redirectRes = await fetchWithBackoff(
        targetUrl,
        { method: "GET", redirect: "follow" },
        { timeoutMs, retries: 2, fetcher, deadline }
      );
      targetUrl = redirectRes.url;
      const reNormalized = normalizeYandexPlaylistUrl(targetUrl);
      currentOwner = reNormalized.owner;
      currentKind = reNormalized.kind;
      currentUuid = reNormalized.uuid;
    } catch (e) {
      if (e instanceof YandexPlaylistError) throw e;
      // Continue with targetUrl
    }
  }

  // Step 2: If we have owner and kind, try public web and mobile API endpoints
  if (currentOwner && currentKind) {
    // Attempt A: Web handlers endpoint
    try {
      const handlersUrl = `https://music.yandex.ru/handlers/playlist.jsx?owner=${encodeURIComponent(currentOwner)}&kinds=${encodeURIComponent(currentKind)}&light=true`;
      const response = await fetchWithBackoff(
        handlersUrl,
        {
          headers: {
            "Accept": "application/json",
            "X-Retpath-Y": `https://music.yandex.ru/users/${encodeURIComponent(currentOwner)}/playlists/${currentKind}`,
          },
        },
        { timeoutMs, retries, fetcher, deadline }
      );

      const text = await response.text();
      if (text.startsWith("{")) {
        const data = JSON.parse(text) as {
          playlist?: {
            title?: string;
            trackCount?: number;
            tracks?: unknown[];
            visibility?: string;
          };
        };

        if (data.playlist?.visibility === "private") {
          throw new YandexPlaylistError("private", "Плейлист приватный или доступ ограничен. Сделайте его публичным в настройках.");
        }

        if (Array.isArray(data.playlist?.tracks) && data.playlist.tracks.length > 0) {
          const parsedTracks: YandexTrack[] = [];
          for (const item of data.playlist.tracks) {
            const parsed = parseRawTrack(item);
            if (parsed) parsedTracks.push(parsed);
          }
          const deduplicated = deduplicateTracks(parsedTracks);
          return cacheAndReturn(
            normalized.canonicalUrl,
            {
              id: `${currentOwner}:${currentKind}`,
              title: data.playlist.title ?? `Плейлист ${currentOwner}`,
              owner: currentOwner,
              trackCount: deduplicated.length,
              tracks: deduplicated,
            },
            options.skipCache
          );
        }
      }
    } catch (e) {
      if (e instanceof YandexPlaylistError && (e.code === "private" || e.code === "not_found")) {
        throw e;
      }
      // Fall through to page inspection
    }

    // Attempt B: api.music.yandex.net endpoint
    try {
      const apiUrl = `https://api.music.yandex.net/users/${encodeURIComponent(currentOwner)}/playlists/${currentKind}`;
      const response = await fetchWithBackoff(
        apiUrl,
        {
          headers: {
            "Accept": "application/json",
            "X-Yandex-Music-Client": "YandexMusicAndroid/24023251",
          },
        },
        { timeoutMs, retries, fetcher, deadline }
      );

      const text = await response.text();
      if (text.startsWith("{")) {
        const data = JSON.parse(text) as {
          result?: {
            title?: string;
            trackCount?: number;
            tracks?: unknown[];
            visibility?: string;
          };
          error?: { name?: string; message?: string };
        };

        if (data.error?.name === "not-found") {
          throw new YandexPlaylistError("not_found", "Плейлист не найден. Проверьте правильность ссылки.");
        }
        if (data.result?.visibility === "private") {
          throw new YandexPlaylistError("private", "Плейлист приватный или доступ ограничен. Сделайте его публичным в настройках.");
        }

        if (Array.isArray(data.result?.tracks) && data.result.tracks.length > 0) {
          const parsedTracks: YandexTrack[] = [];
          for (const item of data.result.tracks) {
            const parsed = parseRawTrack(item);
            if (parsed) parsedTracks.push(parsed);
          }
          const deduplicated = deduplicateTracks(parsedTracks);
          return cacheAndReturn(
            normalized.canonicalUrl,
            {
              id: `${currentOwner}:${currentKind}`,
              title: data.result.title ?? `Плейлист ${currentOwner}`,
              owner: currentOwner,
              trackCount: deduplicated.length,
              tracks: deduplicated,
            },
            options.skipCache
          );
        }
      }
    } catch (e) {
      if (e instanceof YandexPlaylistError && (e.code === "private" || e.code === "not_found")) {
        throw e;
      }
      // Fall through to page inspection
    }
  }

  // Step 3: Fetch the HTML page (covers UUID playlists, page fallbacks, and redirects)
  const pageUrl = currentUuid
    ? `https://music.yandex.ru/playlists/${encodeURIComponent(currentUuid)}`
    : targetUrl;

  const htmlResponse = await fetchWithBackoff(
    pageUrl,
    {
      headers: {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
    },
    { timeoutMs, retries, fetcher, deadline }
  );

  const html = await htmlResponse.text();
  const extracted = extractFromHtmlState(html);

  // If HTML revealed owner & kind that we didn't have before, try API one more time
  if ((!currentOwner || !currentKind) && extracted.owner && extracted.kind) {
    try {
      const nested = await fetchYandexPlaylist(
        `https://music.yandex.ru/users/${encodeURIComponent(extracted.owner)}/playlists/${extracted.kind}`,
        options
      );
      if (nested.tracks.length > 0) {
        return cacheAndReturn(normalized.canonicalUrl, nested, options.skipCache);
      }
    } catch {
      // Use tracks extracted directly from HTML
    }
  }

  if (extracted.tracks.length === 0) {
    // If no tracks extracted and page has not-found indicators
    if (html.includes("404") || html.includes("notFound")) {
      throw new YandexPlaylistError("not_found", "Плейлист не найден. Проверьте правильность ссылки.");
    }
    throw new YandexPlaylistError(
      "upstream_error",
      "Не удалось извлечь треки из плейлиста. Убедитесь, что плейлист публичный и содержит треки."
    );
  }

  return cacheAndReturn(
    normalized.canonicalUrl,
    {
      id: currentUuid ?? (extracted.kind ? `${extracted.owner}:${extracted.kind}` : "yandex-playlist"),
      title: extracted.title ?? "Плейлист Яндекс Музыки",
      owner: extracted.owner ?? currentOwner,
      trackCount: extracted.tracks.length,
      tracks: extracted.tracks,
    },
    options.skipCache
  );
}
