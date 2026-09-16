import { describe, expect, test, vi } from "vitest";
import {
  normalizeYandexPlaylistUrl,
  extractFromHtmlState,
  fetchYandexPlaylist,
  YandexPlaylistError,
  clearYandexPlaylistCache,
} from "../../src/lib/yandex/playlist";

describe("Yandex Music playlist URL normalization", () => {
  test("normalizes classic user playlist URL and strips tracking params", () => {
    const url = "https://music.yandex.ru/users/yamusic-top/playlists/1076?utm_source=share&utm_medium=copy_link";
    const res = normalizeYandexPlaylistUrl(url);
    expect(res.canonicalUrl).toBe("https://music.yandex.ru/users/yamusic-top/playlists/1076");
    expect(res.type).toBe("user_playlist");
    expect(res.owner).toBe("yamusic-top");
    expect(res.kind).toBe("1076");
  });

  test("normalizes modern UUID playlist URL", () => {
    const url = "https://music.yandex.ru/playlists/72b84cfd-f06b-4e8c-85e6-7ceab2eb8466";
    const res = normalizeYandexPlaylistUrl(url);
    expect(res.canonicalUrl).toBe("https://music.yandex.ru/playlists/72b84cfd-f06b-4e8c-85e6-7ceab2eb8466");
    expect(res.type).toBe("uuid_playlist");
    expect(res.uuid).toBe("72b84cfd-f06b-4e8c-85e6-7ceab2eb8466");
  });

  test("normalizes playlist with prefix like ps.uuid", () => {
    const url = "music.yandex.ru/playlists/ps.1a2b3c4d";
    const res = normalizeYandexPlaylistUrl(url);
    expect(res.canonicalUrl).toBe("https://music.yandex.ru/playlists/ps.1a2b3c4d");
    expect(res.type).toBe("uuid_playlist");
    expect(res.uuid).toBe("ps.1a2b3c4d");
  });

  test("normalizes legacy query format ?owner=...&kinds=...", () => {
    const url = "https://music.yandex.ru/?owner=music-lover&kinds=42";
    const res = normalizeYandexPlaylistUrl(url);
    expect(res.canonicalUrl).toBe("https://music.yandex.ru/users/music-lover/playlists/42");
    expect(res.type).toBe("user_playlist");
    expect(res.owner).toBe("music-lover");
    expect(res.kind).toBe("42");
  });

  test("normalizes lk.music.yandex.ru redirect link", () => {
    const url = "https://lk.music.yandex.ru/short/xyz123";
    const res = normalizeYandexPlaylistUrl(url);
    expect(res.type).toBe("lk_redirect");
  });

  test("throws YandexPlaylistError on invalid or empty URLs", () => {
    expect(() => normalizeYandexPlaylistUrl("")).toThrow(YandexPlaylistError);
    expect(() => normalizeYandexPlaylistUrl("https://spotify.com/playlist/123")).toThrow(YandexPlaylistError);
    expect(() => normalizeYandexPlaylistUrl("not-a-valid-domain")).toThrow(YandexPlaylistError);
  });
});

describe("extractFromHtmlState", () => {
  test("extracts tracks from JSON-LD Schema.org MusicPlaylist", () => {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Плейлист дня — Яндекс Музыка</title>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "MusicPlaylist",
          "name": "Плейлист дня",
          "track": [
            {
              "@type": "MusicRecording",
              "name": "Starman",
              "byArtist": { "@type": "MusicGroup", "name": "David Bowie" }
            },
            {
              "@type": "MusicRecording",
              "name": "Heroes",
              "author": "David Bowie"
            }
          ]
        }
        </script>
      </head>
      <body></body>
      </html>
    `;
    const res = extractFromHtmlState(html);
    expect(res.title).toBe("Плейлист дня");
    expect(res.tracks).toHaveLength(2);
    expect(res.tracks[0]?.title).toBe("Starman");
    expect(res.tracks[0]?.artists).toEqual(["David Bowie"]);
    expect(res.tracks[1]?.title).toBe("Heroes");
  });

  test("throws on 404 page content", () => {
    const html = "<html><body>404: This page could not be found</body></html>";
    expect(() => extractFromHtmlState(html)).toThrow(YandexPlaylistError);
  });
});

describe("fetchYandexPlaylist error handling and deduplication", () => {
  test("returns tracks successfully from public handlers endpoint", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          playlist: {
            title: "Рок 80-х",
            visibility: "public",
            tracks: [
              {
                id: 101,
                title: "Blue Monday",
                artists: [{ name: "New Order" }],
              },
              {
                id: 102,
                title: "Blue Monday",
                artists: [{ name: "New Order" }],
              },
              {
                id: 103,
                title: "Love Will Tear Us Apart",
                artists: [{ name: "Joy Division" }],
              },
            ],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const res = await fetchYandexPlaylist("https://music.yandex.ru/users/tester/playlists/1", {
      fetcher: mockFetcher,
      timeoutMs: 1000,
      retries: 1,
    });

    expect(res.title).toBe("Рок 80-х");
    expect(res.tracks).toHaveLength(2);
    expect(res.tracks[0]?.title).toBe("Blue Monday");
    expect(res.tracks[0]?.artists).toEqual(["New Order"]);
    expect(res.tracks[1]?.title).toBe("Love Will Tear Us Apart");
  });

  test("throws private error when playlist visibility is private", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          playlist: {
            visibility: "private",
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    await expect(
      fetchYandexPlaylist("https://music.yandex.ru/users/tester/playlists/2", {
        fetcher: mockFetcher,
        timeoutMs: 1000,
        retries: 1,
      })
    ).rejects.toThrowError(/приватный/i);
  });

  test("throws not_found error on HTTP 404", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(new Response("Not Found", { status: 404 }));

    await expect(
      fetchYandexPlaylist("https://music.yandex.ru/users/tester/playlists/999999", {
        fetcher: mockFetcher,
        timeoutMs: 1000,
        retries: 1,
      })
    ).rejects.toThrowError(/не найден/i);
  });

  test("serves repeated playlist requests from in-memory cache without hitting fetcher", async () => {
    clearYandexPlaylistCache();
    const mockFetcher = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            playlist: {
              title: "Кеш-тест",
              visibility: "public",
              tracks: [
                {
                  id: 1,
                  title: "Track 1",
                  artists: [{ name: "Artist 1" }],
                },
              ],
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
    );

    const res1 = await fetchYandexPlaylist("https://music.yandex.ru/users/cache-user/playlists/10", {
      fetcher: mockFetcher,
      timeoutMs: 1000,
      retries: 1,
    });
    expect(res1.title).toBe("Кеш-тест");
    expect(mockFetcher).toHaveBeenCalledTimes(1);

    // Second call to same URL should hit in-memory cache
    const res2 = await fetchYandexPlaylist("https://music.yandex.ru/users/cache-user/playlists/10", {
      fetcher: mockFetcher,
      timeoutMs: 1000,
      retries: 1,
    });
    expect(res2.title).toBe("Кеш-тест");
    expect(mockFetcher).toHaveBeenCalledTimes(1); // Still 1!

    // After clearing cache, it should hit fetcher again
    clearYandexPlaylistCache();
    await fetchYandexPlaylist("https://music.yandex.ru/users/cache-user/playlists/10", {
      fetcher: mockFetcher,
      timeoutMs: 1000,
      retries: 1,
    });
    expect(mockFetcher).toHaveBeenCalledTimes(2);
  });
});
