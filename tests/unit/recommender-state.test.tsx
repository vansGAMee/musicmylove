// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import MusicRecommender from "../../src/components/MusicRecommender";

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); localStorage.clear(); });

test("prefetches each selection and automatically returns results after the fifth", async () => {
  vi.useFakeTimers();
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/search")) {
      const title = new URL(url, "http://x").searchParams.get("q")!;
      return new Response(JSON.stringify([{ mbid: `seed-${title}`, title, artist: `Artist ${title}` }]));
    }
    if (url.startsWith("/api/tastelift")) {
      return new Response(JSON.stringify({
        recommendations: Array.from({ length: 40 }, (_, index) => ({
          mbid: `candidate-${index}`,
          title: `Candidate ${index}`,
          artist: `Artist ${index}`,
          score: 100 - index,
          strongestTasteHead: index % 4,
          seedSupport: 2,
          popularityPercentile: 0.5,
          noveltyLiftScore: 1.2,
          spotifyLink: `https://open.spotify.com/search/Artist%20${index}%20Candidate%20${index}`,
        })),
      }));
    }
    return new Response(JSON.stringify({ id: null }));
  });
  vi.stubGlobal("fetch", fetcher);
  render(<MusicRecommender />);
  expect(screen.queryByRole("button", { name: /generate/i })).not.toBeInTheDocument();
  for (const title of ["one", "two", "three", "four", "five"]) {
    fireEvent.change(screen.getByRole("combobox"), { target: { value: title } });
    await act(async () => { await vi.advanceTimersByTimeAsync(350); });
    expect(screen.getByRole("button", { name: new RegExp(title, "i") })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: new RegExp(title, "i") }));
  }
  await act(async () => { await Promise.resolve(); });
  expect(screen.getAllByTestId("recommendation")).toHaveLength(40);
  expect(screen.getByText("5 / 5")).toBeInTheDocument();
  expect(fetcher.mock.calls.filter(([url]) => String(url).startsWith("/api/tastelift"))).toHaveLength(1);
});

test("handles uploading 417 Russian tracks, passes all 417 seeds to tastelift, and never renders an iframe", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.startsWith("/api/tastelift")) {
      const body = JSON.parse(String(init?.body));
      expect(body.songs.length).toBe(417);
      return new Response(JSON.stringify({
        recommendations: Array.from({ length: 40 }, (_, index) => ({
          mbid: `candidate-${index}`,
          title: `Рекомендация ${index}`,
          artist: `Артист ${index}`,
          score: 100 - index,
          strongestTasteHead: 0,
          seedSupport: 2,
          popularityPercentile: 0.5,
          noveltyLiftScore: 1.2,
          spotifyLink: `https://open.spotify.com/search/Candidate%20${index}`,
        })),
        seeds: body.songs.map((s: { artist: string; title: string }, i: number) => ({
          input: s,
          status: "resolved",
          source: "text",
          track: { mbid: `seed-${i}`, artist: s.artist, title: s.title },
        })),
      }));
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetcher);

  const { container } = render(<MusicRecommender />);

  // Verify no iframe anywhere
  expect(container.querySelector("iframe")).toBeNull();

  // Create a 417-song text content
  const russianTracks = [
    "Михаил Круг — Фраер",
    "ВИА Гра — Притяжения больше нет",
    "Валерий Меладзе — Иностранец",
    "Самоцветы — На дальней станции сойду",
    ...Array.from({ length: 413 }, (_, i) => `Русский Исполнитель ${i} — Трек ${i}`),
  ].join("\n");

  const file = new File([russianTracks], "playlist.txt", { type: "text/plain" });
  file.text = async () => russianTracks;
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  expect(input).not.toBeNull();

  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
  });
  await act(async () => {
    await Promise.resolve();
  });

  // Verify tastelift was called with all 417 seeds
  const tasteliftCalls = fetcher.mock.calls.filter(([url]) => String(url).startsWith("/api/tastelift"));
  expect(tasteliftCalls).toHaveLength(1);
  const sentPayload = JSON.parse(String(tasteliftCalls[0]?.[1]?.body));
  expect(sentPayload.songs).toHaveLength(417);
  expect(sentPayload.songs[0].artist).toBe("Михаил Круг");
  expect(sentPayload.songs[0].title).toBe("Фраер");

  // Verify 40 distinct recommendations rendered
  const recs = screen.getAllByTestId("recommendation");
  expect(recs.length).toBeGreaterThanOrEqual(40);
  expect(screen.getAllByText("Рекомендация 0").length).toBeGreaterThanOrEqual(1);
  expect(screen.getAllByText("Рекомендация 39").length).toBeGreaterThanOrEqual(1);

  // Verify no error message
  expect(screen.queryByText(/pipeline is temporarily unavailable/i)).toBeNull();

  // Verify no iframe is ever created
  expect(container.querySelector("iframe")).toBeNull();
});

test("handles Yandex playlist import successfully without ever rendering an iframe", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.startsWith("/api/yandex/playlist")) {
      return new Response(JSON.stringify({
        ok: true,
        tracks: Array.from({ length: 50 }, (_, i) => ({
          id: `ym-${i}`,
          title: `Яндекс Трек ${i}`,
          artists: [`Яндекс Артист ${i}`],
        })),
      }));
    }
    if (url.startsWith("/api/tastelift")) {
      const body = JSON.parse(String(init?.body));
      expect(body.songs.length).toBe(50);
      return new Response(JSON.stringify({
        recommendations: Array.from({ length: 40 }, (_, index) => ({
          mbid: `candidate-${index}`,
          title: `YM Рекомендация ${index}`,
          artist: `YM Артист ${index}`,
          score: 100 - index,
          strongestTasteHead: 0,
          seedSupport: 2,
          popularityPercentile: 0.5,
          noveltyLiftScore: 1.2,
          spotifyLink: `https://open.spotify.com/search/YM%20${index}`,
        })),
        seeds: body.songs.map((s: { artist: string; title: string }, i: number) => ({
          input: s,
          status: "resolved",
          source: "text",
          track: { mbid: `ym-seed-${i}`, artist: s.artist, title: s.title },
        })),
      }));
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetcher);

  const { container } = render(<MusicRecommender />);

  // Fill in Yandex URL input
  const input = container.querySelector(".yandex-input") as HTMLInputElement;
  fireEvent.change(input, { target: { value: "https://music.yandex.ru/playlists/59b1329f-1186-c562-f8c5-4f8e397656a8" } });

  // Submit Yandex import form
  const submitBtn = container.querySelector(".yandex-submit-btn") as HTMLButtonElement;
  await act(async () => {
    fireEvent.click(submitBtn);
  });

  // Verify tastelift received all 50 seeds
  const tasteliftCalls = fetcher.mock.calls.filter(([url]) => String(url).startsWith("/api/tastelift"));
  expect(tasteliftCalls).toHaveLength(1);
  const sentPayload = JSON.parse(String(tasteliftCalls[0]?.[1]?.body));
  expect(sentPayload.songs).toHaveLength(50);

  // Verify recommendations loaded
  expect(screen.getAllByText("YM Рекомендация 0").length).toBeGreaterThanOrEqual(1);

  // Verify no iframe anywhere
  expect(container.querySelector("iframe")).toBeNull();
});

test("handles Yandex geo-block error gracefully without ever rendering an iframe", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/yandex/playlist")) {
      return new Response(
        JSON.stringify({
          ok: false,
          code: "geo_blocked",
          error: "Сервис Яндекс Музыки недоступен из региона сервера (геоблокировка). Вставьте треки вручную.",
        }),
        { status: 451 }
      );
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetcher);

  const { container } = render(<MusicRecommender />);

  const input = container.querySelector(".yandex-input") as HTMLInputElement;
  fireEvent.change(input, { target: { value: "https://music.yandex.ru/playlists/59b1329f-1186-c562-f8c5-4f8e397656a8" } });

  const submitBtn = container.querySelector(".yandex-submit-btn") as HTMLButtonElement;
  await act(async () => {
    fireEvent.click(submitBtn);
  });

  // Verify notice is shown
  expect(container.querySelector(".yandex-notice")).toHaveTextContent(/недоступен|геоблокировка|region|unavailable/i);

  // Absolutely NO iframe is rendered
  expect(container.querySelector("iframe")).toBeNull();
  expect(container.querySelector(".yandex-iframe-wrapper")).toBeNull();
});

test("displays liked tracks in favorites tab and keeps them in the recommendation list", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/tastelift")) {
      return new Response(JSON.stringify({
        recommendations: [
          { mbid: "rec-1", title: "Loved Song", artist: "Loved Artist", score: 99, strongestTasteHead: 0, seedSupport: 1, popularityPercentile: 0.5, noveltyLiftScore: 1.5, spotifyLink: "" },
          { mbid: "rec-2", title: "Other Song", artist: "Other Artist", score: 95, strongestTasteHead: 1, seedSupport: 1, popularityPercentile: 0.5, noveltyLiftScore: 1.2, spotifyLink: "" },
        ],
        seeds: Array.from({ length: 5 }, (_, i) => ({
          input: { artist: `Seed Artist ${i}`, title: `Seed Title ${i}` },
          status: "resolved",
          track: { mbid: `seed-${i}`, artist: `Seed Artist ${i}`, title: `Seed Title ${i}` },
        })),
      }));
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetcher);

  const { container } = render(<MusicRecommender />);
  const playlistInput = [
    "Artist,Title",
    "Seed Artist 0,Seed Title 0",
    "Seed Artist 1,Seed Title 1",
    "Seed Artist 2,Seed Title 2",
    "Seed Artist 3,Seed Title 3",
    "Seed Artist 4,Seed Title 4",
  ].join("\n");

  const file = new File([playlistInput], "seeds.csv", { type: "text/csv" });
  file.text = async () => playlistInput;
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;

  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
  });
  await act(async () => { await Promise.resolve(); });

  // Recommendations are displayed
  expect(screen.getAllByText("Loved Song").length).toBeGreaterThanOrEqual(1);

  // Find the like button for Loved Song and click it
  const likeBtn = screen.getByRole("button", { name: /^like track: loved song$/i });
  await act(async () => {
    fireEvent.click(likeBtn);
  });

  // The track is still in the playlist (not removed!)
  expect(screen.getAllByText("Loved Song").length).toBeGreaterThanOrEqual(1);

  // Switch to Favorites tab
  const favoritesTabBtn = screen.getAllByRole("button", { name: /Favorites|Избранное/i })[0];
  await act(async () => {
    fireEvent.click(favoritesTabBtn);
  });

  // Favorites tab displays Loved Song!
  expect(screen.getAllByText("Loved Song").length).toBeGreaterThanOrEqual(1);
  // But does NOT display Other Song (which was not liked)
  expect(screen.queryByText("Other Song")).toBeNull();
});

test("excludes tracks from the entire playlist even when playlist has more than 500 tracks", async () => {
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/tastelift")) {
      return new Response(JSON.stringify({
        recommendations: [
          // This track is in the playlist at position 550 (outside the first 500)
          { mbid: "rec-in-playlist", title: "Existing Song 550", artist: "Existing Artist", score: 99, strongestTasteHead: 0, seedSupport: 1, popularityPercentile: 0.5, noveltyLiftScore: 1.5, spotifyLink: "" },
          // This track is genuinely new
          { mbid: "rec-brand-new", title: "Brand New Track", artist: "New Artist", score: 95, strongestTasteHead: 1, seedSupport: 1, popularityPercentile: 0.5, noveltyLiftScore: 1.2, spotifyLink: "" },
        ],
        seeds: Array.from({ length: 500 }, (_, i) => ({
          input: { artist: `Seed Artist ${i}`, title: `Seed Title ${i}` },
          status: "resolved",
          track: { mbid: `seed-${i}`, artist: `Seed Artist ${i}`, title: `Seed Title ${i}` },
        })),
      }));
    }
    return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetcher);

  const { container } = render(<MusicRecommender />);
  // Create 600 tracks with CSV header
  const tracks600 = [
    "Artist,Title",
    ...Array.from({ length: 600 }, (_, i) =>
      i === 550 ? "Existing Artist,Existing Song 550" : `Playlist Artist ${i},Playlist Song ${i}`
    ),
  ].join("\n");

  const file = new File([tracks600], "large_playlist.csv", { type: "text/csv" });
  file.text = async () => tracks600;
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;

  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
  });
  await act(async () => { await Promise.resolve(); });

  // Recommendations should contain Brand New Track, but NOT Existing Song 550
  const recs = screen.getAllByTestId("recommendation");
  const recTitles = recs.map((el) => el.querySelector(".track-title")?.textContent);
  expect(recTitles).toContain("Brand New Track");
  expect(recTitles).not.toContain("Existing Song 550");
});



