import { expect, test } from "vitest";
import { TasteInputError, parseTasteInput } from "../../src/lib/tastelift/input";

const song = (number: number) => ({ artist: `Artist ${number}`, title: `Song ${number}` });

test("accepts 5, 200, and 201 songs so large taste sets retain every seed", () => {
  expect(parseTasteInput({ songs: Array.from({ length: 5 }, (_, index) => song(index)) })).toHaveLength(5);
  expect(parseTasteInput({ songs: Array.from({ length: 200 }, (_, index) => song(index)) })).toHaveLength(200);
  expect(parseTasteInput({ songs: Array.from({ length: 201 }, (_, index) => song(index)) })).toHaveLength(201);
});

test("rejects a batch larger than the 500-song contract", () => {
  expect(() => parseTasteInput({ songs: Array.from({ length: 501 }, (_, index) => song(index)) })).toThrow(TasteInputError);
});

test("normalizes Unicode and whitespace while retaining original distinct seeds", () => {
  expect(parseTasteInput({ songs: [
    { artist: "  Ｂｊ\u00f6rk ", title: " J\u00f3ga  " },
    { artist: "Bj\u00f6rk", title: "J\u00f3ga (Live)" },
    song(2), song(3), song(4),
  ] }).slice(0, 2)).toEqual([
    { artist: "Bj\u00f6rk", title: "J\u00f3ga" },
    { artist: "Bj\u00f6rk", title: "J\u00f3ga (Live)" },
  ]);
});

test("extracts a Spotify track id only from valid Spotify track URLs", () => {
  expect(parseTasteInput({ songs: [
    { artist: "A", title: "B", spotify_url: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC?si=abc" },
    song(1), song(2), song(3), song(4),
  ] })[0]).toEqual({ artist: "A", title: "B", spotifyUrl: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC?si=abc", spotifyId: "4uLU6hMCjMI75M1A2tKUQC" });
  expect(() => parseTasteInput({ songs: [
    { artist: "A", title: "B", spotify_url: "https://example.test/track/not-spotify" },
    song(1), song(2), song(3), song(4),
  ] })).toThrow(/Spotify/i);
});

test("rejects duplicate-only batches instead of silently dropping all seeds", () => {
  expect(() => parseTasteInput({ songs: Array.from({ length: 5 }, () => ({ artist: "The Same", title: "Track" })) })).toThrow(/duplicate/i);
});

test("deduplicates only identical normalized artist-title pairs in first-seen order", () => {
  const parsed = parseTasteInput({ songs: [
    { artist: "A", title: "First" },
    { artist: " A ", title: " First " },
    { artist: "A", title: "First (Live)" },
    song(2), song(3), song(4),
  ] });
  expect(parsed).toEqual([
    { artist: "A", title: "First" },
    { artist: "A", title: "First (Live)" },
    song(2), song(3), song(4),
  ]);
});
