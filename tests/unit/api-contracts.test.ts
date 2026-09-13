import { expect, test } from "vitest";
import { parseSearch, parseSimilar, parseSpotify } from "../../src/lib/listenbrainz";

test("parses only complete recording-search rows", () => {
  expect(parseSearch([{ recording_mbid: "m", recording_name: "Song", artist_credit_name: "Artist", release_name: "Album" }, { nope: true }])).toEqual([
    { mbid: "m", title: "Song", artist: "Artist", release: "Album" },
  ]);
});

test("rejects malformed similarity payloads", () => {
  expect(() => parseSimilar({ error: true })).toThrow(/similarity/i);
});

test("uses an exact Spotify id when present", () => {
  expect(parseSpotify([{ spotify_track_ids: ["abc"] }])).toBe("abc");
  expect(parseSpotify([])).toBeNull();
});
