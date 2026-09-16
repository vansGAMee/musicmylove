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

test("accepts Spotify extended-history JSON and promotes the most-played distinct tracks", () => {
  const history = [
    { master_metadata_album_artist_name: "Rare Artist", master_metadata_track_name: "Deep Cut", spotify_track_uri: "spotify:track:4uLU6hMCjMI75M1A2tKUQC", ms_played: 20_000 },
    { master_metadata_album_artist_name: "Main Artist", master_metadata_track_name: "Hit", spotify_track_uri: "spotify:track:0VjIjW4GlUZAMYd2vXMi3b", ms_played: 5_000 },
    { master_metadata_album_artist_name: "Rare Artist", master_metadata_track_name: "Deep Cut", spotify_track_uri: "spotify:track:4uLU6hMCjMI75M1A2tKUQC", ms_played: 30_000 },
    ...Array.from({ length: 4 }, (_, index) => ({ master_metadata_album_artist_name: `Artist ${index}`, master_metadata_track_name: `Song ${index}`, ms_played: 10_000 - index })),
    { episode_name: "Podcast", ms_played: 999_999 },
  ];

  const parsed = parseTasteInput(history);

  expect(parsed).toHaveLength(6);
  expect(parsed[0]).toEqual({ artist: "Rare Artist", title: "Deep Cut", spotifyUrl: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC", spotifyId: "4uLU6hMCjMI75M1A2tKUQC" });
  expect(parsed.filter((track) => track.artist === "Rare Artist")).toHaveLength(1);
});

test("reduces histories larger than the serving bound to their 500 strongest distinct tracks", () => {
  const history = Array.from({ length: 501 }, (_, index) => ({ artistName: `Artist ${index}`, trackName: `Song ${index}`, msPlayed: index + 1 }));
  const parsed = parseTasteInput(history);
  expect(parsed).toHaveLength(500);
  expect(parsed[0]).toEqual({ artist: "Artist 500", title: "Song 500" });
  expect(parsed.some((track) => track.artist === "Artist 0")).toBe(false);
});

test("accepts Exportify CSV format with track ids and artist names", () => {
  const csv = `Spotify Track Id,Track Name,Artist Name(s),Album Name,Release Date
4uLU6hMCjMI75M1A2tKUQC,Never Gonna Give You Up,Rick Astley,Whenever You Need Somebody,1987-11-12
0VjIjW4GlUZAMYd2vXMi3b,Blinding Lights,The Weeknd,After Hours,2020-03-20
60bd9d53-01ff-4562-8058,Everything in Its Right Place,Radiohead,Kid A,2000-10-02
5566d65b-2089-4cf5-9766,"Hyper-ballad, Remix",Björk,Post,1995-06-13
8a49dba0-253a-4535-b87f,Roads,Portishead,Dummy,1994-08-22
f3bba4cd-8018-468b-902e,Teardrop,Massive Attack,Mezzanine,1998-04-20`;

  const parsed = parseTasteInput(csv);
  expect(parsed).toHaveLength(6);
  expect(parsed[0].artist).toBe("Rick Astley");
  expect(parsed[0].title).toBe("Never Gonna Give You Up");
  expect(parsed[0].spotifyUrl).toBe("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC");
  expect(parsed[0].spotifyId).toBe("4uLU6hMCjMI75M1A2tKUQC");
  expect(parsed[3].title).toBe("Hyper-ballad, Remix");
});

test("accepts Exportify JSON array format with Track Name and Artist Name(s)", () => {
  const json = [
    { "Spotify Track Id": "4uLU6hMCjMI75M1A2tKUQC", "Track Name": "Never Gonna Give You Up", "Artist Name(s)": "Rick Astley" },
    { "Spotify Track Id": "0VjIjW4GlUZAMYd2vXMi3b", "Track Name": "Blinding Lights", "Artist Name(s)": "The Weeknd" },
    { "Track Name": "Roads", "Artist Name(s)": "Portishead" },
    { "Track Name": "Everything in Its Right Place", "Artist Name(s)": "Radiohead" },
    { "Track Name": "Hyper-Ballad", "Artist Name(s)": "Björk" },
  ];
  const parsed = parseTasteInput(json);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({
    artist: "Rick Astley",
    title: "Never Gonna Give You Up",
    spotifyUrl: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
    spotifyId: "4uLU6hMCjMI75M1A2tKUQC",
  });
});

test("accepts Spotify playlist JSON with tracks.items hierarchy", () => {
  const playlist = {
    name: "Indie Mix",
    tracks: {
      items: [
        { track: { name: "Roads", artists: [{ name: "Portishead" }], id: "4uLU6hMCjMI75M1A2tKUQC" } },
        { track: { name: "Teardrop", artists: [{ name: "Massive Attack" }] } },
        { track: { name: "Idioteque", artists: [{ name: "Radiohead" }] } },
        { track: { name: "Army of Me", artists: [{ name: "Björk" }] } },
        { track: { name: "Midnight City", artists: [{ name: "M83" }] } },
      ],
    },
  };
  const parsed = parseTasteInput(playlist);
  expect(parsed).toHaveLength(5);
  expect(parsed[0].artist).toBe("Portishead");
  expect(parsed[0].title).toBe("Roads");
  expect(parsed[0].spotifyId).toBe("4uLU6hMCjMI75M1A2tKUQC");
});

test("accepts plain text tracklist with Artist - Title lines", () => {
  const text = `Portishead - Roads
Massive Attack — Teardrop
Radiohead - Everything in Its Right Place
Björk - Hyperballad
M83 - Midnight City`;

  const parsed = parseTasteInput(text);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Portishead", title: "Roads" });
  expect(parsed[1]).toEqual({ artist: "Massive Attack", title: "Teardrop" });
});

test("supports allowPartial option for 1-4 tracks without error", () => {
  const shortText = `Portishead - Roads\nMassive Attack - Teardrop`;
  const parsed = parseTasteInput(shortText, { allowPartial: true });
  expect(parsed).toHaveLength(2);
  expect(parsed[0].artist).toBe("Portishead");
  expect(parsed[1].artist).toBe("Massive Attack");
});

test("accepts M3U / M3U8 playlist with #EXTINF directives", () => {
  const m3u = `#EXTM3U
#EXTINF:245,Queen - Bohemian Rhapsody
/music/queen/bohemian.mp3
#EXTINF:180,The Beatles - Yesterday
/music/beatles/yesterday.mp3
#EXTINF:300,Pink Floyd - Comfortably Numb
/music/pink_floyd/comfortably.mp3
#EXTINF:210,David Bowie - Heroes
/music/bowie/heroes.mp3
#EXTINF:260,Led Zeppelin - Stairway to Heaven
/music/zeppelin/stairway.mp3`;

  const parsed = parseTasteInput(m3u);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Queen", title: "Bohemian Rhapsody" });
  expect(parsed[1]).toEqual({ artist: "The Beatles", title: "Yesterday" });
  expect(parsed[2]).toEqual({ artist: "Pink Floyd", title: "Comfortably Numb" });
});

test("accepts PLS playlist format", () => {
  const pls = `[playlist]
File1=http://stream.example.com/1
Title1=Queen - Bohemian Rhapsody
Length1=245
File2=http://stream.example.com/2
Title2=The Beatles - Yesterday
Length2=180
File3=http://stream.example.com/3
Title3=Pink Floyd - Comfortably Numb
Length3=300
File4=http://stream.example.com/4
Title4=David Bowie - Heroes
Length4=210
File5=http://stream.example.com/5
Title5=Led Zeppelin - Stairway to Heaven
Length5=260
NumberOfEntries=5
Version=2`;

  const parsed = parseTasteInput(pls);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Queen", title: "Bohemian Rhapsody" });
  expect(parsed[4]).toEqual({ artist: "Led Zeppelin", title: "Stairway to Heaven" });
});

test("accepts numbered and bulleted plain text tracklists", () => {
  const numbered = `1. Queen - Bohemian Rhapsody
02) The Beatles - Yesterday
[3] Pink Floyd - Comfortably Numb
4 - David Bowie - Heroes
• Led Zeppelin - Stairway to Heaven`;

  const parsed = parseTasteInput(numbered);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Queen", title: "Bohemian Rhapsody" });
  expect(parsed[1]).toEqual({ artist: "The Beatles", title: "Yesterday" });
  expect(parsed[2]).toEqual({ artist: "Pink Floyd", title: "Comfortably Numb" });
  expect(parsed[3]).toEqual({ artist: "David Bowie", title: "Heroes" });
  expect(parsed[4]).toEqual({ artist: "Led Zeppelin", title: "Stairway to Heaven" });
});

test("accepts Russian CSV with Исполнитель and Название headers and semicolon delimiter", () => {
  const csv = `Исполнитель;Название;Альбом
Кино;Группа крови;Группа крови
Наутилус Помпилиус;Крылья;Крылья
ДДТ;Что такое осень;Актриса Весна
Аквариум;Поезд в огне;Равноденствие
Сплин;Моё сердце;25 кадр`;

  const parsed = parseTasteInput(csv);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Кино", title: "Группа крови" });
  expect(parsed[1]).toEqual({ artist: "Наутилус Помпилиус", title: "Крылья" });
  expect(parsed[4]).toEqual({ artist: "Сплин", title: "Моё сердце" });
});

test("accepts TSV format without headers", () => {
  const tsv = `Queen\tBohemian Rhapsody
The Beatles\tYesterday
Pink Floyd\tComfortably Numb
David Bowie\tHeroes
Led Zeppelin\tStairway to Heaven`;

  const parsed = parseTasteInput(tsv);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Queen", title: "Bohemian Rhapsody" });
  expect(parsed[1]).toEqual({ artist: "The Beatles", title: "Yesterday" });
});

test("accepts Apple Music JSON export format with data[].attributes", () => {
  const appleJson = {
    data: [
      { attributes: { artistName: "Radiohead", name: "Karma Police" } },
      { attributes: { artistName: "Portishead", name: "Glory Box" } },
      { attributes: { artistName: "Massive Attack", name: "Angel" } },
      { attributes: { artistName: "Björk", name: "Army of Me" } },
      { attributes: { artistName: "M83", name: "Midnight City" } },
    ],
  };

  const parsed = parseTasteInput(appleJson);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Radiohead", title: "Karma Police" });
  expect(parsed[1]).toEqual({ artist: "Portishead", title: "Glory Box" });
});

test("accepts Last.fm JSON format with recenttracks.track and nested artist name", () => {
  const lastfmJson = {
    recenttracks: {
      track: [
        { artist: { "#text": "Radiohead" }, name: "Karma Police" },
        { artist: { name: "Portishead" }, name: "Glory Box" },
        { artist: { "#text": "Massive Attack" }, name: "Angel" },
        { artist: { name: "Björk" }, name: "Army of Me" },
        { artist: { "#text": "M83" }, name: "Midnight City" },
      ],
    },
  };

  const parsed = parseTasteInput(lastfmJson);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Radiohead", title: "Karma Police" });
  expect(parsed[1]).toEqual({ artist: "Portishead", title: "Glory Box" });
});

test("accepts objects with track instead of title and Russian keys", () => {
  const customJson = [
    { artist: "Queen", track: "Bohemian Rhapsody" },
    { Artist: "The Beatles", Track: "Yesterday" },
    { Исполнитель: "Кино", Название: "Группа крови" },
    { author: "David Bowie", song: "Heroes" },
    { performer: "Led Zeppelin", Title: "Stairway to Heaven" },
  ];

  const parsed = parseTasteInput(customJson);
  expect(parsed).toHaveLength(5);
  expect(parsed[0]).toEqual({ artist: "Queen", title: "Bohemian Rhapsody" });
  expect(parsed[1]).toEqual({ artist: "The Beatles", title: "Yesterday" });
  expect(parsed[2]).toEqual({ artist: "Кино", title: "Группа крови" });
  expect(parsed[3]).toEqual({ artist: "David Bowie", title: "Heroes" });
  expect(parsed[4]).toEqual({ artist: "Led Zeppelin", title: "Stairway to Heaven" });
});


