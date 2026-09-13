"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { diversify, rankCandidates } from "../lib/ranking";
import { spotifySearch } from "../lib/listenbrainz";
import type { RankedTrack, SeedTrack, SimilarTrack, Track } from "../lib/types";

const readFeedback = (): Record<string, "like" | "dislike"> => {
  if (typeof window === "undefined") return {};
  try { return JSON.parse(localStorage.getItem("musicmylove:v1:feedback") ?? "{}"); } catch { return {}; }
};

export default function MusicRecommender() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Track[]>([]);
  const [seeds, setSeeds] = useState<SeedTrack[]>([]);
  const [rankedPool, setRankedPool] = useState<RankedTrack[]>([]);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike">>(readFeedback);
  const lists = useRef<Record<string, SimilarTrack[]>>({});
  const pending = useRef<Partial<Record<string, Promise<SimilarTrack[]>>>>({});

  useEffect(() => {
    if (query.trim().length < 2 || seeds.length >= 5) { setSearchResults([]); return; }
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal });
        if (response.ok) setSearchResults(await response.json());
      } catch (error) { if (!(error instanceof DOMException && error.name === "AbortError")) setSearchResults([]); }
    }, 300);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, seeds.length]);

  const getSimilar = (seed: SeedTrack) => {
    const existing = pending.current[seed.mbid];
    if (existing) return existing;
    const cacheKey = `musicmylove:v1:similar:${seed.mbid}`;
    let stale: SimilarTrack[] = [];
    try { stale = JSON.parse(localStorage.getItem(cacheKey) ?? "[]"); } catch { /* ignore corrupt cache */ }
    pending.current[seed.mbid] = fetch(`/api/similar/${seed.mbid}`)
      .then(async (response) => { if (!response.ok) throw new Error("similarity failed"); const value = await response.json() as SimilarTrack[]; localStorage.setItem(cacheKey, JSON.stringify(value)); return value; })
      .catch(() => stale)
      .then((value) => { lists.current[seed.mbid] = value; return value; });
    return pending.current[seed.mbid]!;
  };

  const select = async (track: Track) => {
    if (seeds.some((seed) => seed.mbid === track.mbid) || seeds.length >= 5) return;
    const next = [...seeds, track];
    setSeeds(next); setQuery(""); setSearchResults([]);
    getSimilar(track);
    if (next.length === 5) {
      setLoading(true);
      await Promise.all(next.map(getSimilar));
      const ranked = rankCandidates(next, lists.current, "rrf");
      setRankedPool(ranked);
      setLoading(false);
    }
  };

  const saveFeedback = (mbid: string, value: "like" | "dislike") => {
    const next = { ...feedback, [mbid]: value };
    setFeedback(next); localStorage.setItem("musicmylove:v1:feedback", JSON.stringify(next));
  };
  const visible = useMemo(() => diversify(rankedPool.filter((track) => feedback[track.mbid] !== "dislike"), 20), [rankedPool, feedback]);

  return <main className="shell">
    <header><span className="eyebrow">MUSIC, FOUND BY MUSIC</span><h1>Give me 5 songs you love.<br /><em>I&apos;ll predict what you&apos;ll love next.</em></h1></header>
    <section className="picker" aria-label="Choose songs">
      <div className="progress"><span>{seeds.length} / 5</span><div>{Array.from({ length: 5 }, (_, i) => <i key={i} className={i < seeds.length ? "filled" : ""} />)}</div></div>
      {seeds.length < 5 && <div className="search"><input role="combobox" aria-label="Search for a song" placeholder="Search a song or artist…" value={query} onChange={(event) => setQuery(event.target.value)} autoComplete="off" />
        {searchResults.length > 0 && <div className="menu">{searchResults.slice(0, 8).map((track) => <button key={track.mbid} onClick={() => select(track)}><b>{track.title}</b><span>{track.artist}{track.release ? ` · ${track.release}` : ""}</span></button>)}</div>}
      </div>}
      <div className="seeds">{seeds.map((seed, i) => <div className="seed" key={seed.mbid}><span>{String(i + 1).padStart(2, "0")}</span><b>{seed.title}</b><small>{seed.artist}</small></div>)}</div>
    </section>
    {loading && <p className="loading">Reading the shape of your taste…</p>}
    {rankedPool.length > 0 && <section className="recommendations"><div className="result-heading"><div><span className="eyebrow">YOUR NEXT TWENTY</span><h2>A little familiar.<br />A little unexpected.</h2></div><p>Built from shared listening patterns—not genres, labels, or guesswork.</p></div>
      <div className="cards">{visible.map((track, i) => <article data-testid="recommendation" className="card" key={track.mbid}><span className="number">{String(i + 1).padStart(2, "0")}</span><div className="track"><h3>{track.title}</h3><p>{track.artist}</p><small>Picked from: {track.pickedFrom.map((seed) => seed.title).join(" + ") || "your mix"}</small></div><div className="actions"><a href={spotifySearch(track)} target="_blank" rel="noreferrer">Open in Spotify ↗</a><button className={feedback[track.mbid] === "like" ? "active" : ""} onClick={() => saveFeedback(track.mbid, "like")} aria-label={`Like ${track.title}`}>♥</button><button onClick={() => saveFeedback(track.mbid, "dislike")} aria-label={`Dislike ${track.title}`}>×</button></div></article>)}</div>
    </section>}
  </main>;
}
