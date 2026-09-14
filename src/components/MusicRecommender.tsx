"use client";

import { useEffect, useMemo, useState } from "react";
import type { RankedTrack, SeedTrack, Track } from "../lib/types";

interface TasteLiftApiRecommendation {
  mbid: string;
  artist: string;
  title: string;
  release?: string;
  score: number;
  strongestTasteHead: number;
  seedSupport: number;
  popularityPercentile: number;
  noveltyLiftScore: number;
  spotifyLink: string;
}

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
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike">>(readFeedback);
  const [spotifyLinks, setSpotifyLinks] = useState<Record<string, string>>({});

  useEffect(() => {
    if (query.trim().length < 2 || seeds.length >= 5) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal });
        if (response.ok) setSearchResults(await response.json());
      } catch (error) { if (!(error instanceof DOMException && error.name === "AbortError")) setSearchResults([]); }
    }, 300);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, seeds.length]);

  const select = async (track: Track) => {
    if (seeds.some((seed) => seed.mbid === track.mbid) || seeds.length >= 5) return;
    const next = [...seeds, track];
    setSeeds(next); setQuery(""); setSearchResults([]);
    if (next.length === 5) {
      setLoading(true); setError("");
      try {
        const response = await fetch("/api/tastelift", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ songs: next.map((seed) => ({ artist: seed.artist, title: seed.title })) }) });
        const payload = await response.json() as { error?: string; recommendations?: TasteLiftApiRecommendation[] };
        if (!response.ok || !payload.recommendations) throw new Error(payload.error ?? "TasteLift recommendation failed");
        setRankedPool(payload.recommendations.map((item) => ({
          mbid: item.mbid, artist: item.artist, title: item.title, ...(item.release ? { release: item.release } : {}),
          score: item.score, features: Array(17).fill(0), pickedFrom: [], tasteHeadIndex: item.strongestTasteHead,
          seedSupport: item.seedSupport, popularityPercentile: item.popularityPercentile, liftScore: item.noveltyLiftScore,
        })));
        setSpotifyLinks(Object.fromEntries(payload.recommendations.map((item) => [item.mbid, item.spotifyLink])));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "TasteLift recommendation failed");
      } finally { setLoading(false); }
    }
  };

  const saveFeedback = (mbid: string, value: "like" | "dislike") => {
    const next = { ...feedback, [mbid]: value };
    setFeedback(next); localStorage.setItem("musicmylove:v1:feedback", JSON.stringify(next));
  };
  const visible = useMemo(() => rankedPool.filter((track) => feedback[track.mbid] !== "dislike").slice(0, 40), [rankedPool, feedback]);

  return <main className="shell">
    <header><span className="eyebrow">MUSIC, FOUND BY MUSIC</span><h1>Give me 5 songs you love.<br /><em>I&apos;ll predict what you&apos;ll love next.</em></h1></header>
    <section className="picker" aria-label="Choose songs">
      <div className="progress"><span>{seeds.length} / 5</span><div>{Array.from({ length: 5 }, (_, i) => <i key={i} className={i < seeds.length ? "filled" : ""} />)}</div></div>
      {seeds.length < 5 && <div className="search"><input role="combobox" aria-label="Search for a song" aria-controls="search-results" aria-expanded={searchResults.length > 0} placeholder="Search a song or artist…" value={query} onChange={(event) => { const value = event.target.value; setQuery(value); if (value.trim().length < 2) setSearchResults([]); }} autoComplete="off" />
        {searchResults.length > 0 && <div className="menu" id="search-results">{searchResults.slice(0, 8).map((track) => <button key={track.mbid} onClick={() => select(track)}><b>{track.title}</b><span>{track.artist}{track.release ? ` · ${track.release}` : ""}</span></button>)}</div>}
      </div>}
      <div className="seeds">{seeds.map((seed, i) => <div className="seed" key={seed.mbid}><span>{String(i + 1).padStart(2, "0")}</span><b>{seed.title}</b><small>{seed.artist}</small></div>)}</div>
    </section>
    {loading && <p className="loading">Reading the shape of your taste…</p>}
    {error && <p className="loading" role="alert">{error}</p>}
    {rankedPool.length > 0 && <section className="recommendations"><div className="result-heading"><div><span className="eyebrow">YOUR NEXT FORTY</span><h2>A little familiar.<br />A little unexpected.</h2></div><p>Built from shared listening patterns—not genres, labels, or guesswork.</p></div>
      <div className="cards">{visible.map((track, i) => <article data-testid="recommendation" className="card" key={track.mbid}><span className="number">{String(i + 1).padStart(2, "0")}</span><div className="track"><h3>{track.title}</h3><p>{track.artist}</p><small>Picked from: {track.pickedFrom.map((seed) => seed.title).join(" + ") || "your mix"}</small></div><div className="actions"><a href={spotifyLinks[track.mbid] ?? "#"} target="_blank" rel="noreferrer">Open in Spotify ↗</a><button className={feedback[track.mbid] === "like" ? "active" : ""} onClick={() => saveFeedback(track.mbid, "like")} aria-label={`Like ${track.title}`}>♥</button><button onClick={() => saveFeedback(track.mbid, "dislike")} aria-label={`Dislike ${track.title}`}>×</button></div></article>)}</div>
    </section>}
  </main>;
}
