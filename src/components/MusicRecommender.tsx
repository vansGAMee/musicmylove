"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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

const INITIAL_FIGMA_TRACKS: RankedTrack[] = [
  { mbid: "demo-1", title: "Disorder", artist: "Joy Division", release: "Unknown Pleasures", score: 99, features: [], pickedFrom: [], tasteHeadIndex: 0, popularityPercentile: 0.72, liftScore: 1.4 },
  { mbid: "demo-2", title: "Atmosphere", artist: "Joy Division", release: "Atmosphere", score: 95, features: [], pickedFrom: [], tasteHeadIndex: 1, popularityPercentile: 0.68, liftScore: 1.2 },
  { mbid: "demo-3", title: "Ceremony", artist: "New Order", release: "Ceremony", score: 92, features: [], pickedFrom: [], tasteHeadIndex: 2, popularityPercentile: 0.65, liftScore: 1.1 },
  { mbid: "demo-4", title: "A Forest", artist: "The Cure", release: "Seventeen Seconds", score: 90, features: [], pickedFrom: [], tasteHeadIndex: 0, popularityPercentile: 0.81, liftScore: 0.9 },
  { mbid: "demo-5", title: "Shadowplay", artist: "Joy Division", release: "Unknown Pleasures", score: 88, features: [], pickedFrom: [], tasteHeadIndex: 3, popularityPercentile: 0.61, liftScore: 1.5 },
];

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
  const [activeTab, setActiveTab] = useState<"all" | "favorites" | "recent" | "mood">("all");
  const [activeScreen, setActiveScreen] = useState<"import" | "playlist" | "player">("playlist");
  const [selectedTrack, setSelectedTrack] = useState<RankedTrack>(INITIAL_FIGMA_TRACKS[0]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (query.trim().length < 2 || seeds.length >= 5) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal });
        if (response.ok) setSearchResults(await response.json());
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) setSearchResults([]);
      }
    }, 300);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, seeds.length]);

  const requestRecommendations = async (payloadBody: unknown) => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/tastelift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadBody),
      });
      const payload = await response.json() as { error?: string; recommendations?: TasteLiftApiRecommendation[] };
      if (!response.ok || !payload.recommendations) {
        throw new Error(payload.error ?? "TasteLift recommendation failed");
      }
      const mapped: RankedTrack[] = payload.recommendations.map((item) => ({
        mbid: item.mbid,
        artist: item.artist,
        title: item.title,
        ...(item.release ? { release: item.release } : {}),
        score: item.score,
        features: Array(17).fill(0),
        pickedFrom: [],
        tasteHeadIndex: item.strongestTasteHead,
        seedSupport: item.seedSupport,
        popularityPercentile: item.popularityPercentile,
        liftScore: item.noveltyLiftScore,
      }));
      setRankedPool(mapped);
      setSpotifyLinks(Object.fromEntries(payload.recommendations.map((item) => [item.mbid, item.spotifyLink])));
      if (mapped.length > 0) {
        setSelectedTrack(mapped[0]);
        setActiveScreen("playlist");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "TasteLift recommendation failed");
    } finally {
      setLoading(false);
    }
  };

  const selectSeed = async (track: Track) => {
    if (seeds.some((seed) => seed.mbid === track.mbid) || seeds.length >= 5) return;
    const next = [...seeds, track];
    setSeeds(next);
    setQuery("");
    setSearchResults([]);
    if (next.length === 5) {
      await requestRecommendations({ songs: next.map((seed) => ({ artist: seed.artist, title: seed.title })) });
    }
  };

  const removeSeed = (mbid: string) => {
    setSeeds(seeds.filter((s) => s.mbid !== mbid));
  };

  const parseAndSendJson = (text: string) => {
    try {
      const parsed = JSON.parse(text);
      void requestRecommendations(parsed);
    } catch {
      setError("Invalid JSON format. Please upload Spotify streaming history JSON.");
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      parseAndSendJson(text);
    };
    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      parseAndSendJson(text);
    };
    reader.readAsText(file);
  };

  const saveFeedback = (mbid: string, value: "like" | "dislike") => {
    const next = { ...feedback, [mbid]: feedback[mbid] === value ? undefined : value };
    const cleaned = Object.fromEntries(Object.entries(next).filter(([, v]) => v !== undefined)) as Record<string, "like" | "dislike">;
    setFeedback(cleaned);
    localStorage.setItem("musicmylove:v1:feedback", JSON.stringify(cleaned));
  };

  const displayTracks = useMemo(() => {
    const source = rankedPool.length > 0 ? rankedPool : INITIAL_FIGMA_TRACKS;
    let pool = source.filter((track) => feedback[track.mbid] !== "dislike");
    if (activeTab === "favorites") {
      pool = pool.filter((t) => feedback[t.mbid] === "like");
    }
    return pool.slice(0, 40);
  }, [rankedPool, feedback, activeTab]);

  const activeTrack = selectedTrack;
  const spotifyUrl = spotifyLinks[activeTrack.mbid] ?? `https://open.spotify.com/search/${encodeURIComponent(`${activeTrack.artist} ${activeTrack.title}`)}`;

  return (
    <div className="app-wrapper">
      <header className="top-brand">
        <h1>MOODLIST · JOY DIVISION STYLE</h1>
        <p>WARM MONOCHROME · HARD TYPOGRAPHY · PHYSICAL DEPTH</p>
      </header>

      <main className="cards-container">
        {/* CARD 1: IMPORT */}
        <section
          className={`device-card card-import ${activeScreen !== "import" ? "mobile-hidden" : ""}`}
          aria-label="Import your library"
        >
          <button type="button" className="nav-back" onClick={() => setActiveScreen("playlist")}>
            ← Back
          </button>

          <div className="card-eyebrow">Import</div>
          <h2 className="card-title-lg">your library</h2>

          <div className="card-scrollable">
            <div
              className={`tactile-disc-dropzone ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              aria-label="Upload JSON file"
            >
              <div className="tactile-disc-inner">
                <span>JSON</span>
              </div>
            </div>

            <div className="dropzone-container">
              <p className="dropzone-label">Drop your JSON file here</p>
              <p className="dropzone-sublabel">or tap to browse</p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              style={{ display: "none" }}
              onChange={handleFileUpload}
              id="json-file-input"
            />

            <button
              type="button"
              className="btn-pill"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose File
            </button>

            <div className="manual-search-box">
              <div className="progress-tag">
                <span>Manual search</span>
                <span>{seeds.length} / 5</span>
              </div>

              <div className="search-input-wrapper">
                <input
                  role="combobox"
                  className="search-input"
                  aria-label="Search for a song"
                  aria-controls="search-results-menu"
                  aria-expanded={searchResults.length > 0}
                  placeholder="Search a song or artist…"
                  value={query}
                  onChange={(e) => {
                    const val = e.target.value;
                    setQuery(val);
                    if (val.trim().length < 2) setSearchResults([]);
                  }}
                  autoComplete="off"
                />

                {searchResults.length > 0 && (
                  <div className="search-dropdown" id="search-results-menu">
                    {searchResults.slice(0, 6).map((track) => (
                      <button
                        key={track.mbid}
                        type="button"
                        className="search-item"
                        onClick={() => selectSeed(track)}
                      >
                        <b>{track.title}</b>
                        <span>{track.artist}{track.release ? ` · ${track.release}` : ""}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {seeds.length > 0 && (
                <div className="seed-chips">
                  {seeds.map((s) => (
                    <div key={s.mbid} className="seed-chip">
                      <span>{s.title}</span>
                      <button type="button" onClick={() => removeSeed(s.mbid)} aria-label={`Remove ${s.title}`}>×</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {loading && <div className="loading-indicator">Reading the shape of your taste…</div>}
            {error && <div className="loading-indicator" role="alert">{error}</div>}
          </div>
        </section>

        {/* CARD 2: MY PLAYLIST */}
        <section
          className={`device-card card-playlist ${activeScreen !== "playlist" ? "mobile-hidden" : ""}`}
          aria-label="My Playlist"
        >
          <div className="card-eyebrow">moodlist</div>
          <h2 className="card-title-lg">My Playlist</h2>
          <div className="card-subtitle">
            {rankedPool.length > 0 ? `${displayTracks.length} tracks` : "1,248 tracks"}
          </div>

          <div className="filter-pills-bar">
            <button
              type="button"
              className={`filter-pill ${activeTab === "all" ? "active" : ""}`}
              onClick={() => setActiveTab("all")}
            >
              All
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "favorites" ? "active" : ""}`}
              onClick={() => setActiveTab("favorites")}
            >
              Favorites
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "recent" ? "active" : ""}`}
              onClick={() => setActiveTab("recent")}
            >
              Recent
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "mood" ? "active" : ""}`}
              onClick={() => setActiveTab("mood")}
            >
              Mood
            </button>
          </div>

          <div className="card-scrollable">
            <div className="track-list">
              {displayTracks.map((track) => {
                const isSelected = activeTrack?.mbid === track.mbid;
                const isLiked = feedback[track.mbid] === "like";
                const rowSpotify = spotifyLinks[track.mbid] ?? `https://open.spotify.com/search/${encodeURIComponent(`${track.artist} ${track.title}`)}`;
                return (
                  <article
                    key={track.mbid}
                    data-testid="recommendation"
                    className={`track-row ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      setSelectedTrack(track);
                      setActiveScreen("player");
                    }}
                  >
                    <div className="vinyl-avatar">
                      <div className="vinyl-center-hole" />
                    </div>

                    <div className="track-info">
                      <h3 className="track-title">{track.title}</h3>
                      <p className="track-artist">{track.artist}</p>
                      {rankedPool.length > 0 && (
                        <div className="track-badges">
                          {track.tasteHeadIndex !== undefined && (
                            <span className="mini-badge head">Head {track.tasteHeadIndex + 1}</span>
                          )}
                          {track.popularityPercentile !== undefined && (
                            <span className="mini-badge pop">{Math.round(track.popularityPercentile * 100)}% pop</span>
                          )}
                          {track.liftScore !== undefined && (
                            <span className="mini-badge lift">+{track.liftScore.toFixed(2)} lift</span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="track-row-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className={`btn-icon ${isLiked ? "active" : ""}`}
                        onClick={() => saveFeedback(track.mbid, "like")}
                        aria-label={`Like ${track.title}`}
                      >
                        ♥
                      </button>
                      <button
                        type="button"
                        className="btn-icon"
                        onClick={() => saveFeedback(track.mbid, "dislike")}
                        aria-label={`Dislike ${track.title}`}
                      >
                        ×
                      </button>
                      <a
                        href={rowSpotify}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-icon"
                        aria-label={`Open ${track.title} in Spotify`}
                      >
                        ···
                      </a>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>

          <div className="bottom-tab-bar">
            <button
              type="button"
              className="bottom-tab-link"
              onClick={() => setActiveScreen("import")}
            >
              Library
            </button>
            <button
              type="button"
              className="bottom-tab-link active"
              onClick={() => setActiveScreen("playlist")}
            >
              Playlists
            </button>
          </div>
        </section>

        {/* CARD 3: JOY DIVISION TACTILE WHEEL PLAYER */}
        <section
          className={`device-card card-player ${activeScreen !== "player" ? "mobile-hidden" : ""}`}
          aria-label="Now Playing Player"
        >
          <button
            type="button"
            className="nav-back"
            onClick={() => setActiveScreen("playlist")}
          >
            ← Back
          </button>

          <div className="player-view">
            <div className="player-header">
              <div className="card-eyebrow">{activeTrack.artist}</div>
              <h2 className="card-title-lg">{activeTrack.title}</h2>
            </div>

            <div className="player-wheel-container">
              <button
                type="button"
                className={`wheel-control-btn ${feedback[activeTrack.mbid] === "like" ? "active" : ""}`}
                onClick={() => saveFeedback(activeTrack.mbid, "like")}
                aria-label="Like"
              >
                +
              </button>

              <div className="joy-division-wheel">
                <div className="wheel-inner-hub">
                  <div className="wheel-spool-center" />
                </div>
              </div>

              <button
                type="button"
                className="wheel-control-btn"
                onClick={() => saveFeedback(activeTrack.mbid, "dislike")}
                aria-label="Dislike"
              >
                −
              </button>
            </div>

            <div className="player-meta">
              <p>{activeTrack.release ?? "Unknown Pleasures"}</p>
              <p>From your JSON library</p>
            </div>

            <div className="player-slider">
              <div className="slider-knob" />
            </div>

            <a
              href={spotifyUrl}
              target="_blank"
              rel="noreferrer"
              className="btn-pill"
            >
              Open in Spotify ↗
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
