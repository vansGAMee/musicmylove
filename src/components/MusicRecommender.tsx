"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RankedTrack, SeedTrack, Track } from "../lib/types";
import { downloadTasteCardPng } from "../lib/exportCard";

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

const TRANSLATIONS = {
  ru: {
    brandTitle: "MusicMyLove",
    brandTagline: "ПЕРСОНАЛЬНЫЕ МУЗЫКАЛЬНЫЕ РЕКОМЕНДАЦИИ НА БАЗЕ ИИ",
    themeDark: "Тёмная",
    themeLight: "Светлая",
    themeDarkAria: "Включить тёмную тему",
    themeLightAria: "Включить светлую тему",
    importEyebrow: "Импорт",
    yourLibrary: "Твоя медиатека",
    dropzoneLabel: "Перетащите JSON-файл сюда",
    dropzoneSublabel: "или нажмите для выбора",
    chooseFile: "Выбрать файл",
    manualSearch: "Ручной поиск",
    searchPlaceholder: "Поиск трека или исполнителя…",
    searchAria: "Поиск песни",
    readingTaste: "Считывание вашего вкуса…",
    back: "← Назад",
    uploadAria: "Загрузить JSON-файл",
    tracksCountSuffix: "треков",
    playlistEyebrow: "Плейлист",
    myPlaylist: "Мой плейлист",
    tabAll: "Все",
    tabFavorites: "Избранное",
    tabRecent: "Недавние",
    tabMood: "Настроение",
    navLibrary: "Медиатека",
    navPlaylists: "Плейлисты",
    badgeHead: "Вкус",
    badgePop: "% поп",
    badgeLift: "лифт",
    likeAria: "В избранное",
    dislikeAria: "Скрыть трек",
    spotifyAria: "Открыть в Spotify",
    playerAria: "Плеер",
    removeSeedAria: "Удалить трек",
    fromLibrary: "Из вашей медиатеки",
    fromTasteLift: "Рекомендация MusicMyLove AI",
    openSpotify: "Открыть в Spotify ↗",
    share: "Поделиться",
    shareAria: "Поделиться",
    shareTitle: "Экспорт в PNG",
    shareModalTitle: "Поделиться",
    shareTopSeeds: "Мой выбор",
    shareBottomRecs: "Рекомендации",
    shareDownloadPng: "Скачать PNG",
    shareCopyLink: "Скопировать ссылку",
    shareCopied: "Ссылка скопирована!",
    shareClose: "Закрыть",
    tipText: "Понравились рекомендации? Можно угостить автора чаем ☕",
    tipTextCompact: "Угостить автора чаем ☕",
    tipButton: "Угостить",
    tipDismissAria: "Закрыть навсегда",
    errorInvalidJson: "Неверный формат JSON. Загрузите файл истории прослушиваний Spotify.",
    errorCouldNotParse: "Не удалось прочитать файл. Убедитесь, что это корректный JSON.",
    errorRecommendationFailed: "Ошибка получения рекомендаций",
  },
  en: {
    brandTitle: "MusicMyLove",
    brandTagline: "PERSONALIZED AI MUSIC DISCOVERY",
    themeDark: "Dark",
    themeLight: "Light",
    themeDarkAria: "Switch to dark mode",
    themeLightAria: "Switch to light mode",
    importEyebrow: "Import",
    yourLibrary: "your library",
    dropzoneLabel: "Drop your JSON file here",
    dropzoneSublabel: "or tap to browse",
    chooseFile: "Choose File",
    manualSearch: "Manual search",
    searchPlaceholder: "Search a song or artist…",
    searchAria: "Search for a song",
    readingTaste: "Reading the shape of your taste…",
    back: "← Back",
    uploadAria: "Upload JSON file",
    tracksCountSuffix: "tracks",
    playlistEyebrow: "playlist",
    myPlaylist: "My Playlist",
    tabAll: "All",
    tabFavorites: "Favorites",
    tabRecent: "Recent",
    tabMood: "Mood",
    navLibrary: "Library",
    navPlaylists: "Playlists",
    badgeHead: "Head",
    badgePop: "% pop",
    badgeLift: "lift",
    likeAria: "Like track",
    dislikeAria: "Dislike track",
    spotifyAria: "Open in Spotify",
    playerAria: "Now Playing Player",
    removeSeedAria: "Remove",
    fromLibrary: "From your JSON library",
    fromTasteLift: "MusicMyLove AI Recommendation",
    openSpotify: "Open in Spotify ↗",
    share: "Share",
    shareAria: "Share",
    shareTitle: "Export to PNG",
    shareModalTitle: "Share",
    shareTopSeeds: "My selection",
    shareBottomRecs: "Recommendations",
    shareDownloadPng: "Download PNG",
    shareCopyLink: "Copy Link",
    shareCopied: "Link copied!",
    shareClose: "Close",
    tipText: "Enjoying the recommendations? You can tip the author a tea ☕",
    tipTextCompact: "Tip the author a tea ☕",
    tipButton: "Tip tea",
    tipDismissAria: "Dismiss forever",
    errorInvalidJson: "Invalid JSON format. Please upload Spotify streaming history JSON.",
    errorCouldNotParse: "Could not parse file. Make sure it is valid JSON.",
    errorRecommendationFailed: "TasteLift recommendation failed",
  },
};

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
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [isRussian, setIsRussian] = useState(false);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [showTipBanner, setShowTipBanner] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const t = isRussian ? TRANSLATIONS.ru : TRANSLATIONS.en;

  useEffect(() => {
    try {
      if (localStorage.getItem("musicmylove:v1:tip-dismissed") === "true") return;
    } catch {
      return;
    }
    const timer = setTimeout(() => {
      setShowTipBanner(true);
    }, 40000);
    return () => clearTimeout(timer);
  }, []);

  const dismissTipForever = () => {
    setShowTipBanner(false);
    try {
      localStorage.setItem("musicmylove:v1:tip-dismissed", "true");
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    try {
      if (typeof navigator !== "undefined") {
        const langs = navigator.languages ? Array.from(navigator.languages) : [navigator.language];
        const hasRu = langs.some((l) => typeof l === "string" && l.toLowerCase().startsWith("ru"));
        if (hasRu) {
          setIsRussian(true);
          document.documentElement.lang = "ru";
        } else {
          document.documentElement.lang = "en";
        }
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      const savedTheme = localStorage.getItem("musicmylove:v1:theme");
      if (savedTheme === "dark" || savedTheme === "light") {
        setTheme(savedTheme);
        document.documentElement.setAttribute("data-theme", savedTheme);
      } else {
        document.documentElement.setAttribute("data-theme", "light");
      }
    } catch {
      document.documentElement.setAttribute("data-theme", "light");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
    try {
      localStorage.setItem("musicmylove:v1:theme", nextTheme);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (query.trim().length < 2 || seeds.length >= 5) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal });
        if (response.ok) {
          const raw = await response.json();
          const items: Track[] = Array.isArray(raw) ? raw : (raw.results ?? []);
          setSearchResults(items);
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) setSearchResults([]);
      }
    }, 200);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
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
        throw new Error(payload.error ?? t.errorRecommendationFailed);
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
      setError(cause instanceof Error ? cause.message : t.errorRecommendationFailed);
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
      setError(t.errorInvalidJson);
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

  const shareSeeds = useMemo(() => {
    if (seeds.length > 0) return seeds.map((s) => ({ title: s.title, artist: s.artist }));
    const liked = Object.keys(feedback).filter((id) => feedback[id] === "like");
    if (liked.length > 0) {
      const pool = rankedPool.length > 0 ? rankedPool : INITIAL_FIGMA_TRACKS;
      return pool
        .filter((t) => liked.includes(t.mbid))
        .slice(0, 5)
        .map((t) => ({ title: t.title, artist: t.artist }));
    }
    return INITIAL_FIGMA_TRACKS.slice(0, 4).map((t) => ({ title: t.title, artist: t.artist }));
  }, [seeds, feedback, rankedPool]);

  const shareRecs = useMemo(() => {
    const pool = rankedPool.length > 0 ? rankedPool : INITIAL_FIGMA_TRACKS;
    return pool.slice(0, 5).map((t) => ({
      title: t.title,
      artist: t.artist,
      headIndex: t.tasteHeadIndex,
      liftScore: t.liftScore,
      popularityPercentile: t.popularityPercentile,
    }));
  }, [rankedPool]);

  const handleDownloadPng = () => {
    downloadTasteCardPng({
      seeds: shareSeeds,
      recommendations: shareRecs,
      isRussian,
      theme,
    });
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.origin);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch {
      // ignore
    }
  };

  return (
    <div className="app-wrapper">
      {showTipBanner && (
        <aside className="tip-toast-banner" role="complementary" aria-label="Support">
          <span className="tip-toast-text tip-toast-text-desktop">{t.tipText}</span>
          <span className="tip-toast-text tip-toast-text-mobile">{t.tipTextCompact}</span>
          <a
            href="https://pay.cloudtips.ru/p/45660cf3"
            target="_blank"
            rel="noopener noreferrer"
            className="tip-toast-btn"
          >
            {t.tipButton}
          </a>
          <button
            type="button"
            className="tip-toast-close"
            onClick={dismissTipForever}
            aria-label={t.tipDismissAria}
            title={t.tipDismissAria}
          >
            ×
          </button>
        </aside>
      )}

      <header className="top-brand">
        <div className="top-brand-bar">
          <div>
            <h1>{t.brandTitle}</h1>
            <p>{t.brandTagline}</p>
          </div>
          <div className="top-brand-actions">
            <button
              type="button"
              className="share-btn"
              onClick={() => setIsShareModalOpen(true)}
              aria-label={t.shareAria}
              title={t.shareTitle}
            >
              <span>↗</span>
              <span>{t.share}</span>
            </button>
            <button
              type="button"
              className="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label={theme === "light" ? t.themeDarkAria : t.themeLightAria}
              title={theme === "light" ? t.themeDarkAria : t.themeLightAria}
            >
              <span>{theme === "light" ? "☾" : "☼"}</span>
              <span>{theme === "light" ? t.themeDark : t.themeLight}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="cards-container">
        {/* CARD 1: IMPORT */}
        <section
          className={`device-card card-import ${activeScreen !== "import" ? "mobile-hidden" : ""}`}
          aria-label={t.yourLibrary}
        >
          <button type="button" className="nav-back" onClick={() => setActiveScreen("playlist")}>
            {t.back}
          </button>

          <div className="card-eyebrow">{t.importEyebrow}</div>
          <h2 className="card-title-lg">{t.yourLibrary}</h2>

          <div className="card-scrollable">
            <div
              className={`tactile-disc-dropzone ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              aria-label={t.uploadAria}
            >
              <div className="tactile-disc-inner">
                <span>JSON</span>
              </div>
            </div>

            <div className="dropzone-container">
              <p className="dropzone-label">{t.dropzoneLabel}</p>
              <p className="dropzone-sublabel">{t.dropzoneSublabel}</p>
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
              {t.chooseFile}
            </button>

            <div className="manual-search-box">
              <div className="progress-tag">
                <span>{t.manualSearch}</span>
                <span>{seeds.length} / 5</span>
              </div>

              <div className="search-input-wrapper">
                <input
                  role="combobox"
                  className="search-input"
                  aria-label={t.searchAria}
                  aria-controls="search-results-menu"
                  aria-expanded={searchResults.length > 0}
                  placeholder={t.searchPlaceholder}
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
                      <button type="button" onClick={() => removeSeed(s.mbid)} aria-label={`${t.removeSeedAria}: ${s.title}`}>×</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {loading && <div className="loading-indicator">{t.readingTaste}</div>}
            {error && <div className="loading-indicator" role="alert">{error}</div>}
          </div>
        </section>

        {/* CARD 2: MY PLAYLIST */}
        <section
          className={`device-card card-playlist ${activeScreen !== "playlist" ? "mobile-hidden" : ""}`}
          aria-label={t.myPlaylist}
        >
          <div className="card-eyebrow">{t.playlistEyebrow}</div>
          <h2 className="card-title-lg">{t.myPlaylist}</h2>
          <div className="card-subtitle">
            {rankedPool.length > 0 ? `${displayTracks.length} ${t.tracksCountSuffix}` : `1,248 ${t.tracksCountSuffix}`}
          </div>

          <div className="filter-pills-bar">
            <button
              type="button"
              className={`filter-pill ${activeTab === "all" ? "active" : ""}`}
              onClick={() => setActiveTab("all")}
            >
              {t.tabAll}
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "favorites" ? "active" : ""}`}
              onClick={() => setActiveTab("favorites")}
            >
              {t.tabFavorites}
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "recent" ? "active" : ""}`}
              onClick={() => setActiveTab("recent")}
            >
              {t.tabRecent}
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTab === "mood" ? "active" : ""}`}
              onClick={() => setActiveTab("mood")}
            >
              {t.tabMood}
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
                            <span className="mini-badge head">{t.badgeHead} {track.tasteHeadIndex + 1}</span>
                          )}
                          {track.popularityPercentile !== undefined && (
                            <span className="mini-badge pop">{Math.round(track.popularityPercentile * 100)}{t.badgePop}</span>
                          )}
                          {track.liftScore !== undefined && (
                            <span className="mini-badge lift">+{track.liftScore.toFixed(2)} {t.badgeLift}</span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="track-row-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className={`btn-icon ${isLiked ? "active" : ""}`}
                        onClick={() => saveFeedback(track.mbid, "like")}
                        aria-label={`${t.likeAria}: ${track.title}`}
                      >
                        ♥
                      </button>
                      <button
                        type="button"
                        className="btn-icon"
                        onClick={() => saveFeedback(track.mbid, "dislike")}
                        aria-label={`${t.dislikeAria}: ${track.title}`}
                      >
                        ×
                      </button>
                      <a
                        href={rowSpotify}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-icon"
                        aria-label={`${t.spotifyAria}: ${track.title}`}
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
              {t.navLibrary}
            </button>
            <button
              type="button"
              className="bottom-tab-link active"
              onClick={() => setActiveScreen("playlist")}
            >
              {t.navPlaylists}
            </button>
          </div>
        </section>

        {/* CARD 3: JOY DIVISION TACTILE WHEEL PLAYER */}
        <section
          className={`device-card card-player ${activeScreen !== "player" ? "mobile-hidden" : ""}`}
          aria-label={t.playerAria}
        >
          <button
            type="button"
            className="nav-back"
            onClick={() => setActiveScreen("playlist")}
          >
            {t.back}
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
                aria-label={t.likeAria}
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
                aria-label={t.dislikeAria}
              >
                −
              </button>
            </div>

            <div className="player-meta">
              <p>{activeTrack.release ?? "Unknown Pleasures"}</p>
              <p>{activeTrack.liftScore !== undefined ? t.fromTasteLift : t.fromLibrary}</p>
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
              {t.openSpotify}
            </a>
          </div>
        </section>
      </main>

      {isShareModalOpen && (
        <div className="share-modal-backdrop" onClick={() => setIsShareModalOpen(false)}>
          <div
            className="share-modal-dialog"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="share-modal-title"
          >
            <div className="share-modal-header">
              <h3 id="share-modal-title" className="share-modal-title">{t.shareModalTitle}</h3>
              <button
                type="button"
                className="share-modal-close"
                onClick={() => setIsShareModalOpen(false)}
                aria-label={t.shareClose}
              >
                ×
              </button>
            </div>

            <div className="taste-card-preview-container">
              <div className="taste-card-preview">
                <div className="preview-brand-row">
                  <span className="preview-brand-title">MUSICMYLOVE</span>
                </div>

                <div className="preview-waveform">
                  <div className="preview-wave-line" />
                  <div className="preview-wave-line" />
                  <div className="preview-wave-line" />
                  <div className="preview-wave-line" />
                  <div className="preview-wave-line" />
                </div>

                <div className="preview-section">
                  <div className="preview-section-title">{t.shareTopSeeds}</div>
                  <div className="preview-track-list">
                    {shareSeeds.map((s, idx) => (
                      <div key={idx} className="preview-track-row">
                        <span className="preview-track-idx">{idx + 1 < 10 ? `0${idx + 1}` : idx + 1}</span>
                        <div className="preview-track-meta">
                          <span className="preview-track-title">{s.title}</span>
                          <span className="preview-track-artist">{s.artist}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="preview-divider-line" />

                <div className="preview-section">
                  <div className="preview-section-title">{t.shareBottomRecs}</div>
                  <div className="preview-track-list">
                    {shareRecs.map((r, idx) => (
                      <div key={idx} className="preview-track-row">
                        <span className="preview-track-idx">{idx + 1 < 10 ? `0${idx + 1}` : idx + 1}</span>
                        <div className="preview-track-meta">
                          <span className="preview-track-title">{r.title}</span>
                          <span className="preview-track-artist">{r.artist}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="preview-footer">
                  <span>musicmylove.vercel.app</span>
                </div>
              </div>
            </div>

            <div className="share-modal-actions">
              <button
                type="button"
                className="btn-pill share-download-btn"
                onClick={handleDownloadPng}
              >
                <span>⬇</span>
                <span>{t.shareDownloadPng}</span>
              </button>
              <button
                type="button"
                className="share-copy-btn"
                onClick={handleCopyLink}
              >
                <span>🔗</span>
                <span>{copySuccess ? t.shareCopied : t.shareCopyLink}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
