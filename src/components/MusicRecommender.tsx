"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RankedTrack, SeedTrack, Track } from "../lib/types";
import { downloadTasteCardPng } from "../lib/exportCard";
import { parseFileContentToSongs } from "../lib/tastelift/input";

const SUPPORTED_FORMATS = ["JSON", "CSV", "TXT", "M3U"] as const;

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
    dropzoneLabel: "Перетащите файл сюда",
    dropzoneSublabel: "JSON, CSV, TXT или M3U список треков",
    chooseFile: "Выбрать файл",
    manualSearch: "Ручной поиск",
    searchPlaceholder: "Поиск трека или исполнителя…",
    searchAria: "Поиск песни",
    readingTaste: "Считывание вашего вкуса…",
    back: "← Назад",
    uploadAria: "Загрузить файл с треками (JSON, CSV, TXT, M3U)",
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
    youtubeAria: "Открыть в YouTube",
    playerAria: "Плеер",
    removeSeedAria: "Удалить трек",
    fromLibrary: "Из вашей медиатеки",
    fromTasteLift: "Рекомендация MusicMyLove AI",
    openSpotify: "Spotify ↗",
    openYouTube: "YouTube ↗",
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
    analyzingTaste: "Считывание вкуса…",
    analyzingSubtitle: "Формируем персональный плейлист…",
    buttonAnalyzing: "Считывание…",
    howToGetFile: "Как получить файл?",
    helpExportifyPre: "Экспорт плейлиста в CSV: ",
    helpExportifyPost: " (в 1 клик через веб-сайт)",
    helpSpotifyArchive: "Либо запросите архив данных в настройках аккаунта Spotify (JSON)",
    yandexPlaceholder: "Ссылка на плейлист Яндекс Музыки…",
    yandexSubmit: "Импортировать",
    yandexImporting: "Импорт треков из Яндекс Музыки…",
    yandexNotFound: "Плейлист не найден. Проверьте правильность ссылки.",
    yandexPrivate: "Этот плейлист приватный. Сделайте его публичным в настройках.",
    yandexInvalidUrl: "Некорректная ссылка на плейлист Яндекс Музыки",
    yandexRateLimited: "Слишком много запросов. Подождите немного перед повторным импортом.",
    yandexGeoBlocked: "Яндекс Музыка недоступна из региона сервера. Вставьте треки вручную (Исполнитель — Название, по одному на строку).",
    yandexImportedSuccess: "Импортировано треков: {count}",
    yandexEmpty: "В плейлисте не найдено треков",
    importedBadge: "Импортировано: {count}",
    moreTracksEllipsis: "+ ещё {count}…",
    shareDisabledTooltip: "Сначала добавьте свои треки, чтобы поделиться вкусом",
    errorInvalidJson: "Неверный формат файла. Загрузите JSON, CSV, TXT или M3U список треков.",
    errorCouldNotParse: "Не удалось прочитать файл. Убедитесь, что это корректный JSON, CSV, TXT или M3U.",
    errorNeedMoreSeeds: "Найдено треков: {count}. Добавьте еще через поиск ниже до 5.",
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
    dropzoneLabel: "Drop your music file here",
    dropzoneSublabel: "or tap to browse",
    chooseFile: "Choose File",
    manualSearch: "Manual search",
    searchPlaceholder: "Search a song or artist…",
    searchAria: "Search for a song",
    readingTaste: "Reading the shape of your taste…",
    back: "← Back",
    uploadAria: "Upload track file (JSON, CSV, TXT, M3U)",
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
    youtubeAria: "Open in YouTube",
    playerAria: "Now Playing Player",
    removeSeedAria: "Remove",
    fromLibrary: "From your library",
    fromTasteLift: "MusicMyLove AI Recommendation",
    openSpotify: "Spotify ↗",
    openYouTube: "YouTube ↗",
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
    analyzingTaste: "Reading your taste…",
    analyzingSubtitle: "Generating your playlist…",
    buttonAnalyzing: "Analyzing…",
    howToGetFile: "How to get a file?",
    helpExportifyPre: "Export playlist to CSV: ",
    helpExportifyPost: " (1 click via browser)",
    helpSpotifyArchive: "Or request your listening history from Spotify settings (JSON)",
    yandexPlaceholder: "Yandex Music playlist link…",
    yandexSubmit: "Import",
    yandexImporting: "Importing from Yandex Music…",
    yandexNotFound: "Playlist not found. Check the link.",
    yandexPrivate: "This playlist is private. Please make it public in settings.",
    yandexInvalidUrl: "Invalid Yandex Music playlist link",
    yandexRateLimited: "Too many requests. Please wait a moment before trying again.",
    yandexGeoBlocked: "Yandex Music is unavailable from the server region. Paste tracks manually (Artist — Title, one per line).",
    yandexImportedSuccess: "Imported tracks: {count}",
    yandexEmpty: "No tracks found in this playlist",
    importedBadge: "Imported: {count}",
    moreTracksEllipsis: "+ {count} more…",
    shareDisabledTooltip: "Import or select your tracks first to share taste",
    errorInvalidJson: "Invalid file format. Please upload a JSON, CSV, TXT, or M3U tracklist.",
    errorCouldNotParse: "Could not parse file. Make sure it is valid JSON, CSV, TXT, or M3U.",
    errorNeedMoreSeeds: "Found {count} tracks. Add more via search below to reach 5.",
    errorRecommendationFailed: "TasteLift recommendation failed",
  },
};

interface MusicFact {
  text: string;
  isTea?: boolean;
}

const MUSIC_FACTS_RU: MusicFact[] = [
  { text: "Обложка альбома Joy Division — это радиосигналы первого открытого пульсара CP 1919" },
  { text: "На одной стороне винила всего одна непрерывная спиральная канавка длиной около 500 метров" },
  { text: "Музыкальные «мурашки» (флиссон) вызывают мощный выброс дофамина в вентральном стриатуме мозга" },
  { text: "Человеческое ухо способно различить два звуковых щелчка с разницей всего в 2 миллисекунды" },
  { text: "Люди ярче всего помнят музыку своей юности из-за психологического эффекта reminiscence bump" },
  { text: "Шумоподавление работает по принципу противофазы: волна гасится своей зеркально инвертированной копией" },
  { text: "До 1939 года не было стандарта 440 Гц — нота Ля в разных странах настраивалась от 400 до 460 Гц" },
  { text: "Мозг музыканта бессознательно продолжает считывать ритм даже во время пауз и тишины" },
  { text: "Формат долгоиграющих пластинок LP (33⅓ об/мин) был впервые представлен в 1948 году" },
  { text: "Слуховая кора перерабатывает музыку быстрее, чем зрительная кора распознает изображения" },
  { text: "Глубокий суббас ощущается телом физически благодаря рецепторам вибрации кожи Пачини" },
  { text: "Пока нейросеть вычисляет форму вкуса, можно угостить автора чаем ☕", isTea: true },
];

const MUSIC_FACTS_EN: MusicFact[] = [
  { text: "The Joy Division cover depicts radio waves from CP 1919, the first discovered pulsar" },
  { text: "A vinyl record side consists of a single continuous groove roughly 500 meters long" },
  { text: "Musical chills (frisson) trigger a surge of dopamine in the brain's ventral striatum" },
  { text: "The human auditory system can distinguish sounds separated by just 2 milliseconds" },
  { text: "We recall music from our youth most vividly due to the reminiscence bump phenomenon" },
  { text: "Active noise cancellation mirrors sound waves in anti-phase to cancel ambient noise" },
  { text: "Before 1939's 440 Hz standard, concert pitch varied across Europe from 400 to 460 Hz" },
  { text: "The auditory cortex stays rhythmically synchronized even during musical pauses and silences" },
  { text: "The 33⅓ RPM LP format was introduced in 1948, revolutionizing long-form albums" },
  { text: "The auditory cortex processes musical signals faster than the visual cortex processes images" },
  { text: "Low-end sub bass is perceived physically through Pacinian vibration corpuscles in the skin" },
  { text: "While the AI calculates your taste topology, you can tip the author a tea ☕", isTea: true },
];

export default function MusicRecommender() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Track[]>([]);
  const [seeds, setSeeds] = useState<SeedTrack[]>([]);
  const [importedSeeds, setImportedSeeds] = useState<{ artist: string; title: string }[]>([]);
  const [rankedPool, setRankedPool] = useState<RankedTrack[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike">>({});
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
  const [formatIndex, setFormatIndex] = useState(0);
  const [isFormatFading, setIsFormatFading] = useState(false);
  const [showHelpPopover, setShowHelpPopover] = useState(false);
  const [yandexUrl, setYandexUrl] = useState("");
  const [yandexNotice, setYandexNotice] = useState("");
  const [yandexLoading, setYandexLoading] = useState(false);
  const [yandexIframeUrl, setYandexIframeUrl] = useState<string | null>(null);
  const [factIndex, setFactIndex] = useState(0);
  const [isFactFading, setIsFactFading] = useState(false);
  const facts = isRussian ? MUSIC_FACTS_RU : MUSIC_FACTS_EN;

  useEffect(() => {
    if (!loading) {
      setIsFactFading(false);
      return;
    }

    const nonTea = facts.filter((f) => !f.isTea);
    const teaIdx = facts.findIndex((f) => f.isTea);

    setFactIndex(Math.floor(Math.random() * nonTea.length));

    const interval = setInterval(() => {
      setIsFactFading(true);
      setTimeout(() => {
        const showTea = Math.random() < 0.12 && teaIdx !== -1;
        if (showTea) {
          setFactIndex(teaIdx);
        } else {
          setFactIndex((prev) => {
            let next: number;
            do {
              next = Math.floor(Math.random() * nonTea.length);
            } while (next === prev && nonTea.length > 1);
            return next;
          });
        }
        setIsFactFading(false);
      }, 250);
    }, 3200);

    return () => clearInterval(interval);
  }, [loading, isRussian, facts]);

  useEffect(() => {
    if (!showHelpPopover) return;
    const timer = setTimeout(() => {
      setShowHelpPopover(false);
    }, 6000);
    return () => clearTimeout(timer);
  }, [showHelpPopover]);

  useEffect(() => {
    if (!yandexNotice) return;
    const timer = setTimeout(() => {
      setYandexNotice("");
    }, 4500);
    return () => clearTimeout(timer);
  }, [yandexNotice]);

  useEffect(() => {
    const handleYandexMessage = (event: MessageEvent) => {
      if (typeof event.origin === "string" && !event.origin.includes("yandex.")) return;
      try {
        const data = event.data;
        if (!data || typeof data !== "object") return;
        const payload = (data as { payload?: Record<string, unknown> }).payload ?? data;
        const rawTrack = (payload as Record<string, unknown>).currentTrack ?? (payload as Record<string, unknown>).track;
        if (rawTrack && typeof rawTrack === "object") {
          const tObj = rawTrack as Record<string, unknown>;
          const title = String(tObj.title ?? tObj.name ?? "").trim();
          let artist = "";
          if (Array.isArray(tObj.artists)) {
            artist = tObj.artists
              .map((a: unknown) => (typeof a === "object" && a !== null && "name" in a ? String((a as { name?: string }).name) : String(a)))
              .filter(Boolean)
              .join(", ");
          } else if (typeof tObj.artist === "string") {
            artist = tObj.artist;
          }
          if (title) {
            const finalArtist = artist || "Unknown Artist";
            setImportedSeeds((prev) => {
              const exists = prev.some((s) => s.title.toLowerCase() === title.toLowerCase() && s.artist.toLowerCase() === finalArtist.toLowerCase());
              if (exists) return prev;
              const next = [...prev, { title, artist: finalArtist }];
              setSeeds(next.slice(0, 5).map((s, idx) => ({
                mbid: `seed-ym-${idx}-${Date.now()}`,
                title: s.title,
                artist: s.artist,
                score: 100,
                features: [],
                pickedFrom: [],
              })));
              return next;
            });
          }
        }
      } catch {}
    };
    window.addEventListener("message", handleYandexMessage);
    return () => window.removeEventListener("message", handleYandexMessage);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsFormatFading(true);
      setTimeout(() => {
        setFormatIndex((prev) => (prev + 1) % SUPPORTED_FORMATS.length);
        setIsFormatFading(false);
      }, 400);
    }, 3800);
    return () => clearInterval(interval);
  }, []);

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

  useEffect(() => {
    try {
      const saved = localStorage.getItem("musicmylove:v1:feedback");
      if (saved) {
        const parsed = JSON.parse(saved) as Record<string, string>;
        const cleaned = Object.fromEntries(
          Object.entries(parsed).filter(([id, v]) => !id.startsWith("demo-") || v === "like")
        );
        setFeedback(cleaned as Record<string, "like" | "dislike">);
      }
    } catch {
      // ignore
    }
  }, []);

  const isTogglingThemeRef = useRef(false);
  const toggleTheme = () => {
    if (isTogglingThemeRef.current) return;
    isTogglingThemeRef.current = true;
    setTimeout(() => {
      isTogglingThemeRef.current = false;
    }, 180);
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
    if (query.trim().length < 2 || seeds.length >= 50) return;
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
      // Automatically incorporate user favorites ("like") into taste seed songs
      let bodyToSend = payloadBody;
      const likedMbids = Object.keys(feedback).filter((id) => feedback[id] === "like");
      if (likedMbids.length > 0 && typeof payloadBody === "object" && payloadBody !== null) {
        const bodyObj = payloadBody as { songs?: Array<{ artist: string; title: string; spotify_url?: string }> };
        if (Array.isArray(bodyObj.songs)) {
          const existingKeys = new Set(
            bodyObj.songs.map((s) => `${s.artist.toLowerCase()}:::${s.title.toLowerCase()}`)
          );
          const extraLikedSongs: Array<{ artist: string; title: string }> = [];
          for (const mbid of likedMbids) {
            const track = rankedPool.find((t) => t.mbid === mbid) ?? seeds.find((s) => s.mbid === mbid) ?? INITIAL_FIGMA_TRACKS.find((t) => t.mbid === mbid);
            if (track) {
              const key = `${track.artist.toLowerCase()}:::${track.title.toLowerCase()}`;
              if (!existingKeys.has(key)) {
                existingKeys.add(key);
                extraLikedSongs.push({ artist: track.artist, title: track.title });
              }
            }
          }
          if (extraLikedSongs.length > 0) {
            bodyToSend = {
              ...bodyObj,
              songs: [...bodyObj.songs, ...extraLikedSongs].slice(0, 500),
            };
          }
        }
      }

      const response = await fetch("/api/tastelift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyToSend),
      });

      let payload: {
        error?: string;
        recommendations?: TasteLiftApiRecommendation[];
        seeds?: Array<{
          artist?: string;
          title?: string;
          mbid?: string;
          status?: string;
          input?: { artist: string; title: string };
          track?: { mbid?: string; artist: string; title: string; release?: string };
        }>;
      } | null = null;

      try {
        const text = await response.text();
        if (text && (text.startsWith("{") || text.startsWith("["))) {
          payload = JSON.parse(text);
        }
      } catch {
        payload = null;
      }

      if (!response.ok || !payload || !payload.recommendations) {
        throw new Error(payload?.error ?? t.errorRecommendationFailed);
      }
      if (payload.seeds && payload.seeds.length > 0) {
        const resolvedSeeds: SeedTrack[] = payload.seeds.slice(0, 5).map((s, idx) => {
          const artist = s.track?.artist ?? s.input?.artist ?? s.artist ?? "";
          const title = s.track?.title ?? s.input?.title ?? s.title ?? "";
          return {
            mbid: s.track?.mbid ?? s.mbid ?? `seed-resolved-${idx}`,
            title,
            artist,
            score: 100,
            features: [],
            pickedFrom: [],
          };
        });
        setSeeds(resolvedSeeds);
        setImportedSeeds((prev) =>
          prev.length > 0
            ? prev
            : payload!.seeds!.map((s) => ({
                artist: s.track?.artist ?? s.input?.artist ?? s.artist ?? "",
                title: s.track?.title ?? s.input?.title ?? s.title ?? "",
              }))
        );
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
      const msg = cause instanceof Error ? cause.message : "";
      if (!msg || /token|json|syntaxerror/i.test(msg)) {
        setError(t.errorRecommendationFailed);
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const sParam = params.get("s") ?? params.get("seeds");
      if (!sParam) return;

      const parsedItems = sParam
        .split(",")
        .map((item) => {
          const colonIdx = item.indexOf(":");
          if (colonIdx !== -1) {
            return {
              artist: item.slice(0, colonIdx).replace(/\+/g, " ").trim(),
              title: item.slice(colonIdx + 1).replace(/\+/g, " ").trim(),
            };
          }
          return {
            artist: "",
            title: item.replace(/\+/g, " ").trim(),
          };
        })
        .filter((x) => x.title);

      if (parsedItems.length === 0) return;

      const seedTracks: SeedTrack[] = parsedItems.slice(0, 5).map((t, idx) => ({
        mbid: `seed-shared-${idx}-${Date.now()}`,
        title: t.title,
        artist: t.artist,
        score: 100,
        features: [],
        pickedFrom: [],
      }));
      setSeeds(seedTracks);
      setImportedSeeds(parsedItems);

      if (parsedItems.length >= 5) {
        void requestRecommendations({ songs: parsedItems });
      }
    } catch {
      // ignore
    }
  }, []);

  const selectSeed = async (track: Track) => {
    if (seeds.some((seed) => seed.mbid === track.mbid) || seeds.length >= 50) return;
    const next = [...seeds, track];
    setSeeds(next);
    setQuery("");
    setSearchResults([]);
    if (next.length === 5 || (importedSeeds.length > 0 && next.length > seeds.length)) {
      const allSongs = [
        ...importedSeeds,
        ...next.map((seed) => ({ artist: seed.artist, title: seed.title })),
      ];
      await requestRecommendations({ songs: allSongs.slice(0, 500) });
    }
  };

  const removeSeed = (mbid: string) => {
    setSeeds(seeds.filter((s) => s.mbid !== mbid));
  };

  const handleUploadedContent = async (text: string) => {
    setError("");
    try {
      const parsedSongs = parseFileContentToSongs(text, { allowPartial: true });
      if (parsedSongs.length === 0) {
        setError(t.errorCouldNotParse);
        return;
      }

      const uploadSeeds: SeedTrack[] = parsedSongs.slice(0, 5).map((s, idx) => ({
        mbid: `seed-upload-${idx}-${Date.now()}`,
        title: s.title,
        artist: s.artist,
        score: 100,
        features: [],
        pickedFrom: [],
      }));
      setSeeds(uploadSeeds);
      setImportedSeeds(parsedSongs.map((s) => ({ artist: s.artist, title: s.title })));

      if (parsedSongs.length >= 5) {
        await requestRecommendations({
          songs: parsedSongs.slice(0, 50).map((s) => ({
            artist: s.artist,
            title: s.title,
            ...(s.spotifyUrl ? { spotify_url: s.spotifyUrl } : {}),
          })),
        });
      } else {
        setError("");
      }
    } catch {
      setError(t.errorCouldNotParse);
    }
  };

  const readUploadedFileAsText = async (file: File): Promise<string> => {
    try {
      const buffer = await file.arrayBuffer();
      const utf8Decoder = new TextDecoder("utf-8", { fatal: false });
      const text = utf8Decoder.decode(buffer);
      const replacementCount = (text.match(/\uFFFD/gu) || []).length;
      if (replacementCount > 0) {
        try {
          const cp1251Decoder = new TextDecoder("windows-1251");
          const cp1251Text = cp1251Decoder.decode(buffer);
          const cp1251Replacements = (cp1251Text.match(/\uFFFD/gu) || []).length;
          if (cp1251Replacements < replacementCount) {
            return cp1251Text;
          }
        } catch {
          // Keep UTF-8 text
        }
      }
      return text;
    } catch {
      return await file.text();
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.target;
    const file = input.files?.[0];
    if (!file) return;
    void (async () => {
      try {
        const text = await readUploadedFileAsText(file);
        await handleUploadedContent(text);
      } finally {
        input.value = "";
      }
    })();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    void (async () => {
      const text = await readUploadedFileAsText(file);
      await handleUploadedContent(text);
    })();
  };

  useEffect(() => {
    const handleGlobalPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.isContentEditable) return;
      if (loading) return;
      const text = e.clipboardData?.getData("text");
      if (text && text.trim().length > 3) {
        if (text.includes("\n") || text.includes(" - ") || text.includes(" — ")) {
          void handleUploadedContent(text);
        }
      }
    };
    window.addEventListener("paste", handleGlobalPaste);
    return () => window.removeEventListener("paste", handleGlobalPaste);
  }, [loading]);

  const handleYandexImport = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanUrl = yandexUrl.trim();
    if (!cleanUrl || yandexLoading || loading) return;

    setYandexLoading(true);
    setYandexNotice(t.yandexImporting);
    setError("");

    try {
      let res: Response;
      try {
        // Use GET first to leverage Vercel Edge CDN cache (0 serverless function executions on repeat requests)
        res = await fetch(`/api/yandex/playlist?url=${encodeURIComponent(cleanUrl)}`, {
          method: "GET",
          headers: { "Accept": "application/json" },
        });

        if (!res.ok && res.status === 405) {
          // Fallback to POST only if GET method is not allowed
          res = await fetch("/api/yandex/playlist", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({ url: cleanUrl }),
          });
        }
      } catch {
        setYandexNotice(t.yandexGeoBlocked);
        return;
      }

      let data: {
        ok?: boolean;
        error?: string;
        code?: string;
        tracks?: Array<{ id: string; title: string; artists: string[] }>;
        playlist?: { tracks?: Array<{ id: string; title: string; artists: string[] }> };
      } | null = null;

      try {
        const text = await res.text();
        if (text && (text.startsWith("{") || text.startsWith("["))) {
          data = JSON.parse(text);
        }
      } catch {
        data = null;
      }

      if (!res.ok || !data || !data.ok) {
        let msg = data?.error ?? t.yandexGeoBlocked;
        const code = data?.code;
        if (code === "rate_limited" || res.status === 429) msg = t.yandexRateLimited;
        else if (code === "geo_blocked" || res.status === 451 || res.status >= 500 || !data) {
          msg = t.yandexGeoBlocked;
          const fallbackIframe = (data as { iframeUrl?: string })?.iframeUrl ?? (() => {
            const uuidMatch = /\/playlists\/([A-Za-z0-9._-]+)/i.exec(cleanUrl);
            return uuidMatch ? `https://music.yandex.ru/iframe/playlist/${uuidMatch[1]}` : null;
          })();
          if (fallbackIframe) setYandexIframeUrl(fallbackIframe);
        }
        else if (code === "not_found" || res.status === 404) msg = t.yandexNotFound;
        else if (code === "private" || res.status === 403) msg = t.yandexPrivate;
        else if (code === "invalid_url" || res.status === 400) msg = t.yandexInvalidUrl;
        setYandexNotice(msg);
        return;
      }

      setYandexIframeUrl(null);

      const tracksList: Array<{ id: string; title: string; artists: string[] }> =
        Array.isArray(data.tracks) ? data.tracks : (data.playlist?.tracks ?? []);

      if (tracksList.length === 0) {
        setYandexNotice(t.yandexEmpty);
        return;
      }

      const parsedSongs = tracksList.map((tr) => ({
        title: tr.title,
        artist: Array.isArray(tr.artists) ? tr.artists.join(", ") : (tr.artists || "Unknown Artist"),
      }));

      setImportedSeeds(parsedSongs);

      const uploadSeeds: SeedTrack[] = parsedSongs.slice(0, 5).map((s, idx) => ({
        mbid: `seed-ym-${idx}-${Date.now()}`,
        title: s.title,
        artist: s.artist,
        score: 100,
        features: [],
        pickedFrom: [],
      }));
      setSeeds(uploadSeeds);
      setYandexNotice(t.yandexImportedSuccess.replace("{count}", String(parsedSongs.length)));

      await requestRecommendations({
        songs: parsedSongs.slice(0, 50).map((s) => ({
          artist: s.artist,
          title: s.title,
        })),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      if (!msg || /token|json|syntaxerror|fetch/i.test(msg)) {
        setYandexNotice(t.yandexGeoBlocked);
        const uuidMatch = /\/playlists\/([A-Za-z0-9._-]+)/i.exec(cleanUrl);
        if (uuidMatch) setYandexIframeUrl(`https://music.yandex.ru/iframe/playlist/${uuidMatch[1]}`);
      } else {
        setYandexNotice(msg);
      }
    } finally {
      setYandexLoading(false);
    }
  };

  const saveFeedback = (mbid: string, value: "like" | "dislike") => {
    if (rankedPool.length === 0 && value === "dislike") {
      return; // Keep demo preview tracks visible until real tracks are uploaded
    }
    const isCurrentlyLiked = feedback[mbid] === "like";
    const nextValue = feedback[mbid] === value ? undefined : value;
    const next = { ...feedback, [mbid]: nextValue };
    const cleaned = Object.fromEntries(Object.entries(next).filter(([, v]) => v !== undefined)) as Record<string, "like" | "dislike">;
    setFeedback(cleaned);
    try {
      localStorage.setItem("musicmylove:v1:feedback", JSON.stringify(cleaned));
    } catch {
      // ignore
    }

    // Auto-incorporate favorites into taste seeds:
    if (value === "like" && !isCurrentlyLiked) {
      const candidate = rankedPool.find((t) => t.mbid === mbid) ?? INITIAL_FIGMA_TRACKS.find((t) => t.mbid === mbid);
      if (candidate) {
        setSeeds((prev) => {
          if (
            prev.some(
              (s) =>
                s.mbid === candidate.mbid ||
                (s.title.toLowerCase() === candidate.title.toLowerCase() &&
                  s.artist.toLowerCase() === candidate.artist.toLowerCase())
            )
          ) {
            return prev;
          }
          return [
            ...prev,
            {
              mbid: candidate.mbid,
              title: candidate.title,
              artist: candidate.artist,
              score: 100,
              features: [],
              pickedFrom: [],
            },
          ];
        });
        setImportedSeeds((prev) => {
          if (
            prev.some(
              (s) =>
                s.title.toLowerCase() === candidate.title.toLowerCase() &&
                s.artist.toLowerCase() === candidate.artist.toLowerCase()
            )
          ) {
            return prev;
          }
          return [...prev, { artist: candidate.artist, title: candidate.title }];
        });
      }
    } else if (value === "like" && isCurrentlyLiked) {
      // User un-liked the track: remove from seeds if it was added
      setSeeds((prev) => prev.filter((s) => s.mbid !== mbid));
    }
  };

  const displayTracks = useMemo(() => {
    if (rankedPool.length > 0) {
      let pool = rankedPool.filter((track) => feedback[track.mbid] !== "dislike");

      // Strictly filter out any track already in the user's seeds / playlist
      const seedMbids = new Set(seeds.map((s) => s.mbid));
      const seedIdentities = new Set([
        ...seeds.map((s) => `${s.artist.toLowerCase()}:::${s.title.toLowerCase()}`),
        ...importedSeeds.map((s) => `${s.artist.toLowerCase()}:::${s.title.toLowerCase()}`),
      ]);

      pool = pool.filter((track) => {
        if (seedMbids.has(track.mbid)) return false;
        const identity = `${track.artist.toLowerCase()}:::${track.title.toLowerCase()}`;
        if (seedIdentities.has(identity)) return false;
        return true;
      });

      if (activeTab === "favorites") {
        pool = pool.filter((t) => feedback[t.mbid] === "like");
      }
      return pool.slice(0, 40);
    }
    // When no recommendation pool is loaded yet, ALWAYS show the initial demo tracks so the playlist is never empty!
    if (activeTab === "favorites") {
      const liked = INITIAL_FIGMA_TRACKS.filter((t) => feedback[t.mbid] === "like");
      return liked.length > 0 ? liked : INITIAL_FIGMA_TRACKS;
    }
    return INITIAL_FIGMA_TRACKS;
  }, [rankedPool, feedback, activeTab, seeds, importedSeeds]);

  const activeTrack = selectedTrack;
  const spotifyUrl = spotifyLinks[activeTrack.mbid] ?? `https://open.spotify.com/search/${encodeURIComponent(`${activeTrack.artist} ${activeTrack.title}`)}`;
  const youtubeUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(`${activeTrack.artist} ${activeTrack.title}`)}`;

  const canShare = useMemo(() => {
    return seeds.length > 0 || importedSeeds.length > 0 || rankedPool.length > 0;
  }, [seeds.length, importedSeeds.length, rankedPool.length]);

  const shareSeeds = useMemo(() => {
    if (seeds.length > 0) {
      return seeds.slice(0, 5).map((s) => ({ title: s.title, artist: s.artist }));
    }
    if (importedSeeds.length > 0) {
      return importedSeeds.slice(0, 5);
    }
    const liked = Object.keys(feedback).filter((id) => feedback[id] === "like");
    if (liked.length > 0 && rankedPool.length > 0) {
      return rankedPool
        .filter((t) => liked.includes(t.mbid))
        .slice(0, 5)
        .map((t) => ({ title: t.title, artist: t.artist }));
    }
    // NEVER fall back to dummy/initial tracks in Share!
    return [];
  }, [seeds, importedSeeds, feedback, rankedPool]);

  const shareRecs = useMemo(() => {
    if (rankedPool.length === 0) {
      // NEVER fall back to dummy/initial tracks in Share!
      return [];
    }
    const pool = rankedPool;
    if (pool.length <= 5) {
      return pool.map((t) => ({
        title: t.title,
        artist: t.artist,
        headIndex: t.tasteHeadIndex,
        liftScore: t.liftScore,
        popularityPercentile: t.popularityPercentile,
      }));
    }

    // 1. Top 2 recognizable anchor hits from the head of recommendations
    const topAnchors = pool.slice(0, 2);
    const usedMbids = new Set(topAnchors.map((t) => t.mbid));
    const usedArtists = new Set(topAnchors.map((t) => t.artist.toLowerCase()));

    // 2. Candidates from the remainder of the pool (indices 2..end)
    const remainder = pool.slice(2).filter((t) => !usedMbids.has(t.mbid));

    // Sort remainder by highest novelty discovery (liftScore) & deeper rarity (lower popularity)
    const sortedGems = [...remainder].sort((a, b) => {
      const liftA = a.liftScore ?? 0;
      const liftB = b.liftScore ?? 0;
      if (Math.abs(liftB - liftA) > 0.08) {
        return liftB - liftA;
      }
      return (a.popularityPercentile ?? 0.5) - (b.popularityPercentile ?? 0.5);
    });

    const deepGems: RankedTrack[] = [];
    for (const track of sortedGems) {
      const artistKey = track.artist.toLowerCase();
      if (!usedArtists.has(artistKey)) {
        deepGems.push(track);
        usedArtists.add(artistKey);
        if (deepGems.length === 3) break;
      }
    }

    // Fallback if strict artist diversity didn't find 3
    if (deepGems.length < 3) {
      for (const track of sortedGems) {
        if (!deepGems.some((g) => g.mbid === track.mbid)) {
          deepGems.push(track);
          if (deepGems.length === 3) break;
        }
      }
    }

    const combined = [...topAnchors, ...deepGems];
    return combined.slice(0, 5).map((t) => ({
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

  const buildShareUrl = () => {
    const origin = typeof window !== "undefined" ? window.location.origin : "https://musicmylove.vercel.app";
    const tracksToShare = shareSeeds.length > 0 ? shareSeeds : importedSeeds.slice(0, 5);
    if (tracksToShare.length === 0) return origin;
    const query = tracksToShare
      .slice(0, 5)
      .map((t) => `${t.artist.replace(/[:;,]/g, " ").trim()}:${t.title.replace(/[:;,]/g, " ").trim()}`.replace(/\s+/g, "+"))
      .join(",");
    return `${origin}/?s=${query}`;
  };

  const handleCopyLink = async () => {
    try {
      const shareUrl = buildShareUrl();
      await navigator.clipboard.writeText(shareUrl);
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
              disabled={!canShare}
              onClick={() => {
                if (canShare) setIsShareModalOpen(true);
              }}
              aria-label={t.shareAria}
              title={canShare ? t.shareTitle : t.shareDisabledTooltip}
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
              className={`tactile-disc-dropzone ${isDragging ? "dragging" : ""} ${loading ? "is-loading" : ""}`}
              onClick={() => { if (!loading) fileInputRef.current?.click(); }}
              onDragOver={(e) => { e.preventDefault(); if (!loading) setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { if (!loading) handleDrop(e); }}
              onPaste={(e) => {
                if (loading) return;
                const pasted = e.clipboardData?.getData("text");
                if (pasted && pasted.trim()) {
                  e.preventDefault();
                  void handleUploadedContent(pasted);
                }
              }}
              role="button"
              tabIndex={0}
              aria-label={t.uploadAria}
              aria-busy={loading}
            >
              <div className="tactile-disc-inner">
                {loading ? (
                  <div className="disc-spinner" aria-label={t.analyzingTaste}>
                    <span className="spinner-dot" />
                    <span className="spinner-dot" />
                    <span className="spinner-dot" />
                  </div>
                ) : (
                  <span className={`tactile-disc-format ${isFormatFading ? "fading" : ""}`}>
                    {SUPPORTED_FORMATS[formatIndex]}
                  </span>
                )}
              </div>
            </div>

            <div className="dropzone-container">
              <p className="dropzone-label">{loading ? t.analyzingTaste : t.dropzoneLabel}</p>
              {loading ? (
                <div className="dropzone-fact-container">
                  <p className={`dropzone-fact ${isFactFading ? "fading" : ""}`}>
                    {facts[factIndex]?.isTea ? (
                      <>
                        {isRussian ? (
                          <>
                            Пока нейросеть вычисляет форму вкуса...{" "}
                            <a
                              href="https://pay.cloudtips.ru/p/45660cf3"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="fact-tea-link"
                              onClick={(e) => e.stopPropagation()}
                            >
                              угостить автора чаем ☕
                            </a>
                          </>
                        ) : (
                          <>
                            While the AI charts your taste...{" "}
                            <a
                              href="https://pay.cloudtips.ru/p/45660cf3"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="fact-tea-link"
                              onClick={(e) => e.stopPropagation()}
                            >
                              tip the author a tea ☕
                            </a>
                          </>
                        )}
                      </>
                    ) : (
                      facts[factIndex]?.text ?? t.analyzingSubtitle
                    )}
                  </p>
                </div>
              ) : (
                <p className="dropzone-sublabel">{t.dropzoneSublabel}</p>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.csv,.tsv,.txt,.m3u,.m3u8,.pls,application/json,text/csv,text/tab-separated-values,text/plain,audio/x-mpegurl"
              style={{ display: "none" }}
              onChange={handleFileUpload}
              id="json-file-input"
            />

            {error && (
              <div className="import-error-banner" role="alert">
                <span>{error}</span>
                <button
                  type="button"
                  className="import-error-dismiss"
                  onClick={() => setError("")}
                  aria-label="Закрыть"
                >
                  ×
                </button>
              </div>
            )}

            <button
              type="button"
              className={`btn-pill ${loading ? "btn-pill-loading" : ""}`}
              onClick={() => { if (!loading) fileInputRef.current?.click(); }}
              disabled={loading}
            >
              {loading ? t.buttonAnalyzing : t.chooseFile}
            </button>

            <div className="import-help-wrapper">
              <button
                type="button"
                className="import-help-btn"
                onClick={() => setShowHelpPopover((prev) => !prev)}
                aria-expanded={showHelpPopover}
              >
                <span>{t.howToGetFile}</span>
                <span className="help-icon-circle" aria-hidden="true">i</span>
              </button>

              {showHelpPopover && (
                <div className="import-help-popover" role="tooltip">
                  <div className="import-help-popover-content">
                    <p className="import-help-row">
                      <span>{t.helpExportifyPre}</span>
                      <a
                        href="https://exportify.app/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="import-help-link"
                      >
                        exportify.app
                      </a>
                      <span>{t.helpExportifyPost}</span>
                    </p>
                    <p className="import-help-row import-help-muted">
                      {t.helpSpotifyArchive}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="import-help-close"
                    onClick={() => setShowHelpPopover(false)}
                    aria-label={t.shareClose}
                  >
                    ×
                  </button>
                </div>
              )}
            </div>

            <form
              className="yandex-import-box"
              onSubmit={handleYandexImport}
            >
              <div className="yandex-input-wrapper">
                <svg
                  className="yandex-input-icon"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                </svg>
                <input
                  type="url"
                  className="yandex-input"
                  placeholder={t.yandexPlaceholder}
                  value={yandexUrl}
                  onChange={(e) => {
                    setYandexUrl(e.target.value);
                    if (yandexNotice) setYandexNotice("");
                  }}
                  disabled={yandexLoading}
                  aria-label={t.yandexPlaceholder}
                />
                <button
                  type="submit"
                  className={`yandex-submit-btn ${yandexLoading ? "loading" : ""}`}
                  disabled={!yandexUrl.trim() || yandexLoading}
                  aria-label={t.yandexSubmit}
                  title={t.yandexSubmit}
                >
                  {yandexLoading ? "…" : "→"}
                </button>
              </div>
              {yandexNotice && (
                <div className="yandex-notice" role="status">
                  {yandexNotice}
                </div>
              )}
              {yandexIframeUrl && (
                <div
                  className="yandex-iframe-wrapper"
                  style={{
                    marginTop: "12px",
                    borderRadius: "12px",
                    overflow: "hidden",
                    border: "1px solid rgba(255, 255, 255, 0.12)",
                    background: "rgba(0, 0, 0, 0.4)",
                  }}
                >
                  <iframe
                    src={yandexIframeUrl}
                    width="100%"
                    height="320"
                    frameBorder="0"
                    style={{ display: "block", border: "none", width: "100%", height: "320px", background: "#18181b" }}
                    title="Yandex Music Player"
                    allow="autoplay"
                  />
                  <div
                    style={{
                      padding: "10px 14px",
                      background: "rgba(255, 255, 255, 0.04)",
                      fontSize: "0.82rem",
                      opacity: 0.9,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <span>
                      {isRussian
                        ? "Плейлист открыт в плеере. При воспроизведении треки добавляются автоматически, или скопируйте список треков и вставьте (Ctrl+V)."
                        : "Playlist opened in player. Tracks add automatically as they play, or copy & paste tracklist (Ctrl+V)."}
                    </span>
                    <button
                      type="button"
                      onClick={() => setYandexIframeUrl(null)}
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "inherit",
                        cursor: "pointer",
                        fontSize: "1.2rem",
                        padding: "0 6px",
                      }}
                      title={isRussian ? "Закрыть плеер" : "Close player"}
                      aria-label={isRussian ? "Закрыть плеер" : "Close player"}
                    >
                      ×
                    </button>
                  </div>
                </div>
              )}
            </form>

            <div className="manual-search-box">
              <div className="progress-tag">
                <span>{t.manualSearch}</span>
                {importedSeeds.length > 0 ? (
                  <span className="imported-count-badge">
                    {t.importedBadge.replace("{count}", String(importedSeeds.length))}
                  </span>
                ) : (
                  <span>{seeds.length} / 5</span>
                )}
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
                  {seeds.slice(0, 4).map((s) => (
                    <div key={s.mbid} className="seed-chip">
                      <span>{s.title}</span>
                      <button type="button" onClick={() => removeSeed(s.mbid)} aria-label={`${t.removeSeedAria}: ${s.title}`}>×</button>
                    </div>
                  ))}
                  {((importedSeeds.length > 4 ? importedSeeds.length : seeds.length) > 4) && (
                    <div
                      className="seed-chip-more"
                      title={
                        importedSeeds.length > 4
                          ? importedSeeds.slice(4).map((x) => `${x.artist} - ${x.title}`).slice(0, 10).join("\n") + (importedSeeds.length > 14 ? "\n…" : "")
                          : seeds.slice(4).map((x) => `${x.artist} - ${x.title}`).join("\n")
                      }
                    >
                      {t.moreTracksEllipsis.replace(
                        "{count}",
                        String((importedSeeds.length > 4 ? importedSeeds.length : seeds.length) - 4)
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
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
                const rowYoutube = `https://www.youtube.com/results?search_query=${encodeURIComponent(`${track.artist} ${track.title}`)}`;
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
                        title={t.likeAria}
                      >
                        ♥
                      </button>
                      <button
                        type="button"
                        className="btn-icon"
                        onClick={() => saveFeedback(track.mbid, "dislike")}
                        aria-label={`${t.dislikeAria}: ${track.title}`}
                        title={t.dislikeAria}
                      >
                        ×
                      </button>
                      <a
                        href={rowSpotify}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-icon"
                        aria-label={`${t.spotifyAria}: ${track.title}`}
                        title={`${t.spotifyAria}: ${track.title}`}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                          <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.495 17.308a.747.747 0 0 1-1.026.248c-2.812-1.718-6.353-2.107-10.525-1.155a.748.748 0 0 1-.336-1.458c4.567-1.044 8.49-.603 11.639 1.339.36.22.473.69.248 1.026zm1.467-3.262a.936.936 0 0 1-1.287.309c-3.218-1.978-8.125-2.55-11.93-1.395a.936.936 0 1 1-.546-1.791c4.348-1.32 9.774-.683 13.454 1.58.423.26.556.815.309 1.297zm.126-3.41c-3.856-2.29-10.218-2.502-13.882-1.39a1.123 1.123 0 0 1-.652-2.148c4.218-1.28 11.238-1.03 15.688 1.61a1.123 1.123 0 0 1-1.154 1.928z"/>
                        </svg>
                      </a>
                      <a
                        href={rowYoutube}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-icon"
                        aria-label={`${t.youtubeAria}: ${track.title}`}
                        title={`${t.youtubeAria}: ${track.title}`}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                          <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                        </svg>
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

            <div className="player-external-actions">
              <a
                href={spotifyUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-pill player-btn-external"
                aria-label={`${t.spotifyAria}: ${activeTrack.title}`}
                title={`${t.spotifyAria}: ${activeTrack.title}`}
              >
                <svg className="external-link-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.495 17.308a.747.747 0 0 1-1.026.248c-2.812-1.718-6.353-2.107-10.525-1.155a.748.748 0 0 1-.336-1.458c4.567-1.044 8.49-.603 11.639 1.339.36.22.473.69.248 1.026zm1.467-3.262a.936.936 0 0 1-1.287.309c-3.218-1.978-8.125-2.55-11.93-1.395a.936.936 0 1 1-.546-1.791c4.348-1.32 9.774-.683 13.454 1.58.423.26.556.815.309 1.297zm.126-3.41c-3.856-2.29-10.218-2.502-13.882-1.39a1.123 1.123 0 0 1-.652-2.148c4.218-1.28 11.238-1.03 15.688 1.61a1.123 1.123 0 0 1-1.154 1.928z"/>
                </svg>
                <span>{t.openSpotify}</span>
              </a>
              <a
                href={youtubeUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-pill player-btn-external"
                aria-label={`${t.youtubeAria}: ${activeTrack.title}`}
                title={`${t.youtubeAria}: ${activeTrack.title}`}
              >
                <svg className="external-link-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                </svg>
                <span>{t.openYouTube}</span>
              </a>
            </div>
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
                    {shareSeeds.length > 0 ? (
                      shareSeeds.map((s, idx) => (
                        <div key={idx} className="preview-track-row">
                          <span className="preview-track-idx">{idx + 1 < 10 ? `0${idx + 1}` : idx + 1}</span>
                          <div className="preview-track-meta">
                            <span className="preview-track-title">{s.title}</span>
                            <span className="preview-track-artist">{s.artist}</span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="preview-track-meta" style={{ opacity: 0.6, fontSize: "11px", padding: "4px 0" }}>
                        {t.shareDisabledTooltip}
                      </div>
                    )}
                  </div>
                </div>

                <div className="preview-divider-line" />

                <div className="preview-section">
                  <div className="preview-section-title">{t.shareBottomRecs}</div>
                  <div className="preview-track-list">
                    {shareRecs.length > 0 ? (
                      shareRecs.map((r, idx) => (
                        <div key={idx} className="preview-track-row">
                          <span className="preview-track-idx">{idx + 1 < 10 ? `0${idx + 1}` : idx + 1}</span>
                          <div className="preview-track-meta">
                            <span className="preview-track-title">{r.title}</span>
                            <span className="preview-track-artist">{r.artist}</span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="preview-track-meta" style={{ opacity: 0.6, fontSize: "11px", padding: "4px 0" }}>
                        {t.shareDisabledTooltip}
                      </div>
                    )}
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
