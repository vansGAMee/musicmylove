import { NextResponse } from "next/server";
import { TasteInputError, parseTasteInput } from "../../../src/lib/tastelift/input";
import { createTasteResolverAdapters, resolveTasteSeeds } from "../../../src/lib/tastelift/resolver";
import { recommendTasteSeeds } from "../../../src/lib/tastelift/pipeline";
import { spotifySearch } from "../../../src/lib/listenbrainz";
import { tasteliftRateLimiter, getClientIp } from "../../../src/lib/rate-limit";
import type { TasteLiftFeedback } from "../../../src/lib/tastelift/model";

// Vercel Serverless maximum execution limit (up to 60s supported)
export const maxDuration = 60;

interface TasteLiftRouteDependencies {
  parseTasteInput?: typeof parseTasteInput;
  resolveTasteSeeds?: typeof resolveTasteSeeds;
  createTasteResolverAdapters?: typeof createTasteResolverAdapters;
  recommendTasteSeeds?: typeof recommendTasteSeeds;
}

export async function handleTasteLiftPost(request: Request, dependencies: TasteLiftRouteDependencies = {}) {
  // Rate limit protection against abuse
  const rateLimit = tasteliftRateLimiter.check(getClientIp(request));
  if (!rateLimit.success) {
    return NextResponse.json(
      {
        error: "Too many recommendation requests. Please wait a moment before trying again.",
        code: "rate_limited",
      },
      {
        status: 429,
        headers: {
          "Retry-After": String(rateLimit.resetSeconds),
          "X-RateLimit-Limit": String(rateLimit.limit),
          "X-RateLimit-Remaining": "0",
        },
      }
    );
  }

  const contentLength = Number(request.headers.get("content-length"));
  if (contentLength > 2 * 1024 * 1024) {
    return NextResponse.json({ error: "Payload exceeds 2MB limit" }, { status: 413 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON" }, { status: 400 });
  }

  try {
    const receivedRows = Array.isArray(body) ? body : typeof body === "object" && body !== null && Array.isArray((body as { songs?: unknown[] }).songs) ? (body as { songs: unknown[] }).songs : [];
    const inputs = (dependencies.parseTasteInput ?? parseTasteInput)(body, { allowPartial: true });
    const seeds = await (dependencies.resolveTasteSeeds ?? resolveTasteSeeds)(
      inputs,
      (dependencies.createTasteResolverAdapters ?? createTasteResolverAdapters)()
    );
    const usableSeeds = seeds.filter((seed) => seed.status === "resolved");
    if (usableSeeds.length < 5) {
      return NextResponse.json(
        { error: "At least five seeds must resolve before recommendations can be generated", seeds },
        { status: 422 }
      );
    }
    const normalizedKeys = receivedRows.flatMap((row): string[] => {
      if (typeof row !== "object" || row === null) return [];
      const item = row as { artist?: unknown; title?: unknown };
      if (typeof item.artist !== "string" || !item.artist.trim() || typeof item.title !== "string" || !item.title.trim()) return [];
      const normalize = (value: string) => value.normalize("NFKC").trim().toLowerCase().replace(/\s+/gu, " ");
      return [`${normalize(item.artist)}\u0000${normalize(item.title)}`];
    });
    const duplicates = normalizedKeys.length - new Set(normalizedKeys).size;
    const feedbackRows = typeof body === "object" && body !== null && Array.isArray((body as { feedback?: unknown[] }).feedback) ? (body as { feedback: unknown[] }).feedback : [];
    const feedback: TasteLiftFeedback[] = feedbackRows.slice(0, 200).flatMap((row): TasteLiftFeedback[] => {
      if (typeof row !== "object" || row === null) return [];
      const item = row as { artist?: unknown; title?: unknown; value?: unknown };
      if (typeof item.artist !== "string" || !item.artist.trim() || typeof item.title !== "string" || !item.title.trim() || (item.value !== "like" && item.value !== "dislike")) return [];
      return [{ track: { artist: item.artist.trim(), title: item.title.trim() }, value: item.value }];
    });
    const result = await (dependencies.recommendTasteSeeds ?? recommendTasteSeeds)(usableSeeds, { feedback });
    return NextResponse.json({
      seeds,
      importCounts: {
        received: receivedRows.length,
        parsed: inputs.length,
        resolved: usableSeeds.filter((seed) => seed.source !== "text").length,
        textFallback: usableSeeds.filter((seed) => seed.source === "text").length,
        unresolved: seeds.filter((seed) => seed.status === "unresolved").length + Math.max(0, receivedRows.length - duplicates - inputs.length),
        duplicates,
      },
      candidateCount: result.candidateCount,
      recommendations: result.recommendations.map((track) => ({
        mbid: track.mbid,
        artist: track.artist,
        title: track.title,
        ...(track.release ? { release: track.release } : {}),
        score: track.slateScore ?? track.score,
        strongestTasteHead: track.tasteHeadIndex,
        seedSupport: track.seedSupport ?? 0,
        popularityPercentile: track.popularityPercentile ?? 0.5,
        noveltyLiftScore: track.liftScore ?? 0,
        spotifyLink: spotifySearch(track),
      })),
    });
  } catch (error) {
    if (error instanceof TasteInputError) {
      return NextResponse.json({ error: "Invalid TasteLift input", issues: error.issues }, { status: 400 });
    }
    return NextResponse.json({ error: "Recommendation pipeline is temporarily unavailable" }, { status: 502 });
  }
}

export async function POST(request: Request) {
  return handleTasteLiftPost(request);
}
