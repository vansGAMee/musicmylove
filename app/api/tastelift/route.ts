import { NextResponse } from "next/server";
import { TasteInputError, parseTasteInput } from "../../../src/lib/tastelift/input";
import { createTasteResolverAdapters, resolveTasteSeeds } from "../../../src/lib/tastelift/resolver";
import { recommendTasteSeeds } from "../../../src/lib/tastelift/pipeline";
import { spotifySearch } from "../../../src/lib/listenbrainz";
import { tasteliftRateLimiter, getClientIp } from "../../../src/lib/rate-limit";

// Vercel Serverless maximum execution limit on Hobby plan
export const maxDuration = 10;

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
    const inputs = (dependencies.parseTasteInput ?? parseTasteInput)(body);
    const seeds = await (dependencies.resolveTasteSeeds ?? resolveTasteSeeds)(
      inputs,
      (dependencies.createTasteResolverAdapters ?? createTasteResolverAdapters)()
    );
    if (seeds.some((seed) => seed.status === "unresolved")) {
      return NextResponse.json(
        { error: "Every seed must resolve before recommendations can be generated", seeds },
        { status: 422 }
      );
    }
    const result = await (dependencies.recommendTasteSeeds ?? recommendTasteSeeds)(seeds);
    return NextResponse.json({
      seeds,
      candidateCount: result.candidateCount,
      recommendations: result.recommendations.map((track) => ({
        mbid: track.mbid,
        artist: track.artist,
        title: track.title,
        ...(track.release ? { release: track.release } : {}),
        score: track.slateScore ?? track.score,
        strongestTasteHead: track.tasteHeadIndex,
        seedSupport: track.seedSupport ?? 0,
        popularityPercentile: track.popularityPercentile ?? 0,
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
