import { NextResponse } from "next/server";
import { searchRecordings } from "@/src/lib/listenbrainz";
import { searchRateLimiter, getClientIp } from "@/src/lib/rate-limit";

export const maxDuration = 60;

export async function GET(request: Request) {
  const rateLimit = searchRateLimiter.check(getClientIp(request));
  if (!rateLimit.success) {
    return NextResponse.json(
      { error: "Too many search requests. Please slow down." },
      { status: 429, headers: { "Retry-After": String(rateLimit.resetSeconds) } }
    );
  }

  const query = new URL(request.url).searchParams.get("q")?.trim() ?? "";
  if (query.length < 2) {
    return NextResponse.json([]);
  }

  try {
    const results = await searchRecordings(query);
    return NextResponse.json(results, {
      headers: {
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    });
  } catch {
    return NextResponse.json({ error: "Search is temporarily unavailable" }, { status: 502 });
  }
}
