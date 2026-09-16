import { NextResponse } from "next/server";
import { fetchYandexPlaylist, YandexPlaylistError } from "../../../../src/lib/yandex/playlist";
import { yandexRateLimiter, getClientIp } from "../../../../src/lib/rate-limit";

// Vercel Serverless maximum execution limit on Hobby plan
export const maxDuration = 10;

async function processYandexPlaylist(url: string, isGet: boolean) {
  try {
    const playlist = await fetchYandexPlaylist(url);
    const headers: Record<string, string> = {};
    if (isGet) {
      // Allow Vercel Edge Network to cache public playlist responses for 1 hour, saving serverless executions
      headers["Cache-Control"] = "public, s-maxage=3600, stale-while-revalidate=86400";
    }

    return NextResponse.json(
      {
        ok: true,
        playlist,
        tracks: playlist.tracks,
      },
      { headers }
    );
  } catch (error) {
    if (error instanceof YandexPlaylistError) {
      const statusMap: Record<string, number> = {
        invalid_url: 400,
        not_found: 404,
        private: 403,
        upstream_error: 502,
        geo_blocked: 451,
      };
      return NextResponse.json(
        { ok: false, error: error.message, code: error.code },
        { status: statusMap[error.code] ?? 500 }
      );
    }

    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Не удалось импортировать плейлист из Яндекс Музыки",
        code: "internal_error",
      },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  // Check rate limit per client IP
  const rateLimit = yandexRateLimiter.check(getClientIp(request));
  if (!rateLimit.success) {
    return NextResponse.json(
      {
        ok: false,
        error: "Слишком много запросов на импорт. Пожалуйста, подождите немного перед повторной попыткой.",
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

  const { searchParams } = new URL(request.url);
  const url = searchParams.get("url")?.trim() ?? "";

  if (!url) {
    return NextResponse.json(
      { ok: false, error: "Параметр url обязателен", code: "missing_url" },
      { status: 400 }
    );
  }

  return processYandexPlaylist(url, true);
}

export async function POST(request: Request) {
  // Check rate limit per client IP
  const rateLimit = yandexRateLimiter.check(getClientIp(request));
  if (!rateLimit.success) {
    return NextResponse.json(
      {
        ok: false,
        error: "Слишком много запросов на импорт. Пожалуйста, подождите немного перед повторной попыткой.",
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
  if (contentLength > 1024 * 1024) {
    return NextResponse.json(
      { ok: false, error: "Размер запроса превышает 1 МБ", code: "payload_too_large" },
      { status: 413 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "Тело запроса должно быть корректным JSON", code: "invalid_body" },
      { status: 400 }
    );
  }

  const url = typeof (body as Record<string, unknown>)?.url === "string"
    ? ((body as Record<string, unknown>).url as string).trim()
    : "";

  if (!url) {
    return NextResponse.json(
      { ok: false, error: "Параметр url обязателен", code: "missing_url" },
      { status: 400 }
    );
  }

  return processYandexPlaylist(url, false);
}
