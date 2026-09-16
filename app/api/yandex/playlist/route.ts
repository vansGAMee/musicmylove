import { NextResponse } from "next/server";
import { fetchYandexPlaylist, YandexPlaylistError } from "../../../../src/lib/yandex/playlist";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Тело запроса должно быть корректным JSON", code: "invalid_body" }, { status: 400 });
  }

  const url = typeof (body as Record<string, unknown>)?.url === "string"
    ? ((body as Record<string, unknown>).url as string).trim()
    : "";

  if (!url) {
    return NextResponse.json({ ok: false, error: "Параметр url обязателен", code: "missing_url" }, { status: 400 });
  }

  try {
    const playlist = await fetchYandexPlaylist(url);
    return NextResponse.json({
      ok: true,
      playlist,
      tracks: playlist.tracks,
    });
  } catch (error) {
    if (error instanceof YandexPlaylistError) {
      const statusMap: Record<string, number> = {
        invalid_url: 400,
        not_found: 404,
        private: 403,
        upstream_error: 502,
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
