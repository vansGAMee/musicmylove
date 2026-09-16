import { NextResponse } from "next/server";
import { fetchSimilar } from "@/src/lib/listenbrainz";

export const maxDuration = 10;

export async function GET(_: Request, context: { params: Promise<{ mbid: string }> }) {
  try {
    const { mbid } = await context.params;
    const result = await fetchSimilar(mbid);
    return NextResponse.json(result, {
      headers: {
        "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=604800",
      },
    });
  } catch {
    return NextResponse.json({ error: "Similarity is temporarily unavailable" }, { status: 502 });
  }
}
