import { NextResponse } from "next/server";
import { lookupSpotify } from "@/src/lib/listenbrainz";

export const maxDuration = 60;

export async function GET(_: Request, context: { params: Promise<{ mbid: string }> }) {
  try {
    const { mbid } = await context.params;
    const id = await lookupSpotify(mbid);
    return NextResponse.json(
      { id },
      {
        headers: {
          "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=604800",
        },
      }
    );
  } catch {
    return NextResponse.json({ id: null });
  }
}
