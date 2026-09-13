import { NextResponse } from "next/server";
import { lookupSpotify } from "@/src/lib/listenbrainz";
export async function GET(_: Request, context: { params: Promise<{ mbid: string }> }) { try { return NextResponse.json({ id: await lookupSpotify((await context.params).mbid) }); } catch { return NextResponse.json({ id: null }); } }
