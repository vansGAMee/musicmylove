import { NextResponse } from "next/server";
import { fetchSimilar } from "@/src/lib/listenbrainz";
export async function GET(_: Request, context: { params: Promise<{ mbid: string }> }) { try { return NextResponse.json(await fetchSimilar((await context.params).mbid)); } catch { return NextResponse.json({ error: "Similarity is temporarily unavailable" }, { status: 502 }); } }
