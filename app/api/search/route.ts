import { NextResponse } from "next/server";
import { searchRecordings } from "@/src/lib/listenbrainz";
export async function GET(request: Request) { const query = new URL(request.url).searchParams.get("q")?.trim() ?? ""; if (query.length < 2) return NextResponse.json([]); try { return NextResponse.json(await searchRecordings(query)); } catch { return NextResponse.json({ error: "Search is temporarily unavailable" }, { status: 502 }); } }
