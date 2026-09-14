import { NextResponse } from "next/server";
import { TasteInputError, parseTasteInput } from "../../../src/lib/tastelift/input";
import { createTasteResolverAdapters, resolveTasteSeeds } from "../../../src/lib/tastelift/resolver";

interface TasteLiftRouteDependencies {
  parseTasteInput?: typeof parseTasteInput;
  resolveTasteSeeds?: typeof resolveTasteSeeds;
  createTasteResolverAdapters?: typeof createTasteResolverAdapters;
}

export async function handleTasteLiftPost(request: Request, dependencies: TasteLiftRouteDependencies = {}) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON" }, { status: 400 });
  }
  try {
    const inputs = (dependencies.parseTasteInput ?? parseTasteInput)(body);
    const seeds = await (dependencies.resolveTasteSeeds ?? resolveTasteSeeds)(inputs, (dependencies.createTasteResolverAdapters ?? createTasteResolverAdapters)());
    if (seeds.some((seed) => seed.status === "unresolved")) {
      return NextResponse.json({ error: "Every seed must resolve before recommendations can be generated", seeds }, { status: 422 });
    }
    return NextResponse.json({ seeds });
  } catch (error) {
    if (error instanceof TasteInputError) return NextResponse.json({ error: "Invalid TasteLift input", issues: error.issues }, { status: 400 });
    return NextResponse.json({ error: "Seed resolution is temporarily unavailable" }, { status: 502 });
  }
}

export async function POST(request: Request) { return handleTasteLiftPost(request); }
