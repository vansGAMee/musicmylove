import type { Track } from "../types";

/** Normalizes display text only for deterministic recording identity comparisons. */
export function normalizeIdentityText(value: string): string {
  return value.normalize("NFKC").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

const VERSION_LABEL = "(?:\\d{4}\\s+)?(?:remaster(?:ed)?|remix(?:ed)?|live|demo|acoustic|instrumental|radio\\s+edit|single\\s+version|album\\s+version|mono|stereo|clean|explicit|cover)(?:\\s+\\d{4})?";
const VERSION_QUALIFIER = new RegExp(`^${VERSION_LABEL}$`, "iu");
const DASH_VERSION = new RegExp(`\\s*[-–—]\\s*${VERSION_LABEL}$`, "iu");
const PARENTHETICAL = /\s*[\[(]([^\[\]()]+)[\])]/gu;

/** Removes only unmistakable version labels, leaving meaningful artist/title text intact. */
export function versionlessTitle(title: string): string {
  let result = normalizeIdentityText(title).replace(PARENTHETICAL, (whole, qualifier: string) => VERSION_QUALIFIER.test(qualifier) ? "" : whole);
  result = result.replace(DASH_VERSION, "");
  return result.replace(/\s+/gu, " ").trim();
}

/** Same-artist variants (remasters, edits, covers) share a retrieval identity. */
export function recordingIdentity(track: Pick<Track, "artist" | "title">): string {
  return `${normalizeIdentityText(track.artist)}\u001f${versionlessTitle(track.title)}`;
}

export function hasSameRecordingIdentity(left: Pick<Track, "artist" | "title">, right: Pick<Track, "artist" | "title">): boolean {
  return recordingIdentity(left) === recordingIdentity(right);
}
