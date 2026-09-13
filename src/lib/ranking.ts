import type {
  CandidateEvidence,
  RankedTrack,
  RankerName,
  SeedEvidence,
  SeedTrack,
  SimilarityLists,
} from "./types";

const SEED_COUNT = 5;
const RRF_K = 60;

const artistKey = (artist: string): string => artist.trim().toLocaleLowerCase();

export function mergeCandidates(
  seeds: readonly SeedTrack[],
  lists: SimilarityLists,
): CandidateEvidence[] {
  const seedMbids = new Set(seeds.map((seed) => seed.mbid));
  const merged = new Map<string, CandidateEvidence>();

  for (const seed of [...seeds].sort((a, b) => a.mbid.localeCompare(b.mbid))) {
    const list = lists[seed.mbid] ?? [];
    const maxScore = Math.max(0, ...list.map((item) => item.score));
    list.forEach((track, index) => {
      if (seedMbids.has(track.mbid)) return;
      const evidence: SeedEvidence = {
        seedMbid: seed.mbid,
        rank: index + 1,
        rawScore: track.score,
        normalizedScore: maxScore > 0 ? Math.max(0, track.score) / maxScore : 0,
        reciprocalRank: 1 / (RRF_K + index + 1),
      };
      const candidate = merged.get(track.mbid);
      if (candidate) {
        candidate.evidence.push(evidence);
      } else {
        merged.set(track.mbid, {
          mbid: track.mbid,
          title: track.title,
          artist: track.artist,
          release: track.release,
          evidence: [evidence],
        });
      }
    });
  }

  return [...merged.values()].map((candidate) => ({
    ...candidate,
    evidence: candidate.evidence.sort((a, b) => a.seedMbid.localeCompare(b.seedMbid)),
  }));
}

export function buildFeatures(
  candidate: CandidateEvidence,
  seeds: readonly SeedTrack[],
): number[] {
  const similarities = candidate.evidence
    .map((item) => item.normalizedScore)
    .sort((a, b) => b - a);
  const reciprocalRanks = candidate.evidence
    .map((item) => item.reciprocalRank)
    .sort((a, b) => b - a);
  const paddedSimilarities = [...similarities, ...Array(SEED_COUNT).fill(0)].slice(0, SEED_COUNT);
  const paddedRanks = [...reciprocalRanks, ...Array(SEED_COUNT).fill(0)].slice(0, SEED_COUNT);
  const positives = similarities.filter((value) => value > 0);
  const mean = positives.length
    ? positives.reduce((sum, value) => sum + value, 0) / positives.length
    : 0;
  const variance = positives.length
    ? positives.reduce((sum, value) => sum + (value - mean) ** 2, 0) / positives.length
    : 0;
  const sameArtistCount = seeds.filter(
    (seed) => artistKey(seed.artist) === artistKey(candidate.artist),
  ).length;

  return [
    ...paddedSimilarities,
    ...paddedRanks,
    reciprocalRanks.reduce((sum, value) => sum + value, 0),
    candidate.evidence.length / SEED_COUNT,
    paddedSimilarities[0] ?? 0,
    paddedSimilarities[1] ?? 0,
    mean,
    Math.sqrt(variance),
    sameArtistCount / SEED_COUNT,
  ];
}

function scoreCandidate(features: readonly number[], ranker: RankerName): number {
  if (ranker === "max") return features[12] ?? 0;
  return features[10] ?? 0;
}

export function rankCandidates(
  seeds: readonly SeedTrack[],
  lists: SimilarityLists,
  ranker: RankerName,
): RankedTrack[] {
  const seedByMbid = new Map(seeds.map((seed) => [seed.mbid, seed]));
  return mergeCandidates(seeds, lists)
    .map((candidate) => {
      const features = buildFeatures(candidate, seeds);
      const pickedFrom = [...candidate.evidence]
        .sort(
          (a, b) =>
            b.normalizedScore - a.normalizedScore ||
            b.reciprocalRank - a.reciprocalRank ||
            a.seedMbid.localeCompare(b.seedMbid),
        )
        .slice(0, 2)
        .map((item) => seedByMbid.get(item.seedMbid))
        .filter((seed): seed is SeedTrack => Boolean(seed));
      return {
        mbid: candidate.mbid,
        title: candidate.title,
        artist: candidate.artist,
        release: candidate.release,
        features,
        pickedFrom,
        score: scoreCandidate(features, ranker),
      };
    })
    .sort((a, b) => b.score - a.score || a.mbid.localeCompare(b.mbid));
}

export function diversify(
  ranked: readonly RankedTrack[],
  limit = 20,
): RankedTrack[] {
  const artistCounts = new Map<string, number>();
  const seenMbids = new Set<string>();
  const result: RankedTrack[] = [];
  for (const track of ranked) {
    const key = artistKey(track.artist);
    if (seenMbids.has(track.mbid) || (artistCounts.get(key) ?? 0) >= 2) continue;
    result.push(track);
    seenMbids.add(track.mbid);
    artistCounts.set(key, (artistCounts.get(key) ?? 0) + 1);
    if (result.length === limit) break;
  }
  return result;
}
