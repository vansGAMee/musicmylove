/** Pure TypeScript serving implementation for the JSON-only TasteLift export. */

export interface TasteLiftTrack {
  mbid?: string;
  artist: string;
  title: string;
  /** Allows callers with an empirical value to override the exported MBID map. */
  popularityPercentile?: number;
}

interface TasteLiftArchitecture {
  dim: number;
  heads: number;
  attention_temperature: number;
  affinity_temperature: number;
  normalization_epsilon: number;
  pooling: string;
  track_activation: string;
}

interface TasteLiftHashing {
  normalization: string;
  encoding: string;
  algorithm: string;
  offset_basis: number;
  prime: number;
  buckets: number;
  char_ngram_min: number;
  char_ngram_max: number;
  padding: readonly string[];
  deduplicate: boolean;
  sort: string;
  unicode_units: string;
}

export interface TasteLiftArtifact {
  format: string;
  architecture: TasteLiftArchitecture;
  hashing: TasteLiftHashing;
  vocabulary: { artist: Record<string, number>; title: Record<string, number> };
  weights: {
    "artist_embedding.weight": number[][];
    "title_embedding.weight": number[][];
    "subword_embedding.weight": number[][];
    "track_projection.weight": number[][];
    "track_projection.bias": number[];
    queries: number[][];
    log_score_scale: number;
    log_popularity_weight: number;
  };
  popularity: Record<string, number>;
}

export interface TasteLiftCandidateScore {
  perHeadScores: number[];
  affinity: number;
  popularityPrior: number;
  lift: number;
  strongestHead: number;
  strongestHeadIndex: number;
  popularityPercentile: number;
}

export interface TasteLiftSetScore {
  heads: number[][];
  candidates: TasteLiftCandidateScore[];
}

const normalizeText = (value: string): string => value.normalize("NFKC").toLowerCase().trim().split(/\s+/u).filter(Boolean).join(" ");
const dot = (left: readonly number[], right: readonly number[]): number => left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
const zeros = (width: number): number[] => Array<number>(width).fill(0);

function l2(values: readonly number[], epsilon: number): number[] {
  const norm = Math.sqrt(dot(values, values));
  return values.map((value) => value / Math.max(norm, epsilon));
}

function softplus(value: number): number {
  return value > 20 ? value + Math.log1p(Math.exp(-value)) : Math.log1p(Math.exp(value));
}

function logSumExp(values: readonly number[]): number {
  const maximum = Math.max(...values);
  return maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
}

function softmax(values: readonly number[]): number[] {
  const logDenominator = logSumExp(values);
  return values.map((value) => Math.exp(value - logDenominator));
}

function fnv1aUtf8(value: string, offsetBasis: number, prime: number): number {
  let state = offsetBasis >>> 0;
  for (const byte of new TextEncoder().encode(value)) {
    state ^= byte;
    state = Math.imul(state, prime) >>> 0;
  }
  return state;
}

/** Hashes Unicode code-point ngrams exactly as ml.tastelift_data.hashed_subword_ids. */
export function hashedSubwordIds(value: string, hashing: TasteLiftHashing): number[] {
  if (hashing.padding.length !== 2) throw new Error("TasteLift hash padding must contain two markers");
  const characters = Array.from(`${hashing.padding[0]}${normalizeText(value)}${hashing.padding[1]}`);
  const buckets = new Set<number>();
  for (let width = hashing.char_ngram_min; width <= hashing.char_ngram_max; width += 1) {
    for (let start = 0; start <= characters.length - width; start += 1) {
      buckets.add(fnv1aUtf8(characters.slice(start, start + width).join(""), hashing.offset_basis, hashing.prime) % hashing.buckets);
    }
  }
  return [...buckets].sort((left, right) => left - right);
}

function assertMatrix(name: string, values: readonly number[][], rows: number, columns: number): void {
  if (values.length !== rows || values.some((row) => row.length !== columns)) throw new Error(`TasteLift ${name} dimensions are invalid`);
}

function compareTracks(left: TasteLiftTrack, right: TasteLiftTrack): number {
  const compare = (first: string, second: string): number => first < second ? -1 : first > second ? 1 : 0;
  return compare(left.mbid ?? "", right.mbid ?? "")
    || compare(normalizeText(left.artist), normalizeText(right.artist))
    || compare(normalizeText(left.title), normalizeText(right.title));
}

export class TasteLiftModel {
  private constructor(private readonly artifact: TasteLiftArtifact) {}

  static fromArtifact(artifact: TasteLiftArtifact): TasteLiftModel {
    if (artifact.format !== "tastelift-v1") throw new Error("Unsupported TasteLift export");
    const { architecture, hashing, weights } = artifact;
    if (architecture.dim !== 24 || architecture.heads !== 4 || hashing.normalization !== "NFKC-lower-whitespace" || hashing.encoding !== "UTF-8" || hashing.algorithm !== "FNV-1a-32" || hashing.unicode_units !== "codepoints") {
      throw new Error("Unsupported TasteLift architecture or hashing contract");
    }
    assertMatrix("artist embedding", weights["artist_embedding.weight"], artifact.vocabulary.artist ? Object.keys(artifact.vocabulary.artist).length + 1 : 0, architecture.dim);
    assertMatrix("title embedding", weights["title_embedding.weight"], artifact.vocabulary.title ? Object.keys(artifact.vocabulary.title).length + 1 : 0, architecture.dim);
    assertMatrix("subword embedding", weights["subword_embedding.weight"], hashing.buckets, architecture.dim);
    assertMatrix("projection", weights["track_projection.weight"], architecture.dim, architecture.dim * 4);
    assertMatrix("queries", weights.queries, architecture.heads, architecture.dim);
    if (weights["track_projection.bias"].length !== architecture.dim) throw new Error("TasteLift projection bias dimensions are invalid");
    return new TasteLiftModel(artifact);
  }

  encodeTrack(track: TasteLiftTrack): number[] {
    const { architecture, hashing, vocabulary, weights } = this.artifact;
    const artist = normalizeText(track.artist);
    const title = normalizeText(track.title);
    const meanEmbedding = (ids: readonly number[]): number[] => {
      if (ids.length === 0) return zeros(architecture.dim);
      const total = zeros(architecture.dim);
      for (const id of ids) for (let dimension = 0; dimension < architecture.dim; dimension += 1) total[dimension] += weights["subword_embedding.weight"][id]![dimension]!;
      return total.map((value) => value / ids.length);
    };
    const artistEmbedding = weights["artist_embedding.weight"][vocabulary.artist[artist] ?? 0]!;
    const titleEmbedding = weights["title_embedding.weight"][vocabulary.title[title] ?? 0]!;
    const input = [...artistEmbedding, ...titleEmbedding, ...meanEmbedding(hashedSubwordIds(artist, hashing)), ...meanEmbedding(hashedSubwordIds(title, hashing))];
    const projected = weights["track_projection.weight"].map((row, index) => Math.tanh(dot(row, input) + weights["track_projection.bias"][index]!));
    return l2(projected, architecture.normalization_epsilon);
  }

  /** Sorts the complete active set before every reduction, matching PyTorch encode_set. */
  encodeSet(seeds: readonly TasteLiftTrack[]): number[][] {
    if (seeds.length < 5 || seeds.length > 200) throw new Error("TasteLift requires 5 to 200 seed tracks");
    const vectors = [...seeds].sort(compareTracks).map((seed) => this.encodeTrack(seed));
    const { architecture, weights } = this.artifact;
    const normalizedQueries = weights.queries.map((query) => l2(query, architecture.normalization_epsilon));
    return normalizedQueries.map((query) => {
      const attention = softmax(vectors.map((vector) => dot(query, vector) / architecture.attention_temperature));
      const pooled = zeros(architecture.dim);
      for (let index = 0; index < vectors.length; index += 1) for (let dimension = 0; dimension < architecture.dim; dimension += 1) pooled[dimension] += attention[index]! * vectors[index]![dimension]!;
      return l2(pooled, architecture.normalization_epsilon);
    });
  }

  scoreCandidates(seeds: readonly TasteLiftTrack[], candidates: readonly TasteLiftTrack[]): TasteLiftSetScore {
    const heads = this.encodeSet(seeds);
    const { architecture, popularity, weights } = this.artifact;
    const scale = softplus(weights.log_score_scale);
    const popularityWeight = softplus(weights.log_popularity_weight);
    return {
      heads,
      candidates: candidates.map((candidate) => {
        const vector = this.encodeTrack(candidate);
        const perHeadScores = heads.map((head) => dot(head, vector) * scale);
        const strongestHead = Math.max(...perHeadScores);
        const strongestHeadIndex = perHeadScores.findIndex((score) => score === strongestHead);
        const affinity = architecture.affinity_temperature * logSumExp(perHeadScores.map((score) => score / architecture.affinity_temperature));
        const popularityPercentile = candidate.popularityPercentile ?? (candidate.mbid ? popularity[candidate.mbid] : undefined) ?? 0;
        const popularityPrior = popularityWeight * popularityPercentile;
        return { perHeadScores, affinity, popularityPrior, lift: affinity - popularityPrior, strongestHead, strongestHeadIndex, popularityPercentile };
      }),
    };
  }
}

export const loadTasteLiftModel = (artifact: TasteLiftArtifact): TasteLiftModel => TasteLiftModel.fromArtifact(artifact);
