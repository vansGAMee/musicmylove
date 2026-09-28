export interface RateLimitOptions {
  windowMs: number;
  maxRequests: number;
}

export interface RateLimitResult {
  success: boolean;
  limit: number;
  remaining: number;
  resetSeconds: number;
}

export class MemoryRateLimiter {
  private hits = new Map<string, { count: number; resetAt: number }>();
  private lastPrune = Date.now();

  constructor(private readonly options: RateLimitOptions) {}

  check(identifier: string): RateLimitResult {
    const now = Date.now();

    // Periodically prune expired entries (every 60s or if map grows large)
    if (now - this.lastPrune > 60000 || this.hits.size > 5000) {
      this.prune(now);
    }

    const entry = this.hits.get(identifier);
    if (!entry || now > entry.resetAt) {
      this.hits.set(identifier, { count: 1, resetAt: now + this.options.windowMs });
      return {
        success: true,
        limit: this.options.maxRequests,
        remaining: this.options.maxRequests - 1,
        resetSeconds: Math.ceil(this.options.windowMs / 1000),
      };
    }

    if (entry.count >= this.options.maxRequests) {
      const resetSeconds = Math.max(1, Math.ceil((entry.resetAt - now) / 1000));
      return {
        success: false,
        limit: this.options.maxRequests,
        remaining: 0,
        resetSeconds,
      };
    }

    entry.count += 1;
    const resetSeconds = Math.max(1, Math.ceil((entry.resetAt - now) / 1000));
    return {
      success: true,
      limit: this.options.maxRequests,
      remaining: this.options.maxRequests - entry.count,
      resetSeconds,
    };
  }

  private prune(now: number) {
    this.lastPrune = now;
    for (const [key, entry] of this.hits.entries()) {
      if (now > entry.resetAt) {
        this.hits.delete(key);
      }
    }
  }

  reset() {
    this.hits.clear();
  }
}

export function getClientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    if (first) return first;
  }
  const realIp = request.headers.get("x-real-ip");
  if (realIp) return realIp.trim();
  const cfIp = request.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp.trim();
  return "127.0.0.1";
}

// Global rate limiters for Vercel Free / Serverless tier protection
// 20 Yandex playlist imports / minute per IP
export const yandexRateLimiter = new MemoryRateLimiter({ windowMs: 60_000, maxRequests: 20 });

// 30 TasteLift neural pipeline runs / minute per IP
export const tasteliftRateLimiter = new MemoryRateLimiter({ windowMs: 60_000, maxRequests: 30 });

// 60 Search queries / minute per IP
export const searchRateLimiter = new MemoryRateLimiter({ windowMs: 60_000, maxRequests: 60 });
