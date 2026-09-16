import { describe, expect, test } from "vitest";
import { MemoryRateLimiter, getClientIp } from "../../src/lib/rate-limit";

describe("MemoryRateLimiter", () => {
  test("allows requests up to maxRequests and blocks afterwards", () => {
    const limiter = new MemoryRateLimiter({ windowMs: 10_000, maxRequests: 3 });
    const ip = "192.168.1.100";

    const r1 = limiter.check(ip);
    expect(r1.success).toBe(true);
    expect(r1.remaining).toBe(2);

    const r2 = limiter.check(ip);
    expect(r2.success).toBe(true);
    expect(r2.remaining).toBe(1);

    const r3 = limiter.check(ip);
    expect(r3.success).toBe(true);
    expect(r3.remaining).toBe(0);

    const r4 = limiter.check(ip);
    expect(r4.success).toBe(false);
    expect(r4.remaining).toBe(0);
    expect(r4.resetSeconds).toBeGreaterThan(0);
  });

  test("isolates rate limits by IP address", () => {
    const limiter = new MemoryRateLimiter({ windowMs: 10_000, maxRequests: 1 });
    expect(limiter.check("1.1.1.1").success).toBe(true);
    expect(limiter.check("1.1.1.1").success).toBe(false);
    expect(limiter.check("2.2.2.2").success).toBe(true);
  });

  test("extracts client IP from x-forwarded-for or fallback", () => {
    const req1 = new Request("http://localhost", { headers: { "x-forwarded-for": "203.0.113.195, 70.41.3.18" } });
    expect(getClientIp(req1)).toBe("203.0.113.195");

    const req2 = new Request("http://localhost", { headers: { "x-real-ip": "198.51.100.1" } });
    expect(getClientIp(req2)).toBe("198.51.100.1");

    const req3 = new Request("http://localhost");
    expect(getClientIp(req3)).toBe("127.0.0.1");
  });
});
