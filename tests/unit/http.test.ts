import { expect, test, vi } from "vitest";
import { fetchWithRetry } from "../../src/lib/http";

test("retries 429 and 5xx with a bounded attempt count", async () => {
  const fetcher = vi.fn()
    .mockResolvedValueOnce(new Response("busy", { status: 429, headers: { "retry-after": "0" } }))
    .mockResolvedValueOnce(new Response("oops", { status: 503 }))
    .mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }));
  const result = await fetchWithRetry("https://example.test", { fetcher, sleep: async () => {}, jitter: () => 0 });
  expect(await result.json()).toEqual({ ok: true });
  expect(fetcher).toHaveBeenCalledTimes(3);
});

test("does not retry a permanent client error", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("bad", { status: 400 }));
  await expect(fetchWithRetry("https://example.test", { fetcher, sleep: async () => {} })).rejects.toThrow(/400/);
  expect(fetcher).toHaveBeenCalledOnce();
});
