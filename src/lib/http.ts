const USER_AGENT = "MusicMyLove/0.1 (music recommendation MVP; contact: dev@example.com)";

interface RetryOptions extends RequestInit {
  fetcher?: typeof fetch;
  sleep?: (milliseconds: number) => Promise<void>;
  jitter?: () => number;
  attempts?: number;
  timeoutMs?: number;
}

export async function fetchWithRetry(url: string, options: RetryOptions = {}): Promise<Response> {
  const { fetcher = fetch, sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), jitter = Math.random, attempts = 3, timeoutMs = 5000, ...init } = options;
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetcher(url, { ...init, signal: init.signal ?? controller.signal, headers: { Accept: "application/json", "User-Agent": USER_AGENT, ...init.headers } });
      if (response.ok) return response;
      const retryable = response.status === 429 || response.status >= 500;
      if (!retryable || attempt === attempts - 1) throw new Error(`Upstream request failed with ${response.status}`);
      const retryAfter = Number(response.headers.get("retry-after"));
      await sleep(Number.isFinite(retryAfter) ? retryAfter * 1000 : 150 * 2 ** attempt + jitter() * 50);
    } catch (error) {
      lastError = error;
      if (attempt === attempts - 1 || (error instanceof Error && /with 4\d\d/.test(error.message))) throw error;
      await sleep(150 * 2 ** attempt + jitter() * 50);
    } finally { clearTimeout(timer); }
  }
  throw lastError instanceof Error ? lastError : new Error("Upstream request failed");
}
