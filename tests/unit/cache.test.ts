import { expect, test } from "vitest";
import { VersionedCache } from "../../src/lib/cache";

test("returns stale data separately and isolates schema versions", () => {
  let now = 1000;
  const storage = new Map<string, string>();
  const cache = new VersionedCache("v1", { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) }, () => now);
  cache.set("x", { value: 1 }, 50);
  expect(cache.get("x")).toEqual({ value: { value: 1 }, stale: false });
  now = 1100;
  expect(cache.get("x")).toEqual({ value: { value: 1 }, stale: true });
  expect(new VersionedCache("v2", cache.storage, () => now).get("x")).toBeNull();
});
