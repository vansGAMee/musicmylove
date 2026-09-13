interface StorageLike { getItem(key: string): string | null; setItem(key: string, value: string): void }
interface Entry<T> { value: T; expiresAt: number }

export class VersionedCache<T> {
  constructor(public readonly version: string, public readonly storage: StorageLike, private readonly now = Date.now) {}
  private key(key: string) { return `${this.version}:${key}`; }
  set(key: string, value: T, ttlMs: number) { this.storage.setItem(this.key(key), JSON.stringify({ value, expiresAt: this.now() + ttlMs })); }
  get(key: string): { value: T; stale: boolean } | null {
    const raw = this.storage.getItem(this.key(key));
    if (!raw) return null;
    try { const entry = JSON.parse(raw) as Entry<T>; return { value: entry.value, stale: entry.expiresAt < this.now() }; }
    catch { return null; }
  }
}
