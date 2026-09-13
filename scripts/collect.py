import argparse, json, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

UA = "MusicMyLove/0.1 (music recommendation research; contact: dev@example.com)"

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("usernames"); parser.add_argument("--limit", type=int, default=400); args = parser.parse_args()
    cache = Path("data/cache/users"); cache.mkdir(parents=True, exist_ok=True)
    names = [line.strip() for line in Path(args.usernames).read_text().splitlines() if line.strip()][:args.limit]
    for name in names:
        target = cache / f"{urllib.parse.quote(name, safe='')}.json"
        if target.exists(): continue
        url = f"https://api.listenbrainz.org/1/stats/user/{urllib.parse.quote(name)}/recordings?range=all_time&count=100"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=10) as response:
                    target.write_text(json.dumps(json.load(response), indent=2) + "\n"); break
            except (urllib.error.URLError, TimeoutError): time.sleep((2 ** attempt) + 0.2)
        time.sleep(0.35)
    print(f"cached={len(list(cache.glob('*.json')))} requested={len(names)}")

if __name__ == "__main__": main()
