"""Download, verify, and index legal audio datasets (FMA, MTG-Jamendo, etc.).
No scraping of commercial streaming services. Creative Commons / Public Domain only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile
import subprocess

try:
    from .config import DATA, write_json
    from .prepare_data import normalize
except ImportError:
    from config import DATA, write_json
    from prepare_data import normalize

USER_AGENT = os.environ.get('AUDIO_USER_AGENT', 'MusicMyLoveResearch/0.2 (audio representation MIR research)')

AUDIO_SOURCES = {
    "fma_metadata": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "source": "Free Music Archive (FMA) Metadata"
    },
    "fma_small": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "license": "Creative Commons (per-track in metadata)",
        "source": "Free Music Archive (FMA) Small (8,000 clips)"
    }
}


def download_file(url: str, destination: Path, user_agent: str = USER_AGENT) -> Path:
    """Download a file with resume support and descriptive User-Agent."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + '.part')

    initial_size = part_path.stat().st_size if part_path.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': user_agent})
    if initial_size > 0:
        req.add_header('Range', f'bytes={initial_size}-')

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            mode = 'ab' if initial_size > 0 and resp.status == 206 else 'wb'
            with part_path.open(mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        part_path.replace(destination)
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    return destination


def build_audio_manifest(audio_dir: Path, output_json: Path, source_name: str = "fma_small",
                         license_name: str = "Creative Commons", max_tracks: int = 0) -> dict:
    """Scan audio directory for MP3/OGG/FLAC files and index them into an audio manifest."""
    audio_dir = Path(audio_dir)
    files = sorted(list(audio_dir.rglob('*.mp3')) + list(audio_dir.rglob('*.ogg')) + list(audio_dir.rglob('*.flac')))
    if not files:
        root_audio = Path(__file__).resolve().parent / 'data' / 'cache' / 'audio'
        if root_audio.exists() and root_audio != audio_dir:
            fallback_files = sorted(list(root_audio.rglob('*.mp3')) + list(root_audio.rglob('*.ogg')) + list(root_audio.rglob('*.flac')))
            if fallback_files:
                audio_dir.mkdir(parents=True, exist_ok=True)
                for f in fallback_files:
                    target_file = audio_dir / f.name
                    if not target_file.exists():
                        try:
                            target_file.symlink_to(f.resolve())
                        except OSError:
                            shutil.copy2(f, target_file)
                files = sorted(list(audio_dir.rglob('*.mp3')) + list(audio_dir.rglob('*.ogg')) + list(audio_dir.rglob('*.flac')))
    if max_tracks > 0:
        files = files[:max_tracks]

    manifest_tracks = []
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        sha256 = h.hexdigest()

        stem = p.stem
        track_id = f"audio:{source_name}:{stem}"

        tags = {}
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(p)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            probe_data = json.loads(proc.stdout.decode('utf-8'))
            tags = probe_data.get("format", {}).get("tags", {})
        except Exception:
            pass

        artist = tags.get("artist", tags.get("ARTIST", f"artist_{stem}")).strip()
        title = tags.get("title", tags.get("TITLE", f"track_{stem}")).strip()
        comment = tags.get("comment", license_name).strip()

        manifest_tracks.append({
            "track_id": track_id,
            "filename": p.name,
            "artist_name": artist,
            "track_title": title,
            "rel_path": str(p.relative_to(audio_dir)),
            "abs_path": str(p.resolve()),
            "sha256": sha256,
            "source": source_name,
            "license": comment if "creativecommons" in comment.lower() else license_name,
            "audio_available": True
        })

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "license": license_name,
        "track_count": len(manifest_tracks),
        "tracks": manifest_tracks
    }
    write_json(output_json, result)
    return result


import io


class HttpZipStream(io.RawIOBase):
    """Seekable HTTP stream using Range requests to read remote ZIP files without full download."""
    def __init__(self, url: str, user_agent: str = USER_AGENT):
        self.url = url
        self.user_agent = user_agent
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._size = int(resp.headers['Content-Length'])
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def readinto(self, b) -> int:
        l = len(b)
        if self._pos >= self._size or l == 0:
            return 0
        end = min(self._size - 1, self._pos + l - 1)
        req = urllib.request.Request(self.url, headers={
            'Range': f'bytes={self._pos}-{end}',
            'User-Agent': self.user_agent
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self._pos += n
        return n


def stream_fma_tracks(audio_dir: Path, count: int = 100, user_agent: str = USER_AGENT) -> int:
    """Stream-extract real MP3 tracks from remote FMA archive via HTTP Range requests."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    url = AUDIO_SOURCES['fma_small']['url']
    print(f"Connecting to remote FMA archive via HTTP Range stream: {url}")
    stream = HttpZipStream(url, user_agent=user_agent)
    zf = zipfile.ZipFile(stream)
    mp3_names = [n for n in zf.namelist() if n.endswith('.mp3')]
    print(f"Discovered {len(mp3_names)} MP3 tracks in remote FMA archive. Stream-downloading {count} tracks...")

    downloaded = 0
    for name in mp3_names:
        if downloaded >= count:
            break
        dest = audio_dir / Path(name).name
        if dest.exists() and dest.stat().st_size > 10000:
            downloaded += 1
            continue
        try:
            data = zf.read(name)
            dest.write_bytes(data)
            downloaded += 1
            if downloaded % 10 == 0 or downloaded == count:
                print(f"  [{downloaded}/{count}] Downloaded {dest.name} ({len(data)} bytes)")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")
            continue

    return downloaded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['download', 'index', 'status', 'stream-fma'], default='status')
    p.add_argument('--source', default='fma_small', choices=list(AUDIO_SOURCES.keys()))
    p.add_argument('--audio-dir', type=Path, default=DATA / 'cache' / 'audio')
    p.add_argument('--max-tracks', type=int, default=100)
    args = p.parse_args()

    audio_dir = args.audio_dir.resolve()
    manifest_path = audio_dir / 'audio_manifest.json'

    if args.action == 'stream-fma':
        count = args.max_tracks if args.max_tracks > 0 else 100
        downloaded = stream_fma_tracks(audio_dir, count=count)
        print(f"Stream-downloaded {downloaded} real FMA audio tracks.")
        manifest = build_audio_manifest(audio_dir, manifest_path, source_name="fma_small",
                                        license_name=AUDIO_SOURCES['fma_small']['license'])
        print(f"Indexed {manifest['track_count']} audio tracks into {manifest_path}")

    elif args.action == 'download':
        src_info = AUDIO_SOURCES[args.source]
        zip_dest = audio_dir / f"{args.source}.zip"
        print(f"Downloading {args.source} from {src_info['url']} to {zip_dest}...")
        download_file(src_info['url'], zip_dest)
        print("Download complete. Extracting archive...")
        with zipfile.ZipFile(zip_dest, 'r') as zf:
            zf.extractall(audio_dir)
        print("Extraction complete. Indexing manifest...")
        manifest = build_audio_manifest(audio_dir, manifest_path, source_name=args.source,
                                        license_name=src_info['license'], max_tracks=args.max_tracks)
        print(f"Indexed {manifest['track_count']} audio tracks into {manifest_path}")

    elif args.action == 'index':
        src_info = AUDIO_SOURCES.get(args.source, {"source": args.source, "license": "Creative Commons"})
        manifest = build_audio_manifest(audio_dir, manifest_path, source_name=args.source,
                                        license_name=src_info['license'], max_tracks=args.max_tracks)
        print(f"Indexed {manifest['track_count']} audio tracks into {manifest_path}")

    elif args.action == 'status':
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            print(f"Audio manifest found: {manifest_path}")
            print(f"Tracks: {data['track_count']}, Source: {data.get('source')}, License: {data.get('license')}")
        else:
            print(f"No audio manifest at {manifest_path}")


if __name__ == '__main__':
    main()
