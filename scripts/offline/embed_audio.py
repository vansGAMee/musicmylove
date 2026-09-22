"""Extract real audio embeddings. Optional heavy offline dependencies, never production.
Input JSON list: {mbid, path, source, license}. Downloaded audio belongs in data/cache/.
Usage: python scripts/offline/embed_audio.py manifest.json checkpoint.pt output.json
Install separately: pip install laion-clap
"""
import hashlib
import json
import sys
from pathlib import Path


def main():
    import laion_clap
    rows = sorted(json.loads(Path(sys.argv[1]).read_text()), key=lambda r: r['mbid'])
    if not rows or len({r['mbid'] for r in rows}) != len(rows):
        raise ValueError('Nonempty, unique audio manifest required')
    for row in rows:
        if not row.get('source') or not row.get('license') or not Path(row['path']).is_file():
            raise ValueError('Every recording needs real audio, source and license')
    checkpoint = Path(sys.argv[2])
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(str(checkpoint))
    vectors = []
    for offset in range(0, len(rows), 16):
        vectors.extend(model.get_audio_embedding_from_filelist(x=[r['path'] for r in rows[offset:offset+16]], use_tensor=False).tolist())
    result = {'ids': [r['mbid'] for r in rows], 'vectors': vectors,
              'encoder': 'laion-clap:' + hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'provenance': [{**r, 'sha256': hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()} for r in rows]}
    Path(sys.argv[3]).write_text(json.dumps(result, allow_nan=False))

if __name__ == '__main__':
    main()
