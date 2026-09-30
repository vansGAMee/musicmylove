"""Transparent metadata matching for CLI profiles, without merging recording IDs."""
from collections import defaultdict
import re
from .recommend import match_inputs, match_key


def remaster_key(value):
    value = match_key(value)
    # Only mastering labels. Live, remix, acoustic and arbitrary brackets remain.
    value = re.sub(r'\s*[\[(](?:(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?)[\])]\s*$', '', value)
    return re.sub(r'\s+-\s+(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?\s*$', '', value)


def resolve_profile(lines, tracks, strict=False):
    by_mastering = defaultdict(set)
    if not strict:
        for i, track in enumerate(tracks):
            for artist, title in [(track['artist'], track['title']), *track.get('aliases', [])]:
                by_mastering[remaster_key(artist + ' - ' + title)].add(i)
    seeds, matches, unresolved = set(), [], []
    for line, exact in match_inputs(lines, tracks):
        candidates, kind = exact, 'exact_name_or_id'
        if not candidates and not strict:
            candidates = by_mastering.get(remaster_key(line), set())
            kind = 'remaster_name'
        selected = next(iter(candidates)) if len(candidates) == 1 else None
        if len(candidates) > 1 and not strict:
            # Prefer the catalog's explicit metadata-only record for this name.
            # This does NOT infer an MBID, fuse vectors, or assert recording identity.
            metadata = [i for i in candidates if tracks[i]['id'].startswith('fallback:')
                        and match_key(tracks[i]['artist'] + ' - ' + tracks[i]['title']) == match_key(line)]
            if len(metadata) == 1:
                selected = metadata[0]
                kind = 'metadata_name'
        if selected is None:
            unresolved.append(line)
        else:
            seeds.add(selected)
            matches.append({'input': line, 'id': tracks[selected]['id'], 'kind': kind,
                            'alternatives': len(candidates)})
    return {'seeds': sorted(seeds), 'matches': matches, 'unresolved': unresolved}
