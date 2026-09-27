"""Read only ranker/DEV labels and catalog-related identities from prepared JSON.

The input remains on disk; memory is the retained data plus one decoded row and
a read buffer. Sessions are discarded with each user row. Both dataset writers
emit tracks before users, allowing immediate filtering without renumbering IDs.
"""
import json
from pathlib import Path
import re

from .discovery_engine import song_key
from .profile_input import remaster_key

_WHITESPACE = re.compile(r'[ \t\r\n]*')


class _JSONStream:
    def __init__(self, stream, chunk_size):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.position, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        if self.eof:
            return False
        block = self.stream.read(self.chunk_size)
        self.buffer = self.buffer[self.position:] + block
        self.position = 0
        self.eof = not block
        return bool(block)

    def peek(self):
        while True:
            self.position = _WHITESPACE.match(self.buffer, self.position).end()
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if not self.fill():
                return ''

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f'Invalid or truncated dataset JSON: expected {character!r}')
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError('Truncated dataset JSON')
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if not self.fill():
                    raise ValueError('Invalid or truncated dataset JSON') from error
            else:
                self.position = end
                return value

    def rows(self):
        self.expect('[')
        if self.peek() == ']':
            self.position += 1
            return
        while True:
            # Prepared tracks and users are objects; do not decode entire arrays.
            if self.peek() != '{':
                raise ValueError('Expected a track or user object in dataset JSON')
            yield self.value()
            if self.peek() == ']':
                self.position += 1
                return
            self.expect(',')


def _keys(track):
    yield song_key(track['artist'], track['title'])
    for artist, title in track.get('aliases', []):
        yield song_key(artist, title)


def _dataset_rows(path, chunk_size):
    """Yield (field, array index, value), never materializing a top-level array."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    fields = set()
    with Path(path).open(encoding='utf-8') as source:
        reader = _JSONStream(source, chunk_size)
        reader.expect('{')
        while True:
            if reader.peek() != '"':
                raise ValueError('Expected dataset field name')
            field = reader.value()
            if field in fields or field not in {'fingerprint', 'tracks', 'users'}:
                raise ValueError(f'Unexpected or duplicate dataset field: {field}')
            reader.expect(':')
            if field == 'fingerprint':
                if reader.peek() != '"':
                    raise ValueError('Dataset fingerprint must be a string')
                yield field, None, reader.value()
            else:
                if field == 'users' and 'tracks' not in fields:
                    raise ValueError('Prepared dataset must contain tracks before users')
                for index, row in enumerate(reader.rows()):
                    yield field, index, row
            fields.add(field)
            if reader.peek() == '}':
                reader.position += 1
                break
            reader.expect(',')
        if reader.peek():
            raise ValueError('Trailing input after dataset JSON')
    if fields != {'fingerprint', 'tracks', 'users'}:
        raise ValueError('Dataset is missing required fields')


def load_ranker_data(path, meta, *, chunk_size=65536):
    """Return compact data compatible with mapped_users and raw_protection.

    Track dictionary keys are original raw integer indices. In addition to the
    graph vocabulary, retain every raw identity sharing a canonical/alias key
    with that vocabulary, so out-of-catalog positives still protect negatives.
    Only ranker_train and dev users are retained; FINAL labels are not used.
    """
    if len(meta['seen']) != len(meta['tracks']) or len(set(meta['seen'])) != len(meta['seen']):
        raise ValueError('Invalid graph vocabulary')
    expected = dict(zip(meta['seen'], meta['tracks']))
    catalog_keys = {key for track in meta['tracks'] for key in _keys(track)}
    tracks, users, fingerprint = {}, [], None
    for field, raw, value in _dataset_rows(path, chunk_size):
        if field == 'fingerprint':
            fingerprint = value
        elif field == 'tracks':
            if raw in expected and value['id'] != expected[raw]['id']:
                raise ValueError(f'Graph vocabulary mismatch at raw track {raw}')
            if raw in expected or any(key in catalog_keys for key in _keys(value)):
                tracks[raw] = value
        elif value['split'] in ('ranker_train', 'dev'):
            users.append({'id': value['id'], 'split': value['split'],
                          'tracks': [raw for raw in value['tracks'] if raw in tracks]})
    if not expected.keys() <= tracks.keys():
        raise ValueError('Dataset is missing graph vocabulary tracks')
    if 'fingerprint' in meta and fingerprint != meta['fingerprint']:
        raise ValueError('Dataset fingerprint differs from graph metadata')
    return {'fingerprint': fingerprint, 'tracks': tracks, 'users': users}


def load_coverage_data(path, meta, lines, *, chunk_size=65536):
    """Project raw identities and representation-user support for diagnose().

    Unlike ranker data, coverage needs a compact track list, with user indices
    remapped into that list. Original encounter order preserves raw_ids in the
    report. Held-out users never contribute to representation support.
    """
    wanted = {remaster_key(line) for line in lines}
    representation = set(meta['train_users'])
    tracks, users, indices, fingerprint = [], [], {}, None
    for field, raw, value in _dataset_rows(path, chunk_size):
        if field == 'fingerprint':
            fingerprint = value
        elif field == 'tracks':
            names = [(value['artist'], value['title']), *value.get('aliases', [])]
            if any(remaster_key(artist + ' - ' + title) in wanted for artist, title in names):
                indices[raw] = len(tracks)
                tracks.append(value)
        elif value['id'] in representation:
            users.append({'id': value['id'],
                          'tracks': [indices[raw] for raw in value['tracks'] if raw in indices]})
    if 'fingerprint' in meta and fingerprint != meta['fingerprint']:
        raise ValueError('Dataset fingerprint differs from graph metadata')
    return {'fingerprint': fingerprint, 'tracks': tracks, 'users': users}
