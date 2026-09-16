#!/usr/bin/env npx tsx
/**
 * CRITICAL TEST: Yandex Music Playlist Extraction Fix
 * Проверяет все улучшения в работе с Яндекс Музыкой
 * 
 * Запуск: npx tsx scripts/test-yandex-fix.ts
 */

import { 
  normalizeYandexPlaylistUrl,
  extractFromHtmlState,
  YandexPlaylistError 
} from "../src/lib/yandex/playlist";

console.log('\n' + '='.repeat(70));
console.log('🎵 CRITICAL YANDEX MUSIC FIX VALIDATION');
console.log('='.repeat(70) + '\n');

let passed = 0;
let failed = 0;
const results: { name: string; status: 'pass' | 'fail'; error?: string }[] = [];

function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`✅ ${name}`);
    results.push({ name, status: 'pass' });
    passed++;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.log(`❌ ${name}`);
    console.log(`   → ${msg}\n`);
    results.push({ name, status: 'fail', error: msg });
    failed++;
  }
}

// ============================================
// PART 1: URL NORMALIZATION
// ============================================
console.log('📋 PART 1: URL NORMALIZATION\n');

test('Classic playlist URL', () => {
  const res = normalizeYandexPlaylistUrl('https://music.yandex.ru/users/test/playlists/123');
  if (res.type !== 'user_playlist' || res.owner !== 'test' || res.kind !== '123') {
    throw new Error('Failed to normalize classic URL');
  }
});

test('UUID playlist URL', () => {
  const res = normalizeYandexPlaylistUrl('https://music.yandex.ru/playlists/72b84cfd-f06b-4e8c');
  if (res.type !== 'uuid_playlist') {
    throw new Error('Failed to normalize UUID URL');
  }
});

test('URL with tracking params (should strip)', () => {
  const res = normalizeYandexPlaylistUrl('https://music.yandex.ru/users/test/playlists/123?utm_source=share&utm_medium=copy');
  if (res.canonicalUrl.includes('utm')) {
    throw new Error('Failed to strip UTM params');
  }
});

test('Invalid URL rejection', () => {
  try {
    normalizeYandexPlaylistUrl('https://spotify.com/playlist/123');
    throw new Error('Should have rejected Spotify URL');
  } catch (e) {
    if (!(e instanceof YandexPlaylistError)) {
      throw new Error('Did not throw YandexPlaylistError');
    }
  }
});

// ============================================
// PART 2: JSON-LD EXTRACTION
// ============================================
console.log('\n📋 PART 2: JSON-LD SCHEMA EXTRACTION\n');

test('Extract from JSON-LD MusicPlaylist', () => {
  const html = `
    <script type="application/ld+json">
    {
      "@type": "MusicPlaylist",
      "name": "Test List",
      "track": [
        {"name": "Track 1", "byArtist": {"name": "Artist 1"}},
        {"name": "Track 2", "byArtist": {"name": "Artist 2"}}
      ]
    }
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 2) throw new Error(`Got ${res.tracks.length} tracks, expected 2`);
  if (res.title !== 'Test List') throw new Error('Title not extracted');
});

test('Multiple JSON-LD blocks', () => {
  const html = `
    <script type="application/ld+json">
    {"@type": "MusicPlaylist", "track": [{"name": "T1", "byArtist": {"name": "A1"}}]}
    </script>
    <script type="application/ld+json">
    {"@type": "MusicPlaylist", "track": [{"name": "T2", "byArtist": {"name": "A2"}}]}
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 2) throw new Error('Failed to parse multiple JSON-LD blocks');
});

// ============================================
// PART 3: CRITICAL - NEXT.JS EXTRACTION
// ============================================
console.log('\n📋 PART 3: CRITICAL - NEXT.JS SELF.__NEXT_F.PUSH EXTRACTION\n');

test('Extract from self.__next_f.push (Next.js streaming)', () => {
  const html = `
    <script>
      self.__next_f.push([1,"{\\"tracks\\":[{\\"id\\":\\"1\\",\\"title\\":\\"Song One\\",\\"artists\\":[{\\"name\\":\\"Artist One\\"}]},{\\"id\\":\\"2\\",\\"title\\":\\"Song Two\\",\\"artists\\":[{\\"name\\":\\"Artist Two\\"}]}]}"]);
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 2) throw new Error(`Got ${res.tracks.length} tracks, expected 2+`);
  if (!res.tracks.some(t => t.title === 'Song One')) throw new Error('Failed to extract Song One');
  if (!res.tracks.some(t => t.artists.includes('Artist One'))) throw new Error('Failed to extract Artist One');
});

test('Next.js with complex nesting', () => {
  const html = `
    <script>
      self.__next_f.push([1,"{\\"props\\":{\\"playlist\\":{\\"tracks\\":[{\\"title\\":\\"Nested Song\\",\\"artists\\":[{\\"name\\":\\"Nested Artist\\"}]}]}}}"]);
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 1) throw new Error('Failed to extract from nested Next.js structure');
});

// ============================================
// PART 4: CRITICAL - DEEPLY NESTED JSON
// ============================================
console.log('\n📋 PART 4: CRITICAL - DEEPLY NESTED JSON STRUCTURES\n');

test('Recursively extract from deep nesting', () => {
  const html = `
    <script>
      {
        "props": {
          "pageProps": {
            "dehydratedState": {
              "queries": [{
                "state": {
                  "data": {
                    "playlist": {
                      "title": "Deep Playlist",
                      "tracks": [
                        {"id": "1", "title": "Deep Song", "artists": [{"name": "Deep Artist"}]}
                      ]
                    }
                  }
                }
              }]
            }
          }
        }
      }
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 1) throw new Error('Failed to extract from deep nesting');
  if (!res.tracks.some(t => t.title === 'Deep Song')) throw new Error('Deep song not found');
});

// ============================================
// PART 5: CRITICAL - ESCAPED CHARACTERS
// ============================================
console.log('\n📋 PART 5: CRITICAL - ESCAPED JSON HANDLING\n');

test('Handle escaped quotes in JSON', () => {
  const html = `
    <script>
      {
        "playlist": {
          "tracks": [
            {"title": "Song \\"With\\" Quotes", "artists": [{"name": "Artist \\"Name\\""}]}
          ]
        }
      }
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 1) throw new Error('Failed with escaped quotes');
});

test('Handle escaped backslashes', () => {
  const html = `
    <script>
      {
        "tracks": [
          {"title": "Path\\\\Like\\\\String", "artists": [{"name": "Normal"}]}
        ]
      }
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length < 1) throw new Error('Failed with escaped backslashes');
});

// ============================================
// PART 6: CRITICAL - ARTIST FIELD VARIANTS
// ============================================
console.log('\n📋 PART 6: CRITICAL - ALTERNATIVE ARTIST FIELD NAMES\n');

test('Extract from "author" field', () => {
  const html = `
    <script>
      {"tracks": [{"title": "By Author", "author": "Author Artist"}]}
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (!res.tracks.some(t => t.artists.includes('Author Artist'))) {
    throw new Error('Failed to extract from author field');
  }
});

test('Extract from "performer" field', () => {
  const html = `
    <script>
      {"tracks": [{"title": "By Performer", "performer": "Performer Artist"}]}
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (!res.tracks.some(t => t.artists.includes('Performer Artist'))) {
    throw new Error('Failed to extract from performer field');
  }
});

test('Extract from "artists" array', () => {
  const html = `
    <script>
      {"tracks": [{"title": "By Artists", "artists": [{"name": "Array Artist"}]}]}
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (!res.tracks.some(t => t.artists.includes('Array Artist'))) {
    throw new Error('Failed to extract from artists array');
  }
});

// ============================================
// PART 7: CRITICAL - ERROR HANDLING
// ============================================
console.log('\n📋 PART 7: CRITICAL - ERROR HANDLING & FALLBACK\n');

test('Malformed JSON (graceful fallback)', () => {
  const html = `
    <script>
      {incomplete json
      "tracks": [
        {"id": 1, "title": "Track", "artists": [{"name": "Artist"}]}
      ]
    </script>
  `;
  // Should not throw
  extractFromHtmlState(html);
});

test('Detect 404 page', () => {
  const html = '<html>404: This page could not be found</html>';
  try {
    extractFromHtmlState(html);
    throw new Error('Should have thrown for 404');
  } catch (e) {
    if (!(e instanceof YandexPlaylistError) || e.code !== 'not_found') {
      throw new Error('Did not throw correct 404 error');
    }
  }
});

test('Empty playlist handling', () => {
  const html = `
    <script>
      {"playlist": {"title": "Empty", "tracks": []}}
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length !== 0) throw new Error('Should have 0 tracks for empty playlist');
});

// ============================================
// PART 8: DEDUPLICATION
// ============================================
console.log('\n📋 PART 8: DEDUPLICATION\n');

test('Remove duplicate tracks (same artist & title)', () => {
  const html = `
    <script>
      {
        "tracks": [
          {"title": "Song", "artists": [{"name": "Artist"}]},
          {"title": "Song", "artists": [{"name": "Artist"}]},
          {"title": "Different", "artists": [{"name": "Artist"}]}
        ]
      }
    </script>
  `;
  const res = extractFromHtmlState(html);
  if (res.tracks.length > 2) throw new Error(`Expected 2 tracks after dedup, got ${res.tracks.length}`);
});

// ============================================
// SUMMARY
// ============================================
console.log('\n' + '='.repeat(70));
console.log(`📊 FINAL RESULTS: ${passed} passed, ${failed} failed\n`);

if (failed > 0) {
  console.log('⚠️  FAILED TESTS:\n');
  results.filter(r => r.status === 'fail').forEach(r => {
    console.log(`  ❌ ${r.name}`);
    if (r.error) console.log(`     ${r.error}`);
  });
}

console.log('\n' + '='.repeat(70) + '\n');

if (failed === 0) {
  console.log('🎉 ✅ ALL TESTS PASSED!\n');
  console.log('✨ Yandex Music playlist extraction is working correctly!\n');
  process.exit(0);
} else {
  console.log(`⚠️  ${failed} test(s) failed. Please review above.\n`);
  process.exit(1);
}
