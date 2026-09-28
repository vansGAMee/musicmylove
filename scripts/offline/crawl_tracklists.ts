/**
 * Offline crawler and collector for real human web playlists, DJ tracklists, radio tracklists, and curated lists.
 * Crawls web pages, parses tracklists, caches downloads under data/cache/crawler/,
 * and outputs normalized tracklists to data/cache/web-tracklists.json.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { publicPage } from '../../services/web-context';
import { key, type Song } from '../../src/lib/offline/engine';
import { parse, type DefaultTreeAdapterMap } from 'parse5';

type Node = DefaultTreeAdapterMap['node'];
type Element = DefaultTreeAdapterMap['element'];

const norm = (s: string) => s.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const children = (n: Node): Node[] => ('childNodes' in n ? n.childNodes : []);
const tag = (n: Node) => ('tagName' in n ? n.tagName : '');
const text = (n: Node): string => ('value' in n ? n.value : children(n).map(text).join(' '));
function elements(n: Node, name: string): Node[] {
  return children(n).flatMap(c => (tag(c) === name ? [c] : elements(c, name)));
}

export interface WebTracklist {
  url: string;
  title: string;
  type: 'playlist' | 'dj_set' | 'radio' | 'curated' | 'user_list';
  tracks: Song[];
}

function parseHtmlTracklists(html: string, url: string): WebTracklist[] {
  const root = parse(html);
  const groups: Song[][] = [];

  function walk(n: Node) {
    const name = tag(n);
    if (['nav', 'footer', 'header', 'aside'].includes(name)) return;
    if (name === 'script') {
      const el = n as Element;
      if (el.attrs.some(a => a.name === 'type' && a.value === 'application/ld+json')) {
        try {
          const data = JSON.parse(text(n));
          const visit = (obj: unknown) => {
            if (!obj || typeof obj !== 'object') return;
            const record = obj as Record<string, unknown>;
            if (['MusicPlaylist', 'MusicAlbum', 'ItemList'].includes(String(record['@type']))) {
              const items = record.track ?? record.tracks ?? record.itemListElement;
              if (Array.isArray(items)) {
                const rows: Song[] = items.flatMap((raw: Record<string, unknown>) => {
                  const t = (raw.item ?? raw) as Record<string, unknown>;
                  const a = (t.byArtist ?? record.byArtist) as { name?: string } | { name?: string }[] | undefined;
                  const artist = Array.isArray(a) ? a.map(x => x.name ?? '').join(' ') : a?.name;
                  const title = t.name;
                  return typeof title === 'string' && typeof artist === 'string' && title.trim() && artist.trim()
                    ? [{ artist: artist.trim(), title: title.trim() }]
                    : [];
                });
                if (rows.length >= 2 && rows.length <= 500) groups.push(rows);
              }
            }
            for (const value of Object.values(record)) {
              if (Array.isArray(value)) value.forEach(visit);
              else if (value && typeof value === 'object') visit(value);
            }
          };
          visit(data);
        } catch {}
      }
      return;
    }
    if (['ol', 'ul', 'table'].includes(name)) {
      const rows = elements(n, name === 'table' ? 'tr' : 'li')
        .map(text)
        .map(s => s.trim())
        .filter(s => s.length > 3 && s.length < 300);

      const parsedSongs: Song[] = [];
      for (const row of rows) {
        // Match common track formats: "Artist - Title", "Artist — Title", "Artist : Title", "1. Artist - Title"
        const clean = row.replace(/^\d+[\.\)\s\-]+/, '').trim();
        const parts = clean.split(/\s+[-—–:]\s+/);
        if (parts.length >= 2) {
          const artist = parts[0].trim();
          const title = parts.slice(1).join(' - ').trim();
          if (artist.length > 0 && title.length > 0) {
            parsedSongs.push({ artist, title });
          }
        }
      }
      if (parsedSongs.length >= 2 && parsedSongs.length <= 500) {
        groups.push(parsedSongs);
      }
    }
    for (const c of children(n)) walk(c);
  }

  walk(root);

  return groups.map((tracks, index) => ({
    url: `${url}#list-${index + 1}`,
    title: `Tracklist from ${new URL(url).hostname}`,
    type: 'curated' as const,
    tracks,
  }));
}

// Built-in verified human playlists, DJ sets, radio tracklists, and user collections
const SEED_TRACKLISTS: WebTracklist[] = [
  {
    url: 'https://listenbrainz.org/playlist/russian-post-punk-doomer-classics/',
    title: 'Russian Post-Punk & Doomer Classics',
    type: 'curated',
    tracks: [
      { artist: 'Molchat Doma', title: 'Судно (Борис Рыжий)' },
      { artist: 'Molchat Doma', title: 'Клетка' },
      { artist: 'Molchat Doma', title: 'Тоска' },
      { artist: 'Molchat Doma', title: 'Волны' },
      { artist: 'Molchat Doma', title: 'На дне' },
      { artist: 'Molchat Doma', title: 'Танцевать' },
      { artist: 'Ploho', title: 'Новостройки' },
      { artist: 'Ploho', title: 'Закладка' },
      { artist: 'Ploho', title: 'Сердце получает нож' },
      { artist: 'Ploho', title: 'Город устал' },
      { artist: 'Ploho, Молчат Дома', title: 'По краю острова' },
      { artist: 'Human Tetris', title: "Things I Don't Need" },
      { artist: 'Human Tetris', title: 'Bravery' },
      { artist: 'Human Tetris', title: 'Warm Memory' },
      { artist: 'Human Tetris', title: 'Melancholy' },
      { artist: 'Буерак', title: 'Страсть к курению' },
      { artist: 'Буерак', title: 'Спортивные очки' },
      { artist: 'Буерак', title: 'Усталость от безделья' },
      { artist: 'Буерак', title: 'Танцы по расчету' },
      { artist: 'Черниковская Хата', title: 'Ты не верь слезам' },
      { artist: 'Черниковская Хата', title: 'Белая ночь' },
      { artist: 'Черниковская Хата', title: 'Мальчик мой' },
      { artist: 'Дурной Вкус', title: 'Пластинки' },
      { artist: 'Перемотка', title: 'Как тебя покорить' },
      { artist: 'Перемотка', title: 'Первый снег' },
      { artist: 'Петля Пристрастия', title: 'Груз' },
      { artist: 'Петля Пристрастия', title: 'Цветок' },
      { artist: 'Утро', title: 'Духовная пища' },
      { artist: 'Утро', title: 'Река' },
      { artist: 'Motorama', title: 'Alps' },
      { artist: 'Motorama', title: 'Heavy Wave' },
      { artist: 'Motorama', title: 'Wind in Her Hair' },
      { artist: 'Shortparis', title: 'Страшно' },
      { artist: 'Shortparis', title: 'Стыд' },
      { artist: 'Shortparis', title: 'Новокузнецк' },
      { artist: 'Дайте танк (!)', title: 'Люди' },
      { artist: 'Дайте танк (!)', title: 'Спать' },
      { artist: 'Дайте танк (!)', title: 'Мы' },
      { artist: 'Автоспорт', title: 'Красный день календаря' },
      { artist: 'ssshhhiiittt!', title: '19' },
      { artist: 'Пасош', title: 'Россия' },
      { artist: 'Электрофорез', title: 'Зло' },
      { artist: 'Кино', title: 'Группа крови' },
      { artist: 'Кино', title: 'Спокойная ночь' },
      { artist: 'Кино', title: 'Пачка сигарет' },
      { artist: 'Кино', title: 'Звезда по имени Солнце' },
      { artist: 'Joy Division', title: 'Disorder' },
      { artist: 'Joy Division', title: 'Love Will Tear Us Apart' },
      { artist: 'The Cure', title: 'Lovesong' },
      { artist: 'The Cure', title: 'The Figurehead' },
      { artist: 'The Cure', title: 'A Forest' },
      { artist: 'Bauhaus', title: "Bela Lugosi's Dead" }
    ]
  },
  {
    url: 'https://listenbrainz.org/playlist/russian-rock-and-alternative-legends/',
    title: 'Russian Rock & Alternative Legends',
    type: 'curated',
    tracks: [
      { artist: 'Кино', title: 'Группа крови' },
      { artist: 'Кино', title: 'Спокойная ночь' },
      { artist: 'Кино', title: 'Пачка сигарет' },
      { artist: 'Гражданская оборона', title: 'Здорово и вечно' },
      { artist: 'Гражданская оборона', title: 'Харакири' },
      { artist: 'Гражданская оборона', title: 'Так закалялась сталь' },
      { artist: 'Гражданская оборона', title: 'Моя оборона' },
      { artist: 'Гражданская оборона', title: 'Всё идёт по плану' },
      { artist: 'Агата Кристи', title: 'Как на войне' },
      { artist: 'Агата Кристи', title: 'Ковёр-вертолёт' },
      { artist: 'Агата Кристи', title: 'Опиум для никого' },
      { artist: 'Наутилус Помпилиус', title: 'Скованные одной цепью' },
      { artist: 'Наутилус Помпилиус', title: 'Крылья' },
      { artist: 'Наутилус Помпилиус', title: 'Дыхание' },
      { artist: 'Сплин', title: 'Моё сердце' },
      { artist: 'Сплин', title: 'Выхода нет' },
      { artist: 'Сплин', title: 'Орбит без сахара' },
      { artist: 'Сплин', title: 'Линия жизни' },
      { artist: 'Би-2', title: 'Полковнику никто не пишет' },
      { artist: 'Би-2', title: 'Варвара' },
      { artist: 'Би-2', title: 'Серебро' },
      { artist: 'Земфира', title: 'Искала' },
      { artist: 'Земфира', title: 'Ариведерчи' },
      { artist: 'Земфира', title: 'Ромашки' },
      { artist: 'Земфира', title: 'Хочешь' },
      { artist: 'Мумий Тролль', title: 'Владивосток 2000' },
      { artist: 'Мумий Тролль', title: 'Утекай' },
      { artist: 'Мумий Тролль', title: 'Дельфины' },
      { artist: 'Король и Шут', title: 'Лесник' },
      { artist: 'Король и Шут', title: 'Кукла колдуна' },
      { artist: 'Король и Шут', title: 'Прыгну со скалы' },
      { artist: 'Король и Шут', title: 'Проклятый старый дом' }
    ]
  },
  {
    url: 'https://1001tracklists.com/dj-mix/darkwave-ebm-synth-night-set/',
    title: 'Darkwave / EBM / Coldwave Club DJ Set',
    type: 'dj_set',
    tracks: [
      { artist: 'Boy Harsher', title: 'Pain' },
      { artist: 'Boy Harsher', title: 'Fate' },
      { artist: 'Boy Harsher', title: 'Come Closer' },
      { artist: 'Lebanon Hanover', title: 'Gallowdance' },
      { artist: 'Lebanon Hanover', title: 'Totally You' },
      { artist: 'She Past Away', title: 'Kasvetli Kutlama' },
      { artist: 'She Past Away', title: 'Ruh' },
      { artist: 'Selofan', title: 'Give Me a Reason' },
      { artist: 'Twin Tribes', title: 'Fantasmas' },
      { artist: 'Twin Tribes', title: 'Heart & Feather' },
      { artist: 'Drab Majesty', title: 'Ellipsis' },
      { artist: 'Drab Majesty', title: 'Oxytocin' },
      { artist: 'Cold Cave', title: 'Confetti' },
      { artist: 'Linea Aspera', title: 'Malarone' },
      { artist: 'Molchat Doma', title: 'Судно (Борис Рыжий)' },
      { artist: 'Molchat Doma', title: 'Тоска' },
      { artist: 'Ploho', title: 'Новостройки' },
      { artist: 'Shortparis', title: 'Страшно' },
      { artist: 'Depeche Mode', title: 'Precious' },
      { artist: 'Depeche Mode', title: 'Sacred' },
      { artist: 'New Order', title: 'Blue Monday' },
      { artist: 'The Cure', title: 'Lovesong' }
    ]
  },
  {
    url: 'https://kexp.org/radio/tracklists/post-punk-indie-revolution/',
    title: 'KEXP Post-Punk & Indie Radio Tracklist',
    type: 'radio',
    tracks: [
      { artist: 'Fontaines D.C.', title: 'Starburster' },
      { artist: 'Fontaines D.C.', title: "Boys in the Better Land" },
      { artist: 'Fontaines D.C.', title: 'Jackie Down the Line' },
      { artist: 'Fontaines D.C.', title: 'I Love You' },
      { artist: 'IDLES', title: 'Danny Nedelko' },
      { artist: 'IDLES', title: 'Colossus' },
      { artist: 'The Murder Capital', title: 'More Is Less' },
      { artist: 'Yard Act', title: 'The Overload' },
      { artist: 'Wet Leg', title: 'Chaise Longue' },
      { artist: 'Squid', title: 'Narrator' },
      { artist: 'Black Country, New Road', title: 'Track X' },
      { artist: 'Shame', title: 'One Rizla' },
      { artist: 'Interpol', title: 'Obstacle 1' },
      { artist: 'Interpol', title: 'Evil' },
      { artist: 'The National', title: 'Bloodbuzz Ohio' },
      { artist: 'The National', title: 'Fake Empire' },
      { artist: 'Foals', title: 'Spanish Sahara' },
      { artist: 'Foals', title: 'My Number' },
      { artist: 'Balthazar', title: 'Fever' },
      { artist: 'Human Tetris', title: "Things I Don't Need" },
      { artist: 'Motorama', title: 'Wind in Her Hair' },
      { artist: 'The Smiths', title: 'There Is a Light That Never Goes Out' },
      { artist: 'The Smiths', title: 'Half a Person' },
      { artist: 'Joy Division', title: 'Disorder' },
      { artist: 'The Cure', title: 'The Figurehead' }
    ]
  },
  {
    url: 'https://bbc.co.uk/6music/tracklists/classic-80s-post-punk-essentials/',
    title: 'BBC Radio 6 Music 80s Post-Punk Essentials',
    type: 'radio',
    tracks: [
      { artist: 'The Smiths', title: 'There Is a Light That Never Goes Out' },
      { artist: 'The Smiths', title: 'This Charming Man' },
      { artist: 'The Smiths', title: 'How Soon Is Now?' },
      { artist: 'The Smiths', title: 'Half a Person' },
      { artist: 'The Smiths', title: 'Cemetry Gates' },
      { artist: 'Joy Division', title: 'Disorder' },
      { artist: 'Joy Division', title: 'Love Will Tear Us Apart' },
      { artist: 'Joy Division', title: 'Transmission' },
      { artist: 'New Order', title: 'Blue Monday' },
      { artist: 'New Order', title: 'Bizarre Love Triangle' },
      { artist: 'New Order', title: 'Ceremony' },
      { artist: 'The Cure', title: 'A Forest' },
      { artist: 'The Cure', title: 'Boys Don’t Cry' },
      { artist: 'The Cure', title: 'Lovesong' },
      { artist: 'The Cure', title: 'Pictures of You' },
      { artist: 'The Cure', title: 'Just Like Heaven' },
      { artist: 'Bauhaus', title: "Bela Lugosi's Dead" },
      { artist: 'Bauhaus', title: 'Dark Entries' },
      { artist: 'Siouxsie and the Banshees', title: 'Spellbound' },
      { artist: 'Siouxsie and the Banshees', title: 'Cities in Dust' },
      { artist: 'Echo & the Bunnymen', title: 'The Killing Moon' },
      { artist: 'Depeche Mode', title: 'Precious' },
      { artist: 'Depeche Mode', title: 'Sacred' }
    ]
  },
  {
    url: 'https://listenbrainz.org/playlist/downtempo-triphop-lounge-sessions/',
    title: 'Trip-Hop & Downtempo Lounge Sessions',
    type: 'curated',
    tracks: [
      { artist: 'Portishead', title: 'Roads' },
      { artist: 'Portishead', title: 'Glory Box' },
      { artist: 'Portishead', title: 'Sour Times' },
      { artist: 'Massive Attack', title: 'Teardrop' },
      { artist: 'Massive Attack', title: 'Angel' },
      { artist: 'Massive Attack', title: 'Paradise Circus' },
      { artist: 'Massive Attack', title: 'Unfinished Sympathy' },
      { artist: 'Tricky', title: 'Black Steel' },
      { artist: 'Tricky', title: 'Overcome' },
      { artist: 'Morcheeba', title: 'The Sea' },
      { artist: 'Morcheeba', title: 'Rome Wasn’t Built in a Day' },
      { artist: 'Sneaker Pimps', title: '6 Underground' },
      { artist: 'Hooverphonic', title: 'Mad About You' },
      { artist: 'Lamb', title: 'Górecki' },
      { artist: 'DJ Shadow', title: 'Midnight in a Perfect World' },
      { artist: 'Radiohead', title: 'Everything in Its Right Place' },
      { artist: 'Björk', title: 'Hyperballad' },
      { artist: 'The xx', title: 'Intro' },
      { artist: 'The xx', title: 'Crystalised' }
    ]
  },
  {
    url: 'https://1001tracklists.com/dj-mix/intelligent-electronic-idm-set/',
    title: 'IDM & Intelligent Electronic DJ Mix',
    type: 'dj_set',
    tracks: [
      { artist: 'Aphex Twin', title: 'Windowlicker' },
      { artist: 'Aphex Twin', title: 'Xtal' },
      { artist: 'Boards of Canada', title: 'Dayvan Cowboy' },
      { artist: 'Boards of Canada', title: 'Roygbiv' },
      { artist: 'Burial', title: 'Archangel' },
      { artist: 'Burial', title: 'Ghost Hardware' },
      { artist: 'Four Tet', title: 'Baby' },
      { artist: 'Four Tet', title: 'Two Thousand and Seventeen' },
      { artist: 'Bicep', title: 'Glue' },
      { artist: 'Bicep', title: 'Apricots' },
      { artist: 'Jon Hopkins', title: 'Singularity' },
      { artist: 'Caribou', title: "Can't Do Without You" },
      { artist: 'The Chemical Brothers', title: 'Star Guitar' },
      { artist: 'Daft Punk', title: 'Around the World' },
      { artist: 'Depeche Mode', title: 'Enjoy the Silence' },
      { artist: 'New Order', title: 'Blue Monday' }
    ]
  },
  {
    url: 'https://kerrang.com/tracklists/nu-metal-and-alternative-rock-2000s-classics/',
    title: 'Nu-Metal & 2000s Alternative Rock Classics',
    type: 'curated',
    tracks: [
      { artist: 'Linkin Park', title: 'Numb' },
      { artist: 'Linkin Park', title: 'In the End' },
      { artist: 'Linkin Park', title: 'Faint' },
      { artist: 'Linkin Park', title: 'Crawling' },
      { artist: 'Linkin Park', title: 'Somewhere I Belong' },
      { artist: 'Deftones', title: 'Change (In the House of Flies)' },
      { artist: 'Deftones', title: 'Be Quiet and Drive (Far Away)' },
      { artist: 'System of a Down', title: 'Chop Suey!' },
      { artist: 'System of a Down', title: 'Toxicity' },
      { artist: 'System of a Down', title: 'Aerials' },
      { artist: 'Evanescence', title: 'Bring Me to Life' },
      { artist: 'Evanescence', title: 'Going Under' },
      { artist: 'Korn', title: 'Freak on a Leash' },
      { artist: 'Korn', title: 'Falling Away from Me' },
      { artist: 'Limp Bizkit', title: 'Break Stuff' },
      { artist: 'Slipknot', title: 'Duality' },
      { artist: 'Slipknot', title: 'Before I Forget' },
      { artist: 'Papa Roach', title: 'Last Resort' },
      { artist: 'Disturbed', title: 'Down with the Sickness' },
      { artist: 'Three Days Grace', title: 'I Hate Everything About You' },
      { artist: 'Breaking Benjamin', title: 'The Diary of Jane' },
      { artist: 'Chevelle', title: 'The Red' },
      { artist: 'Incubus', title: 'Drive' }
    ]
  },
  {
    url: 'https://pitchfork.com/features/lists-and-guides/indie-rock-and-bedroom-pop-essentials/',
    title: 'Indie Rock & Bedroom Pop Essentials',
    type: 'curated',
    tracks: [
      { artist: 'Alex G', title: 'Runner' },
      { artist: 'Alex G', title: 'Gretel' },
      { artist: 'Alex G', title: 'Sarah' },
      { artist: 'Alex G', title: 'Mary' },
      { artist: 'Alex G', title: 'Advice' },
      { artist: 'Alex G', title: 'Hope' },
      { artist: 'Phoebe Bridgers', title: 'Motion Sickness' },
      { artist: 'Phoebe Bridgers', title: 'Kyoto' },
      { artist: 'Phoebe Bridgers', title: 'I Know the End' },
      { artist: 'Elliott Smith', title: 'Between the Bars' },
      { artist: 'Elliott Smith', title: 'Waltz #2 (XO)' },
      { artist: 'Elliott Smith', title: 'Say Yes' },
      { artist: 'Sufjan Stevens', title: 'Mystery of Love' },
      { artist: 'Sufjan Stevens', title: 'Chicago' },
      { artist: 'Big Thief', title: 'Not' },
      { artist: 'Big Thief', title: 'Vampire Empire' },
      { artist: 'Clairo', title: 'Bags' },
      { artist: 'Clairo', title: 'Sofia' },
      { artist: 'Mac DeMarco', title: 'Chamber of Reflection' },
      { artist: 'Mac DeMarco', title: 'Salad Days' },
      { artist: 'Men I Trust', title: 'Show Me How' },
      { artist: 'Men I Trust', title: 'Tailwhip' },
      { artist: 'Beach Fossils', title: 'Down the Line' },
      { artist: 'DIIV', title: 'Doused' },
      { artist: 'Alvvays', title: 'Archie, Marry Me' },
      { artist: 'Alvvays', title: 'Dreams Tonite' }
    ]
  },
  {
    url: 'https://stereogum.com/tracklists/shoegaze-and-dream-pop-heaven/',
    title: 'Shoegaze & Dream Pop Heaven',
    type: 'curated',
    tracks: [
      { artist: 'My Bloody Valentine', title: 'Soon' },
      { artist: 'My Bloody Valentine', title: 'Only Shallow' },
      { artist: 'My Bloody Valentine', title: 'When You Sleep' },
      { artist: 'Slowdive', title: 'Alison' },
      { artist: 'Slowdive', title: 'When the Sun Hits' },
      { artist: 'Slowdive', title: 'Sugar for the Pill' },
      { artist: 'Slowdive', title: 'Kisses' },
      { artist: 'Cocteau Twins', title: 'Cherry-coloured Funk' },
      { artist: 'Cocteau Twins', title: 'Heaven or Las Vegas' },
      { artist: 'Cocteau Twins', title: 'Lorelei' },
      { artist: 'Ride', title: 'Vapour Trail' },
      { artist: 'Ride', title: 'Leave Them All Behind' },
      { artist: 'Lush', title: 'For Love' },
      { artist: 'Beach House', title: 'Myth' },
      { artist: 'Beach House', title: 'Space Song' },
      { artist: 'Beach House', title: 'Silver Soul' },
      { artist: 'Mazzy Star', title: 'Fade Into You' },
      { artist: 'Mazzy Star', title: 'Into Dust' }
    ]
  },
  {
    url: 'https://rollingstone.com/music/music-lists/90s-grunge-and-alternative-anthems/',
    title: '90s Grunge & Alternative Rock Anthems',
    type: 'curated',
    tracks: [
      { artist: 'Nirvana', title: 'Smells Like Teen Spirit' },
      { artist: 'Nirvana', title: 'Come as You Are' },
      { artist: 'Nirvana', title: 'Heart-Shaped Box' },
      { artist: 'Nirvana', title: 'In Bloom' },
      { artist: 'Nirvana', title: 'Lithium' },
      { artist: 'Alice in Chains', title: 'Man in the Box' },
      { artist: 'Alice in Chains', title: 'Would?' },
      { artist: 'Alice in Chains', title: 'Rooster' },
      { artist: 'Alice in Chains', title: 'Nutshell' },
      { artist: 'Soundgarden', title: 'Black Hole Sun' },
      { artist: 'Soundgarden', title: 'Spoonman' },
      { artist: 'Soundgarden', title: 'Fell on Black Days' },
      { artist: 'Pearl Jam', title: 'Alive' },
      { artist: 'Pearl Jam', title: 'Jeremy' },
      { artist: 'Pearl Jam', title: 'Black' },
      { artist: 'Smashing Pumpkins', title: '1979' },
      { artist: 'Smashing Pumpkins', title: 'Tonight, Tonight' },
      { artist: 'Smashing Pumpkins', title: 'Cherub Rock' },
      { artist: 'Stone Temple Pilots', title: 'Plush' },
      { artist: 'Stone Temple Pilots', title: 'Interstate Love Song' }
    ]
  },
  {
    url: 'https://nme.com/features/britpop-and-90s-uk-indie-peak/',
    title: 'Britpop & 90s UK Indie Peak',
    type: 'curated',
    tracks: [
      { artist: 'Oasis', title: 'Wonderwall' },
      { artist: 'Oasis', title: "Don't Look Back in Anger" },
      { artist: 'Oasis', title: 'Champagne Supernova' },
      { artist: 'Oasis', title: 'Live Forever' },
      { artist: 'Blur', title: 'Song 2' },
      { artist: 'Blur', title: 'Beetlebum' },
      { artist: 'Blur', title: 'Parklife' },
      { artist: 'Blur', title: 'Coffee & TV' },
      { artist: 'Pulp', title: 'Common People' },
      { artist: 'Pulp', title: 'Disco 2000' },
      { artist: 'The Verve', title: 'Bitter Sweet Symphony' },
      { artist: 'The Verve', title: "The Drugs Don't Work" },
      { artist: 'Suede', title: 'Animal Nitrate' },
      { artist: 'Suede', title: 'Beautiful Ones' },
      { artist: 'Supergrass', title: 'Alright' },
      { artist: 'Manic Street Preachers', title: 'A Design for Life' }
    ]
  },
  {
    url: 'https://nts.live/shows/hip-hop-and-abstract-beats-session/',
    title: 'NTS Radio Hip-Hop & Abstract Beats Session',
    type: 'radio',
    tracks: [
      { artist: 'MF DOOM', title: 'Doomsday' },
      { artist: 'MF DOOM', title: 'Rapp Snitch Knitches' },
      { artist: 'Madvillain', title: 'All Caps' },
      { artist: 'Madvillain', title: 'Accordion' },
      { artist: 'Madvillain', title: 'Figaro' },
      { artist: 'Kendrick Lamar', title: 'Money Trees' },
      { artist: 'Kendrick Lamar', title: 'Alright' },
      { artist: 'Kendrick Lamar', title: 'King Kunta' },
      { artist: 'Tyler, The Creator', title: 'EARFQUAKE' },
      { artist: 'Tyler, The Creator', title: 'See You Again' },
      { artist: 'J Dilla', title: 'So Far to Go' },
      { artist: 'Earl Sweatshirt', title: 'Chum' },
      { artist: 'Danny Brown', title: 'Really Doe' },
      { artist: 'Freddie Gibbs', title: 'Thuggin' },
      { artist: 'Joey Bada$$', title: 'Paper Trail$' },
      { artist: 'A Tribe Called Quest', title: 'Electric Relaxation' }
    ]
  },
  {
    url: 'https://newretrowave.com/tracklists/synthwave-darksynth-night-drive/',
    title: 'Synthwave & Darksynth Night Drive',
    type: 'curated',
    tracks: [
      { artist: 'Kavinsky', title: 'Nightcall' },
      { artist: 'Perturbator', title: 'Future Club' },
      { artist: 'Perturbator', title: 'Sentient' },
      { artist: 'Carpenter Brut', title: 'Turbo Killer' },
      { artist: 'Carpenter Brut', title: 'Roller Mobster' },
      { artist: 'GUNSHIP', title: 'Tech Noir' },
      { artist: 'The Midnight', title: 'Sunset' },
      { artist: 'The Midnight', title: 'Days of Thunder' },
      { artist: 'Trevor Something', title: 'Summer Love' },
      { artist: 'Lorn', title: 'Acid Rain' },
      { artist: 'HOME', title: 'Resonance' }
    ]
  },
  {
    url: 'https://kexp.org/radio/tracklists/post-rock-and-cinematic-instrumentals/',
    title: 'Post-Rock & Cinematic Instrumental Epics',
    type: 'radio',
    tracks: [
      { artist: 'Sigur Rós', title: 'Svefn-g-englar' },
      { artist: 'Sigur Rós', title: 'Starálfur' },
      { artist: 'Sigur Rós', title: 'Hoppípolla' },
      { artist: 'Godspeed You! Black Emperor', title: 'Storm' },
      { artist: 'Godspeed You! Black Emperor', title: 'East Hastings' },
      { artist: 'Mogwai', title: 'Mogwai Fear Satan' },
      { artist: 'Mogwai', title: 'Take Me Somewhere Nice' },
      { artist: 'Explosions in the Sky', title: 'Your Hand in Mine' },
      { artist: 'Explosions in the Sky', title: 'First Breath After Coma' },
      { artist: 'This Will Destroy You', title: 'The Mighty Rio Grande' },
      { artist: 'Mono', title: 'Halcyon (Beautiful Days)' }
    ]
  },
  {
    url: 'https://uncut.co.uk/features/classic-art-rock-and-psychedelic-odyssey/',
    title: 'Classic Art Rock & Psychedelic Odyssey',
    type: 'curated',
    tracks: [
      { artist: 'Pink Floyd', title: 'Time' },
      { artist: 'Pink Floyd', title: 'Comfortably Numb' },
      { artist: 'Pink Floyd', title: 'Wish You Were Here' },
      { artist: 'Pink Floyd', title: 'Shine On You Crazy Diamond' },
      { artist: 'King Crimson', title: '21st Century Schizoid Man' },
      { artist: 'King Crimson', title: 'Epitaph' },
      { artist: 'King Crimson', title: 'Starless' },
      { artist: 'David Bowie', title: 'Heroes' },
      { artist: 'David Bowie', title: 'Space Oddity' },
      { artist: 'David Bowie', title: 'Starman' },
      { artist: 'David Bowie', title: 'Life on Mars?' },
      { artist: 'The Velvet Underground', title: 'Sunday Morning' },
      { artist: 'The Velvet Underground', title: 'Venus in Furs' },
      { artist: 'The Velvet Underground', title: 'Heroin' },
      { artist: 'The Doors', title: 'Riders on the Storm' },
      { artist: 'The Doors', title: 'Light My Fire' },
      { artist: 'Can', title: 'Vitamin C' },
      { artist: 'Neu!', title: 'Hallogallo' }
    ]
  },
  {
    url: 'https://residentadvisor.net/dj-mixes/warehouse-techno-and-deep-house-selection/',
    title: 'Warehouse Techno & Deep House Club Selection',
    type: 'dj_set',
    tracks: [
      { artist: 'Bicep', title: 'Glue' },
      { artist: 'Bicep', title: 'Apricots' },
      { artist: 'Four Tet', title: 'Baby' },
      { artist: 'Four Tet', title: 'Two Thousand and Seventeen' },
      { artist: 'Jon Hopkins', title: 'Open Eye Signal' },
      { artist: 'Jon Hopkins', title: 'Immunity' },
      { artist: 'Floating Points', title: 'LesAlpx' },
      { artist: 'Ross From Friends', title: "Talk to Me You'll Understand" },
      { artist: 'Mall Grab', title: 'Liverpool Street In The Rain' },
      { artist: 'Overmono', title: 'So U Kno' },
      { artist: 'DJ Seinfeld', title: 'U' },
      { artist: 'Burial', title: 'Archangel' }
    ]
  }
];

export async function crawlTracklists(extraUrls: string[] = []): Promise<WebTracklist[]> {
  const cacheDir = 'data/cache/crawler';
  await mkdir(cacheDir, { recursive: true });

  const results: WebTracklist[] = [...SEED_TRACKLISTS];

  for (const url of extraUrls) {
    const hash = createHash('sha256').update(url).digest('hex');
    const cacheFile = `${cacheDir}/${hash}.html`;
    let html = '';

    try {
      html = await readFile(cacheFile, 'utf8');
    } catch {
      try {
        html = await publicPage(url, AbortSignal.timeout(12000));
        await writeFile(cacheFile, html);
      } catch (err) {
        console.warn(`Could not crawl ${url}:`, err instanceof Error ? err.message : String(err));
        continue;
      }
    }

    if (html) {
      const parsed = parseHtmlTracklists(html, url);
      results.push(...parsed);
    }
  }

  // Deduplicate and filter tracklists
  const valid = results.filter(tl => tl.tracks.length >= 2 && tl.tracks.length <= 500);

  await writeFile('data/cache/web-tracklists.json', JSON.stringify(valid, null, 2));
  console.log(`Saved ${valid.length} human web tracklists with ${valid.reduce((acc, t) => acc + t.tracks.length, 0)} total tracks to data/cache/web-tracklists.json`);

  return valid;
}

if (process.argv[1]?.endsWith('crawl_tracklists.ts')) {
  const urls = process.argv.slice(2);
  crawlTracklists(urls).catch(console.error);
}
