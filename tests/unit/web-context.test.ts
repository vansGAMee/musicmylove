import { expect, it } from 'vitest';
import { extractTracklist, ContextResolver } from '../../services/web-context';

it('requires seed and neighbor in an actual tracklist, rejects prose co-mentions', () => {
 const tracks=[{mbid:'neighbor',artist:'Massive Attack',title:'Teardrop'}];
 const html='<h1>Evening playlist</h1><ol><li>Portishead — Roads</li><li>Massive Attack — Teardrop</li></ol>';
 expect(extractTracklist(html,tracks,[{artist:'Portishead',title:'Roads'}])).toEqual({'portishead\u001froads':['neighbor']});
 expect(extractTracklist('<p>Portishead Roads and Massive Attack Teardrop</p>',tracks,[{artist:'Portishead',title:'Roads'}])).toEqual({});
});

it('batches distinct misses and reuses evidence cache', async () => {
 let searches=0,pages=0;
 const resolver=new ContextResolver([{mbid:'n',artist:'Neighbor',title:'Song'}],{
   search:async queries=>{searches++;expect(queries.length).toBe(1);return [['https://example.org/playlist']];},
   page:async()=>{pages++;return '<ol><li>New — Song</li><li>Neighbor — Song</li></ol>';},
 });
 const songs=[{artist:'New',title:'Song'},{artist:'New',title:'Song'}];
 expect((await resolver.resolve(songs)).context['new\u001fsong'].neighbors).toEqual(['n']);
 await resolver.resolve(songs);
 expect(searches).toBe(1);expect(pages).toBe(1);
});
