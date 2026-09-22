import { readFile } from 'node:fs/promises';
import { recommend, type Catalog } from '../../src/lib/offline/engine';
const c=JSON.parse(await readFile('public/data/catalog.json','utf8')) as Catalog;
const songs=c.tracks.filter(t=>['Radiohead','Portishead','Massive Attack','Björk','M83'].includes(t.artist)).slice(0,20);
const start=performance.now();const result=recommend(c,songs,{});
if(result.recommendations.length!==40)throw new Error(`Expected 40, got ${result.recommendations.length}`);
if(JSON.stringify(result)!==JSON.stringify(recommend(c,[...songs].reverse(),{})))throw new Error('Permutation mismatch');
console.log(JSON.stringify({tracks:c.tracks.length,ms:Math.round(performance.now()-start),coverage:result.coverage,candidates:result.candidateCount,top:result.recommendations.slice(0,5).map(t=>t.artist+' - '+t.title)},null,2));
