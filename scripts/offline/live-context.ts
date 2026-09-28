import { readFile } from 'node:fs/promises';
import { ContextResolver, publicPage, searchWeb } from '../../services/web-context';
import { recommend, key, type Catalog } from '../../src/lib/offline/engine';
const c=JSON.parse(await readFile('public/data/catalog.json','utf8')) as Catalog;
const songs=[{artist:'Portishead',title:'Roads'},{artist:'Massive Attack',title:'Teardrop'}];
// Force a genuine cold path at the identity boundary, retaining real neighboring vectors.
const keys=new Set(songs.map(key));
const cold={...c,tracks:c.tracks.map(t=>keys.has(key(t))?{...t,title:'hidden-for-cold-path-test'}:t)};
const resolver=new ContextResolver(cold.tracks,{search:searchWeb,page:publicPage});
const response=await resolver.resolve(songs);
const result=recommend(cold,songs,{},response.context);
console.log(JSON.stringify({evidence:response,coverage:result.coverage,recommendations:result.recommendations.length},null,2));
if(!result.coverage.graph||!result.recommendations.length)process.exitCode=1;
