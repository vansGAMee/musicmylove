/** Input: verified tracklists [{url, tracks:[{artist,title}]}], extracted offline from real pages. */
import { readFile, writeFile } from 'node:fs/promises';
import { key, type Song, type Context } from '../../src/lib/offline/engine';
const catalog=JSON.parse(await readFile('public/data/catalog.json','utf8'));
const known=new Map<string,string>(catalog.tracks.map((t:Song&{mbid:string})=>[key(t),t.mbid]));
const lists=JSON.parse(await readFile(process.argv[2],'utf8')) as {url:string;tracks:Song[]}[];
const context:Record<string,Context>={};
for(const list of lists){
  const url=new URL(list.url);if(!['http:','https:'].includes(url.protocol))throw new Error('Real source URL required');
  if(list.tracks.length<2||list.tracks.length>200)continue;
  for(const song of list.tracks){
    const neighbors=list.tracks.filter(t=>key(t)!==key(song)).flatMap(t=>known.has(key(t))?[known.get(key(t))!]:[]);
    if(!neighbors.length)continue;
    const previous=context[key(song)]??{neighbors:[],sources:[]};
    context[key(song)]={neighbors:[...new Set([...previous.neighbors,...neighbors])].sort().slice(0,200),sources:[...new Set([...previous.sources,url.href])].sort()};
  }
}
await writeFile(process.argv[3]??'data/cache/web-context.json',JSON.stringify(context));
console.log({contextTracks:Object.keys(context).length});
