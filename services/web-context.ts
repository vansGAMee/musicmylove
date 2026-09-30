import { parse, type DefaultTreeAdapterMap } from 'parse5';
import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';
import { key, type Song, type Context } from '../src/lib/offline/engine';
import type { Track } from '../src/lib/types';
type Node=DefaultTreeAdapterMap['node'];
type Element=DefaultTreeAdapterMap['element'];
const norm=(s:string)=>s.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const children=(n:Node):Node[]=>'childNodes' in n?n.childNodes:[];
const tag=(n:Node)=>'tagName' in n?n.tagName:'';
const text=(n:Node):string=>'value' in n?n.value:children(n).map(text).join(' ');
function elements(n:Node,name:string):Node[]{return children(n).flatMap(c=>tag(c)===name?[c]:elements(c,name));}
/** Inverted title anchors: page matching never scans the track catalog. */
class TrackIndex {
 private postings=new Map<string,Track[]>();
 constructor(tracks:Track[]){
  const counts=new Map<string,number>();
  for(const t of tracks)for(const word of new Set(norm(t.title).split(' ')))counts.set(word,(counts.get(word)??0)+1);
  for(const t of tracks){const words=norm(t.title).split(' ').filter(Boolean).sort((a,b)=>(counts.get(a)??0)-(counts.get(b)??0)||(a<b?-1:1));if(!words.length)continue;const list=this.postings.get(words[0])??[];list.push(t);this.postings.set(words[0],list);}
 }
 match(row:string):Track[]{const value=' '+norm(row)+' ',found=new Map<string,Track>();for(const token of new Set(value.trim().split(' ')))for(const t of this.postings.get(token)??[])if(value.includes(' '+norm(t.artist)+' ')&&value.includes(' '+norm(t.title)+' '))found.set(t.mbid,t);return [...found.values()];}
}
function tracklists(html:string):string[][] {
 const root=parse(html),groups:string[][]=[];
 function walk(n:Node){
  const name=tag(n);if(['nav','footer','header','aside'].includes(name))return;
  if(name==='script'){
   const el=n as Element;
   if(el.attrs.some(a=>a.name==='type'&&a.value==='application/ld+json'))try{
    const data=JSON.parse(text(n));
    const visit=(obj:unknown)=>{
     if(!obj||typeof obj!=='object')return;
     const record=obj as Record<string,unknown>;
     if(['MusicPlaylist','MusicAlbum','ItemList'].includes(String(record['@type']))){
      const items=record.track??record.tracks??record.itemListElement;
      if(Array.isArray(items)){
       const rows=items.flatMap((raw:Record<string,unknown>)=>{const t=(raw.item??raw) as Record<string,unknown>;const a=(t.byArtist??record.byArtist) as {name?:string}|{name?:string}[]|undefined;const artist=Array.isArray(a)?a.map(x=>x.name??'').join(' '):a?.name;return typeof t.name==='string'&&artist?[artist+' '+t.name]:[];});
       if(rows.length>=2&&rows.length<=500)groups.push(rows);
      }
     }
     for(const value of Object.values(record))if(Array.isArray(value))value.forEach(visit);else if(value&&typeof value==='object')visit(value);
    };visit(data);
   }catch{/* Invalid structured data is not evidence. */}
   return;
  }
  if(['ol','ul','table'].includes(name)){
   const rows=elements(n,name==='table'?'tr':'li').map(text).map(s=>s.trim()).filter(s=>s.length>3&&s.length<600);
   if(rows.length>=2&&rows.length<=500)groups.push(rows);
  }
  for(const c of children(n))walk(c);
 }
 walk(root);return groups;
}
function extract(html:string,index:TrackIndex,seeds:Song[]):Record<string,string[]> {
 const seedMap = new Map<string, Song>(seeds.map(s => [key(s), s]));
 const seedIndex=new TrackIndex(seeds.map(s=>({...s,mbid:key(s)}))),result:Record<string,string[]>={};
 for(const rows of tracklists(html)){
  const matchedCatalog: Track[] = [], found = new Set<string>();
  for(const row of rows){
    index.match(row).forEach(t=>matchedCatalog.push(t));
    seedIndex.match(row).forEach(t=>found.add(t.mbid));
  }
  if(!matchedCatalog.length)continue;
  for(const k of found){
    const seed = seedMap.get(k);
    // Exclude same-artist connections so albums and discographies cannot create artificial strong links
    const valid = matchedCatalog.filter(t => !seed || norm(t.artist) !== norm(seed.artist)).map(t => t.mbid);
    if (!valid.length) continue;
    result[k]=[...new Set([...(result[k]??[]),...valid])].sort().slice(0,200);
  }
 }
 return result;
}
export const extractTracklist=(html:string,tracks:Track[],seeds:Song[])=>extract(html,new TrackIndex(tracks),seeds);

export interface WebProviders {
 search:(queries:string[],signal:AbortSignal)=>Promise<string[][]>;
 page:(url:string,signal:AbortSignal)=>Promise<string>;
}
export interface ContextCache {get:(key:string)=>Promise<Context|undefined>;set:(key:string,value:Context)=>Promise<void>}
export async function concurrent<T,R>(items:T[],limit:number,fn:(item:T,index:number)=>Promise<R>):Promise<R[]>{
 const result:R[]=new Array(items.length);let next=0;
 await Promise.all(Array.from({length:Math.min(items.length,limit)},async()=>{while(next<items.length){const i=next++;result[i]=await fn(items[i],i);}}));return result;
}
export class ContextResolver {
 private index:TrackIndex;private memory=new Map<string,Context>();
 constructor(tracks:Track[],private providers:WebProviders,private cache?:ContextCache){this.index=new TrackIndex(tracks);}
 async resolve(input:Song[]){
  const songs=[...new Map(input.map(s=>[key(s),s])).values()].sort((a,b)=>key(a)<key(b)?-1:1);
  const context:Record<string,Context>={},unresolved:Song[]=[];
  await concurrent(songs,32,async s=>{const k=key(s),old=this.memory.get(k)??await this.cache?.get(k);if(old)context[k]=old;else unresolved.push(s);});
  unresolved.sort((a,b)=>key(a)<key(b)?-1:1);
  if(!unresolved.length)return {context,unresolved:[],failed:false};
  const signal=AbortSignal.timeout(16000);
  // Group multiple artist/title pairs into every search; provider receives batches.
  const groups:Song[][]=[];for(let i=0;i<unresolved.length;i+=4)groups.push(unresolved.slice(i,i+4));
  const quotes=(s:string)=>'"'+s.replace(/["\\]/g,' ').slice(0,160)+'"';
  const queries=groups.map(g=>'('+g.map(s=>'('+quotes(s.artist)+' '+quotes(s.title)+')').join(' OR ')+') (playlist OR tracklist OR tracklisting OR плейлист)');
  const searches=await this.providers.search(queries,signal);
  if(searches.length!==groups.length)throw new Error('Incomplete search batch');
  const pages=new Map<string,Song[]>();
  searches.forEach((urls,i)=>urls.slice(0,5).forEach(url=>pages.set(url,[...(pages.get(url)??[]),...groups[i]])));
  let failed=false;
  await concurrent([...pages],12,async([url,seeds])=>{
   try{
    const html=await this.providers.page(url,signal),found=extract(html,this.index,seeds);
    for(const [k,neighbors] of Object.entries(found)){
     const old=context[k]??{neighbors:[],sources:[]};
     context[k]={neighbors:[...new Set([...old.neighbors,...neighbors])].sort().slice(0,200),sources:[...new Set([...old.sources,url])].sort()};
    }
   }catch{failed=true;}
  });
  // Cache proven evidence only; failed/empty search must not poison future lookup.
  await concurrent(Object.entries(context),16,async([k,v])=>{if(v.neighbors.length){this.memory.set(k,v);await this.cache?.set(k,v);}});
  return {context,unresolved:unresolved.filter(s=>!context[key(s)]).map(key),failed};
 }
}

const ua='MusicMyLove/0.1 (+https://github.com/vansGAMee/musicmylove; tracklist discovery)';
function publicAddress(ip:string){
 if(ip.includes(':'))return !/^(::|fc|fd|fe[89ab]|ff|2001:db8)/i.test(ip);
 const a=ip.split('.').map(Number);return a.length===4&&!([0,10,127].includes(a[0])||a[0]>=224||(a[0]===169&&a[1]===254)||(a[0]===172&&a[1]>=16&&a[1]<=31)||(a[0]===192&&a[1]===168)||(a[0]===100&&a[1]>=64&&a[1]<=127));
}
export async function publicPage(input:string,signal:AbortSignal):Promise<string>{
 let url=new URL(input);
 for(let redirects=0;redirects<4;redirects++){
  if(!['https:','http:'].includes(url.protocol)||url.username||url.password||url.port||url.hostname==='localhost')throw new Error('Non-public page');
  const host=url.hostname.replace(/^\[|\]$/g,'');
  const addresses=isIP(host)?[{address:host}]:await lookup(host,{all:true});
  if(!addresses.length||addresses.some(a=>!publicAddress(a.address)))throw new Error('Non-public address');
  const r=await fetch(url,{redirect:'manual',headers:{'User-Agent':ua,Accept:'text/html'},signal:AbortSignal.any([signal,AbortSignal.timeout(5000)])});
  if(r.status>=300&&r.status<400){const location=r.headers.get('location');await r.body?.cancel();if(!location)throw new Error('Missing redirect');url=new URL(location,url);continue;}
  if(!r.ok||!r.headers.get('content-type')?.includes('text/html'))throw new Error('Page unavailable');
  if(Number(r.headers.get('content-length'))>2_000_000){await r.body?.cancel();throw new Error('Page too large');}
  const reader=r.body!.getReader();let length=0;const parts:Uint8Array[]=[];
  try{while(true){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>2_000_000)throw new Error('Page too large');parts.push(value);}}finally{await reader.cancel();}
  return Buffer.concat(parts).toString('utf8');
 }
 throw new Error('Too many redirects');
}
export async function searchWeb(queries:string[],signal:AbortSignal):Promise<string[][]>{
  const token=process.env.SERPER_API_KEY;
  if(token){
    try {
      const batches:string[][]=[];for(let i=0;i<queries.length;i+=50)batches.push(queries.slice(i,i+50));
      const results=await concurrent(batches,6,async batch=>{
        const r=await fetch('https://google.serper.dev/search',{method:'POST',headers:{'Content-Type':'application/json','X-API-KEY':token},body:JSON.stringify(batch.map(q=>({q,num:10}))),signal});
        if(!r.ok)return batch.map(()=>[]);
        const rows=await r.json();if(!Array.isArray(rows)||rows.length!==batch.length)return batch.map(()=>[]);
        return rows.map((row:{organic?:{link:string}[]})=>(row.organic??[]).map(x=>x.link).filter(u=>/^https?:\/\//.test(u)));
      });
      return results.flat();
    } catch {
      // Fall through to public HTML search on Serper error
    }
  }
  // Public HTML search works without a key at low volume; challenge pages return empty instead of crashing
  return concurrent(queries,4,async q=>{
    try {
      const r=await fetch('https://html.duckduckgo.com/html/?q='+encodeURIComponent(q),{headers:{'User-Agent':ua},signal});
      if(!r.ok)return [];
      const html=await r.text();if(/anomaly|challenge-form/.test(html))return [];
      const doc=parse(html),urls:string[]=[];
      const visit=(n:Node)=>{if(tag(n)==='a'){
        const a=n as Element,cls=a.attrs.find(x=>x.name==='class')?.value??'',href=a.attrs.find(x=>x.name==='href')?.value;
        if(cls.split(' ').includes('result__a')&&href){const u=new URL(href,'https://duckduckgo.com');const target=u.searchParams.get('uddg')??u.href;if(/^https?:\/\//.test(target))urls.push(target);}
      }children(n).forEach(visit);};visit(doc);return urls;
    } catch {
      return [];
    }
  });
}
