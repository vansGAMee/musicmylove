/** Optional batch gateway. Deploy separately from the static Vercel site. */
import { createServer, type IncomingMessage } from 'node:http';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { fetchYandexPlaylist, normalizeYandexPlaylistUrl } from '../src/lib/yandex/playlist';
import { key, type Context, type Song } from '../src/lib/offline/engine';
import { ContextResolver, publicPage, searchWeb } from './web-context';
import { importYandexBatch } from './yandex-import';
const root='data/cache/gateway';
await mkdir(root,{recursive:true});
const origin=process.env.FRONTEND_ORIGIN??'http://localhost:3000';
const upstream=process.env.CONTEXT_BATCH_URL;
const catalog=JSON.parse(await readFile(process.env.TRACK_CATALOG??'ml/tastelift-catalog.json','utf8'));
const webResolver=new ContextResolver(catalog.tracks,{search:searchWeb,page:publicPage},{
 get:async k=>{try{const row=JSON.parse(await readFile(root+'/web-'+createHash('sha256').update(k).digest('hex')+'.json','utf8'));if(Date.now()-row.at<7*86400000)return row.value;}catch{}},
 set:async(k,value)=>{const file=root+'/web-'+createHash('sha256').update(k).digest('hex')+'.json';const temp=file+'.'+randomUUID();await writeFile(temp,JSON.stringify({at:Date.now(),value}));await rename(temp,file);}
});
const flights=new Map<string,Promise<unknown>>();
async function cached<T>(name:string,ttl:number,fn:()=>Promise<T>):Promise<T>{
  const file=root+'/'+createHash('sha256').update(name).digest('hex')+'.json';
  try{const item=JSON.parse(await readFile(file,'utf8'));if(Date.now()-item.at<ttl)return item.value;}catch{}
  const existing=flights.get(name);if(existing)return existing as Promise<T>;
  const task=(async()=>{const value=await fn();const temp=file+'.'+randomUUID();await writeFile(temp,JSON.stringify({at:Date.now(),value}));await rename(temp,file);return value;})();
  flights.set(name,task);try{return await task;}finally{flights.delete(name);}
}
async function body(req:IncomingMessage){let size=0;const parts:Buffer[]=[];for await(const part of req){size+=part.length;if(size>1024*1024)throw new Error('Request exceeds 1 MB');parts.push(part);}return JSON.parse(Buffer.concat(parts).toString());}
function allowed(url:string){const u=new URL(url);return u.protocol==='https:'&&!u.username&&!u.password&&!u.port&&['music.yandex.ru','music.yandex.com','music.yandex.kz','music.yandex.by','lk.music.yandex.ru','api.music.yandex.net'].includes(u.hostname);}
// Validate every redirect before connecting; never become an arbitrary URL proxy.
const safeFetch:typeof fetch=async(input,init)=>{
  let url=String(input);
  for(let n=0;n<5;n++){
    if(!allowed(url))throw new Error('Unsupported upstream host');
    const response=await fetch(url,{...init,redirect:'manual'});
    if(response.status>=300&&response.status<400){const next=response.headers.get('location');await response.body?.cancel();if(!next)throw new Error('Empty redirect');url=new URL(next,url).href;continue;}
    return response;
  }
  throw new Error('Too many redirects');
};
const clients=new Map<string,{at:number;n:number}>();
createServer(async(req,res)=>{
  const send=(status:number,value:unknown)=>{res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
  if(req.headers.origin&&req.headers.origin!==origin){send(403,{error:'Origin not allowed'});return;}
  res.setHeader('Access-Control-Allow-Origin',origin);res.setHeader('Vary','Origin');
  res.setHeader('Access-Control-Allow-Headers','Content-Type');res.setHeader('Access-Control-Allow-Methods','POST, OPTIONS');
  if(req.method==='OPTIONS'){res.writeHead(204);res.end();return;}
  if(req.url==='/health'){send(200,{ok:true,contextConfigured:true,searchProvider:process.env.SERPER_API_KEY?'serper-batch':'duckduckgo-html'});return;}
  if(req.method!=='POST'){send(405,{error:'POST required'});return;}
  const now=Date.now();for(const [ip,v] of clients)if(now-v.at>60000)clients.delete(ip);
  const ip=req.socket.remoteAddress??'unknown',limit=clients.get(ip)??{at:now,n:0};clients.set(ip,limit);
  if(++limit.n>20){res.setHeader('Retry-After','60');send(429,{error:'Too many requests'});return;}
  try{
    const input=await body(req);
    if(req.url==='/playlist'){
      if(typeof input.url!=='string')throw new Error('Playlist URL required');
      const raw=input.url.startsWith('https://')?input.url:'https://'+input.url;
      if(!allowed(raw))throw new Error('Unsupported playlist host');
      const normalized=normalizeYandexPlaylistUrl(raw);
      const playlist=await cached('playlist:'+normalized.canonicalUrl,3600000,()=>importYandexBatch(raw));
      send(200,{ok:true,playlist,tracks:playlist.tracks});return;
    }
    if(req.url==='/context'){
      if(!Array.isArray(input.tracks)||input.tracks.length>5000||input.tracks.some((s:Song)=>typeof s.artist!=='string'||typeof s.title!=='string'||s.artist.length>500||s.title.length>500))throw new Error('Invalid track batch');
      const songs=[...new Map<string,Song>(input.tracks.map((s:Song)=>[key(s),s])).values()].sort((a,b)=>key(a)<key(b)?-1:1);
      if(!upstream){send(200,await webResolver.resolve(songs));return;}
      // Provider must return actual tracklist evidence, not generated neighbors.
      const context:Record<string,Context>={};
      const missing:Song[]=[];
      for(const s of songs){try{const saved=JSON.parse(await readFile(root+'/context-'+createHash('sha256').update(key(s)).digest('hex')+'.json','utf8'));if(Date.now()-saved.at<7*86400000){context[key(s)]=saved.value;continue;}}catch{}missing.push(s);}
      if(missing.length){
        const response=await fetch(upstream,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.CONTEXT_TOKEN?{Authorization:'Bearer '+process.env.CONTEXT_TOKEN}:{})},body:JSON.stringify({tracks:missing}),signal:AbortSignal.timeout(15000)});
        if(!response.ok)throw new Error('Context provider unavailable');
        const data=await response.json();
        for(const s of missing){const value=data.context?.[key(s)] as Context|undefined;
          if(!value||!Array.isArray(value.neighbors)||!Array.isArray(value.sources)||value.neighbors.length>200||!value.neighbors.every(x=>typeof x==='string')||!value.sources.every(x=>typeof x==='string'&&/^https?:\/\//.test(x))||(value.neighbors.length&&!value.sources.length))continue;
          context[key(s)]=value;const file=root+'/context-'+createHash('sha256').update(key(s)).digest('hex')+'.json';const temp=file+'.'+randomUUID();await writeFile(temp,JSON.stringify({at:Date.now(),value}));await rename(temp,file);
        }
      }
      send(200,{context});return;
    }
    send(404,{error:'Not found'});
  }catch(e){send(502,{ok:false,error:e instanceof Error?e.message:'Gateway error'});}
}).listen(Number(process.env.PORT??8787),'0.0.0.0',()=>console.log('MusicMyLove batch gateway listening'));
