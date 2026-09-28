/** Yandex Cloud Functions adapter: the frontend remains a static Vercel export. */
import { readFile, mkdir, writeFile, rename } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { importYandexBatch } from './yandex-import';
import { ContextResolver, publicPage, searchWeb } from './web-context';
import type { Song, Context } from '../src/lib/offline/engine';
const folder=process.env.CACHE_DIR??'/tmp/musicmylove-cache';
const pathFor=(k:string)=>folder+'/'+createHash('sha256').update(k).digest('hex')+'.json';
async function get<T>(k:string):Promise<T|undefined>{try{const data=JSON.parse(await readFile(pathFor(k),'utf8'));if(Date.now()-data.at<86400000)return data.value;}catch{}}
async function set(k:string,value:unknown){await mkdir(folder,{recursive:true});const file=pathFor(k),temp=file+'.'+randomUUID();await writeFile(temp,JSON.stringify({at:Date.now(),value}));await rename(temp,file);}
let resolver:Promise<ContextResolver>|undefined;
function contextResolver(){return resolver??=readFile(process.env.TRACK_CATALOG??'catalog.json','utf8').then(raw=>new ContextResolver(JSON.parse(raw).tracks,{search:searchWeb,page:publicPage},{get:k=>get<Context>(k),set}));}
interface Event {httpMethod:string;path?:string;url?:string;queryStringParameters?:Record<string,string>;headers?:Record<string,string>;body?:string;isBase64Encoded?:boolean}
export async function handler(event:Event){
 const origin=event.headers?.origin??event.headers?.Origin;
 const allowed=process.env.FRONTEND_ORIGIN;
 const headers={'Content-Type':'application/json','Access-Control-Allow-Origin':allowed??'http://localhost:3000','Access-Control-Allow-Methods':'POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type','Vary':'Origin'};
 const send=(statusCode:number,value:unknown)=>({statusCode,headers,isBase64Encoded:false,body:JSON.stringify(value)});
 if(origin&&origin!==allowed)return send(403,{error:'Origin not allowed'});
 if(event.httpMethod==='OPTIONS')return send(204,{});
 const path=event.queryStringParameters?.route??event.path??'';
 if(path.endsWith('/health'))return send(200,{ok:true,searchProvider:process.env.SERPER_API_KEY?'serper':'duckduckgo'});
 if(event.httpMethod!=='POST')return send(405,{error:'POST required'});
 try{
  const raw=event.isBase64Encoded?Buffer.from(event.body??'','base64').toString():event.body??'';
  if(Buffer.byteLength(raw)>1024*1024)return send(413,{error:'Payload too large'});
  const input=JSON.parse(raw);
  if(path.endsWith('/playlist')){
   if(typeof input.url!=='string')return send(400,{error:'Playlist URL required'});
   const k='playlist:'+input.url;
   const playlist=await get<Awaited<ReturnType<typeof importYandexBatch>>>(k)??await importYandexBatch(input.url);
   await set(k,playlist);return send(200,{ok:true,playlist,tracks:playlist.tracks});
  }
  if(path.endsWith('/context')){
   if(!Array.isArray(input.tracks)||input.tracks.length>5000||input.tracks.some((s:Song)=>!s||typeof s.artist!=='string'||typeof s.title!=='string'||s.artist.length>500||s.title.length>500))return send(400,{error:'Invalid track batch'});
   return send(200,await (await contextResolver()).resolve(input.tracks));
  }
  return send(404,{error:'Not found'});
 }catch(error){return send(502,{ok:false,error:error instanceof Error?error.message:'Gateway error'});}
}
