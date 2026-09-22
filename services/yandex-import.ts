import { normalizeYandexPlaylistUrl, YandexPlaylistError, type YandexPlaylistResult, type YandexTrack } from '../src/lib/yandex/playlist';
import { concurrent } from './web-context';
const hosts=new Set(['music.yandex.ru','music.yandex.com','music.yandex.kz','music.yandex.by','lk.music.yandex.ru','lk.music.yandex.com']);
function check(input:string){const u=new URL(/^https?:\/\//i.test(input)?input:'https://'+input);if(u.protocol!=='https:'||u.username||u.password||u.port||!hosts.has(u.hostname))throw new YandexPlaylistError('invalid_url','Ссылка должна вести на публичный плейлист Яндекс Музыки');return u;}
interface RawTrack {id?:number|string;title?:string;artists?:{name?:string}[]}
function track(raw:RawTrack):YandexTrack|undefined {const artists=raw.artists?.map(a=>a.name?.trim()).filter((a):a is string=>!!a);return raw.id!==undefined&&raw.title?.trim()&&artists?.length?{id:String(raw.id),title:raw.title.trim(),artists}:undefined;}
export async function importYandexBatch(input:string,fetcher:typeof fetch=fetch):Promise<YandexPlaylistResult>{
 const signal=AbortSignal.timeout(22000);let url=check(input);
 // Short links are resolved server-side and every redirect host is checked.
 for(let n=0;url.hostname.startsWith('lk.')&&n<4;n++){
  const r=await fetcher(url.href,{redirect:'manual',signal});const location=r.headers.get('location');await r.body?.cancel();if(!location)throw new Error('Не удалось раскрыть короткую ссылку');url=check(new URL(location,url).href);
 }
 const normalized=normalizeYandexPlaylistUrl(url.href);
 if(!normalized.uuid&&!(normalized.owner&&normalized.kind))throw new YandexPlaylistError('invalid_url','Не удалось распознать плейлист');
 const api='https://api.music.yandex.net';
 async function request(path:string,init:RequestInit={}){
  for(let attempt=0;attempt<2;attempt++){
   const r=await fetcher(api+path,{...init,redirect:'error',signal,headers:{'User-Agent':'MusicMyLove/0.1 (+https://github.com/vansGAMee/musicmylove)','Accept':'application/json','X-Yandex-Music-Client':'YandexMusicAndroid/24023251',...(process.env.YANDEX_MUSIC_TOKEN?{Authorization:'OAuth '+process.env.YANDEX_MUSIC_TOKEN}:{}),...init.headers}});
   if(r.status===451)throw new YandexPlaylistError('geo_blocked','Регион gateway заблокирован Яндексом; нужен gateway в РФ.');
   if(r.status===401||r.status===403)throw new YandexPlaylistError('private','Плейлист недоступен или приватный');
   if(r.status===404)throw new YandexPlaylistError('not_found','Плейлист не найден');
   if((r.status===429||r.status>=500)&&attempt===0){await r.body?.cancel();await new Promise(resolve=>setTimeout(resolve,300));continue;}
   if(!r.ok)throw new Error(`Яндекс HTTP ${r.status}`);
   const data=await r.json();if(data.error)throw new Error(String(data.error.message??data.error.name));return data.result;
  }
  throw new Error('Яндекс недоступен');
 }
 const result=await request(normalized.uuid?'/playlist/'+encodeURIComponent(normalized.uuid)+'?rich-tracks=true':'/users/'+encodeURIComponent(normalized.owner!)+'/playlists/'+normalized.kind+'?rich-tracks=true');
 const playlist=result?.playlist??result;
 if(playlist?.visibility==='private')throw new YandexPlaylistError('private','Плейлист приватный');
 const rows=playlist?.tracks as (RawTrack&{track?:RawTrack})[];
 if(!Array.isArray(rows)||typeof playlist.trackCount!=='number'||rows.length!==playlist.trackCount)throw new Error('Яндекс вернул неполный плейлист; импорт отменён');
 if(rows.length>5000)throw new Error('Максимум 5000 треков за один импорт');
 const resolved=new Map<string,YandexTrack>();
 for(const row of rows){const t=track(row.track??row);if(t)resolved.set(String(row.id??t.id),t);}
 const missing=[...new Set(rows.filter(r=>!resolved.has(String(r.id??r.track?.id))).map(r=>String(r.id??r.track?.id)))];
 if(missing.includes('undefined'))throw new Error('Некорректный идентификатор трека');
 const batches:string[][]=[];for(let i=0;i<missing.length;i+=200)batches.push(missing.slice(i,i+200));
 await concurrent(batches,6,async ids=>{
  const values=await request('/tracks',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({'track-ids':ids.join(','),'with-positions':'false'}).toString()});
  if(!Array.isArray(values))throw new Error('Не удалось получить треки');
  for(const raw of values){const t=track(raw);if(t)resolved.set(t.id,t);}
 });
 const tracks=rows.map(r=>resolved.get(String(r.id??r.track?.id)));
 if(tracks.some(t=>!t))throw new Error('Яндекс вернул неполный набор названий; импорт отменён');
 return {id:normalized.uuid??`${normalized.owner}:${normalized.kind}`,title:playlist.title??'Яндекс-плейлист',owner:normalized.owner,trackCount:rows.length,tracks:tracks as YandexTrack[]};
}
