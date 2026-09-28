import { key, type Context, type Feedback, type Song } from './engine';
import type { recommend } from './engine';
import type { Track } from '../types';
const gateway=process.env.NEXT_PUBLIC_GATEWAY_URL?.replace(/\/$/,'');
function gatewayUrl(path:string){
  const url=new URL(gateway!);
  if(url.hostname==='functions.yandexcloud.net'){url.searchParams.set('route',path);return url.href;}
  return gateway+path;
}
let worker:Worker|undefined,serial=0;
const pending=new Map<number,{resolve:(x:unknown)=>void;reject:(e:Error)=>void}>();
function rpc<T>(method:string,payload:unknown):Promise<T>{
  if(!worker){worker=new Worker(new URL('./worker.ts',import.meta.url), { type: 'module' });worker.onmessage=e=>{const p=pending.get(e.data.id);if(!p)return;pending.delete(e.data.id);if(e.data.error)p.reject(new Error(e.data.error));else p.resolve(e.data.result);};worker.onerror=()=>{for(const p of pending.values())p.reject(new Error('Ошибка локального поиска'));pending.clear();worker?.terminate();worker=undefined;};}
  return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve:resolve as (x:unknown)=>void,reject});worker!.postMessage({id,method,payload});});
}
export async function searchLocal(query: string): Promise<Track[]> {
  const local = await rpc<Track[]>('search', query);
  if (local.length > 0 || query.trim().length < 2) return local;
  try {
    const res = await fetch(`https://labs.api.listenbrainz.org/recording-search/json?query=${encodeURIComponent(query.trim())}`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return local;
    const data = await res.json() as Array<Record<string, unknown>>;
    if (!Array.isArray(data)) return local;
    return data.slice(0, 25).map((item, idx) => ({
      mbid: String(item.recording_mbid || item.mbid || `labs-${idx}`),
      artist: String(item.artist_credit_name || item.artist_name || item.artist || '').trim(),
      title: String(item.recording_name || item.track_name || item.title || '').trim(),
      ...(item.release_name ? { release: String(item.release_name).trim() } : {}),
    })).filter(t => t.artist && t.title);
  } catch {
    return local;
  }
}
export async function importPlaylist(url:string): Promise<Response> {
  if(gateway) {
    try {
      const res = await fetch(gatewayUrl('/playlist'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url}),signal:AbortSignal.timeout(30000)});
      if (res.ok) return res;
    } catch {
      // Fall back to direct browser import on network failure
    }
  }
  try {
    const { fetchYandexPlaylist, YandexPlaylistError } = await import('../yandex/playlist');
    const playlist = await fetchYandexPlaylist(url);
    return new Response(JSON.stringify({ ok: true, playlist, tracks: playlist.tracks }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err: unknown) {
    const isGeo = (err as { code?: string })?.code === 'geo_blocked' || String(err).includes('geo_blocked');
    const isPrivate = (err as { code?: string })?.code === 'private';
    const isNotFound = (err as { code?: string })?.code === 'not_found';
    const status = isGeo ? 451 : isPrivate ? 403 : isNotFound ? 404 : 400;
    const msg = isGeo
      ? 'Не удалось импортировать плейлист. Попробуйте ещё раз.'
      : err instanceof Error ? err.message : 'Ошибка импорта плейлиста';
    return new Response(JSON.stringify({ ok: false, error: msg, code: (err as { code?: string })?.code ?? 'geo_blocked' }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
let context:Record<string,Context>|undefined;
const attempted=new Set<string>();
export async function recommendLocal(songs:Song[],feedback:Feedback){
  if(!context){try{context=JSON.parse(localStorage.getItem('musicmylove:context:v1')??'{}');}catch{context={};}}
  const holes=await rpc<Song[]>('missing',{songs,context});
  const fresh=holes.filter(s=>!attempted.has(key(s)));
  if(gateway&&fresh.length){
    fresh.forEach(s=>attempted.add(key(s)));
    try{
      const response=await fetch(gatewayUrl('/context'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tracks:fresh}),signal:AbortSignal.timeout(20000)});
      if(response.ok){
        const body=(await response.json()) as { context?: Record<string, unknown> };
        const verified: Record<string, Context> = {};
        if (body.context && typeof body.context === 'object') {
          for (const [k, v] of Object.entries(body.context)) {
            const x = v as Context;
            if (Array.isArray(x?.neighbors) && Array.isArray(x?.sources) && x.neighbors.length > 0 && x.sources.length > 0) {
              verified[k] = { neighbors: x.neighbors, sources: x.sources };
            }
          }
        }
        context={...context,...verified};
        try{localStorage.setItem('musicmylove:context:v1',JSON.stringify(context));}catch{/* Storage quota does not break recommendations. */}
      }
    }catch{/* Missing evidence stays missing; graph coverage is returned. */}
  }
  const result=await rpc<ReturnType<typeof recommend>>('recommend',{songs,feedback,context});
  if(!result.recommendations.length)throw new Error(`Недостаточно связей: найдено ${result.coverage.graph} из ${result.coverage.total} треков. Нужен более полный граф или web context.`);
  return result;
}
