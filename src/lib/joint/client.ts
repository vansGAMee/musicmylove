import type {siteRecommend,siteSearch} from './site';
let worker:Worker|undefined,ready:Promise<unknown>|undefined,serial=0;
const pending=new Map<number,{resolve:(value:unknown)=>void;reject:(error:Error)=>void}>();
function rpc<T>(type:string,payload:object):Promise<T>{
  if(!worker){
    worker=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
    worker.onmessage=e=>{const p=pending.get(e.data.id);if(!p)return;pending.delete(e.data.id);if(e.data.error)p.reject(new Error(e.data.error));else p.resolve(e.data.result??e.data);};
    worker.onerror=()=>{for(const p of pending.values())p.reject(new Error('Не удалось запустить модель в браузере'));pending.clear();worker?.terminate();worker=undefined;ready=undefined;};
  }
  return new Promise<T>((resolve,reject)=>{const id=++serial;pending.set(id,{resolve:resolve as (value:unknown)=>void,reject});worker!.postMessage({id,type,...payload});});
}
async function load(){if(!ready)ready=rpc('load',{base:'/joint-model'}).catch(e=>{ready=undefined;throw e;});await ready;}
export async function searchJoint(query:string):Promise<ReturnType<typeof siteSearch>>{await load();return rpc('search',{query});}
export async function recommendJoint(songs:Parameters<typeof siteRecommend>[1],feedback:Parameters<typeof siteRecommend>[2]):Promise<ReturnType<typeof siteRecommend>>{await load();return rpc('siteRecommend',{songs,feedback});}
