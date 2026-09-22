/// <reference lib="webworker" />
import { key, missing, recommend, type Catalog, type Context, type Song } from './engine';
let catalogPromise:Promise<Catalog>|undefined;
let context:Record<string,Context>={};
function catalog(){return catalogPromise??=fetch('/data/catalog.json').then(async r=>{if(!r.ok)throw new Error('Каталог не собран: npm run build:catalog');const c=await r.json() as Catalog;if(c.format!=='human-graph-v1')throw new Error('Неверная версия каталога');return c;}).catch(e=>{catalogPromise=undefined;throw e;});}
self.onmessage=async(event:MessageEvent)=>{
  const {id,method,payload}=event.data;
  try{
    const c=await catalog();
    if(method==='search'){
      const q=String(payload).normalize('NFKC').toLowerCase().trim();
      if(!q){self.postMessage({id,result:[]});return;}
      const cleanQ=q.replace(/[^\p{L}\p{N}]+/gu,' ').trim();
      const tokens=cleanQ.split(/\s+/).filter(Boolean);
      if(!tokens.length){self.postMessage({id,result:[]});return;}
      const scored=c.tracks.flatMap(t=>{
        const artist=t.artist.toLowerCase();
        const title=t.title.toLowerCase();
        const normArtist=artist.replace(/[^\p{L}\p{N}]+/gu,' ').trim();
        const normTitle=title.replace(/[^\p{L}\p{N}]+/gu,' ').trim();
        const full=normArtist+' '+normTitle;
        const matchesAll=tokens.every(tok=>normArtist.includes(tok)||normTitle.includes(tok));
        if(!matchesAll)return [];
        let score=0;
        if(full===cleanQ)score+=350;
        else if(normArtist===cleanQ)score+=300;
        else if(normTitle===cleanQ)score+=250;
        else if(full.startsWith(cleanQ))score+=180;
        else if(normArtist.startsWith(cleanQ))score+=150;
        else if(normTitle.startsWith(cleanQ))score+=120;
        for(const tok of tokens){
          if(normArtist===tok)score+=50;
          else if(normTitle===tok)score+=40;
          else if(normArtist.startsWith(tok))score+=25;
          else if(normTitle.startsWith(tok))score+=20;
          else score+=5;
        }
        score+=Math.round(t.popularity*20);
        return [{...t,searchScore:score}];
      }).sort((a,b)=>b.searchScore-a.searchScore||(a.mbid<b.mbid?-1:1)).slice(0,25).map(({searchScore,...t})=>t);
      self.postMessage({id,result:scored});return;
    }
    context={...context,...payload.context};
    if(method==='missing'){self.postMessage({id,result:missing(c,payload.songs).filter((s:Song)=>!context[key(s)])});return;}
    const result=recommend(c,payload.songs,payload.feedback??{},context);
    self.postMessage({id,result});
  }catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error)});}
};
