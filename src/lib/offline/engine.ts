import { dot, unit, searchHnsw, type Hnsw } from './hnsw';
import type { Track } from '../types';
export type Song=Pick<Track,'artist'|'title'>;
export type Feedback=Record<string,'like'|'dislike'>;
export interface Context { neighbors:string[]; sources:string[] }
export interface Catalog {
  format:'human-graph-v1'; source:string;
  tracks:(Track & {popularity:number})[];
  graph:number[][]; graphIndex:Hnsw;
  audio?:{ids:string[]; vectors:number[][]; index:Hnsw; encoder:string};
  context:Record<string,Context>;
}
export const key=(s:Song)=>[s.artist,s.title].map(x=>x.normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim()).join('\u001f');
interface Seed { track:Song; id?:number; graph?:number[]; audio?:number[] }
const cache=new WeakMap<Catalog,{ids:Map<string,number>; keys:Map<string,number>; audio:Map<string,number>}>();
function maps(c:Catalog){let m=cache.get(c);if(!m){m={ids:new Map(c.tracks.map((t,i)=>[t.mbid,i])),keys:new Map(),audio:new Map(c.audio?.ids.map((id,i)=>[id,i])??[])};c.tracks.forEach((t,i)=>{if(!m!.keys.has(key(t)))m!.keys.set(key(t),i);});cache.set(c,m);}return m;}
export function missing(c:Catalog,songs:Song[]):Song[]{const m=maps(c);return [...new Map(songs.filter(s=>!m.keys.has(key(s))&&!c.context[key(s)]).map(s=>[key(s),s])).values()].sort((a,b)=>key(a)<key(b)?-1:1);}
function mean(vs:number[][]){return unit(vs[0].map((_,i)=>vs.reduce((n,v)=>n+v[i],0)/vs.length));}
function heads(vectors:number[][]):number[][] {
  if(!vectors.length)return [];
  const count=Math.min(vectors.length,12,Math.max(2,Math.ceil(Math.sqrt(vectors.length))));
  let centers=[vectors[0]];
  while(centers.length<count){let best=-1,idx=-1;vectors.forEach((v,i)=>{const d=1-Math.max(...centers.map(c=>dot(c,v)));if(d>best){best=d;idx=i;}});if(best<1e-7)break;centers.push(vectors[idx]);}
  for(let iteration=0;iteration<4;iteration++){
    const groups:number[][][]=centers.map(()=>[]);
    for(const v of vectors){let best=0;for(let i=1;i<centers.length;i++)if(dot(v,centers[i])>dot(v,centers[best]))best=i;groups[best].push(v);}
    centers=groups.filter(g=>g.length).map(mean);
  }
  return centers;
}
export function recommend(c:Catalog,songs:Song[],feedback:Feedback,context:Record<string,Context>={}){
  const m=maps(c), unique=[...new Map(songs.map(s=>[key(s),s])).entries()].sort(([a],[b])=>a<b?-1:a>b?1:0).map(([,s])=>s);
  const seeds:Seed[]=unique.map(track=>{
    const id=m.keys.get(key(track));
    const evidence=c.context[key(track)]??context[key(track)];
    const neighbors=evidence?.sources.length ? [...new Set(evidence.neighbors)].sort().flatMap(id=>{const i=m.ids.get(id);return i===undefined?[]:[c.graph[i]];}) : [];
    const ai=id===undefined?undefined:m.audio.get(c.tracks[id].mbid);
    return {track,id,graph:id===undefined?(neighbors.length?mean(neighbors):undefined):c.graph[id],audio:ai===undefined?undefined:c.audio!.vectors[ai]};
  });
  const positive=Object.keys(feedback).filter(id=>feedback[id]==='like').sort().flatMap(id=>{const i=m.ids.get(id);return i===undefined?[]:[i];});
  const gh=heads([...seeds.flatMap(s=>s.graph?[s.graph]:[]),...positive.map(i=>c.graph[i])]);
  const ah=heads([...seeds.flatMap(s=>s.audio?[s.audio]:[]),...positive.flatMap(i=>{const a=m.audio.get(c.tracks[i].mbid);return a===undefined?[]:[c.audio!.vectors[a]];})]);
  const excluded=new Set(unique.map(key));positive.forEach(i=>excluded.add(key(c.tracks[i])));
  const candidates=new Map<number,number>();
  for(const h of gh)for(const hit of searchHnsw(c.graphIndex,c.graph,h,Math.ceil(2000/Math.max(1,gh.length))))candidates.set(hit.id,Math.max(candidates.get(hit.id)??-1,hit.score));
  if(c.audio)for(const h of ah)for(const hit of searchHnsw(c.audio.index,c.audio.vectors,h,Math.ceil(2000/Math.max(1,ah.length)))){const id=m.ids.get(c.audio.ids[hit.id]);if(id!==undefined)candidates.set(id,Math.max(candidates.get(id)??-1,hit.score));}
  // Heads may retrieve overlapping neighborhoods. Fill the union, not each head's quota.
  if(candidates.size<2000&&gh.length)for(const hit of searchHnsw(c.graphIndex,c.graph,mean(gh),2000))candidates.set(hit.id,Math.max(candidates.get(hit.id)??-1,hit.score));
  const negatives=Object.keys(feedback).filter(id=>feedback[id]==='dislike').sort().flatMap(id=>{const i=m.ids.get(id);return i===undefined?[]:[c.graph[i]];});
  const seedHeadMap=seeds.map(s=>{if(!s.graph||!gh.length)return 0;const sims=gh.map(h=>dot(h,s.graph!));return Math.max(0,sims.indexOf(Math.max(...sims)));});
  const headSeedCounts=gh.map((_,h)=>seedHeadMap.filter(x=>x===h).length);
  const maxAffinityPerHead=gh.map(()=>0.01);
  const preCandidates=[...candidates].sort((a,b)=>b[1]-a[1]||a[0]-b[0]).slice(0,2000).flatMap(([id])=>{
    const t=c.tracks[id];if(excluded.has(key(t))||feedback[t.mbid]==='dislike')return [];
    const candVec=c.graph[id];
    const gs=gh.map(h=>Math.max(0,dot(h,candVec))), g=Math.max(0,...gs), strongestTasteHead=Math.max(0,gs.indexOf(g));
    const seedSims=seeds.flatMap((s,idx)=>s.graph?[{track:s.track,sim:dot(s.graph,candVec),head:seedHeadMap[idx]}]:[]);
    const inHeadSims=seedSims.filter(x=>x.head===strongestTasteHead).sort((a,b)=>b.sim-a.sim);
    const topInHead=inHeadSims.slice(0,Math.min(3,inHeadSims.length));
    const inHeadRelevance=topInHead.length?topInHead.reduce((acc,x)=>acc+Math.max(0,x.sim),0)/topInHead.length:g;
    const rawAffinity=0.50*g+0.50*inHeadRelevance;
    if(rawAffinity>maxAffinityPerHead[strongestTasteHead])maxAffinityPerHead[strongestTasteHead]=rawAffinity;
    return [{id,t,candVec,gs,g,strongestTasteHead,seedSims,inHeadSims,rawAffinity}];
  });
  const pool=preCandidates.map(item=>{
    const {t,candVec,strongestTasteHead,seedSims,inHeadSims,rawAffinity}=item;
    const maxH=maxAffinityPerHead[strongestTasteHead]||1;
    const calibratedAffinity=0.60*rawAffinity+0.40*Math.min(1.0,rawAffinity/Math.max(0.25,maxH));
    const inHeadSupport=inHeadSims.filter(x=>x.sim>0.15).length;
    const supportRatio=inHeadSupport/Math.max(1,headSeedCounts[strongestTasteHead]);
    const supportBonus=0.06*supportRatio;
    const ai=m.audio.get(t.mbid);
    const sound=ai===undefined||!ah.length?undefined:Math.max(0,...ah.map(h=>dot(h,c.audio!.vectors[ai])));
    const tasteScore=sound===undefined?calibratedAffinity:0.70*calibratedAffinity+0.30*sound;
    const rare=Math.max(0,tasteScore*(1-t.popularity));
    const negative=negatives.length?Math.max(0,...negatives.map(v=>dot(v,candVec))):0;
    const score=tasteScore+supportBonus+0.03*rare-0.02*t.popularity-0.35*negative;
    const posSeeds=seedSims.filter(x=>x.sim>0.05).sort((a,b)=>b.sim-a.sim);
    const bestSeeds=posSeeds.length>=2?posSeeds:seedSims.sort((a,b)=>b.sim-a.sim);
    const supportingSeeds=bestSeeds.slice(0,4).map(x=>({artist:x.track.artist,title:x.track.title}));
    return {...t,score,strongestTasteHead,seedSupport:inHeadSupport,supportingSeeds,popularityPercentile:t.popularity,noveltyLiftScore:rare,spotifyLink:`https://open.spotify.com/search/${encodeURIComponent(t.artist+' '+t.title)}`};
  }).sort((a,b)=>b.score-a.score||(a.mbid<b.mbid?-1:1));
  const seen=new Set<string>(),artists=new Map<string,number>();
  const recommendations=pool.filter(t=>{const a=t.artist.toLowerCase(),k=key(t);if(seen.has(k)||(artists.get(a)??0)>=2)return false;seen.add(k);artists.set(a,(artists.get(a)??0)+1);return true;}).slice(0,40);
  return {recommendations,seeds:seeds.map(s=>({input:s.track,status:s.graph||s.audio?'resolved':'unresolved',track:s.id===undefined?undefined:c.tracks[s.id]})),coverage:{total:seeds.length,graph:seeds.filter(s=>s.graph).length,audio:seeds.filter(s=>s.audio).length},candidateCount:candidates.size};
}
