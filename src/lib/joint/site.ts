import {candidateIds, recommend, type Catalog, type JointData} from './runtime';
type Song = {artist:string;title:string};
const normalize=(s:string)=>s.normalize('NFKC').toLowerCase().replace(/ß/g,'ss').replace(/ς/g,'σ').replace(/[’‘]/g,"'").replace(/[‐‑–—]/g,'-').replace(/\s+/gu,' ').trim().replace(/\s+-\s+/g,' - ');
const name=(s:Song)=>normalize(`${s.artist} - ${s.title}`);
const mastering=(s:string)=>s.replace(/\s*[\[(](?:(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?)[\])]\s*$/,'').replace(/\s+-\s+(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?\s*$/,'');
const familyName=(s:Song)=>normalize(s.artist)+'\u001f'+normalize(s.title).replace(/\s*[\[(][^\])]*(?:remaster|live|version|remix|acoustic|edit)[^\])]*[\])]\s*$/,'').replace(/\s+-\s+(?:\d{4}\s+)?(?:remaster|live|version|remix|acoustic|edit).*$/,'').trim();
const cache=new WeakMap<Catalog,{exact:Map<string,Set<number>>;master:Map<string,Set<number>>;family:Map<string,Set<number>>}>();
function indices(c:Catalog){
  let value=cache.get(c);if(value)return value;
  value={exact:new Map(),master:new Map(),family:new Map()};
  const add=(m:Map<string,Set<number>>,k:string,i:number)=>{const s=m.get(k)??new Set();s.add(i);m.set(k,s);};
  c.tracks.forEach((t,i)=>{for(const [artist,title] of [[t.artist,t.title],...(t.aliases??[])]){
    const song={artist,title};add(value!.exact,name(song),i);add(value!.master,mastering(name(song)),i);add(value!.family,familyName(song),i);
  }});cache.set(c,value);return value;
}
export function resolveSongs(c:Catalog,songs:Song[]){
  const m=indices(c),seeds=new Set<number>(),known=new Set<number>(),unresolved:Song[]=[],matches:{input:Song;index:number}[]=[];
  for(const song of songs){
    const key=name(song),ids=m.exact.get(key)??m.master.get(mastering(key))??new Set<number>();
    for(const i of ids)known.add(i);
    for(const i of m.family.get(familyName(song))??[])known.add(i);
    let selected=ids.size===1?[...ids][0]:undefined;
    if(ids.size>1){const fallback=[...ids].filter(i=>c.tracks[i].id.startsWith('fallback:')&&name(c.tracks[i])===key);if(fallback.length===1)selected=fallback[0];}
    if(selected===undefined)unresolved.push(song);else{seeds.add(selected);matches.push({input:song,index:selected});}
  }
  return {seeds:[...seeds].sort((a,b)=>a-b),known:[...known].sort((a,b)=>a-b),unresolved,matches};
}
export function siteSearch(d:JointData,query:string){
  const q=normalize(query);
  return d.catalog.tracks.filter((t,i)=>name(t).includes(q)&&resolveSongs(d.catalog,[t]).seeds.includes(i)).slice(0,25).map(t=>({mbid:t.id,artist:t.artist,title:t.title}));
}
export function siteRecommend(d:JointData,songs:Song[],feedback:Record<string,'like'|'dislike'>){
  if(!songs.length||songs.length>2000)throw Error('Нужен список из 1–2000 треков');
  const profile=resolveSongs(d.catalog,songs);
  if(!profile.seeds.length)throw Error(`Распознано 0/${songs.length}. Добавьте треки из каталога через поиск.`);
  const known=[...profile.known];d.catalog.tracks.forEach((t,i)=>{if(feedback[t.id]==='dislike')known.push(i);});
  const recommendations=recommend(d,profile.seeds,known).map(({index,score})=>{
    const t=d.catalog.tracks[index];return {mbid:t.id,artist:t.artist,title:t.title,score,
      release:undefined as string|undefined,strongestTasteHead:undefined as number|undefined,
      seedSupport:undefined as number|undefined,supportingSeeds:undefined as Song[]|undefined,
      popularityPercentile:undefined as number|undefined,noveltyLiftScore:undefined as number|undefined,
      spotifyLink:`https://open.spotify.com/search/${encodeURIComponent(t.artist+' '+t.title)}`};
  });
  if(!recommendations.length)throw Error('Нет рекомендаций для распознанных треков после исключений.');
  return {recommendations,seeds:songs.map(input=>{
    const match=profile.matches.find(m=>m.input===input),t=match===undefined?undefined:d.catalog.tracks[match.index];
    return {input,track:t?{mbid:t.id,artist:t.artist,title:t.title}:undefined};
  }),coverage:{total:songs.length,graph:profile.matches.length,audio:profile.matches.filter(m=>d.info[m.index*2+1]>0).length},
  candidateCount:candidateIds(d,profile.seeds).length,unresolved:profile.unresolved};
}
