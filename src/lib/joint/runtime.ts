/** Compact pure-browser neural inference. No audio decoding, Python, API or CLAP. */
export interface Catalog {
  tracks: { id: string; artist: string; title: string; aliases?: string[][] }[];
  artists: number[]; families: number[];
}
export interface Weights {w1: number[][]; b1: number[]; w2: number[][]; b2: number[]}
export interface JointData {catalog: Catalog; vectors: Float32Array; neighbors: Int32Array; info: Float32Array; weights: Weights}
export function selectSeeds(input: number[], artists: number[]): number[] {
  const key = (i: number) => Math.imul(i + 1, 2654435761) >>> 0;
  const groups = new Map<number, number[]>();
  for (const i of [...new Set(input)].sort((a,b)=>key(a)-key(b))) {
    const group=groups.get(artists[i]) ?? []; group.push(i); groups.set(artists[i],group);
  }
  const queues=[...groups.values()].sort((a,b)=>key(a[0])-key(b[0])); const out:number[]=[];
  for (let p=0;p<Math.max(0,...queues.map(q=>q.length));p++) {
    for (const q of queues) {if(p<q.length)out.push(q[p]);if(out.length===64)return out;}
  }
  return out;
}
export function candidateIds(d:JointData,input:number[]):number[] {
  const blocked=new Set(input.map(i=>d.catalog.families[i])); const ids=new Set<number>();
  for(const s of selectSeeds(input,d.catalog.artists))for(let k=0;k<24;k++) {
    const i=d.neighbors[s*24+k];if(i>=0&&!blocked.has(d.catalog.families[i]))ids.add(i);
  }
  if(!ids.size&&input.length){
    const seeds=selectSeeds(input,d.catalog.artists),n=d.catalog.tracks.length,scores:{i:number;dot:number}[]=[];
    for(let i=0;i<n;i++){
      if(blocked.has(d.catalog.families[i]))continue;
      let maxDot=-Infinity;
      for(const s of seeds){
        let dot=0;for(let j=0;j<32;j++)dot+=d.vectors[i*32+j]*d.vectors[s*32+j];
        if(dot>maxDot)maxDot=dot;
      }
      scores.push({i,dot:maxDot});
    }
    scores.sort((a,b)=>b.dot-a.dot);
    for(let k=0;k<Math.min(100,scores.length);k++)ids.add(scores[k].i);
  }
  return [...ids].sort((a,b)=>a-b);
}
export function neuralScore(d:JointData,seeds:number[],i:number):number {
  let maximum=-Infinity,total=0,coverage=0,familiar=0;
  for(const s of seeds){
    let dot=0;for(let j=0;j<32;j++)dot+=d.vectors[i*32+j]*d.vectors[s*32+j];
    maximum=Math.max(maximum,dot);total+=dot;coverage+=d.info[s*2+1];
    if(d.catalog.artists[s]===d.catalog.artists[i])familiar=1;
  }
  const x=[maximum,total/seeds.length,d.info[i*2],familiar,d.info[i*2+1],coverage/seeds.length];
  let value=d.weights.b2[0];
  for(let h=0;h<16;h++){
    let v=d.weights.b1[h];for(let j=0;j<6;j++)v+=x[j]*d.weights.w1[h][j];
    value+=Math.max(0,v)*d.weights.w2[0][h];
  }
  return value;
}
export function recommend(d:JointData,input:number[],known:number[]=[]):{index:number;score:number}[] {
  const n=d.catalog.tracks.length;
  if(!input.length||input.length>2000||[...input,...known].some(i=>!Number.isInteger(i)||i<0||i>=n))throw Error('Invalid track indices');
  const seeds=selectSeeds(input,d.catalog.artists),blocked=new Set([...input,...known].map(i=>d.catalog.families[i]));
  const rows=candidateIds(d,input).filter(i=>!blocked.has(d.catalog.families[i])).map(index=>({index,score:neuralScore(d,seeds,index)}));
  rows.sort((a,b)=>b.score-a.score || (d.catalog.tracks[a.index].id<d.catalog.tracks[b.index].id?-1:1));
  const used=new Set<number>(),counts=new Map<number,number>(),out:typeof rows=[];let previous=-1;
  while(rows.length&&out.length<50){
    const pos=rows.findIndex(r=>!used.has(d.catalog.families[r.index])&&(counts.get(d.catalog.artists[r.index])??0)<2&&d.catalog.artists[r.index]!==previous);
    if(pos<0)break;
    const row=rows.splice(pos,1)[0],artist=d.catalog.artists[row.index];out.push(row);
    used.add(d.catalog.families[row.index]);counts.set(artist,(counts.get(artist)??0)+1);previous=artist;
  }
  return out;
}
export async function loadJoint(base:string):Promise<JointData> {
  const response=await fetch(`${base}/manifest.json`);if(!response.ok)throw Error('Missing joint manifest');
  const m=await response.json();
  if(m.schema!=='joint-browser-v1'||m.dimensions!==32||m.neighbors!==24||m.bytes>32*1024*1024)throw Error('Unsupported bundle');
  async function bytes(name:string):Promise<ArrayBuffer>{
    const r=await fetch(`${base}/${name}`);if(!r.ok)throw Error(`Missing ${name}`);
    const b=await r.arrayBuffer();if(b.byteLength!==m.sizes[name])throw Error(`Wrong size ${name}`);
    const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(v=>v.toString(16).padStart(2,'0')).join('');
    if(hash!==m.sha256[name])throw Error(`Corrupt ${name}`);return b;
  }
  const [v,g,info,c,w]=await Promise.all(['vectors.f32','neighbors.i32','info.f32','catalog.json','ranker.json'].map(bytes));
  if(v.byteLength!==m.tracks*32*4||g.byteLength!==m.tracks*24*4||info.byteLength!==m.tracks*2*4)throw Error('Wrong shapes');
  return {vectors:new Float32Array(v),neighbors:new Int32Array(g),info:new Float32Array(info),
    catalog:JSON.parse(new TextDecoder().decode(c)),weights:JSON.parse(new TextDecoder().decode(w))};
}
