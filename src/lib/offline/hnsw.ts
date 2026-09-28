/** Serializable hierarchical navigable small-world index; build offline, query in TS. */
export interface Hnsw { entry: number; levels: number[][][]; m: number }
export interface Hit { id: number; score: number }
export type Vector = ArrayLike<number>;

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) n += a[i] * (b[i] ?? 0);
  return n;
}

export function unit(v: ArrayLike<number>): number[] {
  const n = Math.sqrt(dot(v, v));
  const res = new Array(v.length);
  for (let i = 0; i < v.length; i++) res[i] = n ? v[i] / n : v[i];
  return res;
}

const order = (a: Hit, b: Hit) => b.score - a.score || a.id - b.id;

function insert(list: Hit[], hit: Hit, limit: number) {
  let lo = 0, hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (order(list[mid], hit) <= 0) lo = mid + 1;
    else hi = mid;
  }
  if (lo < limit) {
    list.splice(lo, 0, hit);
    if (list.length > limit) list.pop();
  }
}

function layer(index: Hnsw, vectors: readonly ArrayLike<number>[], q: ArrayLike<number>, entry: number, level: number, ef: number): Hit[] {
  const first = { id: entry, score: dot(q, vectors[entry]) };
  const best = [first], queue = [first], visited = new Set([entry]);
  while (queue.length) {
    const current = queue.shift()!;
    if (best.length >= ef && order(current, best[best.length - 1]) > 0) break;
    for (const id of index.levels[current.id]?.[level] ?? []) {
      if (visited.has(id)) continue;
      visited.add(id);
      const hit = { id, score: dot(q, vectors[id]) };
      if (best.length < ef || order(hit, best[best.length - 1]) < 0) {
        insert(best, hit, ef);
        insert(queue, hit, ef);
      }
    }
  }
  return best;
}

export function searchHnsw(index: Hnsw, vectors: readonly ArrayLike<number>[], q: ArrayLike<number>, k: number, ef = Math.max(64, k)): Hit[] {
  if (index.entry < 0 || k <= 0) return [];
  let entry = index.entry;
  for (let l = index.levels[entry].length - 1; l > 0; l--) entry = layer(index, vectors, q, entry, l, 1)[0].id;
  return layer(index, vectors, q, entry, 0, Math.max(k, ef)).slice(0, k);
}

export function buildHnsw(vectors: readonly ArrayLike<number>[], m = 12, ef = 64): Hnsw {
  const index: Hnsw = { entry: -1, levels: [], m };
  let state = 123456789;
  for(let id=0;id<vectors.length;id++){
    state^=state<<13;state^=state>>>17;state^=state<<5;
    const height=Math.min(12,Math.floor(-Math.log(((state>>>0)+1)/4294967297)/Math.log(m)));
    index.levels.push(Array.from({length:height+1},()=>[]));
    if(index.entry<0){index.entry=id;continue;}
    let entry=index.entry;const top=index.levels[entry].length-1;
    for(let l=top;l>height;l--)entry=layer(index,vectors,vectors[id],entry,l,1)[0].id;
    for(let l=Math.min(height,top);l>=0;l--){
      const hits=layer(index,vectors,vectors[id],entry,l,ef);
      // Diversity selection prevents dense identical vectors monopolizing edges.
      const selected:Hit[]=[];
      for(const hit of hits){if(selected.every(s=>dot(vectors[hit.id],vectors[s.id])<hit.score)){selected.push(hit);if(selected.length===m)break;}}
      for(const hit of hits){if(selected.length===m)break;if(!selected.some(s=>s.id===hit.id))selected.push(hit);}
      index.levels[id][l]=selected.map(h=>h.id);
      for(const {id:other} of selected){
        const links=index.levels[other][l];links.push(id);
        if(links.length>m*2){links.sort((a,b)=>dot(vectors[other],vectors[b])-dot(vectors[other],vectors[a])||a-b);links.length=m*2;}
      }
      entry=hits[0].id;
    }
    if(height>top)index.entry=id;
  }
  // A bounded backbone keeps duplicate-vector clusters reachable after pruning.
  // Without it, equal-score pruning can isolate whole user neighborhoods.
  if(vectors.length>1)for(let id=0;id<vectors.length;id++){
    const links=index.levels[id][0];
    for(const neighbor of [(id+1)%vectors.length,(id+vectors.length-1)%vectors.length])if(!links.includes(neighbor))links.push(neighbor);
  }
  return index;
}
