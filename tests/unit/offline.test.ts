import { describe, it, expect } from 'vitest';
import { buildHnsw, searchHnsw } from '../../src/lib/offline/hnsw';
import { recommend, type Catalog } from '../../src/lib/offline/engine';

describe('offline human graph', () => {
  const vectors = [[1,0], [0.99,0.1], [0,1], [0.1,0.99], [-1,0]];
  it('retrieves actual vector neighbors deterministically', () => {
    const index = buildHnsw(vectors);
    expect(searchHnsw(index, vectors, [1,0], 2).map(x => x.id)).toEqual([0,1]);
    expect(buildHnsw(vectors)).toEqual(index);
  });
  it('is permutation invariant, excludes all positives, and honors explicit dislikes', () => {
    const catalog: Catalog = { format:'human-graph-v1', source:'test-only', tracks:vectors.map((_,i) => ({mbid:String(i), artist:'Artist '+i, title:'Song '+i, popularity:0.1})), graph:vectors, graphIndex:buildHnsw(vectors), context:{} };
    const songs = [catalog.tracks[0],catalog.tracks[2]];
    const a = recommend(catalog, songs, {});
    expect(a).toEqual(recommend(catalog, [...songs].reverse(), {}));
    expect(a.recommendations.every(t => !['0','2'].includes(t.mbid))).toBe(true);
    expect(recommend(catalog, songs, {'1':'dislike'}).recommendations.some(t => t.mbid==='1')).toBe(false);
    expect(a.coverage.audio).toBe(0);
    expect(recommend(catalog, [{artist:'unknown',title:'unknown'}], {}).recommendations).toEqual([]);
  });
});

it('keeps duplicate-vector clusters reachable', () => {
  const vectors=Array.from({length:500},(_,i)=>i<250?[1,0]:[0,1]);
  const index=buildHnsw(vectors);
  expect(searchHnsw(index,vectors,[1,0],400)).toHaveLength(400);
});

it('uses web evidence only with sources and excludes positives beyond row 500', () => {
  const vectors=[[1,0],[0.9,0.1],[0,1]];
  const c:Catalog={format:'human-graph-v1',source:'test-only',tracks:vectors.map((_,i)=>({mbid:String(i),artist:'A'+i,title:'T'+i,popularity:0})),graph:vectors,graphIndex:buildHnsw(vectors),context:{}};
  const songs=Array.from({length:501},(_,i)=>({artist:'Missing',title:String(i)}));
  songs.push(c.tracks[1]);
  expect(recommend(c,songs,{}).recommendations.some(t=>t.mbid==='1')).toBe(false);
  const unknown={artist:'New',title:'New'};
  expect(recommend(c,[unknown],{}, {'new\u001fnew':{neighbors:['0'],sources:[]}}).coverage.graph).toBe(0);
  expect(recommend(c,[unknown],{}, {'new\u001fnew':{neighbors:['0'],sources:['https://example.org/tracklist']}}).coverage.graph).toBe(1);
  expect(recommend(c,[c.tracks[0],c.tracks[0]],{})).toEqual(recommend(c,[c.tracks[0]],{}));
});
