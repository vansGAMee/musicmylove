import {describe,it,expect} from 'vitest';
import {resolveSongs, siteRecommend, siteSearch} from '../../src/lib/joint/site';
import {recommend, type JointData} from '../../src/lib/joint/runtime';

function fixture():JointData {
  return {catalog:{tracks:[
    {id:'a',artist:'Alpha',title:'Song'},
    {id:'b',artist:'Beta',title:'Track',aliases:[['Béta','Alias']]},
    {id:'c',artist:'Gamma',title:'Track'},
    {id:'d',artist:'Gamma',title:'Track'},
    {id:'e',artist:'Other',title:'New'},
  ],artists:[0,1,2,2,3],families:[0,1,2,2,4]},
  vectors:new Float32Array(5*32).fill(.1),neighbors:new Int32Array(5*24).map((_,i)=>i%24<5?i%24:-1),
  info:new Float32Array(10),weights:{w1:Array.from({length:16},()=>Array(6).fill(0)),b1:Array(16).fill(0),w2:[Array(16).fill(0)],b2:[0]}};
}
describe('joint website uses exported model',()=>{
  it('resolves aliases/mastering but refuses ambiguous names and protects all matches',()=>{
    const d=fixture();const r=resolveSongs(d.catalog,[{artist:'Béta',title:'Alias - Remastered 2011'},{artist:'Gamma',title:'Track'}]);
    expect(r.seeds).toEqual([1]);expect(r.known).toEqual([1,2,3]);expect(r.unresolved).toHaveLength(1);
  });
  it('preserves neural order; dislikes exclude whole families without retraining',()=>{
    const d=fixture(),songs=[{artist:'Alpha',title:'Song'}];
    const result=siteRecommend(d,songs,{c:'dislike'});
    expect(result.recommendations.map(r=>r.mbid)).toEqual(recommend(d,[0],[0,2]).map(r=>d.catalog.tracks[r.index].id));
    expect(result.recommendations.every(r=>r.strongestTasteHead===undefined)).toBe(true);
    expect(siteRecommend(d,songs,{}).recommendations).toEqual(siteRecommend(d,songs,{b:'like'}).recommendations);
  });
  it('does not invent recommendations for unmatched input',()=>{
    expect(()=>siteRecommend(fixture(),[{artist:'Absent',title:'Missing'}],{})).toThrow(/Распознано/);
  });
  it('search only offers identities the name-based input can actually resolve',()=>{
    expect(siteSearch(fixture(),'Gamma')).toEqual([]);
    expect(siteSearch(fixture(),'Alpha').map(t=>t.mbid)).toEqual(['a']);
  });
});
