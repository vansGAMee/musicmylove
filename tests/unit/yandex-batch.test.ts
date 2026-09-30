import { expect, it, vi } from 'vitest';
import { importYandexBatch } from '../../services/yandex-import';
it('imports UUID playlists through the API and resolves bare IDs in batches',async()=>{
 const fetcher=vi.fn(async(input:RequestInfo|URL,init?:RequestInit)=>{
  if(String(input).includes('/playlist/'))return Response.json({result:{title:'List',trackCount:2,tracks:[{id:'1'},{id:'2'}]}});
  expect(String(input)).toBe('https://api.music.yandex.net/tracks');
  expect(new URLSearchParams(String(init?.body)).get('track-ids')).toBe('1,2');
  return Response.json({result:[{id:'2',title:'Two',artists:[{name:'B'}]},{id:'1',title:'One',artists:[{name:'A'}]}]});
 });
 const result=await importYandexBatch('https://music.yandex.ru/playlists/abcdef',fetcher);
 expect(result.tracks.map(t=>t.title)).toEqual(['One','Two']);expect(fetcher).toHaveBeenCalledTimes(2);
});
it('never reports a partial playlist as success',async()=>{
 const fetcher=vi.fn(async()=>Response.json({result:{trackCount:3,tracks:[{id:'1',title:'One',artists:[{name:'A'}]}]}}));
 await expect(importYandexBatch('https://music.yandex.ru/playlists/abcdef',fetcher)).rejects.toThrow(/неполный/);
});
it('rejects hostile hosts before any request',async()=>{
 const fetcher=vi.fn();await expect(importYandexBatch('https://music.yandex.ru.evil.test/playlists/x',fetcher)).rejects.toThrow();expect(fetcher).not.toHaveBeenCalled();
});
