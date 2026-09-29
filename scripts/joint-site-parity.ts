import fs from 'node:fs';
import {resolveSongs,siteRecommend} from '../src/lib/joint/site';
import type {JointData} from '../src/lib/joint/runtime';
const root=process.argv[2],songs=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
function buffer(name:string){const b=fs.readFileSync(`${root}/${name}`);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}
const d:JointData={catalog:JSON.parse(fs.readFileSync(`${root}/catalog.json`,'utf8')),
 weights:JSON.parse(fs.readFileSync(`${root}/ranker.json`,'utf8')),vectors:new Float32Array(buffer('vectors.f32')),
 neighbors:new Int32Array(buffer('neighbors.i32')),info:new Float32Array(buffer('info.f32'))};
console.log(JSON.stringify({profile:resolveSongs(d.catalog,songs),result:siteRecommend(d,songs,{})}));
