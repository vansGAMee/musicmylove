/** Build locally: publish compact inference assets, never Python/history/CLAP. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';

const source=path.resolve(process.argv[2]??'python_mvp/data/joint-v1/browser');
const target=path.resolve('public/joint-model');
if(source===target)throw Error('Choose the original browser export as source');
const manifest=JSON.parse(await fs.readFile(path.join(source,'manifest.json'),'utf8'));
if(manifest.schema!=='joint-browser-v1'||manifest.bytes>32*1024*1024)throw Error('Invalid model export');
const files=['vectors.f32','neighbors.i32','info.f32','catalog.json','ranker.json'];
const contents=new Map<string,Buffer>();
for(const file of files){
  const bytes=await fs.readFile(path.join(source,file));
  if(bytes.length!==manifest.sizes[file]||createHash('sha256').update(bytes).digest('hex')!==manifest.sha256[file])throw Error(`Model checksum mismatch: ${file}`);
  contents.set(file,bytes);
}
if([...contents.values()].reduce((sum,b)=>sum+b.length,0)!==manifest.bytes)throw Error('Wrong model payload size');
await fs.rm(target,{recursive:true,force:true});await fs.mkdir(target,{recursive:true});
for(const [file,bytes] of contents)await fs.writeFile(path.join(target,file),bytes);
await fs.copyFile(path.join(source,'manifest.json'),path.join(target,'manifest.json'));
console.log(`Frozen model: ${manifest.tracks} tracks, ${(manifest.bytes/1024/1024).toFixed(1)} MiB; ${manifest.status}. No quality upgrade is asserted.`);
const build=spawnSync('node_modules/.bin/next',['build','--webpack'],{stdio:'inherit',env:{...process.env,NEXT_PUBLIC_JOINT_MODEL:'1'}});
if(build.status!==0)process.exit(build.status??1);
// Static Build Output API; only the generated website is deployed by --prebuilt.
await fs.rm('.vercel/output',{recursive:true,force:true});
await fs.mkdir('.vercel/output',{recursive:true});
await fs.cp('out','.vercel/output/static',{recursive:true,filter:src=>path.relative('out',src).split(path.sep)[0]!=='data'});
await fs.writeFile('.vercel/output/config.json',JSON.stringify({version:3,routes:[{handle:'filesystem'}]},null,2));
console.log('Ready for a PREVIEW: npx vercel deploy --prebuilt');
