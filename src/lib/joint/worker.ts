import {loadJoint,recommend,type JointData} from './runtime';
import {siteRecommend,siteSearch} from './site';
let data:JointData|undefined;
self.onmessage=async(event:MessageEvent)=>{
  const {id,type,base,seeds,known}=event.data;
  try{
    if(type==='load'){data=await loadJoint(base);self.postMessage({id,ready:true});}
    else if(type==='search'&&data)self.postMessage({id,result:siteSearch(data,event.data.query)});
    else if(type==='siteRecommend'&&data)self.postMessage({id,result:siteRecommend(data,event.data.songs,event.data.feedback??{})});
    else if(type==='recommend'&&data)self.postMessage({id,rows:recommend(data,seeds,known??[])});
    else throw Error('Load model before recommendation');
  }catch(error){self.postMessage({id,error:String(error)});}
};
