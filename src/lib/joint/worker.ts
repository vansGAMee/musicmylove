import {loadJoint,recommend,type JointData} from './runtime';
let data:JointData|undefined;
self.onmessage=async(event:MessageEvent)=>{
  const {id,type,base,seeds,known}=event.data;
  try{
    if(type==='load'){data=await loadJoint(base);self.postMessage({id,ready:true,tracks:data.catalog.tracks});}
    else if(type==='recommend'&&data)self.postMessage({id,rows:recommend(data,seeds,known??[])});
    else throw Error('Load model before recommendation');
  }catch(error){self.postMessage({id,error:String(error)});}
};
