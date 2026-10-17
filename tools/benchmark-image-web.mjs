// 对优化前/后实际页面顺序测量；包含RGB准备、HTTP、JSON解码、叠加和目标分离。
import fs from 'node:fs';
import path from 'node:path';
import {withBrowser} from './browser-cdp.mjs';

const output=process.argv[2]||'artifacts/validation/0.11.1/web-performance.json';
fs.mkdirSync(path.dirname(output),{recursive:true});fs.mkdirSync('.cache/image-optimization/fixtures',{recursive:true});
const cases=[{name:'bus',width:810,height:1080},{name:'large',width:2048,height:1536}];
const records=[];
await withBrowser(async({call,evaluate,until,files,errors})=>{
  for(const variant of [{name:'before',url:'http://127.0.0.1:5200'},{name:'after',url:'http://127.0.0.1:5201'}]){
    await call('Page.navigate',{url:variant.url});
    await until(()=>evaluate('document.getElementById("inference-model").options.length>20'),'models');
    await evaluate('window.benchmarkResponse=null;window.benchmarkRequest=null;window.benchmarkAtob=0;window.benchmarkRgbReads=0;const originalFetch=window.fetch,originalAtob=window.atob,originalRead=CanvasRenderingContext2D.prototype.getImageData;window.atob=(...args)=>{window.benchmarkAtob++;return originalAtob(...args)};CanvasRenderingContext2D.prototype.getImageData=function(...args){if(this.canvas.id!=="cutout-foreground"&&this.canvas.id!=="cutout-background")window.benchmarkRgbReads++;return originalRead.apply(this,args)};window.fetch=async(...args)=>{const infer=["/api/infer","/api/infer-rgb"].includes(String(args[0]));if(infer){window.benchmarkResponse=null;window.benchmarkRequest={transport:String(args[0]),bytes:typeof args[1].body==="string"?new TextEncoder().encode(args[1].body).length:args[1].body.byteLength};}const response=await originalFetch(...args);if(infer)window.benchmarkResponse=await response.clone().json();return response;};');
    for(const item of cases){
      const data=await evaluate('(async()=>{const image=new Image();image.src="/samples/bus.jpg";await image.decode();const c=document.createElement("canvas");c.width='+item.width+';c.height='+item.height+';const ctx=c.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,c.width,c.height);const scale=Math.min(c.width/image.width,c.height/image.height);ctx.drawImage(image,(c.width-image.width*scale)/2,(c.height-image.height*scale)/2,image.width*scale,image.height*scale);const rgba=ctx.getImageData(0,0,c.width,c.height).data,rgb=new Uint8Array(c.width*c.height*3);for(let i=0,j=0;i<rgba.length;i+=4){rgb[j++]=rgba[i];rgb[j++]=rgba[i+1];rgb[j++]=rgba[i+2];}const parts=[];for(let i=0;i<rgb.length;i+=8192)parts.push(String.fromCharCode(...rgb.subarray(i,i+8192)));return {rgb:btoa(parts.join("")),image:c.toDataURL("image/png").split(",")[1]};})()');
      const fixture='.cache/image-optimization/fixtures/'+item.name;
      if(variant.name==='before'){fs.writeFileSync(fixture+'.rgb',Buffer.from(data.rgb,'base64'));fs.writeFileSync(fixture+'.png',Buffer.from(data.image,'base64'));}
      await files('#image-file',[fixture+'.png']);await until(()=>evaluate('!document.getElementById("run-inference").disabled && document.getElementById("inference-canvas").width === '+item.width+' && document.getElementById("inference-canvas").height === '+item.height),'image dimensions');
      for(const backend of ['Cpu','Vulkan']){
        await evaluate('(async()=>{const info=await fetch("/api/info").then(r=>r.json());document.getElementById("inference-model").value=info.models.find(m=>m.name==="yolov8n-seg.pt").id;document.getElementById("inference-model").dispatchEvent(new Event("change"));document.querySelector("#inference-backends input[value='+backend+']").checked=true;})()');
        const samples=[];
        for(let iteration=0;iteration<8;iteration++){
          const sample=await evaluate('(async()=>{const start=performance.now(),atob=window.benchmarkAtob,reads=window.benchmarkRgbReads;window.benchmarkResponse=null;document.getElementById("run-inference").click();while(document.getElementById("run-inference").disabled||!window.benchmarkResponse)await new Promise(r=>setTimeout(r,2));if(!document.getElementById("inference-error").hidden)throw new Error(document.getElementById("inference-error").textContent);return {clientMs:performance.now()-start,serverMs:window.benchmarkResponse.elapsedMs,reusedSession:window.benchmarkResponse.reusedSession,instances:window.benchmarkResponse.instances.length,transport:window.benchmarkRequest.transport,requestBytes:window.benchmarkRequest.bytes,maskDecodes:window.benchmarkAtob-atob,canvasReads:window.benchmarkRgbReads-reads};})()');
          if(sample.requestBytes < item.width*item.height*3 || !sample.reusedSession && iteration>=3)throw new Error('Unexpected dimensions or cold session');
          if(variant.name==='after'&&(sample.transport!=='/api/infer-rgb'||sample.requestBytes!==item.width*item.height*3))throw new Error('Binary transport was not exercised');
          if(iteration>=3)samples.push(sample);
        }
        const switched=await evaluate('(()=>{const atob=window.benchmarkAtob,start=performance.now();for(let i=0;i<9;i++){document.getElementById("cutout-target").value=["vehicles","people","all"][i%3];document.getElementById("cutout-target").dispatchEvent(new Event("change"));}return {meanMs:(performance.now()-start)/9,maskDecodes:window.benchmarkAtob-atob};})()');
        if(variant.name==='after'&&switched.maskDecodes!==0)throw new Error('Expected masks to fit the bounded cache');
        const ordered=samples.map(sample=>sample.clientMs).sort((a,b)=>a-b);
        records.push({variant:variant.name,url:variant.url,...item,backend,samples,medianClientMs:ordered[2],switchTarget:switched});
        console.log(JSON.stringify({variant:variant.name,image:item.name,backend,clientMs:ordered[2],requestBytes:samples[0].requestBytes,maskDecodes:samples[0].maskDecodes,switchMaskDecodes:switched.maskDecodes}));
      }
    }
  }
  if(errors.length)throw new Error(errors.join('\n'));
});
fs.writeFileSync(output,JSON.stringify({status:'passed',browser:'Headless Edge --disable-gpu (canvas); .NET Vulkan uses actual GPU',warmup:3,measured:5,scope:'Actual UI; client includes 2ms completion polling and response-clone instrumentation on both variants. Development diagnostic only.',records},null,2));
