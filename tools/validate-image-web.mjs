// 仅使用Node标准库及本机浏览器验证二进制传输、会话复用和有界掩码缓存。
import fs from 'node:fs';
import path from 'node:path';
import {withBrowser} from './browser-cdp.mjs';

const base=process.argv[2]||'http://127.0.0.1:5201';
const output=process.argv[3]||'artifacts/validation/0.11.1/image-web.json';
const checks=[];const info=await fetch(base+'/api/info').then(r=>r.json());
if(!info.binaryRgb)throw new Error('Binary transport unavailable');
const model=info.models.find(m=>m.name==='yolov8n-seg.pt');
const metadata={modelId:model.id,backend:'Cpu',confidence:.25,width:810,height:1080};
const rgb=fs.readFileSync('.cache/image-optimization/fixtures/bus.rgb');
async function post(meta,body,extra={},binary=true){const response=await fetch(base+(binary?'/api/infer-rgb':'/api/infer'),{method:'POST',headers:binary?{'Content-Type':'application/octet-stream','X-SimdYolo':JSON.stringify(meta),...extra}:{'Content-Type':'application/json'},body:binary?body:JSON.stringify({...meta,pixels:rgb.toString('base64')}),...(body instanceof ReadableStream?{duplex:'half'}:{})});return {status:response.status,data:await response.json()};}
const legacy=await post(metadata,null,{},false),binary=await post(metadata,rgb);
if(legacy.status!==200||binary.status!==200)throw new Error('Valid inference failed');
let maxBoxError=0,maxConfidenceError=0,maxMaskByteError=0;
if(legacy.data.instances.length!==binary.data.instances.length)throw new Error('Transport changed instances');
for(let i=0;i<legacy.data.instances.length;i++){
  const a=legacy.data.instances[i],b=binary.data.instances[i];if(a.box.classId!==b.box.classId)throw new Error('Transport changed class');
  for(const key of ['x','y','width','height'])maxBoxError=Math.max(maxBoxError,Math.abs(a.box[key]-b.box[key]));
  maxConfidenceError=Math.max(maxConfidenceError,Math.abs(a.box.confidence-b.box.confidence));
  if(JSON.stringify({...a.mask,probabilities:null})!==JSON.stringify({...b.mask,probabilities:null}))throw new Error('Transport changed mask geometry');
  const ma=Buffer.from(a.mask.probabilities,'base64'),mb=Buffer.from(b.mask.probabilities,'base64');
  for(let j=0;j<ma.length;j++)maxMaskByteError=Math.max(maxMaskByteError,Math.abs(ma[j]-mb[j]));
}
if(maxBoxError>.01||maxConfidenceError>2e-5||maxMaskByteError>1)throw new Error('Legacy/binary numerical mismatch');
checks.push('Legacy JSON and RGB24 transport preserve instances, geometry and masks');
for(const confidence of [.7,1,.25]){const value=await post({...metadata,confidence},rgb);if(value.status!==200||!value.data.reusedSession||value.data.modelLoadMs!==0||value.data.instances.some(i=>i.box.confidence<confidence))throw new Error('Confidence session reuse failed');}
checks.push('Changing confidence reuses loaded model and obeys current threshold');
const tiny={...metadata,width:1,height:1};
const stream=bytes=>new ReadableStream({start(controller){controller.enqueue(new Uint8Array(bytes));controller.close();}});
for(const [name,meta,body,headers,expected] of [
  ['missing metadata',tiny,new Uint8Array(3),{'X-SimdYolo':''},400],
  ['metadata limit',tiny,new Uint8Array(3),{'X-SimdYolo':' '.repeat(4097)},400],
  ['malformed metadata',tiny,new Uint8Array(3),{'X-SimdYolo':'{oops'},400],
  ['upload dimensions',{...tiny,width:2049},new Uint8Array(3),{},400],
  ['RGB content length',tiny,new Uint8Array(2),{},400],
  ['chunked truncated RGB',tiny,stream(2),{},400],
  ['chunked surplus RGB',tiny,stream(4),{},400],
  ['paired original dimensions',{...tiny,originalWidth:1},new Uint8Array(3),{},400],
  ['unknown model',{...tiny,modelId:'unknown'},new Uint8Array(3),{},404],
  ['cross-site request',tiny,new Uint8Array(3),{Origin:'https://example.invalid'},403],
]){const result=await post(meta,body,headers);if(result.status!==expected)throw new Error(name+': '+result.status);checks.push('Reject '+name);}
const chunked=await post({...tiny,confidence:1},stream(3));if(chunked.status!==200)throw new Error('Valid chunked body rejected');checks.push('Valid chunked RGB body accepted');

const originalModule=fs.readFileSync('.cache/image-optimization/baseline/cutout.js').toString('base64');
const canvas=await withBrowser(async({call,evaluate,until,errors})=>{
  await call('Page.navigate',{url:base});await until(()=>evaluate('document.getElementById("inference-model").options.length>20'),'models');
  const result=await evaluate('(async()=>{const current=await import("/cutout.js?v=2"),previous=await import("data:text/javascript;base64,'+originalModule+'");const fail=message=>{throw new Error(message)};const encode=values=>btoa(String.fromCharCode(...values));const mask=values=>({x:.25,y:1.25,width:2,height:2,displayWidth:5.5,displayHeight:6.5,probabilities:encode(values)});const unpadded={...mask([128,255,127,0]),probabilities:encode([128,255,127,0]).replace(/=+$/,String())};const compatibility=new current.MaskCanvasCache();compatibility.get(unpadded);compatibility.clear();const first=mask([0,128,255,127]),cache=new current.MaskCanvasCache(64);const tile=cache.get(first);if(cache.get(first)!==tile||cache.decodedMasks!==1||cache.hits!==1)fail("No cache hit");const old=cache.get(first,[255,0,0],94);if(cache.get(first,[255,0,0],94)!==old)fail("Tint cache missed");first.probabilities=encode([255,255,255,255]);if(cache.get(first)===tile||tile.width!==0)fail("Mutation did not invalidate");cache.clear();if(cache.bytes!==0||cache.entries.size!==0)fail("Cache clear failed");const bounded=new current.MaskCanvasCache(32),retained=bounded.get(mask([255,0,0,0]));for(let i=0;i<5;i++)bounded.get(mask([i,128,255,127]));if(bounded.bytes>32||retained.width!==0)fail("LRU did not evict");bounded.clear();const temporary=new current.MaskCanvasCache(4),oversized=temporary.get(mask([255,255,255,255]));if(temporary.bytes!==0||temporary.entries.size!==0||oversized.width!==2)fail("Oversized entry retained or destroyed");const source=document.createElement("canvas");source.width=32;source.height=24;const ctx=source.getContext("2d");ctx.fillStyle="#90a0b0";ctx.fillRect(0,0,32,24);const instances=[{box:{classId:0},className:"person",mask:mask([0,128,255,127])},{box:{classId:5},className:"bus",mask:{...mask([255,0,128,255]),x:2.5,y:3.5}}];let compared=0;for(const target of ["people","vehicles","all","class:9"]){const a=document.createElement("canvas"),b=document.createElement("canvas"),c=document.createElement("canvas"),d=document.createElement("canvas");const oldResult=previous.renderCutouts(source,instances,32,24,target,a,b),newResult=current.renderCutouts(source,instances,32,24,target,c,d,cache);if(JSON.stringify(oldResult)!==JSON.stringify(newResult))fail("Cutout counts changed");for(const pair of [[a,c],[b,d]]){const x=pair[0].getContext("2d").getImageData(0,0,32,24).data,y=pair[1].getContext("2d").getImageData(0,0,32,24).data;for(let i=0;i<x.length;i++){if(x[i]!==y[i])fail("RGBA changed at "+i);compared++;}}}const decoded=cache.decodedMasks;current.renderCutouts(source,instances,32,24,"all",document.createElement("canvas"),document.createElement("canvas"),cache);if(cache.decodedMasks!==decoded)fail("Repeated composition decoded masks");cache.clear();return {rgbaBytesCompared:compared,exact:true,cacheBudget:true,mutationInvalidation:true,tintReuse:true};})()');
  if(errors.length)throw new Error(errors.join('\n'));return result;
});
checks.push('Mask cache hits, tint reuse, mutation invalidation, LRU budget and oversized temporary result');
checks.push('Original and optimized cutouts match every RGBA byte for fractional, overlapping and empty masks');
fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify({status:'passed',version:info.version,checks,maxBoxError,maxConfidenceError,maxMaskByteError,canvas},null,2));console.log(JSON.stringify({status:'passed',checks:checks.length,canvas}));
