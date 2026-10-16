// 开发验证：只驱动浏览器；模型、标注、指标、校准与导出均由.NET计算。
import fs from 'node:fs';
import path from 'node:path';

export async function validateWorkflows({base,destination,evaluate,call,waitFor,screenshot,checks,info,imagePath}) {
  const model=info.models.find(item=>item.name==='yolov8n.pt');
  const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const source='data:image/jpeg;base64,'+fs.readFileSync(imagePath).toString('base64');
  const created=await evaluate('(async()=>{'+
    'const image=new Image();image.src='+JSON.stringify(source)+';await image.decode();const images=[];'+
    'for(let index=0;index<3;index++){const c=document.createElement("canvas");c.width=240+index;c.height=320;const ctx=c.getContext("2d");ctx.drawImage(image,0,0,c.width,c.height);'+
    'const rgba=ctx.getImageData(0,0,c.width,c.height).data,rgb=new Uint8Array(c.width*c.height*3);for(let i=0,j=0;i<rgba.length;i+=4){rgb[j++]=rgba[i];rgb[j++]=rgba[i+1];rgb[j++]=rgba[i+2];}'+
    'const chunks=[];for(let i=0;i<rgb.length;i+=8192)chunks.push(String.fromCharCode(...rgb.subarray(i,i+8192)));images.push({name:"bus-workflow-"+index+".jpg",width:c.width,height:c.height,pixels:btoa(chunks.join("")),labels:""});}'+
    'const r=await fetch("/api/datasets",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:"网页工作流测试（非独立精度基准）",classes:["person","bus"],images})});if(!r.ok)throw new Error(await r.text());return r.json();})()');
  const set=async(id,value,change=false)=>evaluate('document.getElementById('+JSON.stringify(id)+').value='+JSON.stringify(value)+';'+(change?'document.getElementById('+JSON.stringify(id)+').dispatchEvent(new Event("change"));':''));
  const click=id=>evaluate('document.getElementById('+JSON.stringify(id)+').click()');
  const request=async(url,body,method='POST')=>{const response=await fetch(base+url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!response.ok)throw new Error(await response.text());return response.json();};
  await click('refresh-datasets');await evaluate('location.hash="annotation"');
  await waitFor(()=>evaluate('!!document.querySelector("#annotation-dataset option[value=\\"'+created.id+'\\"]")'),'Annotation dataset list refreshed');
  await set('annotation-dataset',created.id,true);
  await waitFor(()=>evaluate('document.getElementById("annotation-status").textContent.includes("bus-workflow-0")'),'Actual RGB loaded into annotation canvas');
  await evaluate('document.getElementById("annotation-canvas").scrollIntoView({block:"start"})');await pause(200);
  const bounds=await evaluate('(()=>{const r=document.getElementById("annotation-canvas").getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()');
  async function pointer(type,x,y,buttons=0){await call('Input.dispatchMouseEvent',{type,x:bounds.x+bounds.w*x,y:bounds.y+bounds.h*y,button:type==='mouseMoved'?'none':'left',buttons,clickCount:type==='mouseMoved'?0:1});}
  await pointer('mousePressed',.2,.2,1);await pointer('mouseMoved',.45,.55,1);await pointer('mouseReleased',.45,.55);
  await waitFor(()=>evaluate('document.querySelectorAll("#annotation-boxes button").length===1 && document.getElementById("annotation-status").textContent.includes("未保存")'),'Pointer draw created actual annotation');
  await set('annotation-class','1',true);await click('annotation-save');
  const saved=()=>evaluate('document.getElementById("annotation-status").textContent.includes("已保存") && !document.getElementById("annotation-status").textContent.includes("处理中")');
  await waitFor(saved,'Edited class and box saved');
  const stored=await fetch(base+'/api/datasets/'+created.id+'/images/0').then(r=>r.json()),parts=stored.labels.trim().split(/\s+/).map(Number);
  if(parts[0]!==1||Math.abs(parts[1]-.325)>.01||Math.abs(parts[2]-.375)>.01)throw new Error('Saved annotation differs from pointer operation: '+stored.labels);
  const stale=await fetch(base+'/api/datasets/'+created.id+'/images/0/labels',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({labels:stored.labels,revision:'stale'})});
  if(stale.status!==409)throw new Error('Stale annotation revision accepted');
  await set('annotation-model',model.id);await click('annotation-prelabel');
  await waitFor(()=>evaluate('document.getElementById("annotation-status").textContent.includes("未保存") && !document.getElementById("annotation-status").textContent.includes("处理中") && document.querySelectorAll("#annotation-boxes button").length>1'),'Real model generated editable prelabels',90000);
  const unchanged=await fetch(base+'/api/datasets/'+created.id+'/images/0').then(r=>r.json());
  if(unchanged.labels!==stored.labels)throw new Error('Prelabels saved without user action');
  await screenshot('web-annotation.png');await click('annotation-save');await waitFor(saved,'Reviewed prelabels persisted');
  const zip=await fetch(base+'/api/datasets/'+created.id+'/export').then(r=>r.arrayBuffer());
  if(new DataView(zip).getUint32(0,true)!==0x04034b50)throw new Error('YOLO export is not ZIP');
  fs.writeFileSync(path.join(destination,'annotation-dataset.zip'),Buffer.from(zip));
  checks.push('Annotation pointer/class/save, stale revision rejection, actual prelabel review, YOLO PNG ZIP export');
  await evaluate('location.hash="training"');await set('training-dataset',created.id,true);await click('training-split');
  await waitFor(()=>evaluate('!!document.getElementById("training-validation").value'),'UI created disjoint training and validation datasets');
  await set('training-model',model.id);await set('training-epochs','1');await set('training-size','128');await set('training-patience','1');await click('start-training');
  await waitFor(()=>evaluate('document.getElementById("training-status").textContent.includes("已完成") && !document.getElementById("download-best").classList.contains("disabled") && document.getElementById("training-validation-summary").textContent.includes("→")'),'Training evaluated baseline/epoch and registered best model',90000);
  const training=await fetch(base+'/api/training').then(r=>r.json()),trained=training.find(job=>job.validationHistory.length>=2&&job.status==='completed');
  if(!trained?.bestModelId)throw new Error('Best model not registered');
  const validationId=await evaluate('document.getElementById("training-validation").value');
  await screenshot('web-training-validation.png');
  await evaluate('location.hash="evaluate"');await set('evaluation-dataset',validationId);await set('evaluation-model',model.id);await set('evaluation-other',trained.bestModelId);await set('evaluation-size','128');await click('evaluation-compare');
  await waitFor(()=>evaluate('document.getElementById("workflow-status").textContent.includes("模型比较 · 已完成") && document.querySelector("#evaluation-results tbody")?.rows.length===2'),'Web compared original and real best checkpoint',90000);
  await screenshot('web-evaluation.png');
  const compared=(await fetch(base+'/api/workflows').then(r=>r.json())).find(job=>job.request.kind==='compare'&&job.status==='completed');
  const report=await fetch(base+compared.downloadUrl).then(r=>r.json());
  if(!report.before.samples.length||report.datasetSha256.length!==64)throw new Error('Comparison lacks actual samples/provenance');
  fs.writeFileSync(path.join(destination,'web-model-comparison.json'),JSON.stringify(report,null,2));
  await set('tools-model',trained.bestModelId);await set('tools-size','128');await click('tools-export');
  await waitFor(()=>evaluate('document.getElementById("workflow-status").textContent.includes("ONNX导出 · 已完成")'),'Web exported best checkpoint as ONNX',90000);
  const exported=(await fetch(base+'/api/workflows').then(r=>r.json())).find(job=>job.request.kind==='export'&&job.status==='completed');
  if(!exported?.result?.modelId||!exported.downloadUrl)throw new Error('Export lacks registered model/download');
  await set('tools-model',model.id);await set('calibration-dataset',created.id);await click('tools-calibrate');
  await waitFor(()=>evaluate('document.getElementById("workflow-status").textContent.includes("INT8校准 · 已完成")'),'Web generated static INT8 calibration',90000);
  const calibration=(await fetch(base+'/api/workflows').then(r=>r.json())).find(job=>job.request.kind==='calibrate'&&job.status==='completed');
  await evaluate('location.hash="inference"');await set('inference-model',model.id,true);await set('inference-precision','Int8',true);
  await waitFor(()=>evaluate('!!document.querySelector("#inference-calibration option[value=\\"'+calibration.id+'\\"]")'),'Calibration available in inference UI');
  await set('inference-calibration',calibration.id);await click('run-inference');
  await waitFor(()=>evaluate('document.getElementById("actual-backend").textContent.includes("INT8") && !document.getElementById("run-inference").disabled'),'Actual calibrated INT8 inference displayed',90000);
  const forced=await evaluate('({cpu:document.querySelector("#inference-backends input[value=Cpu]").checked,gpuDisabled:[...document.querySelectorAll("#inference-backends input")].filter(x=>x.value!=="Cpu").every(x=>x.disabled),error:document.getElementById("inference-error").textContent})');
  if(!forced.cpu||!forced.gpuDisabled||forced.error)throw new Error('Low precision device UI inconsistent '+JSON.stringify(forced));
  await screenshot('web-int8-inference.png');
  await set('inference-precision','Float16',true);await click('run-inference');
  await waitFor(()=>evaluate('document.getElementById("actual-backend").textContent.includes("FP16") && !document.getElementById("run-inference").disabled'),'Actual FP16 mixed inference displayed',90000);
  await set('inference-precision','Float32',true);await set('training-validation','',true);
  checks.push('Real holdout validation and best checkpoint; full model comparison; ONNX registration; static INT8 and FP16 CPU execution');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  for(const view of ['annotation','evaluate']){
    await evaluate('location.hash='+JSON.stringify(view));await pause(200);
    const size=await evaluate('({viewport:innerWidth,width:document.documentElement.scrollWidth})');
    if(size.width>size.viewport+1)throw new Error('New view mobile overflow '+view+' '+JSON.stringify(size));
  }
  await screenshot('web-workflows-mobile.png');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  return {status:'passed',datasetId:created.id,trainingId:trained.id,comparisonId:compared.id,exportId:exported.id,calibrationId:calibration.id,prelabelReview:true,pureManagedServer:true};
}
