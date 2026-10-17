// 使用 Node 内置 CDP 连接自有无头浏览器，验证真实页面操作；无需 npm 依赖。
import {parseArgs} from 'node:util';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {validateWorkflows} from './validate-web-workflows.mjs';

const {values}=parseArgs({options:{url:{type:'string',default:'http://127.0.0.1:5188'},image:{type:'string'},output:{type:'string',default:'artifacts/validation/0.10.0'},'training-model':{type:'string',default:'yolov8n.pt'},'large-images':{type:'boolean',default:false},cutout:{type:'boolean',default:false},features:{type:'boolean',default:false},browser:{type:'string',default:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}}});
if(!values.image)throw new Error('--image 必须指定一张真实目标照片，例如官方 bus.jpg');
const base=values.url.replace(/\/$/,''), root=process.cwd(), destination=path.resolve(values.output);
const profile=path.join(root,'.cache/training-validation/edge-profile-'+Date.now());
fs.mkdirSync(destination,{recursive:true});
const browser=spawn(values.browser,[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
  '--user-data-dir='+profile,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let socket, next=1; const pending=new Map(), errors=[], checks=[], largeImages=[];let cutout=null;
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(fn, label, ms=45000){const start=Date.now();while(Date.now()-start<ms){try{if(await fn()){checks.push(label);return;}}catch{}await pause(150);}throw new Error('Timeout: '+label);}
function call(method,params={}){return new Promise((resolve,reject)=>{const id=next++; const timeout=setTimeout(()=>{pending.delete(id);reject(new Error(method+' timed out'));},30000); pending.set(id,{resolve,reject,timeout});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const result=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw new Error(result.exceptionDetails.text+' '+JSON.stringify(result.exceptionDetails.exception));return result.result.value;}
async function screenshot(name){const response=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(path.join(destination,name),Buffer.from(response.data,'base64'));}
async function setFiles(selector,files){const {root:document}=await call('DOM.getDocument');const {nodeId}=await call('DOM.querySelector',{nodeId:document.nodeId,selector});await call('DOM.setFileInputFiles',{nodeId,files:files.map(p=>path.resolve(p))});}
try{
  let targets;
  await waitFor(async()=>{const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];targets=await fetch(`http://127.0.0.1:${port}/json/list`).then(r=>r.json());return targets.some(t=>t.type==='page');},'Headless Edge started',15000);
  socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.id){const request=pending.get(value.id);if(!request)return;pending.delete(value.id);clearTimeout(request.timeout);value.error?request.reject(new Error(JSON.stringify(value.error))):request.resolve(value.result);}else if(value.method==='Runtime.exceptionThrown')errors.push(value.params.exceptionDetails.text);else if(value.method==='Log.entryAdded'&&value.params.entry.level==='error')errors.push(value.params.entry.text);};
  await call('Page.enable');await call('Runtime.enable');await call('Log.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:base});
  await waitFor(()=>evaluate("document.querySelector('#inference-model')?.options.length>20 && document.querySelector('#connection-text')?.textContent.includes('连接')"),'Actual API populated models');
  const info=await fetch(base+'/api/info').then(r=>r.json());
  const model=info.models.find(m=>m.name==='yolov8n.pt');
  const trainingModel=info.models.find(m=>m.name===values['training-model']);
  if(!trainingModel?.trainable)throw new Error('Requested training model unavailable');
  for(const name of ['yolov8n.pt','yolov9t.pt','yolov10n.pt','yolo11n.pt','yolo26n.pt'])if(!info.models.some(m=>m.name===name&&m.trainable))throw new Error('Missing training family '+name);
  checks.push('All five training families exposed');
  await evaluate(`document.querySelector('#inference-model').value=${JSON.stringify(model.id)};document.querySelector('#inference-model').dispatchEvent(new Event('change',{bubbles:true}));`);
  await setFiles('#image-file',[values.image]);
  await waitFor(()=>evaluate("!document.querySelector('#run-inference').disabled"),'Real JPEG decoded in browser');
  await evaluate("document.querySelector('#run-inference').click()");
  await waitFor(()=>evaluate("Number(document.querySelector('#instance-count').textContent)>0"),'Actual inference boxes displayed');
  checks.push('Inference instances '+await evaluate("document.querySelector('#instance-count').textContent"));
  const bounds=await evaluate("(()=>{const c=document.querySelector('#inference-canvas').getBoundingClientRect(),s=document.querySelector('#image-dropzone').getBoundingClientRect();return {canvas:{width:c.width,height:c.height},stage:{width:s.width,height:s.height}}})()");
  if(bounds.canvas.width>bounds.stage.width+1||bounds.canvas.height>bounds.stage.height+1)throw new Error('Image preview cropped '+JSON.stringify(bounds));
  checks.push('Portrait image fully fits preview');
  await screenshot('web-inference.png');
  if(values['large-images']){
    // 用真实照片生成大尺寸JPEG，并操作实际文件输入框；禁止前端读取超出有界尺寸的RGBA。
    await evaluate(`(()=>{
      const originalFetch=window.fetch;
      window.largeImageReadbacks=[];
      window.largeImageDraws={boxes:[],points:[],masks:[]};
      window.largeImageOriginalFetch=(...args)=>originalFetch(...args);
      window.fetch=async (...args)=>{
        const inference=['/api/infer','/api/infer-rgb'].includes(String(args[0]));
        if(inference){
          const binary=String(args[0])==='/api/infer-rgb';
          if(binary){const bytes=args[1].body,parts=[];for(let i=0;i<bytes.length;i+=8192)parts.push(String.fromCharCode(...bytes.subarray(i,i+8192)));window.largeImageRequest={...JSON.parse(args[1].headers['X-SimdYolo']),pixels:btoa(parts.join(''))};}
          else window.largeImageRequest=JSON.parse(args[1].body);
          window.largeImageResponse=null;
        }
        const response=await originalFetch(...args);
        if(inference)window.largeImageResponse=await response.clone().json();
        return response;
      };
      const read=CanvasRenderingContext2D.prototype.getImageData;
      CanvasRenderingContext2D.prototype.getImageData=function(...args){
        window.largeImageReadbacks.push({width:args[2],height:args[3]});
        if(args[2]>2048||args[3]>2048)throw new Error('Unbounded RGBA readback');
        return read.apply(this,args);
      };
      for(const [method,kind] of [['strokeRect','boxes'],['arc','points'],['drawImage','masks']]){
        const draw=CanvasRenderingContext2D.prototype[method];
        CanvasRenderingContext2D.prototype[method]=function(...args){
          if(this.canvas.id==='inference-canvas'&&(method!=='drawImage'||args.length===5)){
            const matrix=this.getTransform();window.largeImageDraws[kind].push({scaleX:matrix.a,scaleY:matrix.d});
          }
          return draw.apply(this,args);
        };
      }
    })()`);
    const source='data:image/jpeg;base64,'+fs.readFileSync(values.image).toString('base64');
    const cases=[
      {name:'4k-detection-cpu',width:4001,height:5333,model:'yolov8n.pt',backend:'Cpu'},
      {name:'8k-segmentation-cpu',width:8191,height:5463,model:'yolov8n-seg.pt',backend:'Cpu'},
      {name:'8k-pose-vulkan',width:6143,height:8191,model:'yolov8n-pose.pt',backend:'Vulkan'},
    ];
    const fixtures=path.join(root,'.cache/web-large-images/fixtures');fs.mkdirSync(fixtures,{recursive:true});
    for(const item of cases){
      if(!info.backends.some(b=>b.id===item.backend&&b.inference))throw new Error('Required backend unavailable: '+item.backend);
      const jpeg=await evaluate(`(async()=>{
        const image=new Image();image.src=${JSON.stringify(source)};await image.decode();
        const canvas=document.createElement('canvas');canvas.width=${item.width};canvas.height=${item.height};
        const ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
        const ratio=Math.min(canvas.width/image.width,canvas.height/image.height),w=image.width*ratio,h=image.height*ratio;
        ctx.drawImage(image,(canvas.width-w)/2,(canvas.height-h)/2,w,h);
        const result=canvas.toDataURL('image/jpeg',.91);canvas.width=canvas.height=0;return result.split(',')[1];
      })()`);
      const file=path.join(fixtures,item.name+'.jpg');fs.writeFileSync(file,Buffer.from(jpeg,'base64'));
      const selected=info.models.find(m=>m.name===item.model);if(!selected)throw new Error('Model unavailable '+item.model);
      await evaluate(`document.querySelector('#inference-model').value=${JSON.stringify(selected.id)};document.querySelector('#inference-model').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#inference-backends input[value="${item.backend}"]').checked=true;`);
      await setFiles('#image-file',[file]);
      await waitFor(()=>evaluate(`document.querySelector('#image-details').textContent.includes('${item.width} × ${item.height}') && !document.querySelector('#run-inference').disabled`),'Large image decoded '+item.name);
      const preview=await evaluate(`({width:document.querySelector('#inference-canvas').width,height:document.querySelector('#inference-canvas').height})`);
      if(Math.max(preview.width,preview.height)>2048)throw new Error('Unbounded preview '+JSON.stringify(preview));
      await evaluate("window.largeImageDraws={boxes:[],points:[],masks:[]};document.querySelector('#run-inference').click()");
      await waitFor(()=>evaluate("window.largeImageResponse && !document.querySelector('#run-inference').disabled && document.querySelector('#image-state').textContent.includes('已完成')"),'Large image inference '+item.name,90000);
      const metrics=await evaluate(`(async()=>{
        const request=window.largeImageRequest,result=window.largeImageResponse;
        if(request.originalWidth!==${item.width}||request.originalHeight!==${item.height}||Math.max(request.width,request.height)>2048)throw new Error('Original/upload dimensions mismatch');
        if(result.image?.coordinateSystem!=='original'||!result.image.resized||!result.instances.length)throw new Error('Missing mapped results');
        const {originalWidth,originalHeight,...legacy}=request;
        const response=await window.largeImageOriginalFetch('/api/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(legacy)});
        if(!response.ok)throw new Error('Legacy RGB request rejected');
        const reference=await response.json(),sx=originalWidth/request.width,sy=originalHeight/request.height;
        const draws=window.largeImageDraws;
        if(draws.boxes.length!==result.instances.length)throw new Error('Boxes were not drawn');
        for(const draw of [...draws.boxes,...draws.points,...draws.masks])if(Math.abs(draw.scaleX-1/sx)>1e-6||Math.abs(draw.scaleY-1/sy)>1e-6)throw new Error('Canvas overlay transform mismatch');
        if(reference.instances.length!==result.instances.length)throw new Error('Instance count changed');
        // 两次实际前向可能存在并行FP32归约的末位差异；按像素误差验证坐标，单独记录置信度/量化掩码误差。
        let maxCoordinateError=0,maxConfidenceError=0,maxMaskByteError=0,masks=0,keypoints=0;
        const compare=(actual,expected)=>{maxCoordinateError=Math.max(maxCoordinateError,Math.abs(actual-expected));};
        const confidence=(a,b)=>{maxConfidenceError=Math.max(maxConfidenceError,Math.abs(a-b));};
        result.instances.forEach((instance,index)=>{
          const expected=reference.instances[index];
          if(instance.box.classId!==expected.box.classId)throw new Error('Class changed');
          confidence(instance.box.confidence,expected.box.confidence);
          for(const field of ['x','width'])compare(instance.box[field],expected.box[field]*sx);
          for(const field of ['y','height'])compare(instance.box[field],expected.box[field]*sy);
          instance.keypoints.forEach((p,k)=>{const q=expected.keypoints[k];compare(p.x,q.x*sx);compare(p.y,q.y*sy);confidence(p.confidence,q.confidence);keypoints++;});
          if(instance.mask){
            const a=instance.mask,b=expected.mask;
            if(a.width!==b.width||a.height!==b.height)throw new Error('Mask bitmap size changed');
            const ap=atob(a.probabilities),bp=atob(b.probabilities);
            if(ap.length!==bp.length||ap.length!==a.width*a.height)throw new Error('Mask bitmap was expanded');
            for(let k=0;k<ap.length;k++)maxMaskByteError=Math.max(maxMaskByteError,Math.abs(ap.charCodeAt(k)-bp.charCodeAt(k)));
            compare(a.x,b.x*sx);compare(a.y,b.y*sy);compare(a.displayWidth,b.width*sx);compare(a.displayHeight,b.height*sy);masks++;
          }
        });
        if(maxCoordinateError>.01)throw new Error('Coordinate mapping error '+maxCoordinateError);
        if(maxConfidenceError>1e-5||maxMaskByteError>1)throw new Error('FP32 repeat inference tolerance exceeded');
        if(masks&&!draws.masks.length||keypoints&&!draws.points.length)throw new Error('Mask/pose overlays were not drawn');
        return {originalWidth,originalHeight,uploadWidth:request.width,uploadHeight:request.height,encodedRgbBytes:request.pixels.length,instances:result.instances.length,masks,keypoints,maxCoordinateError,maxConfidenceError,maxMaskByteError,canvasDraws:{boxes:draws.boxes.length,points:draws.points.length,masks:draws.masks.length},canvasTransformsVerified:true,readbacks:window.largeImageReadbacks.slice(),reusedSession:reference.reusedSession};
      })()`);
      if(item.model.includes('-seg')&&!metrics.masks)throw new Error('No segmentation masks');
      if(item.model.includes('-pose')&&!metrics.keypoints)throw new Error('No pose keypoints');
      largeImages.push({...item,...metrics,preview,passed:true});
      await screenshot(item.name+'.png');console.log(JSON.stringify({largeImagePassed:item.name,...metrics}));
    }
    // 元数据只控制坐标变换：负值、缺失配对、溢出和超大总面积必须明确拒绝。
    // 负例从Node发送，避免预期HTTP 400混入页面的浏览器错误记录。
    const invalid=[];
    const invalidBase={modelId:model.id,backend:'Cpu',width:1,height:1,pixels:'AAAA',originalWidth:6000,originalHeight:4000};
    for(const item of [{originalWidth:0},{originalHeight:-1},{originalWidth:65536},{originalWidth:20000,originalHeight:20000},{originalWidth:2147483647},{originalHeight:null},{width:0},{width:2049},{pixels:'AA'}]){
      const response=await fetch(base+'/api/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...invalidBase,...item})});
      invalid.push(response.status);await response.text();
    }
    if(invalid.some(code=>code!==400))throw new Error('Invalid image request accepted '+JSON.stringify(invalid));
    checks.push('Original-coordinate parity for 4K detection, 8K masks and 8K Vulkan pose; bounded previews/RGB; legacy API; nine invalid requests rejected');
  }
  if(values.cutout){
    // 使用实际示例按钮，确认检测模型自动切换为实例分割，车辆仅靠真实轮廓抠图。
    await evaluate("document.querySelector('#cutout-target').value='vehicles';document.querySelector('#load-cutout-sample').click()");
    await waitFor(()=>evaluate("!document.querySelector('#cutout-results').hidden && !document.querySelector('#download-foreground').disabled"),'Real vehicle/background cutouts generated',90000);
    const pixelMetrics=await evaluate(`(()=>{
      const fg=document.querySelector('#cutout-foreground'),bg=document.querySelector('#cutout-background');
      if(fg.width!==bg.width||fg.height!==bg.height||Math.max(fg.width,fg.height)>2048)throw new Error('Invalid cutout size');
      const a=fg.getContext('2d').getImageData(0,0,fg.width,fg.height).data,b=bg.getContext('2d').getImageData(0,0,bg.width,bg.height).data;
      let selected=0,removed=0,maxAlphaError=0;
      for(let i=3;i<a.length;i+=4){if(a[i]>0)selected++;if(b[i]===0)removed++;maxAlphaError=Math.max(maxAlphaError,Math.abs(a[i]+b[i]-255));}
      if(selected<100||removed<100||selected>=fg.width*fg.height||maxAlphaError>1)throw new Error('Foreground/background alpha mismatch');
      return {width:fg.width,height:fg.height,selectedPixels:selected,removedBackgroundPixels:removed,maxAlphaError,status:document.querySelector('#cutout-status').textContent};
    })()`);
    await screenshot('cutout-vehicles.png');
    // 验证真实PNG下载按钮，重新解码下载文件并检查alpha互补。
    const downloadDirectory=path.join(root,'.cache/web-cutout/downloads-'+Date.now());fs.mkdirSync(downloadDirectory,{recursive:true});
    await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloadDirectory});
    await evaluate("document.querySelector('#download-foreground').click();document.querySelector('#download-background').click()");
    await waitFor(async()=>fs.readdirSync(downloadDirectory).filter(n=>n.endsWith('.png')).length===2,'Both transparent PNG downloads completed',30000);
    const pngFiles=fs.readdirSync(downloadDirectory).filter(n=>n.endsWith('.png'));
    for(const name of pngFiles){const bytes=fs.readFileSync(path.join(downloadDirectory,name));if(!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw new Error('Invalid PNG download');fs.copyFileSync(path.join(downloadDirectory,name),path.join(destination,name.includes('-foreground.png')?'cutout-foreground.png':'cutout-background.png'));}
    const downloaded=await evaluate(`(async()=>{
      const files=${JSON.stringify(pngFiles.map(name=>({name,data:fs.readFileSync(path.join(downloadDirectory,name)).toString('base64')})))};
      const layers={};for(const file of files){const image=new Image();image.src='data:image/png;base64,'+file.data;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.getContext('2d').drawImage(image,0,0);layers[file.name.includes('-foreground.png')?'foreground':'background']=canvas;}
      const fg=layers.foreground,bg=layers.background;if(fg.width!==bg.width||fg.height!==bg.height)throw new Error('PNG dimensions mismatch');
      const a=fg.getContext('2d').getImageData(0,0,fg.width,fg.height).data,b=bg.getContext('2d').getImageData(0,0,bg.width,bg.height).data;let error=0;
      for(let i=3;i<a.length;i+=4)error=Math.max(error,Math.abs(a[i]+b[i]-255));if(error>1)throw new Error('Downloaded PNG alpha mismatch');
      return {width:fg.width,height:fg.height,maxAlphaError:error,files:2};
    })()`);
    // 精确小图验证：自定义类别顺序、掩码孔洞、多实例合并和原图坐标投影。
    const semantics=await evaluate(`(async()=>{
      const {renderCutouts}=await import('/cutout.js?v=1');const source=document.createElement('canvas');source.width=8;source.height=4;
      const context=source.getContext('2d');context.fillStyle='#c06020';context.fillRect(0,0,8,4);
      const fg=document.createElement('canvas'),bg=document.createElement('canvas');
      const instances=[{box:{classId:0},className:'car',mask:{x:10,y:0,width:2,height:2,displayWidth:20,displayHeight:20,probabilities:btoa(String.fromCharCode(255,0,255,255))}},
        {box:{classId:7},className:'person',mask:{x:50,y:10,width:2,height:2,displayWidth:20,displayHeight:20,probabilities:btoa(String.fromCharCode(255,255,255,255))}}];
      const alpha=(canvas,x,y)=>canvas.getContext('2d').getImageData(x,y,1,1).data[3];
      let result=renderCutouts(source,instances,80,40,'vehicles',fg,bg);
      if(result.instances!==1||alpha(fg,1,0)!==255||alpha(fg,2,0)!==0||alpha(fg,5,1)!==0||alpha(bg,1,0)!==0||alpha(bg,5,1)!==255)throw new Error('Vehicle filter/mask holes/projection failed');
      result=renderCutouts(source,instances,80,40,'people',fg,bg);if(result.instances!==1||alpha(fg,5,1)!==255||alpha(fg,1,0)!==0)throw new Error('Custom person class order failed');
      result=renderCutouts(source,instances,80,40,'all',fg,bg);if(result.instances!==2||alpha(fg,1,0)!==255||alpha(fg,5,1)!==255)throw new Error('Multi-instance union failed');
      result=renderCutouts(source,instances,80,40,'class:7',fg,bg);if(result.instances!==1||alpha(fg,5,1)!==255||alpha(fg,1,0)!==0)throw new Error('Specific class filter failed');
      result=renderCutouts(source,instances,80,40,'class:99',fg,bg);if(result.instances!==0||result.pixels!==0)throw new Error('Missing target was fabricated');
      let rejected=false;try{renderCutouts(source,[{...instances[0],mask:{...instances[0].mask,probabilities:'AAAA'}}],80,40,'vehicles',fg,bg);}catch{rejected=true;}if(!rejected)throw new Error('Invalid mask accepted');
      return {customClassOrder:true,maskHoles:true,coordinateProjection:true,multipleInstances:true,specificClass:true,missingTarget:true,invalidMaskRejected:true};
    })()`);
    // 切换为人物时沿用同一次分割，不触发额外模型推理。
    await evaluate("window.cutoutFetches=0;const previous=window.fetch;window.fetch=(...args)=>{if(['/api/infer','/api/infer-rgb'].includes(String(args[0])))window.cutoutFetches++;return previous(...args);};document.querySelector('#cutout-target').value='people';document.querySelector('#cutout-target').dispatchEvent(new Event('change',{bubbles:true}));");
    await waitFor(()=>evaluate("document.querySelector('#cutout-foreground-title').textContent==='只保留人物' && !document.querySelector('#download-foreground').disabled"),'Person cutout updated without inference');
    if(await evaluate('window.cutoutFetches')!==0)throw new Error('Changing target repeated inference');
    await screenshot('cutout-people.png');
    await evaluate("document.querySelector('#image-file').files");
    await setFiles('#image-file',[values.image]);
    await waitFor(()=>evaluate("document.querySelector('#download-foreground').disabled && document.querySelector('#cutout-results').hidden"),'Changing image clears stale cutout downloads');
    cutout={status:'passed',vehicles:pixelMetrics,downloaded,semantics,peopleSelectionWithoutInference:true,clearedAfterImageChange:true};
    checks.push('Vehicle/person cutout previews; complementary PNG downloads; class filters; mask holes; bounded buffers; no stale exports');
  }
  const features=values.features?await validateWorkflows({base,destination,evaluate,call,waitFor,screenshot,checks,info,imagePath:values.image}):null;
  await evaluate("document.querySelector('[data-view=training]').click()");
  await evaluate(`document.querySelector('#training-model').value=${JSON.stringify(trainingModel.id)};document.querySelector('#training-model').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#training-dataset').value='demo-squares';document.querySelector('#training-dataset').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#training-epochs').value='2';document.querySelector('#training-size').value='128';document.querySelector('#training-lr').value='.003';document.querySelector('#start-training').click();`);
  await waitFor(()=>evaluate("document.querySelector('#training-status').textContent.includes('已完成') && document.querySelector('#chart-step-label').textContent.includes('16')"),'UI started and completed 16 real training steps',90000);
  await waitFor(()=>evaluate("!document.querySelector('#download-checkpoint').classList.contains('disabled')"),'Checkpoint download available');
  await pause(6500);await screenshot('web-training.png');
  await evaluate("document.querySelector('#training-epochs').value='3';document.querySelector('#resume-training').click()");
  await waitFor(()=>evaluate("document.querySelector('#training-status').textContent.includes('已完成') && document.querySelector('#epoch-label').textContent.includes('3')"),'UI resumed to epoch 3',90000);
  await evaluate("document.querySelector('#training-epochs').value='100';document.querySelector('#start-training').click()");
  await waitFor(()=>evaluate("document.querySelector('#training-status').textContent.includes('训练中') && !document.querySelector('#cancel-training').disabled"),'Cancellation task running');
  await evaluate("document.querySelector('#cancel-training').click()");
  await waitFor(()=>evaluate("document.querySelector('#training-status').textContent.includes('已取消')"),'UI cancelled actual training');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await pause(300);await screenshot('web-mobile.png');
  const mobile=await evaluate("({viewport:innerWidth,width:document.documentElement.scrollWidth})");
  if(mobile.width>mobile.viewport+1)throw new Error('Mobile overflow '+JSON.stringify(mobile));
  checks.push('Mobile layout has no horizontal overflow');
  const result={status:'passed',browser:'Microsoft Edge headless',trainingModel:trainingModel.name,checks,largeImages,cutout,features,errors,mobile};
  if(errors.length)throw new Error('Browser errors: '+errors.join('\n'));
  fs.writeFileSync(path.join(destination,'web-browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{
  if(socket?.readyState===WebSocket.OPEN){try{await call('Browser.close');}catch{}socket.close();}
  browser.kill();
}
