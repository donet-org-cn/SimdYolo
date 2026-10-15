// 使用 Node 内置 CDP 连接自有无头浏览器，验证真实页面操作；无需 npm 依赖。
import {parseArgs} from 'node:util';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const {values}=parseArgs({options:{url:{type:'string',default:'http://127.0.0.1:5188'},image:{type:'string'},output:{type:'string',default:'artifacts/validation/0.9.0'},browser:{type:'string',default:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}}});
if(!values.image)throw new Error('--image 必须指定一张真实目标照片，例如官方 bus.jpg');
const base=values.url.replace(/\/$/,''), root=process.cwd(), destination=path.resolve(values.output);
const profile=path.join(root,'.cache/training-validation/edge-profile-'+Date.now());
fs.mkdirSync(destination,{recursive:true});
const browser=spawn(values.browser,[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
  '--user-data-dir='+profile,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let socket, next=1; const pending=new Map(), errors=[], checks=[];
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
  await evaluate("document.querySelector('[data-view=training]').click()");
  await evaluate(`document.querySelector('#training-model').value=${JSON.stringify(model.id)};document.querySelector('#training-model').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#training-dataset').value='demo-squares';document.querySelector('#training-dataset').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#training-epochs').value='2';document.querySelector('#training-size').value='128';document.querySelector('#training-lr').value='.003';document.querySelector('#start-training').click();`);
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
  const result={status:'passed',browser:'Microsoft Edge headless',checks,errors,mobile};
  if(errors.length)throw new Error('Browser errors: '+errors.join('\n'));
  fs.writeFileSync(path.join(destination,'web-browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{
  if(socket?.readyState===WebSocket.OPEN){try{await call('Browser.close');}catch{}socket.close();}
  browser.kill();
}
