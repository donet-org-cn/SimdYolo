// 仅使用Node标准库和本机浏览器驱动开发验证，不进入应用运行依赖。
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';

export async function withBrowser(action) {
  const profile=path.resolve('.cache/image-optimization/browser-'+Date.now());fs.mkdirSync(profile,{recursive:true});
  const browser=spawn(process.env.SIMDYOLO_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    ['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:'ignore'});
  const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));let socket,next=0;const requests=new Map(),errors=[];
  function call(method,params={}) { return new Promise((resolve,reject)=>{const id=++next;const timeout=setTimeout(()=>{requests.delete(id);reject(new Error(method+' timed out'));},60000);requests.set(id,{resolve,reject,timeout});socket.send(JSON.stringify({id,method,params}));}); }
  async function evaluate(expression) { const value=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value; }
  async function until(test,label,timeout=60000) { const start=Date.now();let error;while(Date.now()-start<timeout){try{if(await test())return;}catch(value){error=value;}await pause(50);}throw new Error('Timeout '+label+(error?': '+error.message:'')); }
  async function files(selector,files) { const {root}=await call('DOM.getDocument');const {nodeId}=await call('DOM.querySelector',{nodeId:root.nodeId,selector});await call('DOM.setFileInputFiles',{nodeId,files:files.map(file=>path.resolve(file))}); }
  try {
    let port;
    await until(async()=>{try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];return /^\d+$/.test(port);}catch{return false;}},'browser port',15000);
    const targets=await fetch('http://127.0.0.1:'+port+'/json/list').then(response=>response.json());
    socket=new WebSocket(targets.find(item=>item.type==='page').webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
    socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.id){const request=requests.get(value.id);if(request){requests.delete(value.id);clearTimeout(request.timeout);value.error?request.reject(new Error(JSON.stringify(value.error))):request.resolve(value.result);}}else if(value.method==='Runtime.exceptionThrown')errors.push(value.params.exceptionDetails.text);};
    await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    return await action({call,evaluate,until,files,pause,errors});
  } finally { if(socket?.readyState===WebSocket.OPEN){try{await call('Browser.close');}catch{}socket.close();}browser.kill(); }
}
