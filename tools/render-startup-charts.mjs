// 将已有SVG性能图按原尺寸渲染为PNG；只用于文档，不进入产品运行依赖。
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {withBrowser} from './browser-cdp.mjs';

await withBrowser(async ({call,evaluate,until}) => {
  await call('Emulation.setDeviceMetricsOverride',{width:1500,height:948,deviceScaleFactor:1,mobile:false});
  for (const stem of ['startup-source-0.11.3','startup-publish-0.11.3']) {
    const svg=path.resolve('artifacts',stem+'.svg'), png=path.resolve('artifacts',stem+'.png');
    const url=pathToFileURL(svg).href;
    await call('Page.navigate',{url});
    await until(()=>evaluate(`document.URL===${JSON.stringify(url)} && document.documentElement.localName==='svg'`),'SVG document');
    await evaluate('document.fonts.ready.then(()=>true)');
    const {data}=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:1500,height:948,scale:1}});
    fs.writeFileSync(png,Buffer.from(data,'base64')); console.log(png);
  }
});
