// 仅从.NET原始测量JSON排版文档，不产生或修改精度/计时数据。
import fs from 'node:fs';
import path from 'node:path';

const source=process.argv[2]||'artifacts/validation/0.11.0/precision.json';
const destination=process.argv[3]||path.join(path.dirname(source),'precision-table.md');
const report=JSON.parse(fs.readFileSync(source,'utf8'));
if(report.status!=='completed'||report.runs.length!==20)throw new Error('Expected complete five-model/four-mode measurement');
const modes=['FP32','FP16-mixed','INT8-dynamic','INT8-calibrated'];
const models=[...new Set(report.runs.map(run=>run.model))],marks=['①','②','③','④'];
const lines=['# CPU低精度诊断：320输入、四线程','',
  '同一真实图片输入，预热'+report.warmup+'次、测量'+report.iterations+'次取中位数，单位ms。仅网络前向；单进程串行诊断，不替换历史640尺寸排名。**本机最快均为FP32，低精度仍属实验功能。**','',
  '| 模型 | FP32 | FP16混合 | INT8动态 | INT8校准 |',
  '| --- | ---: | ---: | ---: | ---: |'];
for(const model of models){
  const values=modes.map(mode=>report.runs.find(run=>run.model===model&&run.mode===mode));
  const ranking=values.slice().sort((a,b)=>a.medianForwardMs-b.medianForwardMs);
  lines.push('| '+model+' | '+values.map(value=>{const rank=ranking.indexOf(value);const label=value.medianForwardMs.toFixed(2)+' '+marks[rank];return rank===0?'**'+label+'**':label;}).join(' | ')+' |');
}
lines.push('','mAP50:95（百分比）：32张固定COCO128留出图，校准仅取训练侧8张。预训练已见过数据来源，仅作实现诊断；小样本上下波动不能解释为量化提升精度。','',
  '| 模型 | FP32 | FP16混合 | INT8动态 | INT8校准 |',
  '| --- | ---: | ---: | ---: | ---: |');
for(const model of models)lines.push('| '+model+' | '+modes.map(mode=>(100*report.runs.find(run=>run.model===model&&run.mode===mode).Map50To95).toFixed(2)+'%').join(' | ')+' |');
lines.push('','FP16使用Half操作数、FP32累加；INT8使用实际托管整数内核。GPU仍为FP32。校准采用最大绝对激活，不保证优于动态量化。','',
  '[原始采样与摘要](precision.json) · [设置和限制](README.md)','','核心DLL SHA256：'+report.coreAssemblySha256,'','训练DLL SHA256：'+report.trainingAssemblySha256,'');
fs.writeFileSync(destination,lines.join('\n'),'utf8');
