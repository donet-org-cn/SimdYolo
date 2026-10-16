"use strict";

// 此模块只负责交互与结果展示；指标、校准和ONNX图均由.NET服务计算。
export function createWorkflows({ api, post, toast, loadInfo, refreshDatasets }) {
  const $ = id => document.getElementById(id);
  const live = job => ["queued", "running"].includes(job?.status);
  let models = [], datasets = [], jobs = [], active = null, timer, refreshing = false;
  const names = { evaluate: "检测评估", compare: "模型比较", export: "ONNX导出", calibrate: "INT8校准", prelabel: "预标注" };
  const statuses = { queued: "排队中", running: "执行中", completed: "已完成", failed: "失败", cancelled: "已取消" };
  const precisionFields = document.createElement("div");
  precisionFields.innerHTML = `<label class="field"><span>计算精度</span><select id="inference-precision"><option value="Float32">FP32（默认）</option><option value="Float16">CPU FP16操作数 / FP32累加</option><option value="Int8">CPU INT8 / INT32累加</option></select></label><label class="field" id="calibration-field" hidden><span>INT8激活尺度</span><select id="inference-calibration"><option value="">动态量化（无需校准）</option></select></label><p id="precision-note" class="form-footnote" hidden>低精度使用纯.NET CPU计算。FP16采用FP32累加；实际速度与精度需按模型验证。</p>`;
  $("inference-backends").closest("fieldset").after(precisionFields);
  const validationFields = document.createElement("div");
  validationFields.innerHTML = `<label class="field"><span>独立验证集（可选）</span><select id="training-validation"><option value="">不验证</option></select></label><div class="inline-actions"><button id="training-split" class="button button-secondary button-small" type="button">拆分当前数据：80%训练 / 20%验证</button></div><label class="field"><span>早停耐心轮数（0关闭）</span><input id="training-patience" type="number" value="0" min="0" max="500" disabled></label><p class="form-footnote">验证集与训练集禁止复用相同图像；按mAP50:95保存最佳模型。</p>`;
  $("training-dataset").closest(".field").after(validationFields);
  const best = document.createElement("a"); best.id = "download-best"; best.className = "button button-secondary button-small disabled"; best.download = ""; best.textContent = "下载最佳模型"; $("download-checkpoint").after(best);
  const metrics = document.createElement("p"); metrics.id = "training-validation-summary"; metrics.className = "text-small muted"; $("job-message").before(metrics);

  function options(id, values, empty) {
    const select = $(id), previous = select.value; select.replaceChildren();
    if (empty) select.add(new Option(empty, ""));
    values.forEach(value => select.add(new Option(value.name, value.id)));
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
  }
  function updatePrecision() {
    const low = $("inference-precision").value !== "Float32";
    for (const input of $("inference-backends").querySelectorAll("input")) { input.disabled = low && input.value !== "Cpu"; if (low && input.value === "Cpu") input.checked = true; }
    $("precision-note").hidden = !low; $("calibration-field").hidden = $("inference-precision").value !== "Int8";
  }
  function calibrations() {
    const modelId = $("inference-model").value;
    options("inference-calibration", jobs.filter(job => job.status === "completed" && job.request.kind === "calibrate" && job.request.modelId === modelId).map(job => ({ id: job.id, name: `${job.result?.images ?? "?"}张 · ${job.result?.inputShape?.slice(2).join("×") || "校准"} · ${job.id.slice(0, 8)}` })), "动态量化（无需校准）");
  }
  function updateModels(info) {
    models = info.models || [];
    options("evaluation-model", models.filter(model => /detect/i.test(model.task)));
    options("evaluation-other", models.filter(model => /detect/i.test(model.task)));
    options("tools-model", models);
    options("evaluation-backend", (info.backends || []).filter(value => value.inference).map(value => ({ id: value.id, name: value.name })));
    updatePrecision(); calibrations();
  }
  function updateDatasets(values) {
    datasets = values;
    for (const id of ["evaluation-dataset", "calibration-dataset"]) options(id, values);
    options("training-validation", values, "不验证");
    $("training-patience").disabled = !$("training-validation").value;
  }
  function trainingView(job) {
    const history = job?.validationHistory || [], last = history.at(-1), baseline = history[0];
    $("training-validation-summary").textContent = last ? `验证 mAP50:95：${percent(baseline.map50To95)} → ${percent(last.map50To95)} · P ${percent(last.precision)} / R ${percent(last.recall)} · 最佳轮次 ${last.bestEpoch}${job.stoppedEarly ? " · 已早停" : ""}` : "未启用验证：可选择独立验证集获取训练前后指标。";
    link($("download-best"), job?.bestCheckpointUrl);
  }
  function link(element, url) {
    const valid = typeof url === "string" && /^\/api\//.test(url);
    if (valid) element.href = url; else element.removeAttribute("href");
    element.classList.toggle("disabled", !valid); element.setAttribute("aria-disabled", String(!valid));
  }
  function percent(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—"; }
  function table(target, headers, rows) {
    const element = document.createElement("table"), head = element.createTHead().insertRow();
    headers.forEach(value => { const th = document.createElement("th"); th.textContent = value; head.append(th); });
    const body = element.createTBody();
    rows.forEach(values => { const row = body.insertRow(); values.forEach(value => { const td = row.insertCell(); td.textContent = String(value); }); });
    $(target).replaceChildren(element);
  }
  function select(job) { active = job; render(); }
  function render() {
    const job = active;
    $("workflow-cancel").disabled = !live(job); link($("workflow-download"), job?.downloadUrl);
    $("workflow-status").textContent = job ? `${names[job.request.kind]} · ${statuses[job.status]} · ${job.done} / ${job.total || "—"}` : "尚未开始";
    $("workflow-message").textContent = job?.error || "";
    $("evaluation-results").replaceChildren(); $("evaluation-classes").replaceChildren(); $("evaluation-confusion").hidden = true;
    if (job?.status === "completed" && job.result) {
      const result = job.result;
      if (result.metrics || result.before) {
        const values = result.metrics ? [["当前模型", result.metrics]] : [["原模型", result.before], ["对照模型", result.after]];
        table("evaluation-results", ["模型", "图像", "Precision", "Recall", "mAP50", "mAP50:95", "验证耗时"], values.map(([name, value]) => [name, value.images, percent(value.precision), percent(value.recall), percent(value.map50), percent(value.map50To95), `${value.elapsedMs.toFixed(0)} ms`]));
        const latest = values.at(-1)[1];
        table("evaluation-classes", ["对照 / 当前类别", "真值", "TP", "误检 FP", "漏检 FN", "P", "R", "AP50", "AP50:95"], latest.classes.map(value => [value.name, value.groundTruth, value.truePositive, value.falsePositive, value.falseNegative, percent(value.precision), percent(value.recall), percent(value.map50), percent(value.map50To95)]));
        const labels = [...latest.classes.map(value => value.name), "背景"];
        table("evaluation-matrix", ["真值 ↓ / 预测 →", ...labels], latest.confusionMatrix.map((row, index) => [labels[index], ...row]));
        $("evaluation-confusion").hidden = false;
        $("workflow-message").textContent = result.before ? `mAP50:95 变化 ${((result.after.map50To95 - result.before.map50To95) * 100).toFixed(2)} 个百分点；相同数据摘要 ${result.datasetSha256.slice(0, 12)}。完整逐图框、真值和两个模型的逐类结果见下载JSON。` : "完整逐图预测及真值见下载JSON。混淆矩阵含背景行/列。";
      } else if (job.request.kind === "export") $("workflow-message").textContent = `ONNX导出完成：${result.nodes}个节点，输入 ${result.inputShape.join("×")}，SHA256 ${result.sha256}。已加入推理模型列表。`;
      else if (job.request.kind === "calibrate") $("workflow-message").textContent = `校准完成：${result.images}张图像、${result.layers}层卷积；推理页选择同一模型后可使用。`;
    }
    $("workflow-history").replaceChildren();
    for (const item of jobs.filter(value => value.request.kind !== "prelabel")) {
      const button = document.createElement("button"); button.type = "button"; button.className = "button button-secondary button-small";
      const name = models.find(model => model.id === item.request.modelId)?.name || item.request.modelId;
      button.textContent = `${names[item.request.kind]} · ${name} · ${statuses[item.status]} · ${item.id.slice(0, 8)}`;
      button.addEventListener("click", () => select(item)); $("workflow-history").append(button);
    }
  }
  async function refresh() {
    if (refreshing) return; refreshing = true;
    try { jobs = await api("/api/workflows"); if (active) active = jobs.find(job => job.id === active.id) || active; render(); calibrations(); }
    finally { refreshing = false; }
  }
  async function start(kind) {
    $("workflow-error").hidden = true;
    try {
      const tools = ["export", "calibrate"].includes(kind), precision = tools ? "Float32" : $("evaluation-precision").value;
      const job = await post("/api/workflows", { kind, modelId: $(tools ? "tools-model" : "evaluation-model").value, otherModelId: kind === "compare" ? $("evaluation-other").value : null, datasetId: $(kind === "calibrate" ? "calibration-dataset" : "evaluation-dataset").value || null, imageSize: Number($(tools ? "tools-size" : "evaluation-size").value), backend: tools || precision !== "Float32" ? "Cpu" : $("evaluation-backend").value, precision });
      active = job; await refresh(); toast("后台任务已创建，可在进度和记录中查看。");
    } catch (error) { $("workflow-error").textContent = error.message; $("workflow-error").hidden = false; }
  }
  async function poll() {
    try {
      const previous = new Set(jobs.filter(live).map(job => job.id));
      if (previous.size) {
        await refresh();
        if (jobs.some(job => previous.has(job.id) && job.status === "completed" && job.request.kind === "export")) await loadInfo();
      }
    } catch (error) { $("workflow-message").textContent = error.message; }
    finally { timer = setTimeout(poll, 1500); }
  }
  async function wait(job) {
    while (live(job)) { await new Promise(resolve => setTimeout(resolve, 500)); job = await api(`/api/workflows/${job.id}`); }
    if (job.status !== "completed") throw new Error(job.error || "任务已取消。");
    return job;
  }
  $("inference-precision").addEventListener("change", updatePrecision);
  $("inference-model").addEventListener("change", calibrations);
  $("training-validation").addEventListener("change", () => { $("training-patience").disabled = !$("training-validation").value; if ($("training-patience").disabled) $("training-patience").value = "0"; });
  $("training-split").addEventListener("click", async () => {
    try {
      const result = await post(`/api/datasets/${encodeURIComponent($("training-dataset").value)}/split`, { validationFraction: .2, seed: Number($("training-seed").value) });
      await refreshDatasets(); $("training-dataset").value = result.trainingId; $("training-dataset").dispatchEvent(new Event("change")); $("training-validation").value = result.validationId; $("training-validation").dispatchEvent(new Event("change"));
      toast(`已拆分：${result.trainingCount}张训练、${result.validationCount}张独立验证。`);
    } catch (error) { toast(error.message, true); }
  });
  [["evaluation-run", "evaluate"], ["evaluation-compare", "compare"], ["tools-export", "export"], ["tools-calibrate", "calibrate"]].forEach(([id, kind]) => $(id).addEventListener("click", () => start(kind)));
  $("workflow-cancel").addEventListener("click", async () => { if (active) try { select(await post(`/api/workflows/${active.id}/cancel`, {})); } catch (error) { toast(error.message, true); } });
  $("workflow-refresh").addEventListener("click", () => refresh().catch(error => toast(error.message, true)));
  window.addEventListener("pagehide", () => clearTimeout(timer));
  refresh().catch(error => toast(error.message, true)).finally(poll);
  return { updateModels, updateDatasets, trainingView, wait, refresh, precisionKey: () => `${$("inference-precision").value}:${$("inference-calibration").value}`, inferenceOptions: () => ({ precision: $("inference-precision").value, calibrationId: $("inference-precision").value === "Int8" ? $("inference-calibration").value || null : null }) };
}
