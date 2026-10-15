"use strict";

// 本示例只展示服务端真实返回的数据；训练曲线不插入演示点或模拟结果。
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    info: null, datasets: [], jobs: [], activeJobId: null, activeJob: null,
    image: null, imageName: "", imageVersion: 0, result: null,
    inferenceBusy: false, modelUploadBusy: false, trainingBusy: false,
    datasetUploadBusy: false, view: "inference", pollTimer: null, pollFailures: 0,
  };
  const colors = ["#8c70df", "#e29268", "#57a6a6", "#b99b53", "#d87b9e", "#729ac9", "#75a76c"];
  const statusLabels = { queued: "等待中", running: "训练中", completed: "已完成", cancelled: "已取消", failed: "失败", idle: "未开始" };
  const taskLabels = { detect: "目标检测", detection: "目标检测", segment: "实例分割", segmentation: "实例分割", pose: "姿态估计" };
  const skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    svg.classList.add("icon");
    svg.setAttribute("aria-hidden", "true");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function toast(message, error = false) {
    const item = document.createElement("div");
    item.className = `toast${error ? " error" : ""}`;
    item.setAttribute("role", error ? "alert" : "status");
    const text = document.createElement("span");
    text.textContent = message;
    const close = document.createElement("button");
    close.type = "button";
    close.setAttribute("aria-label", "关闭通知");
    close.append(icon("close"));
    close.addEventListener("click", () => item.remove());
    item.append(icon(error ? "info" : "check"), text, close);
    $("toast-region").append(item);
    if (!error) window.setTimeout(() => item.remove(), 6500);
    while ($("toast-region").children.length > 4) $("toast-region").firstElementChild.remove();
  }

  function showError(id, error) {
    const target = $(id);
    target.textContent = error ? error.message || String(error) : "";
    target.hidden = !error;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { cache: "no-store", ...options });
    const contentType = response.headers.get("Content-Type") || "";
    let body;
    if (contentType.includes("json")) body = await response.json();
    else body = await response.text();
    if (!response.ok) {
      let reason = typeof body === "string" ? body : body?.error || body?.detail || body?.message || body?.title;
      if (reason && typeof reason !== "string") reason = JSON.stringify(reason);
      throw new Error(reason || `请求失败（HTTP ${response.status}）`);
    }
    return body;
  }

  function post(path, body) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }

  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const numberText = (value, digits = 3) => finite(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "—";
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const isLive = (job) => job && ["queued", "running"].includes(job.status);
  const modelName = (id) => state.info?.models?.find((m) => m.id === id)?.name || "未知模型";
  const jobModelName = (job) => job.modelName || modelName(job.modelId);
  const jobDatasetName = (job) => job.datasetName || state.datasets.find((dataset) => dataset.id === job.datasetId)?.name || "未知数据集";
  const jobName = (job) => job.name || `${jobModelName(job)} · ${jobDatasetName(job)}`;
  const backendName = (id) => state.info?.backends?.find((b) => b.id === id)?.name || id || "—";

  function navigate() {
    const candidate = location.hash.slice(1);
    state.view = ["inference", "training", "jobs"].includes(candidate) ? candidate : "inference";
    for (const name of ["inference", "training", "jobs"]) $("view-" + name).hidden = name !== state.view;
    for (const link of document.querySelectorAll(".nav-link")) {
      const active = link.dataset.view === state.view;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    }
    $("breadcrumb-current").textContent = { inference: "模型推理", training: "模型训练", jobs: "任务记录" }[state.view];
    document.title = `${$("breadcrumb-current").textContent} · SimdYolo`;
    if (state.view === "training") window.requestAnimationFrame(drawChart);
    if (state.view === "jobs") refreshJobs().catch((e) => toast(e.message, true));
  }

  function populateSelect(select, items, previous, emptyText) {
    select.replaceChildren();
    if (!items.length) {
      select.add(new Option(emptyText, ""));
      select.disabled = true;
      return;
    }
    for (const item of items) select.add(new Option(item.name, item.id));
    if (items.some((item) => item.id === previous)) select.value = previous;
    select.disabled = false;
  }

  function renderModels() {
    const models = state.info?.models || [];
    populateSelect($("inference-model"), models, $("inference-model").value, "尚无模型，请先上传");
    populateSelect($("training-model"), models.filter((m) => m.trainable), $("training-model").value, "没有可训练模型");
    updateModelDetail();
    updateButtons();
  }

  function updateModelDetail() {
    const model = state.info?.models?.find((m) => m.id === $("inference-model").value);
    $("model-detail").textContent = model ? `${String(model.format || "").toUpperCase()} · ${taskLabels[model.task] || model.task || "模型推理"}${model.trainable ? " · 支持检测头微调" : ""}` : "上传 .pt、.onnx 或训练检查点以开始推理。";
  }

  function renderBackends(kind) {
    const container = $(kind + "-backends");
    const previous = container.querySelector("input:checked")?.value;
    const items = (state.info?.backends || []).filter((b) => b[kind] === true);
    container.replaceChildren();
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const label = document.createElement("label");
      label.className = "backend-option";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = kind + "-backend";
      input.value = item.id;
      input.checked = previous ? item.id === previous : i === 0;
      label.append(input, icon("cpu"), document.createTextNode(item.name));
      container.append(label);
    }
    if (!container.querySelector("input:checked") && container.querySelector("input")) container.querySelector("input").checked = true;
    if (!items.length) {
      const note = document.createElement("span");
      note.className = "text-small muted";
      note.textContent = "没有可用后端";
      container.append(note);
    }
  }

  function selectedBackend(kind) {
    return $(kind + "-backends").querySelector("input:checked")?.value || null;
  }

  function updateButtons() {
    const inferenceReady = !!(state.image && $("inference-model").value && selectedBackend("inference"));
    $("run-inference").disabled = !inferenceReady || state.inferenceBusy || state.modelUploadBusy;
    $("run-inference").querySelector("span").textContent = state.inferenceBusy ? "正在推理…" : "开始推理";
    $("choose-image").disabled = state.inferenceBusy;
    $("load-sample").disabled = state.inferenceBusy;
    $("upload-model").disabled = state.modelUploadBusy || state.inferenceBusy;
    $("inference-overlay").hidden = !state.inferenceBusy;
    $("inference-form").setAttribute("aria-busy", String(state.inferenceBusy));
    $("start-training").disabled = state.trainingBusy || state.datasetUploadBusy || !$("training-model").value || !$("training-dataset").value || !selectedBackend("training");
    $("start-training").querySelector("span").textContent = state.trainingBusy ? "正在创建任务…" : "创建训练任务";
    $("choose-dataset").disabled = state.datasetUploadBusy;
    $("cancel-training").disabled = state.trainingBusy || !isLive(state.activeJob);
    $("resume-training").disabled = state.trainingBusy || !state.activeJob?.checkpointUrl || isLive(state.activeJob);
  }

  async function loadInfo() {
    const info = await api("/api/info");
    state.info = info;
    $("app-version").textContent = info.version ? `v${String(info.version).replace(/^v/, "")}` : "";
    $("connection-dot").className = "connection-dot online";
    $("connection-text").textContent = "本机服务已连接";
    if (info.trainingScope) $("training-scope").textContent = info.trainingScope;
    renderModels();
    renderBackends("inference");
    renderBackends("training");
    updateButtons();
  }

  // RGB 以分块方式编码，避免对大图使用一次性展开导致调用栈溢出。
  function bytesToBase64(bytes) {
    const parts = [];
    for (let i = 0; i < bytes.length; i += 8192) parts.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
    return btoa(parts.join(""));
  }

  function base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  async function decodeImage(blob) {
    if (typeof createImageBitmap === "function") return createImageBitmap(blob);
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return image;
    } finally { URL.revokeObjectURL(url); }
  }

  function rgbPayload(image) {
    const canvas = document.createElement("canvas");
    const width = image.width || image.naturalWidth;
    const height = image.height || image.naturalHeight;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, width, height).data;
    const rgb = new Uint8Array(width * height * 3);
    for (let source = 0, target = 0; source < rgba.length; source += 4) {
      rgb[target++] = rgba[source];
      rgb[target++] = rgba[source + 1];
      rgb[target++] = rgba[source + 2];
    }
    const pixels = bytesToBase64(rgb);
    canvas.width = canvas.height = 0;
    return { width, height, pixels };
  }

  async function setImage(blob, name) {
    if (state.inferenceBusy) return;
    const image = await decodeImage(blob);
    if (!(image.width > 0 && image.height > 0)) throw new Error("图像尺寸无效，请选择 JPG 或 PNG 图像。");
    state.image?.close?.();
    state.image = image;
    state.imageName = name;
    state.imageVersion++;
    state.result = null;
    $("image-empty").hidden = true;
    $("inference-canvas").hidden = false;
    $("image-details").textContent = `${name} · ${image.width} × ${image.height}`;
    $("image-state").textContent = "图像已就绪";
    $("inference-time").textContent = "—";
    $("instance-count").textContent = "—";
    $("actual-backend").textContent = "—";
    $("result-note").textContent = "运行后显示服务端实测数据。";
    showError("inference-error", null);
    renderResults();
    drawInference();
    updateButtons();
  }

  function drawInference() {
    if (!state.image) return;
    const canvas = $("inference-canvas");
    canvas.width = state.image.width;
    canvas.height = state.image.height;
    const context = canvas.getContext("2d");
    context.drawImage(state.image, 0, 0);
    const instances = state.result?.instances || [];
    const scale = Math.max(1, Math.min(canvas.width, canvas.height) / 450);
    if ($("show-masks").checked) {
      for (let index = 0; index < instances.length; index++) {
        const instance = instances[index];
        if (!instance.mask) continue;
        const mask = instance.mask;
        if (!mask.probabilities || !Number.isInteger(mask.width) || !Number.isInteger(mask.height) || mask.width <= 0 || mask.height <= 0) continue;
        try {
          const bytes = base64ToBytes(mask.probabilities);
          if (bytes.length !== mask.width * mask.height) continue;
          const rgb = hexRgb(colors[index % colors.length]);
          const image = new ImageData(mask.width, mask.height);
          for (let i = 0; i < bytes.length; i++) {
            image.data[i * 4] = rgb[0]; image.data[i * 4 + 1] = rgb[1]; image.data[i * 4 + 2] = rgb[2];
            image.data[i * 4 + 3] = bytes[i] >= 128 ? 94 : 0;
          }
          const offscreen = document.createElement("canvas");
          offscreen.width = mask.width; offscreen.height = mask.height;
          offscreen.getContext("2d").putImageData(image, 0, 0);
          context.drawImage(offscreen, mask.x || 0, mask.y || 0);
        } catch { /* 单个异常掩码不影响检测框及其他实例的可视化。 */ }
      }
    }
    for (let index = 0; index < instances.length; index++) {
      const instance = instances[index];
      const box = instance.box;
      const color = colors[index % colors.length];
      if (!box || ![box.x, box.y, box.width, box.height].every(finite)) continue;
      context.strokeStyle = color;
      context.lineWidth = 2 * scale;
      context.strokeRect(box.x, box.y, box.width, box.height);
      const label = `${instance.className || `类别 ${box.classId}`} ${numberText(box.confidence * 100, 1)}%`;
      context.font = `600 ${12 * scale}px "Segoe UI", "Microsoft YaHei", sans-serif`;
      const labelWidth = context.measureText(label).width + 12 * scale;
      const labelHeight = 23 * scale;
      const labelX = clamp(box.x - scale, 0, Math.max(0, canvas.width - labelWidth));
      const labelY = box.y > labelHeight ? box.y - labelHeight : clamp(box.y, 0, Math.max(0, canvas.height - labelHeight));
      context.fillStyle = color;
      context.fillRect(labelX, labelY, labelWidth, labelHeight);
      context.fillStyle = "#fff";
      context.fillText(label, labelX + 6 * scale, labelY + 16 * scale);
      const points = instance.keypoints || [];
      const visible = (p) => p && finite(p.x) && finite(p.y) && (p.confidence == null || p.confidence >= 0.25);
      if (points.length === 17) {
        context.strokeStyle = "#ffd27b";
        context.lineWidth = 2 * scale;
        for (const [a, b] of skeleton) {
          if (!visible(points[a]) || !visible(points[b])) continue;
          context.beginPath(); context.moveTo(points[a].x, points[a].y); context.lineTo(points[b].x, points[b].y); context.stroke();
        }
      }
      for (const point of points) {
        if (!visible(point)) continue;
        context.fillStyle = "#ffe5a3";
        context.strokeStyle = "#e7a063";
        context.lineWidth = scale;
        context.beginPath(); context.arc(point.x, point.y, 2.8 * scale, 0, Math.PI * 2); context.fill(); context.stroke();
      }
    }
  }

  function hexRgb(hex) { return [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16)); }

  function cell(text, className) {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) td.className = className;
    return td;
  }

  function renderResults() {
    const instances = state.result?.instances || [];
    $("results-badge").textContent = instances.length;
    $("results-table-wrap").hidden = instances.length === 0;
    $("results-empty").hidden = instances.length > 0;
    $("results-empty").textContent = state.result ? "在当前置信度阈值下没有检测到目标。可以更换图像或调整阈值后重新推理。" : "推理完成后，这里会显示每个目标的类别、置信度和位置。";
    $("results-body").replaceChildren();
    const sorted = instances.map((instance, index) => ({ instance, index })).sort((a, b) => (b.instance.box?.confidence || 0) - (a.instance.box?.confidence || 0));
    for (const { instance, index } of sorted) {
      const tr = document.createElement("tr");
      const identity = cell("");
      const name = document.createElement("span"); name.className = "result-name";
      const swatch = document.createElement("i"); swatch.className = "color-swatch"; swatch.style.background = colors[index % colors.length];
      name.append(swatch, document.createTextNode(String(index + 1).padStart(2, "0"))); identity.append(name);
      const box = instance.box || {};
      const confidence = cell(`${numberText(box.confidence * 100, 1)}%`);
      const track = document.createElement("span"); track.className = "confidence-track";
      const fill = document.createElement("span"); fill.style.width = `${clamp((box.confidence || 0) * 100, 0, 100)}%`; track.append(fill); confidence.append(track);
      const position = `x ${numberText(box.x, 0)} · y ${numberText(box.y, 0)} · ${numberText(box.width, 0)} × ${numberText(box.height, 0)}`;
      const extras = [instance.mask ? "分割掩码" : null, instance.keypoints?.length ? `${instance.keypoints.length} 个关键点` : null].filter(Boolean).join(" · ") || "—";
      tr.append(identity, cell(instance.className || `类别 ${box.classId ?? "—"}`), confidence, cell(position), cell(extras));
      $("results-body").append(tr);
    }
  }

  async function infer(event) {
    event.preventDefault();
    if (state.inferenceBusy || !state.image) return;
    state.inferenceBusy = true;
    showError("inference-error", null);
    updateButtons();
    try {
      const imageVersion = state.imageVersion;
      const payload = rgbPayload(state.image);
      const result = await post("/api/infer", { modelId: $("inference-model").value, backend: selectedBackend("inference"), ...payload, confidence: Number($("confidence").value) });
      if (imageVersion !== state.imageVersion) return;
      state.result = result;
      $("inference-time").textContent = numberText(result.elapsedMs, 2);
      $("instance-count").textContent = result.instances?.length ?? 0;
      $("actual-backend").textContent = backendName(result.backend);
      $("image-state").textContent = "推理已完成";
      $("result-note").textContent = `${taskLabels[result.task] || result.task || "推理"} · 耗时由服务端返回，浏览器解码与上传不计入。`;
      renderResults();
      drawInference();
    } catch (error) { showError("inference-error", error); }
    finally { state.inferenceBusy = false; updateButtons(); }
  }

  async function uploadModel(file) {
    if (!file || state.modelUploadBusy) return;
    if (!/\.(pt|onnx|sytrain)$/i.test(file.name)) throw new Error("请选择 .pt、.onnx 或 .sytrain 文件。");
    state.modelUploadBusy = true;
    $("upload-model").textContent = "正在上传模型…";
    updateButtons();
    try {
      const model = await api("/api/models", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) }, body: file });
      await loadInfo();
      if (model?.id) {
        $("inference-model").value = model.id;
        if (model.trainable) $("training-model").value = model.id;
        updateModelDetail();
      }
      toast(`已加载模型：${model.name || file.name}`);
    } finally {
      state.modelUploadBusy = false;
      $("upload-model").replaceChildren(icon("upload"), document.createTextNode("加载自己的 .pt / .onnx / .sytrain"));
      updateButtons();
    }
  }

  function renderDatasets() {
    populateSelect($("training-dataset"), state.datasets, $("training-dataset").value, "暂无数据集，请先导入");
    updateDatasetDetail();
    updateButtons();
  }

  function updateDatasetDetail() {
    const dataset = state.datasets.find((d) => d.id === $("training-dataset").value);
    $("dataset-detail").textContent = dataset ? `${dataset.count ?? "—"} 张图像 · ${Array.isArray(dataset.classes) ? dataset.classes.length : "—"} 个类别${Array.isArray(dataset.classes) && dataset.classes.length ? ` · ${dataset.classes.slice(0, 5).join("、")}${dataset.classes.length > 5 ? "…" : ""}` : ""}` : "使用内置演示数据体验完整流程，或上传自己的标注数据。";
  }

  async function refreshDatasets() {
    state.datasets = await api("/api/datasets");
    renderDatasets();
  }

  // 目录上传同时支持同目录图片/标签，以及常见 images/ 与 labels/ 分离目录。
  async function uploadDataset(files) {
    if (state.datasetUploadBusy || !files.length) return;
    const classes = $("dataset-classes").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!classes.length) throw new Error("请先填写类别名称，每行一个，顺序对应标注中的类别 ID。");
    const images = files.filter((file) => /\.(jpg|jpeg|png)$/i.test(file.name));
    if (!images.length) throw new Error("选中的目录中没有 JPG 或 PNG 图像。");
    const labelsByPath = new Map();
    const labelsByStem = new Map();
    for (const file of files.filter((item) => /\.txt$/i.test(item.name))) {
      const path = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
      labelsByPath.set(path.toLowerCase(), file);
      const stem = file.name.replace(/\.txt$/i, "").toLowerCase();
      if (!labelsByStem.has(stem)) labelsByStem.set(stem, []);
      labelsByStem.get(stem).push(file);
    }
    state.datasetUploadBusy = true;
    updateButtons();
    try {
      const data = [];
      let emptyCount = 0;
      for (let i = 0; i < images.length; i++) {
        const file = images[i];
        $("dataset-upload-status").textContent = `正在解码 ${i + 1} / ${images.length}：${file.name}`;
        const path = (file.webkitRelativePath || file.name).replace(/\\/g, "/").replace(/\.(jpg|jpeg|png)$/i, ".txt");
        const stem = file.name.replace(/\.(jpg|jpeg|png)$/i, "").toLowerCase();
        let label = labelsByPath.get(path.toLowerCase()) || labelsByPath.get(path.replace(/\/images\//i, "/labels/").toLowerCase());
        if (!label && labelsByStem.get(stem)?.length === 1) label = labelsByStem.get(stem)[0];
        if (!label && labelsByStem.get(stem)?.length > 1) throw new Error(`${file.name} 对应多个同名标注，请调整目录，避免匹配错误。`);
        if (!label) emptyCount++;
        const image = await decodeImage(file);
        try { data.push({ name: file.webkitRelativePath || file.name, ...rgbPayload(image), labels: label ? await label.text() : "" }); }
        finally { image.close?.(); }
        // 给浏览器机会绘制导入进度，较大的目录不会长时间卡住界面。
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
      $("dataset-upload-status").textContent = `正在上传 ${data.length} 张图像${emptyCount ? `（${emptyCount} 张无标签，作为背景样本）` : ""}…`;
      const dataset = await post("/api/datasets", { name: $("dataset-name").value.trim() || "自定义数据集", classes, images: data });
      await refreshDatasets();
      if (dataset?.id) $("training-dataset").value = dataset.id;
      updateDatasetDetail();
      $("dataset-upload-status").textContent = `已导入 ${data.length} 张图像${emptyCount ? `；${emptyCount} 张背景样本` : ""}。`;
      toast("数据集已导入，可以创建训练任务。");
    } catch (error) {
      $("dataset-upload-status").textContent = "导入未完成。";
      throw error;
    } finally { state.datasetUploadBusy = false; updateButtons(); }
  }

  function statusBadge(status) {
    const badge = document.createElement("span");
    const known = Object.hasOwn(statusLabels, status) ? status : "idle";
    badge.className = `status-badge status-${known}`;
    badge.append(document.createElement("span"), document.createTextNode(statusLabels[known]));
    return badge;
  }

  function progressOf(job) {
    if (!job) return 0;
    if (job.status === "completed") return 100;
    return job.totalSteps > 0 ? clamp(100 * (job.step || 0) / job.totalSteps, 0, 100) : 0;
  }

  function mergeJob(job) {
    if (!job?.id) return;
    const index = state.jobs.findIndex((item) => item.id === job.id);
    if (index < 0) state.jobs.unshift(job); else state.jobs[index] = { ...state.jobs[index], ...job };
    if (job.id === state.activeJobId) { state.activeJob = state.jobs.find((item) => item.id === job.id); renderActiveJob(); }
    renderJobs();
  }

  function selectJob(job) {
    state.activeJobId = job.id;
    state.activeJob = job;
    renderActiveJob();
    if (state.view !== "training") location.hash = "training";
  }

  function renderActiveJob() {
    const job = state.activeJob;
    const badge = statusBadge(job?.status || "idle");
    badge.id = "training-status";
    $("training-status").replaceWith(badge);
    if (!job) { updateButtons(); drawChart(); return; }
    $("active-job-name").textContent = jobName(job);
    $("active-job-meta").textContent = [jobModelName(job), jobDatasetName(job), backendName(job.backend), `#${String(job.id).slice(0, 8)}`, job.startedUtc ? formatDate(job.startedUtc) : null].filter(Boolean).join(" · ");
    const progress = progressOf(job);
    $("epoch-label").textContent = `Epoch ${job.epoch ?? 0} / ${job.epochs ?? "—"} · Step ${job.step ?? 0} / ${job.totalSteps ?? "—"}`;
    $("progress-value").textContent = `${numberText(progress, 0)}%`;
    $("training-progress-fill").style.width = `${progress}%`;
    $("training-progress").setAttribute("aria-valuenow", progress.toFixed(1));
    const history = job.history || [];
    const last = history[history.length - 1];
    $("metric-total").textContent = numberText(last?.totalLoss, 4);
    $("metric-box").textContent = numberText(last?.boxLoss, 4);
    $("metric-class").textContent = numberText(last?.classLoss, 4);
    $("metric-dfl").textContent = numberText(last?.dflLoss, 4);
    $("chart-step-label").textContent = `${history.length} 个训练 step`;
    $("job-message").textContent = job.error ? `任务失败：${job.error}` : ({ queued: "任务已排队，等待计算资源。", running: "训练进行中，可以取消并保留已经保存的检查点。", completed: "训练已完成。下载检查点，或在模型推理中选择训练结果。", cancelled: "任务已取消。存在检查点时可以继续训练。", failed: "任务失败，请查看服务端返回的错误信息。" }[job.status] || "正在同步训练状态。");
    const download = $("download-checkpoint");
    const url = safeLocalUrl(job.checkpointUrl);
    download.classList.toggle("disabled", !url);
    download.setAttribute("aria-disabled", String(!url));
    if (url) download.href = url; else download.removeAttribute("href");
    updateButtons();
    drawChart();
  }

  function safeLocalUrl(value) {
    if (!value) return null;
    try { const url = new URL(value, location.href); return url.origin === location.origin && ["http:", "https:"].includes(url.protocol) ? url.href : null; }
    catch { return null; }
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
  }

  function renderJobs() {
    $("jobs-count").textContent = state.jobs.length;
    $("jobs-total").textContent = state.jobs.length;
    $("jobs-empty").hidden = state.jobs.length > 0;
    $("jobs-table-wrap").hidden = state.jobs.length === 0;
    const body = $("jobs-body");
    body.replaceChildren();
    for (const job of state.jobs) {
      const tr = document.createElement("tr");
      const identity = cell("");
      const button = document.createElement("button"); button.className = "job-name-button"; button.type = "button"; button.textContent = jobName(job); button.addEventListener("click", () => selectJob(job));
      const id = document.createElement("span"); id.className = "job-id"; id.textContent = `#${String(job.id).slice(0, 8)}`; id.title = job.id;
      identity.append(button, id);
      const status = cell(""); status.append(statusBadge(job.status));
      const progress = cell(`${numberText(progressOf(job), 0)}%`);
      const track = document.createElement("div"); track.className = "table-progress";
      const fill = document.createElement("span"); fill.style.width = `${progressOf(job)}%`; track.append(fill); progress.append(track);
      const history = job.history || [];
      const actions = cell(""); const wrapper = document.createElement("div"); wrapper.className = "table-action";
      const open = document.createElement("button"); open.className = "button button-secondary"; open.type = "button"; open.textContent = "查看"; open.addEventListener("click", () => selectJob(job)); wrapper.append(open);
      const url = safeLocalUrl(job.checkpointUrl);
      if (url) {
        const link = document.createElement("a"); link.className = "button button-secondary"; link.href = url; link.download = ""; link.setAttribute("aria-label", `下载任务 ${job.id} 的检查点`); link.append(icon("download")); wrapper.append(link);
      }
      actions.append(wrapper);
      tr.append(identity, cell(backendName(job.backend)), status, progress, cell(numberText(history[history.length - 1]?.totalLoss, 4)), cell(formatDate(job.startedUtc)), actions);
      body.append(tr);
    }
  }

  async function refreshJobs() {
    const previouslyRunning = new Set(state.jobs.filter(isLive).map((job) => job.id));
    const jobs = await api("/api/training");
    state.jobs = Array.isArray(jobs) ? jobs : [];
    if (!state.activeJobId && state.jobs.length) {
      const job = state.jobs.find(isLive) || state.jobs[0];
      state.activeJobId = job.id;
      state.activeJob = job;
    } else if (state.activeJobId) {
      const current = state.jobs.find((item) => item.id === state.activeJobId);
      if (current) state.activeJob = current;
    }
    renderJobs();
    renderActiveJob();
    if (state.jobs.some((job) => previouslyRunning.has(job.id) && job.status === "completed")) {
      await loadInfo();
      toast("训练完成，检查点已可下载，模型列表已刷新。");
    }
  }

  async function startTraining(event) {
    event.preventDefault();
    if (state.trainingBusy || !$("training-form").reportValidity()) return;
    state.trainingBusy = true;
    showError("training-error", null);
    updateButtons();
    try {
      const job = await post("/api/training", { modelId: $("training-model").value, datasetId: $("training-dataset").value, backend: selectedBackend("training"), imageSize: Number($("training-size").value), epochs: Number($("training-epochs").value), batchSize: Number($("training-batch").value), learningRate: Number($("training-lr").value), seed: Number($("training-seed").value), resumeId: null });
      mergeJob(job);
      selectJob(job);
      toast("训练任务已创建，进度将每秒同步。");
    } catch (error) { showError("training-error", error); }
    finally { state.trainingBusy = false; updateButtons(); }
  }

  async function cancelTraining() {
    const job = state.activeJob;
    if (!job || state.trainingBusy) return;
    state.trainingBusy = true; updateButtons();
    try { mergeJob(await post(`/api/training/${encodeURIComponent(job.id)}/cancel`, {})); toast("已请求取消，正在等待当前计算步骤结束。"); }
    catch (error) { toast(error.message, true); }
    finally { state.trainingBusy = false; updateButtons(); }
  }

  async function resumeTraining() {
    const job = state.activeJob;
    if (!job || state.trainingBusy) return;
    const epochs = Number($("training-epochs").value);
    if (!Number.isInteger(epochs) || epochs < 1) { toast("请在训练配置中填写有效的训练轮数。", true); return; }
    state.trainingBusy = true; updateButtons();
    try {
      const resumed = await post(`/api/training/${encodeURIComponent(job.id)}/resume`, { epochs });
      mergeJob(resumed); selectJob(resumed); toast("已从检查点创建继续训练任务。");
    } catch (error) { toast(error.message, true); }
    finally { state.trainingBusy = false; updateButtons(); }
  }

  // 采用串行轮询，慢请求不会叠加；离开页面后由浏览器自然降低后台频率。
  async function poll() {
    try {
      if (state.view === "jobs") await refreshJobs();
      else if (state.activeJobId && isLive(state.activeJob)) {
        const before = state.activeJob.status;
        const job = await api(`/api/training/${encodeURIComponent(state.activeJobId)}`);
        mergeJob(job);
        if (isLive({ status: before }) && !isLive(job)) {
          if (job.status === "completed") { toast("训练完成，检查点已可下载。"); await loadInfo(); }
          else if (job.status === "failed") toast(job.error || "训练任务失败，请查看任务详情。", true);
        }
      }
      state.pollFailures = 0;
    } catch (error) {
      state.pollFailures++;
      if (state.pollFailures === 1) toast(`任务状态同步失败：${error.message}。正在自动重试。`, true);
    } finally { state.pollTimer = window.setTimeout(poll, 1000); }
  }

  function drawChart() {
    const canvas = $("loss-chart");
    if (!canvas || $("view-training").hidden) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const width = rect.width, height = rect.height;
    const padding = { left: 47, right: 19, top: 22, bottom: 39 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const history = state.activeJob?.history || [];
    const series = [{ key: "totalLoss", color: "#8d76cc", width: 2 }, { key: "boxLoss", color: "#eda68b", width: 1.3 }, { key: "classLoss", color: "#69b7b2", width: 1.3 }, { key: "dflLoss", color: "#adc278", width: 1.3 }];
    const points = history.filter((point) => finite(point.step));
    const values = points.flatMap((point) => series.map((item) => point[item.key]).filter(finite));
    const hasData = values.length > 0;
    $("chart-empty").hidden = hasData;
    // 使用归约代替展开参数，长时间训练的历史记录不会触发参数数量上限。
    const maxValue = hasData ? values.reduce((current, value) => Math.max(current, value), 0.001) * 1.12 : 1;
    const minValue = hasData ? values.reduce((current, value) => Math.min(current, value), 0) : 0;
    const minStep = points.length ? points.reduce((current, point) => Math.min(current, point.step), Infinity) : 0;
    const maxStep = points.length ? points.reduce((current, point) => Math.max(current, point.step), minStep + 1) : 1;
    const x = (step) => padding.left + ((step - minStep) / (maxStep - minStep)) * plotWidth;
    const y = (value) => padding.top + plotHeight - ((value - minValue) / (maxValue - minValue)) * plotHeight;
    context.font = '9px "Segoe UI", "Microsoft YaHei", sans-serif';
    context.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const yy = padding.top + plotHeight * i / 4;
      context.strokeStyle = "#efedf5";
      context.lineWidth = 1;
      context.setLineDash([3, 4]); context.beginPath(); context.moveTo(padding.left, yy); context.lineTo(width - padding.right, yy); context.stroke(); context.setLineDash([]);
      if (hasData) { context.fillStyle = "#b2a9c1"; context.fillText(((maxValue - minValue) * (1 - i / 4) + minValue).toFixed(2), padding.left - 10, yy + 3); }
    }
    context.textAlign = "center";
    if (hasData) for (let i = 0; i <= 4; i++) {
      const step = Math.round(minStep + (maxStep - minStep) * i / 4);
      context.fillStyle = "#b2a9c1"; context.fillText(String(step), x(step), height - 20);
    }
    context.fillStyle = "#c0b7cd"; context.textAlign = "right"; context.fillText("Step", width - padding.right, height - 4);
    if (!hasData) { canvas.setAttribute("aria-label", "训练损失曲线，目前没有训练数据"); return; }
    context.save(); context.beginPath(); context.rect(padding.left - 2, padding.top - 2, plotWidth + 4, plotHeight + 4); context.clip();
    // 图像宽度限制绘制点数；每个实际 step 仍完整保留在服务端任务记录中。
    const stride = Math.max(1, Math.ceil(points.length / Math.max(400, plotWidth * 2)));
    for (const item of series) {
      context.strokeStyle = item.color; context.fillStyle = item.color; context.lineWidth = item.width;
      context.lineJoin = "round"; context.beginPath();
      let started = false;
      for (let i = 0; i < points.length; i++) {
        if (i % stride !== 0 && i !== points.length - 1) continue;
        const point = points[i];
        if (!finite(point[item.key])) { started = false; continue; }
        if (!started) { context.moveTo(x(point.step), y(point[item.key])); started = true; }
        else context.lineTo(x(point.step), y(point[item.key]));
      }
      context.stroke();
      const last = points[points.length - 1];
      if (last && finite(last[item.key])) { context.beginPath(); context.arc(x(last.step), y(last[item.key]), 2.5, 0, Math.PI * 2); context.fill(); }
    }
    context.restore();
    const last = points[points.length - 1];
    canvas.setAttribute("aria-label", `${history.length} 个训练 step 的实际损失曲线。最近总损失 ${numberText(last?.totalLoss, 4)}，框损失 ${numberText(last?.boxLoss, 4)}，分类损失 ${numberText(last?.classLoss, 4)}，DFL 损失 ${numberText(last?.dflLoss, 4)}。`);
  }

  $("choose-image").addEventListener("click", () => $("image-file").click());
  $("image-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) try { await setImage(file, file.name); } catch (error) { toast(error.message, true); }
    event.target.value = "";
  });
  $("load-sample").addEventListener("click", async () => {
    const button = $("load-sample"); button.disabled = true; button.textContent = "正在加载…";
    try {
      const response = await fetch("/api/sample", { cache: "no-store" });
      if (!response.ok) throw new Error(`示例图加载失败（HTTP ${response.status}）`);
      await setImage(await response.blob(), "内置示例图");
    } catch (error) { toast(error.message, true); }
    finally { button.textContent = "使用示例图"; updateButtons(); }
  });
  const dropzone = $("image-dropzone");
  for (const eventName of ["dragenter", "dragover"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); if (!state.inferenceBusy) dropzone.classList.add("dragover"); });
  for (const eventName of ["dragleave", "drop"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragover"); });
  dropzone.addEventListener("drop", async (event) => {
    const file = event.dataTransfer.files[0];
    if (!file || state.inferenceBusy) return;
    if (!/^image\/(png|jpeg)$/.test(file.type) && !/\.(jpg|jpeg|png)$/i.test(file.name)) { toast("请拖入 JPG 或 PNG 图像。", true); return; }
    try { await setImage(file, file.name); } catch (error) { toast(error.message, true); }
  });
  $("inference-form").addEventListener("submit", infer);
  $("inference-model").addEventListener("change", updateModelDetail);
  $("confidence").addEventListener("input", (event) => $("confidence-value").value = Number(event.target.value).toFixed(2));
  $("show-masks").addEventListener("change", drawInference);
  $("upload-model").addEventListener("click", () => $("model-file").click());
  $("model-file").addEventListener("change", async (event) => { try { await uploadModel(event.target.files[0]); } catch (error) { toast(error.message, true); } finally { event.target.value = ""; } });
  $("training-dataset").addEventListener("change", updateDatasetDetail);
  $("refresh-datasets").addEventListener("click", () => refreshDatasets().catch((e) => toast(e.message, true)));
  $("choose-dataset").addEventListener("click", () => {
    if (!$("dataset-classes").value.trim()) { toast("请先填写类别名称，再选择数据集文件夹。", true); $("dataset-classes").focus(); return; }
    $("dataset-files").click();
  });
  $("dataset-files").addEventListener("change", async (event) => { try { await uploadDataset([...event.target.files]); } catch (error) { toast(error.message, true); } finally { event.target.value = ""; } });
  $("training-form").addEventListener("submit", startTraining);
  $("cancel-training").addEventListener("click", cancelTraining);
  $("resume-training").addEventListener("click", resumeTraining);
  $("refresh-jobs").addEventListener("click", () => refreshJobs().catch((e) => toast(e.message, true)));
  document.addEventListener("change", (event) => { if (event.target.matches("select,input[type=radio]")) updateButtons(); });
  window.addEventListener("hashchange", navigate);
  if (typeof ResizeObserver === "function") new ResizeObserver(() => window.requestAnimationFrame(drawChart)).observe($("loss-chart").parentElement);
  else window.addEventListener("resize", drawChart);
  window.addEventListener("pagehide", () => { window.clearTimeout(state.pollTimer); state.image?.close?.(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) { state.image = null; state.result = null; $("image-empty").hidden = false; $("inference-canvas").hidden = true; updateButtons(); poll(); } });

  navigate();
  Promise.allSettled([loadInfo(), refreshDatasets(), refreshJobs()]).then((results) => {
    results.forEach((result, index) => {
      if (result.status !== "rejected") return;
      if (index === 0) { $("connection-dot").className = "connection-dot offline"; $("connection-text").textContent = "连接失败，请刷新重试"; }
      toast(`${["服务信息", "数据集", "任务记录"][index]}加载失败：${result.reason.message}`, true);
    });
    poll();
  });
})();
