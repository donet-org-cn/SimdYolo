"use strict";

// 归一化框由浏览器编辑；保存、版本冲突校验及YOLO/PNG归档由纯.NET完成。
export function createAnnotation({ api, post, toast, workflows }) {
  const $ = id => document.getElementById(id), canvas = $("annotation-canvas"), context = canvas.getContext("2d");
  const colors = ["#6655d9", "#248d7c", "#cf7951", "#c94e92", "#346cc4"];
  let catalog = null, image = null, background = null, boxes = [], selected = -1, undo = [], dirty = false, busy = false, drag = null, loadVersion = 0;
  let datasetId = "", imageIndex = -1;
  const clamp = value => Math.max(0, Math.min(1, value));
  function error(value) { $("annotation-error").hidden = !value; $("annotation-error").textContent = value?.message || ""; }
  function options(id, values) {
    const select = $(id), previous = select.value; select.replaceChildren(); values.forEach(value => select.add(new Option(value.name, value.id)));
    if (values.some(value => value.id === previous)) select.value = previous;
  }
  function updateModels(info) { options("annotation-model", (info.models || []).filter(model => /detect/i.test(model.task))); }
  function updateDatasets(values) {
    options("annotation-dataset", values);
    if (!catalog && values.length) loadDataset().catch(error);
  }
  function parse(labels) {
    return labels.trim().split(/\r?\n/).filter(Boolean).map(line => { const [cls, x, y, w, h] = line.trim().split(/\s+/).map(Number); return { cls, x: clamp(x - w / 2), y: clamp(y - h / 2), w, h }; });
  }
  function snapshot() { undo.push(JSON.stringify(boxes)); if (undo.length > 50) undo.shift(); }
  function modified() { dirty = true; render(); }
  function render() {
    if (!background) return;
    context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(background, 0, 0);
    const scale = canvas.width / Math.max(200, canvas.getBoundingClientRect().width), handle = 4 * scale;
    boxes.forEach((box, index) => {
      const x = box.x * canvas.width, y = box.y * canvas.height, w = box.w * canvas.width, h = box.h * canvas.height, color = colors[box.cls % colors.length];
      context.strokeStyle = color; context.fillStyle = `${color}18`; context.lineWidth = (index === selected ? 2.5 : 1.4) * scale;
      context.fillRect(x, y, w, h); context.strokeRect(x, y, w, h);
      context.font = `${12 * scale}px "Microsoft YaHei",sans-serif`; const label = `${index + 1} · ${image.classes[box.cls]}`;
      const labelWidth = context.measureText(label).width + 8 * scale; context.fillStyle = color; context.fillRect(x, Math.max(0, y - 19 * scale), labelWidth, 19 * scale); context.fillStyle = "#fff"; context.fillText(label, x + 4 * scale, Math.max(14 * scale, y - 5 * scale));
      if (index === selected) { context.fillStyle = color; for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) context.fillRect(cx - handle, cy - handle, 2 * handle, 2 * handle); }
    });
    $("annotation-status").textContent = `${image.name} · ${boxes.length}个框 · ${dirty ? "未保存" : "已保存"}${busy ? " · 处理中" : ""}`;
    $("annotation-boxes").replaceChildren();
    boxes.forEach((box, index) => {
      const button = document.createElement("button"); button.type = "button"; button.className = `annotation-box${index === selected ? " selected" : ""}`; button.textContent = `${index + 1}. ${image.classes[box.cls]}`;
      button.addEventListener("click", () => { selected = index; $("annotation-class").value = String(box.cls); render(); }); $("annotation-boxes").append(button);
    });
    $("annotation-delete").disabled = busy || selected < 0; $("annotation-undo").disabled = busy || !undo.length;
    $("annotation-save").disabled = busy || !dirty; $("annotation-prelabel").disabled = busy || !$("annotation-model").value;
  }
  async function loadDataset() {
    const version = ++loadVersion; busy = true; error(null);
    try {
      const id = $("annotation-dataset").value; if (!id) return;
      const data = await api(`/api/datasets/${encodeURIComponent(id)}/images`); if (version !== loadVersion) return;
      catalog = data; datasetId = id; options("annotation-image", data.images.map(value => ({ id: String(value.index), name: `${value.index + 1}. ${value.name}` })));
      $("annotation-image").value = "0"; $("annotation-export").href = `/api/datasets/${encodeURIComponent(id)}/export`; $("annotation-export").classList.remove("disabled");
      await loadImage();
    } finally { busy = false; render(); }
  }
  async function loadImage() {
    const version = ++loadVersion; busy = true; error(null);
    try {
      const index = Number($("annotation-image").value), loaded = await api(`/api/datasets/${encodeURIComponent(datasetId)}/images/${index}`); if (version !== loadVersion) return;
      const raw = atob(loaded.pixels), rgba = new Uint8ClampedArray(loaded.width * loaded.height * 4);
      for (let source = 0, target = 0; source < raw.length; source += 3, target += 4) { rgba[target] = raw.charCodeAt(source); rgba[target + 1] = raw.charCodeAt(source + 1); rgba[target + 2] = raw.charCodeAt(source + 2); rgba[target + 3] = 255; }
      background = document.createElement("canvas"); background.width = canvas.width = loaded.width; background.height = canvas.height = loaded.height;
      background.getContext("2d").putImageData(new ImageData(rgba, loaded.width, loaded.height), 0, 0);
      image = loaded; imageIndex = index; boxes = parse(loaded.labels); selected = -1; undo = []; dirty = false;
      options("annotation-class", loaded.classes.map((name, id) => ({ id: String(id), name })));
    } finally { busy = false; render(); }
  }
  function point(event) { const rect = canvas.getBoundingClientRect(); return { x: clamp((event.clientX - rect.left) / rect.width), y: clamp((event.clientY - rect.top) / rect.height) }; }
  canvas.addEventListener("pointerdown", event => {
    if (!image || busy || event.button !== 0) return; event.preventDefault(); canvas.focus({ preventScroll: true }); canvas.setPointerCapture(event.pointerId);
    const start = point(event), rect = canvas.getBoundingClientRect(), thresholdX = 9 / rect.width, thresholdY = 9 / rect.height; let corner = -1;
    if (selected >= 0 && !event.shiftKey) { const box = boxes[selected]; corner = [[box.x, box.y], [box.x + box.w, box.y], [box.x, box.y + box.h], [box.x + box.w, box.y + box.h]].findIndex(([x, y]) => Math.abs(start.x - x) < thresholdX && Math.abs(start.y - y) < thresholdY); }
    if (corner >= 0) { snapshot(); drag = { mode: "resize", corner, start, original: { ...boxes[selected] } }; }
    else {
      selected = event.shiftKey ? -1 : boxes.findLastIndex(box => start.x >= box.x && start.x <= box.x + box.w && start.y >= box.y && start.y <= box.y + box.h);
      snapshot();
      if (selected < 0) { selected = boxes.length; boxes.push({ cls: Number($("annotation-class").value), x: start.x, y: start.y, w: 0, h: 0 }); drag = { mode: "new", start }; }
      else { drag = { mode: "move", start, original: { ...boxes[selected] } }; $("annotation-class").value = String(boxes[selected].cls); }
    }
    render();
  });
  canvas.addEventListener("pointermove", event => {
    if (!drag || selected < 0) return;
    const now = point(event), box = boxes[selected];
    if (drag.mode === "move") { box.x = Math.max(0, Math.min(1 - box.w, drag.original.x + now.x - drag.start.x)); box.y = Math.max(0, Math.min(1 - box.h, drag.original.y + now.y - drag.start.y)); }
    else {
      const original = drag.original, anchor = drag.mode === "new" ? drag.start : { x: original.x + (drag.corner % 2 === 0 ? original.w : 0), y: original.y + (drag.corner < 2 ? original.h : 0) };
      box.x = Math.min(anchor.x, now.x); box.y = Math.min(anchor.y, now.y); box.w = Math.abs(now.x - anchor.x); box.h = Math.abs(now.y - anchor.y);
    }
    modified();
  });
  function endDrag() {
    if (!drag) return; const box = boxes[selected];
    if (box.w * image.width < 2 || box.h * image.height < 2) { boxes = JSON.parse(undo.pop()); selected = -1; }
    drag = null; render();
  }
  canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);
  function remove() { if (busy || selected < 0) return; snapshot(); boxes.splice(selected, 1); selected = -1; modified(); }
  function revert() { if (busy || !undo.length) return; boxes = JSON.parse(undo.pop()); selected = -1; modified(); }
  canvas.addEventListener("keydown", event => { if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); remove(); } if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); revert(); } });
  $("annotation-delete").addEventListener("click", remove); $("annotation-undo").addEventListener("click", revert);
  $("annotation-class").addEventListener("change", () => { if (selected >= 0 && !busy) { snapshot(); boxes[selected].cls = Number($("annotation-class").value); modified(); } });
  $("annotation-dataset").addEventListener("change", () => { if (busy || dirty) { $("annotation-dataset").value = datasetId; toast("请先保存当前标注，再切换数据集。", true); return; } loadDataset().catch(error); });
  $("annotation-image").addEventListener("change", () => { if (busy || dirty) { $("annotation-image").value = String(imageIndex); toast("请先保存当前标注，再切换图像。", true); return; } loadImage().catch(error); });
  $("annotation-save").addEventListener("click", async () => {
    if (!image || busy || drag) return; busy = true; render(); error(null);
    try {
      const labels = boxes.map(box => `${box.cls} ${(box.x + box.w / 2).toFixed(8)} ${(box.y + box.h / 2).toFixed(8)} ${box.w.toFixed(8)} ${box.h.toFixed(8)}`).join("\n");
      const result = await api(`/api/datasets/${encodeURIComponent(datasetId)}/images/${imageIndex}/labels`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ labels, revision: image.revision }) });
      image.revision = catalog.revision = result.revision; image.labels = result.labels; dirty = false; undo = []; toast("标注已保存，后续训练和导出将使用新标签。");
    } catch (value) { error(value); } finally { busy = false; render(); }
  });
  $("annotation-prelabel").addEventListener("click", async () => {
    if (!image || busy) return; busy = true; render(); error(null);
    try {
      const job = await post("/api/workflows", { kind: "prelabel", modelId: $("annotation-model").value, datasetId, imageIndex, imageSize: 640, backend: "Cpu", confidence: Number($("annotation-confidence").value) });
      const completed = await workflows.wait(job), proposal = completed.result;
      if (proposal.revision !== image.revision) throw new Error("数据集版本已变化，请重新加载后预标注，避免覆盖他人的修改。");
      snapshot(); boxes = parse(proposal.labels); selected = -1; modified(); toast(`已生成${boxes.length}个建议框，请检查并保存。未匹配的类别名称不会生成框。`);
    } catch (value) { error(value); } finally { busy = false; render(); workflows.refresh().catch(error); }
  });
  window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("resize", render);
  return { updateModels, updateDatasets, refresh: render };
}
