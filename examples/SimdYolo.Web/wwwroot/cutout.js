// 使用真实实例掩码合成互补的透明图层；处理尺寸沿用网页的大图预览上限。
const groups = {
  people: new Set(["person", "people", "human", "人", "人物", "人员", "行人"]),
  vehicles: new Set(["car", "bus", "truck", "vehicle", "automobile", "汽车", "小汽车", "轿车", "公交车", "巴士", "卡车", "货车", "车辆"]),
};

/** 当前结果的有界掩码画布缓存；更换图片/结果时clear，内容或尺寸变化自动失效。 */
export class MaskCanvasCache {
  /** @param {number} budgetBytes 常驻画布像素字节预算，不包含当前临时结果。 */
  constructor(budgetBytes = 32 * 1024 * 1024) {
    this.budgetBytes = budgetBytes; this.entries = new Map(); this.bytes = 0; this.hits = 0; this.decodedMasks = 0;
  }
  /**
   * @param {object} mask 含8位Base64概率与位图尺寸的掩码。
   * @param {number[]|null} color 可选RGB叠加颜色。
   * @param {number} alpha 彩色层的不透明度，范围0到255。
   * @returns {HTMLCanvasElement} 本次可立即绘制的缓存或临时画布。
   * @throws {Error} 掩码或画布无效。
   */
  get(mask, color = null, alpha = 255) {
    if (![mask.width, mask.height].every(n => Number.isInteger(n) && n > 0 && n <= 2048) ||
        typeof mask.probabilities !== "string" || mask.probabilities.length > Math.ceil(mask.width * mask.height / 3) * 4) throw new Error("实例掩码无效，请重新运行分割推理。");
    let entry = this.entries.get(mask);
    if (entry && (entry.signature !== mask.probabilities || entry.width !== mask.width || entry.height !== mask.height)) { this.remove(mask); entry = null; }
    if (!entry) {
      const raw = atob(mask.probabilities);
      if (raw.length !== mask.width * mask.height) throw new Error("实例掩码长度与尺寸不一致。");
      const image = new ImageData(mask.width, mask.height), pixels = new Uint32Array(image.data.buffer);
      for (let i = 0; i < raw.length; i++) pixels[i] = raw.charCodeAt(i) >= 128 ? 0xffffffff : 0;
      const canvas = document.createElement("canvas"); canvas.width = mask.width; canvas.height = mask.height;
      const context = canvas.getContext("2d"); if (!context) throw new Error("浏览器无法创建掩码缓冲区。");
      context.putImageData(image, 0, 0);
      entry = { signature: mask.probabilities, width: mask.width, height: mask.height, canvas, tinted: null, tint: null, size: raw.length * 4 };
      this.entries.set(mask, entry); this.bytes += entry.size; this.decodedMasks++;
    } else { this.hits++; this.entries.delete(mask); this.entries.set(mask, entry); }
    let result = entry.canvas;
    if (color) {
      const key = color.join(",") + ":" + alpha;
      if (entry.tint !== key) {
        if (entry.tinted) { entry.tinted.width = entry.tinted.height = 0; this.bytes -= entry.width * entry.height * 4; entry.size -= entry.width * entry.height * 4; }
        const canvas = document.createElement("canvas"); canvas.width = entry.width; canvas.height = entry.height;
        const context = canvas.getContext("2d"); if (!context) throw new Error("浏览器无法创建彩色掩码。");
        context.drawImage(entry.canvas, 0, 0); context.globalCompositeOperation = "source-in";
        context.fillStyle = "rgba(" + color.join(",") + "," + alpha / 255 + ")"; context.fillRect(0, 0, canvas.width, canvas.height);
        entry.tinted = canvas; entry.tint = key; entry.size += entry.width * entry.height * 4; this.bytes += entry.width * entry.height * 4;
      }
      result = entry.tinted;
    }
    while (this.bytes > this.budgetBytes && this.entries.size > 1) this.remove(this.entries.keys().next().value);
    // 超预算的单张图作为临时结果返回，不能清空正在返回的画布。
    if (this.bytes > this.budgetBytes) { this.entries.delete(mask); this.bytes -= entry.size; }
    return result;
  }
  /** @param {object} mask 要移除的键；清空其常驻画布并更新字节计数。 */
  remove(mask) {
    const entry = this.entries.get(mask); if (!entry) return;
    entry.canvas.width = entry.canvas.height = 0;
    if (entry.tinted) entry.tinted.width = entry.tinted.height = 0;
    this.entries.delete(mask); this.bytes -= entry.size;
  }
  /** 释放全部常驻画布，保留累计命中/解码计数供开发验证。 */
  clear() { for (const mask of this.entries.keys()) this.remove(mask); }
}

/**
 * 根据实际类别名或用户选定的类别ID筛选实例，兼容自定义类别顺序。
 * @param {object} instance 服务端返回的实例。
 * @param {string} target 人物、车辆、全部目标或class:ID。
 * @returns {boolean} 实例是否属于指定目标。
 */
export function matchesCutoutTarget(instance, target) {
  if (target === "all") return true;
  if (target.startsWith("class:")) return String(instance.box?.classId) === target.slice(6);
  return groups[target]?.has(String(instance.className || "").trim().toLowerCase()) || false;
}

/**
 * 以掩码阈值0.5合并目标实例，生成仅保留目标、移除目标两张互补RGBA画布。
 * @param {CanvasImageSource} source 已解码且最长边不超过2048的预览图像。
 * @param {object[]} instances 真实分割实例；检测框不作为抠图区域。
 * @param {number} originalWidth 原图宽度，用于掩码位置映射。
 * @param {number} originalHeight 原图高度，用于掩码位置映射。
 * @param {string} target 类别筛选条件。
 * @param {HTMLCanvasElement} foreground 目标RGBA输出画布。
 * @param {HTMLCanvasElement} background 背景RGBA输出画布。
 * @param {MaskCanvasCache|null} cache 可选的有界共享缓存，切换类别时复用已解码掩码。
 * @returns {{instances:number,pixels:number,width:number,height:number}} 匹配实例数及非透明掩码像素数。
 * @throws {Error} 图像、原图尺寸或掩码无效，或浏览器无法创建画布。
 */
export function renderCutouts(source, instances, originalWidth, originalHeight, target, foreground, background, cache = null) {
  const width = source.naturalWidth || source.width, height = source.naturalHeight || source.height;
  if (![width, height].every(n => Number.isInteger(n) && n > 0 && n <= 2048) ||
      ![originalWidth, originalHeight].every(n => Number.isFinite(n) && n > 0)) throw new Error("目标分离图像尺寸无效，请重新上传图片。");
  const matte = document.createElement("canvas"), masks = cache || new MaskCanvasCache();
  matte.width = width; matte.height = height;
  const context = matte.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("浏览器无法创建目标分离缓冲区。");
  let count = 0;
  try {
    for (const instance of instances) {
      if (!instance.mask || !matchesCutoutTarget(instance, target)) continue;
      const mask = instance.mask, displayWidth = mask.displayWidth ?? mask.width, displayHeight = mask.displayHeight ?? mask.height;
      if (![mask.width, mask.height].every(n => Number.isInteger(n) && n > 0 && n <= 2048) ||
          ![mask.x, mask.y, displayWidth, displayHeight].every(Number.isFinite) || displayWidth <= 0 || displayHeight <= 0 ||
          typeof mask.probabilities !== "string" || mask.probabilities.length > Math.ceil(mask.width * mask.height / 3) * 4) throw new Error("实例掩码无效，请重新运行分割推理。");
      const tile = masks.get(mask);
      context.drawImage(tile, mask.x * width / originalWidth, mask.y * height / originalHeight,
        displayWidth * width / originalWidth, displayHeight * height / originalHeight);
      count++;
    }
    // 两个输出共用一张合并掩码，交叠实例不会误删彼此，背景保持其他类别。
    const rgba = context.getImageData(0, 0, width, height).data;
    let pixels = 0;
    for (let i = 3; i < rgba.length; i += 4) if (rgba[i]) pixels++;
    for (const [canvas, operation] of [[foreground, "destination-in"], [background, "destination-out"]]) {
      canvas.width = width; canvas.height = height;
      const output = canvas.getContext("2d");
      if (!output) throw new Error("浏览器无法创建透明PNG画布。");
      output.drawImage(source, 0, 0, width, height);
      output.globalCompositeOperation = operation;
      output.drawImage(matte, 0, 0);
      output.globalCompositeOperation = "source-over";
    }
    return { instances: count, pixels, width, height };
  } finally { matte.width = matte.height = 0; if (!cache) masks.clear(); }
}
