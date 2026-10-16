// 使用真实实例掩码合成互补的透明图层；处理尺寸沿用网页的大图预览上限。
const groups = {
  people: new Set(["person", "people", "human", "人", "人物", "人员", "行人"]),
  vehicles: new Set(["car", "bus", "truck", "vehicle", "automobile", "汽车", "小汽车", "轿车", "公交车", "巴士", "卡车", "货车", "车辆"]),
};

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
 * @returns {{instances:number,pixels:number,width:number,height:number}} 匹配实例数及非透明掩码像素数。
 * @throws {Error} 图像、原图尺寸或掩码无效，或浏览器无法创建画布。
 */
export function renderCutouts(source, instances, originalWidth, originalHeight, target, foreground, background) {
  const width = source.naturalWidth || source.width, height = source.naturalHeight || source.height;
  if (![width, height].every(n => Number.isInteger(n) && n > 0 && n <= 2048) ||
      ![originalWidth, originalHeight].every(n => Number.isFinite(n) && n > 0)) throw new Error("目标分离图像尺寸无效，请重新上传图片。");
  const matte = document.createElement("canvas"), tile = document.createElement("canvas");
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
      const bytes = atob(mask.probabilities);
      if (bytes.length !== mask.width * mask.height) throw new Error("实例掩码长度与尺寸不一致。");
      const image = new ImageData(mask.width, mask.height);
      for (let i = 0; i < bytes.length; i++) {
        image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = 255;
        image.data[i * 4 + 3] = bytes.charCodeAt(i) >= 128 ? 255 : 0;
      }
      tile.width = mask.width; tile.height = mask.height;
      const tileContext = tile.getContext("2d");
      if (!tileContext) throw new Error("浏览器无法创建实例掩码缓冲区。");
      tileContext.putImageData(image, 0, 0);
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
  } finally { matte.width = matte.height = tile.width = tile.height = 0; }
}
