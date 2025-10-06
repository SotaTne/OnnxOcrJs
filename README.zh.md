# OnnxOcrJS

基于 **ONNX** 的 OCR 库，兼容 **PaddleOCR** 模型。  
可在 **Node.js** 与 **浏览器** 中运行。

---

## 🌐 Languages

- [English (default)](./README.md)
- [日本語 (Japanese)](./README.ja.md)
- [中文 (Chinese)](./README.zh.md)

---

## 安装

```bash
npm install onnx-ocr-js
```

或

```bash
pnpm add onnx-ocr-js
```

---

## 使用方法

### Node.js 示例

```ts
import { ONNXPaddleOCR } from "onnx-ocr-js";
import * as fs from "fs/promises";
import * as ort from "onnxruntime-node";
import cvReadyPromise from "@techstark/opencv-js";
import { Jimp } from "jimp";

const cv = await cvReadyPromise;
const detModel = await fs.readFile("./models/ppocrv5/det/det.onnx");
const recModel = await fs.readFile("./models/ppocrv5/rec/rec.onnx");
const clsModel = await fs.readFile("./models/ppocrv5/cls/cls.onnx");
const charset = await fs.readFile("./models/ppocrv5/ppocrv5_dict.txt", "utf-8");

const ocr = new ONNXPaddleOCR({ use_angle_cls: true });

const textSystem = await ocr.init({
  cv,
  ort,
  det_model_array_buffer: detModel,
  rec_model_array_buffer: recModel,
  cls_model_array_buffer: clsModel,
  rec_char_dict: charset,
});

// 使用 OpenCV.js 转换图像
const jimpImage = await Jimp.read("./test.png");
const mat = cv.matFromImageData(jimpImage.bitmap);
const mat3ch = new cv.Mat();
cv.cvtColor(mat, mat3ch, cv.COLOR_RGBA2BGR);

const results = await ocr.ocr(textSystem, mat3ch, true, true, true);
console.log(results);
```

### 浏览器示例

```html
<script type="module">
  import { ONNXPaddleOCR } from "onnx-ocr-js";
  import cvReadyPromise from "@techstark/opencv-js";
  import * as ort from "onnxruntime-web";

  const cv = await cvReadyPromise;

  const detModel = await fetch("/models/ppocrv5/det/det.onnx").then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
  const recModel = await fetch("/models/ppocrv5/rec/rec.onnx").then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
  const clsModel = await fetch("/models/ppocrv5/cls/cls.onnx").then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
  const charset = await fetch("/models/ppocrv5/ppocrv5_dict.txt").then(r => r.text());

  const ocr = new ONNXPaddleOCR({ use_angle_cls: true });

  const textSystem = await ocr.init({
    cv,
    ort,
    det_model_array_buffer: detModel,
    rec_model_array_buffer: recModel,
    cls_model_array_buffer: clsModel,
    rec_char_dict: charset,
  });

  const img = document.getElementById("input-img");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  const mat = cv.matFromImageData(imageData);
  const mat3ch = new cv.Mat();
  cv.cvtColor(mat, mat3ch, cv.COLOR_RGBA2BGR);

  const results = await ocr.ocr(textSystem, mat3ch, true, true, true);
  console.log(results);
</script>
```

---

### 模型获取方法

本库在 OCR 推理中使用 PaddleOCR 兼容的 ONNX 模型。  

- 示例模型已包含在本仓库的 [`/models`](./models) 目录下。  
- 你也可以从 OnnxOCR 的 GitHub 仓库获取官方转换的模型：  
  👉 [OnnxOCR/models](https://github.com/jingsongliujing/OnnxOCR/tree/main/onnxocr/models)

⚠️ 由于模型文件体积较大，**npm 包中未包含模型**。  
请根据需要手动下载。

#### 单个文件下载示例

```bash
# 使用 curl
curl -L https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -o det.onnx

# 使用 wget
wget https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -O det.onnx
```

## 注意事项

- **ONNX Runtime**:  
  Node.js 使用 `onnxruntime-node`，浏览器使用 `onnxruntime-web`，React Native 使用 `onnxruntime-react-native`。  

- **OpenCV.js**:  
  本库仅依赖 `@techstark/opencv-js@^4.11.0` 的 **类型定义**。  
  OpenCV.js 的实际加载方式（CDN / npm / 本地部署）由用户自行决定。  

- **模型文件**:  
  `.onnx` 模型需以 `Buffer` (Node.js) 或 `ArrayBuffer` (浏览器) 加载。  

- **许可证**:  
  - 本库: **Apache-2.0**  
  - 模型与算法遵循原项目的许可证:  
    - [PaddleOCR (Apache-2.0)](https://github.com/PaddlePaddle/PaddleOCR)  
    - [ONNXOCR (Apache-2.0)](https://github.com/jingsongliujing/OnnxOCR)  
