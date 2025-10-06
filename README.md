# OnnxOcrJS

An ONNX-based OCR library compatible with **PaddleOCR** models.  
Runs on **Node.js** and **Browser**.

---

## 🌐 Languages

- [English (default)](./README.md)
- [日本語 (Japanese)](./README.ja.md)
- [中文 (Chinese)](./README.zh.md)

---

## Installation

```bash
npm install onnx-ocr-js
```

or

```bash
pnpm add onnx-ocr-js
```

---

## Usage

### Node.js Example

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

// Convert image with OpenCV.js
const jimpImage = await Jimp.read("./test.png");
const mat = cv.matFromImageData(jimpImage.bitmap);
const mat3ch = new cv.Mat();
cv.cvtColor(mat, mat3ch, cv.COLOR_RGBA2BGR);

const results = await ocr.ocr(textSystem, mat3ch, true, true, true);
console.log(results);
```

### Browser Example

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

### How to Get Models

This library uses PaddleOCR-compatible ONNX models for OCR inference.  

- Sample models are included in this repository under the [`/models`](./models) directory.  
- You can also obtain official converted models from the OnnxOCR GitHub repository:  
  👉 [OnnxOCR/models](https://github.com/jingsongliujing/OnnxOCR/tree/main/onnxocr/models)

⚠️ Since model files are large, they are **not included in the npm package**.  
Please download them manually as needed.

#### Example: Downloading individual files

```bash
# Using curl
curl -L https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -o det.onnx

# Using wget
wget https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -O det.onnx
```

## Notes

- **ONNX Runtime**:  
  Use `onnxruntime-node` for Node.js, `onnxruntime-web` for Browser, and `onnxruntime-react-native` for React Native.  

- **OpenCV.js**:  
  This library **only depends on `@techstark/opencv-js@^4.11.0` for type definitions**.  
  You can load OpenCV.js in any way you prefer (CDN, npm, or self-host).  

- **Models**:  
  `.onnx` models must be loaded as `Buffer` (Node.js) or `ArrayBuffer` (Browser).  

- **Licenses**:  
  - This library: **Apache-2.0**  
  - Models and algorithms follow their original projects:  
    - [PaddleOCR (Apache-2.0)](https://github.com/PaddlePaddle/PaddleOCR)  
    - [ONNXOCR (Apache-2.0)](https://github.com/jingsongliujing/OnnxOCR)  
