# OnnxOcrJS

ONNX ベースの OCR ライブラリです。PaddleOCR 互換のモデルを利用でき、Node.js とブラウザ両方で動作します。

---

## 🌐 Languages

- [English (default)](./README.md)
- [日本語 (Japanese)](./README.ja.md)
- [中文 (Chinese)](./README.zh.md)

---

## インストール

```bash
npm install onnx-ocr-js
```

または

```bash
pnpm add onnx-ocr-js
```

---

## 使い方

### Node.js での利用

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

const ocr = new ONNXPaddleOCR({
  limit_side_len: 960,
  det_db_thresh: 0.3,
  det_db_box_thresh: 0.6,
  det_db_unclip_ratio: 1.5,
  det_db_score_mode: "fast",
  det_box_type: "quad",
  cls_image_shape: [3, 48, 192],
  rec_image_shape: [3, 48, 320],
  drop_score: 0.5,
  rec_algorithm: "SVTR_LCNet",
  use_angle_cls: true,
});

const textSystem = await ocr.init({
  cv,
  ort,
  det_model_array_buffer: detModel,
  rec_model_array_buffer: recModel,
  cls_model_array_buffer: clsModel,
  rec_char_dict: charset,
});

// OpenCV.js で画像を Mat に変換
const jimpImage = await Jimp.read("./test.png");
const mat = cv.matFromImageData(jimpImage.bitmap);
const mat3ch = new cv.Mat();
cv.cvtColor(mat, mat3ch, cv.COLOR_RGBA2BGR);

const results = await ocr.ocr(textSystem, mat3ch, true, true, true);
console.log(results);
```

### ブラウザでの利用

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

### モデルの入手方法

本ライブラリは OCR の推論に PaddleOCR 互換の ONNX モデルを利用します。  

- サンプルモデルはこのリポジトリの [`/models`](./models) ディレクトリに含まれています。  
- また、公式変換済みモデルは OnnxOCR の GitHub リポジトリからも入手できます:  
  👉 [OnnxOCR/models](https://github.com/jingsongliujing/OnnxOCR/tree/main/onnxocr/models)

⚠️ モデルはサイズが大きいため、npm パッケージには含まれていません。必要に応じて手動で取得してください。

#### 個別ファイルのダウンロード例

```bash
# curl の場合
curl -L https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -o det.onnx

# wget の場合
wget https://raw.githubusercontent.com/SotaTne/OnnxOcrJS/main/models/ppocrv5/det/det.onnx -O det.onnx
```

## 注意事項

- **ONNX Runtime**:  
  Node.js では `onnxruntime-node`、ブラウザでは `onnxruntime-web`、ReactNative では `onnxruntime-react-native` を利用可能です。

- **OpenCV.js**:  
  本ライブラリは **型情報としてのみ** `@techstark/opencv-js@^4.11.0` に依存しています。  
  実際の OpenCV.js の読み込み（CDN / npm / self-host など）は利用者が自由に選択してください。

- **モデルファイル**:  
  `.onnx` モデルは `ArrayBuffer` または `Buffer` として読み込んでください。

- **ライセンス**:  
  本ライブラリのライセンスは **Apache-2.0** です。  
  ただしアルゴリズムやモデルはそれぞれ以下のライセンスに従います:  
  - [PaddleOCR (Apache-2.0)](https://github.com/PaddlePaddle/PaddleOCR)  
  - [ONNXOCR (Apache-2.0)](https://github.com/jingsongliujing/OnnxOCR)
