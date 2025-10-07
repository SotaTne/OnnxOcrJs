import { TextClassifier, TextClassifierParams } from "../src/predict_cls.js";

import { join } from "path";

import { beforeAll, describe, expect, it } from "vitest";
import { TextDetector, TextDetectorParams } from "../src/predict_det.js";
import { model_dir, test_image_dir } from "./config.js";
import fs from "fs/promises";
import type cvReadyPromiseType from "@techstark/opencv-js";
import { Box, ORT } from "../src/types/type.js";
import { Jimp } from "jimp";
import type { Mat } from "@techstark/opencv-js";

import { TextRecognizer, TextRecognizerParams } from "../src/predict_rec.js";
import { getImgCropList, sortedBoxes } from "../src/utils/funcs/get-crop.js";

let cv: Awaited<typeof cvReadyPromiseType>;
let ort: ORT;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("TextRecognition / TextDetector", () => {
  it("recognition: 簡単なケース", async () => {
    // det

    // const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    // const det_model_buffer = await fs.readFile(det_model_src);

    const image_src = join(test_image_dir, "onnxocr_logo.png");
    const jimpImage = await Jimp.read(image_src);

    const charset_path = join(model_dir, "ppocrv5", "ppocrv5_dict.txt");
    const charset = (await fs.readFile(charset_path, "utf-8")).toString();

    const imageMat = cv.matFromImageData(jimpImage.bitmap);

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const imageMat3Ch = new cv.Mat();
    cv.cvtColor(imageMat, imageMat3Ch, cv.COLOR_RGBA2BGR);

    const rec_model_src = join(model_dir, "ppocrv5", "rec", "rec.onnx");
    const rec_model_buffer = await fs.readFile(rec_model_src);

    const det_box_type = "quad";

    // const textDetectorParams: TextDetectorParams = {
    //   limit_side_len: 960,
    //   det_limit_type: "max",
    //   det_db_thresh: 0.3, // 閾値下げる
    //   det_db_box_thresh: 0.6, // ボックス閾値も下げる
    //   det_db_unclip_ratio: 1.5,
    //   use_dilation: false,
    //   det_db_score_mode: "fast",
    //   det_box_type: det_box_type,
    //   cv,
    //   ort,
    //   det_model_array_buffer: det_model_buffer,
    //   use_gpu: false,
    // };
    // const detector = await TextDetector.create(textDetectorParams);

    // const boxes = await detector.execute(imageMat3Ch);

    // expect(boxes).toBeDefined();
    // expect(boxes!.length).toBeGreaterThanOrEqual(3);

    // if (!boxes || boxes.length === 0) {
    //   throw new Error("No boxes detected");
    // }

    // det

    const boxes = [
      [
        [89, 417],
        [379, 428],
        [376, 523],
        [86, 512],
      ],
      [
        [21, 229],
        [559, 229],
        [559, 341],
        [21, 341],
      ],
      [
        [276, 121],
        [332, 121],
        [332, 180],
        [276, 180],
      ],
    ];

    const { crops: img_crop_list } = getImgCropList(
      imageMat3Ch.clone(),
      boxes as Box[],
      det_box_type,
      cv,
    );

    const textRecognizerParams: TextRecognizerParams = {
      drop_score: 0.5,
      rec_batch_num: 6,
      rec_algorithm: "SVTR_LCNet",
      rec_image_shape: [3, 48, 320],
      cv: cv,
      ort: ort,
      rec_model_array_buffer: rec_model_buffer,
      rec_char_dict: charset,
      use_space_char: false,
      use_gpu: false,
    };

    const textRecognizer = await TextRecognizer.create(textRecognizerParams);
    const rec_res = await textRecognizer.execute(img_crop_list);
    console.log("rec_res:", rec_res);
  });
});
