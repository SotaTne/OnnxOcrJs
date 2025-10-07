"use strict";
import { ONNXPaddleOCR } from "../src/index";
import * as fs from "fs/promises";
import { Jimp } from "jimp";
import { CV2, ORT } from "../src/types/type";
import { beforeAll, describe, it } from "vitest";
import { join } from "path";
import { model_dir, test_image_dir } from "./config";

let cv: CV2;
let ort: ORT;

beforeAll(async () => {
  /// @ts-ignore
  const cvReadyPromise = require("@techstark/opencv-js");
  const nodeORT = require("onnxruntime-node");
  ort = nodeORT;
  cv = await cvReadyPromise;
});

describe("ONNXPaddleOCR", () => {
  it("日本語認識", async () => {
    const det_model_src = join(model_dir, "ppocrv5", "det", "det.onnx");
    const det_model_buffer = await fs.readFile(det_model_src);

    const image_src = join(test_image_dir, "japan_2.jpg");
    const jimpImage = await Jimp.read(image_src);

    if (!jimpImage.bitmap) {
      throw new Error("No image");
    }
    console.log("jimpImage.bitmap:", jimpImage.bitmap);

    const charset_path = join(model_dir, "ppocrv5", "ppocrv5_dict.txt");
    const charset = (await fs.readFile(charset_path, "utf-8")).toString();

    const imageMat = cv.matFromImageData(jimpImage.bitmap);

    // pngでは透過分がalphaチャンネルに入るので、3チャンネルに変換しておく
    const imageMat3Ch = new cv.Mat();
    cv.cvtColor(imageMat, imageMat3Ch, cv.COLOR_RGBA2BGR);

    const cls_model_src = join(model_dir, "ppocrv5", "cls", "cls.onnx");
    const cls_model_buffer = await fs.readFile(cls_model_src);

    const rec_model_src = join(model_dir, "ppocrv5", "rec", "rec.onnx");
    const rec_model_buffer = await fs.readFile(rec_model_src);

    const ocr = new ONNXPaddleOCR({ use_angle_cls: true });

    const textSystem = await ocr.init({
      cv,
      ort,
      det_model_array_buffer: det_model_buffer,
      rec_model_array_buffer: rec_model_buffer,
      cls_model_array_buffer: cls_model_buffer,
      rec_char_dict: charset,
    });

    const results = await ocr.ocr(textSystem, imageMat3Ch, true, true, true);

    console.log(results);
  });
});
