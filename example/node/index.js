"use strict"
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
const jimpImage = await Jimp.read("./japan_2.jpg");
const mat = cv.matFromImageData(jimpImage.bitmap);
const mat3ch = new cv.Mat();
cv.cvtColor(mat, mat3ch, cv.COLOR_RGBA2BGR);

const results = await ocr.ocr(textSystem, mat3ch, true, true, true);
console.log(results);