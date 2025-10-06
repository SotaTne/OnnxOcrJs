import type {
  DROP_SCORE,
  REC_ALGORITHM,
  REC_BATCH_NUM,
  REC_IMAGE_SHAPE_NUMBER,
  USE_GCU,
  USE_SPACE_CHAR,
} from "./types/paddle_types.js";
import type { Mat } from "@techstark/opencv-js";
import type {
  CV2,
  ORT,
  ORTBufferType,
  ORTRunFetchesType,
  ORTSessionReturnType,
} from "./types/type.js";
import {
  argsort,
  cloneNdArray,
  matToLine,
  matToNdArray,
  ndArrayToList,
  tensorToNdArray,
} from "./utils/func.js";
import { PredictBase } from "./predict_base.js";
import { create_onnx_session_fn } from "./onnx_runtime.js";
import { CTCLabelDecode } from "./rec_postprocess.js";
import ndarray from "ndarray";
import ops from "ndarray-ops";

let _use_gpu_rec_onnx_session: ORTSessionReturnType | undefined = undefined;
let _use_gpu_session_hash: string | undefined = undefined;
let _use_cpu_rec_onnx_session: ORTSessionReturnType | undefined = undefined;
let _use_cpu_session_hash: string | undefined = undefined;

export type TextRecognizerParams = {
  drop_score: DROP_SCORE | null;
  rec_batch_num: REC_BATCH_NUM | null;
  rec_algorithm: REC_ALGORITHM | null;

  rec_image_shape: REC_IMAGE_SHAPE_NUMBER | null;
  cv: CV2;
  ort: ORT;
  rec_model_array_buffer: ORTBufferType;
  rec_char_dict: string | null;

  use_space_char: USE_SPACE_CHAR;
  use_gpu: USE_GCU;
};

export class TextRecognizer extends PredictBase {
  cv: CV2;
  rec_image_shape: REC_IMAGE_SHAPE_NUMBER;
  rec_batch_num: REC_BATCH_NUM;
  rec_onnx_session: ORTSessionReturnType;
  rec_algorithm: REC_ALGORITHM;

  postprocess_op: CTCLabelDecode;
  ort: ORT;
  rec_input_name: string[];
  rec_output_name: string[];

  constructor(
    params: TextRecognizerParams & {
      rec_onnx_session: ORTSessionReturnType;
    },
  ) {
    super();
    this.cv = params.cv;
    this.rec_image_shape = params.rec_image_shape ?? [3, 48, 320];
    this.rec_batch_num = params.rec_batch_num ?? 6;
    this.rec_algorithm = params.rec_algorithm ?? "SVTR_LCNet";
    this.rec_onnx_session = params.rec_onnx_session;
    this.ort = params.ort;
    this.postprocess_op = new CTCLabelDecode({
      character_str: params.rec_char_dict,
      use_space_char: params.use_space_char,
    });
    this.rec_onnx_session = params.rec_onnx_session;
    this.rec_input_name = this.get_input_name(this.rec_onnx_session);
    this.rec_output_name = this.get_output_name(this.rec_onnx_session);
  }

  static async create(params: TextRecognizerParams) {
    const rec_onnx_session = await TextRecognizer.get_onnx_session(
      params.rec_model_array_buffer,
      params.use_gpu,
      params.ort,
    );
    return new TextRecognizer({ ...params, rec_onnx_session });
  }

  resize_norm_img(img: Mat, max_hw_ratio: number): number[][][] {
    const imgC = this.rec_image_shape[0]!;
    const imgH = this.rec_image_shape[1]!;
    let imgW = this.rec_image_shape[2]!;
    if (this.rec_algorithm === "NRTR" || this.rec_algorithm === "ViTSTR") {
      const det_cvt = new this.cv.Mat();
      this.cv.cvtColor(img, det_cvt, this.cv.COLOR_BGR2GRAY);
      const uint8_img = this.cv.matFromArray(
        img.rows,
        img.cols,
        this.cv.CV_8UC1,
        matToLine(det_cvt, this.cv).data,
      );
      det_cvt.delete();
      const det_resized_img = new this.cv.Mat();
      if (this.rec_algorithm === "ViTSTR") {
        this.cv.resize(
          uint8_img,
          det_resized_img,
          new this.cv.Size(imgW, imgH),
          0,
          0,
          this.cv.INTER_CUBIC,
        );
      } else {
        this.cv.resize(
          uint8_img,
          det_resized_img,
          new this.cv.Size(imgW, imgH),
          0,
          0,
          this.cv.INTER_LANCZOS4,
        );
      }
      uint8_img.delete();
      const det_resized_3c_img = new this.cv.Mat();
      this.cv.cvtColor(
        det_resized_img,
        det_resized_3c_img,
        this.cv.COLOR_GRAY2BGR,
      );
      det_resized_img.delete();

      const ndarray_img = matToNdArray(det_resized_3c_img, this.cv);
      det_resized_3c_img.delete();

      const img_3c_list = ndArrayToList(
        cloneNdArray(ndarray_img),
      ) as number[][][];
      const ndarray_img_3c = ndarray(Float32Array.from(img_3c_list.flat(2)), [
        ndarray_img.shape[0]!,
        ndarray_img.shape[1]!,
        3,
      ]); // (H,W,3)
      const transposed = ndarray_img_3c.transpose(2, 0, 1); // (3,H,W)
      const list_img = ndArrayToList(transposed) as number[][][];
      const norm_img =
        this.rec_algorithm === "ViTSTR"
          ? list_img.map((row) => row.map((col) => col.map((v) => v / 255.0)))
          : list_img.map((row) =>
              row.map((col) => col.map((v) => v / 128.0 - 1.0)),
            );
      // (3,H,W)
      return norm_img;
    } else if (this.rec_algorithm === "RFL") {
      const det_cvt = new this.cv.Mat();
      this.cv.cvtColor(img, det_cvt, this.cv.COLOR_BGR2GRAY);
      const det_resized_img = new this.cv.Mat();
      this.cv.resize(
        det_cvt,
        det_resized_img,
        new this.cv.Size(imgW, imgH),
        0,
        0,
        this.cv.INTER_CUBIC,
      );
      det_cvt.delete();
      const det_resized_3c_img = new this.cv.Mat();
      this.cv.cvtColor(
        det_resized_img,
        det_resized_3c_img,
        this.cv.COLOR_GRAY2BGR,
      );
      det_resized_img.delete();
      const ndarray_img = matToNdArray(det_resized_3c_img, this.cv);
      det_resized_3c_img.delete();
      const img_3c_list = ndArrayToList(
        cloneNdArray(ndarray_img),
      ) as number[][][];
      const ndarray_img_3c = ndarray(Float32Array.from(img_3c_list.flat(2)), [
        ndarray_img.shape[0]!,
        ndarray_img.shape[1]!,
        3,
      ]); // (H,W,3)
      const transposed = ndarray_img_3c.transpose(2, 0, 1); // (3,H,W)
      const list_img = ndArrayToList(transposed) as number[][][];
      const norm_img = list_img.map((row) =>
        row.map((col) => col.map((v) => (v / 255.0 - 0.5) / 0.5)),
      );
      return norm_img;
    }
    if (imgC !== img.channels()) {
      throw new Error(
        `input image channels ${img.channels()} must be equal to rec_image_shape ${imgC}`,
      );
    }

    // 🔴 BGR → RGB
    const rgb = new this.cv.Mat();
    this.cv.cvtColor(img, rgb, this.cv.COLOR_BGR2RGB);

    imgW = Math.trunc(imgH * max_hw_ratio);
    const h = rgb.rows;
    const w = rgb.cols;
    const ratio = w / h;
    let resized_w = Math.ceil(imgH * ratio);
    if (resized_w > imgW) {
      resized_w = imgW;
    } else {
      resized_w = Math.trunc(resized_w);
    }

    if (this.rec_algorithm === "RARE") {
      if (resized_w > this.rec_image_shape[2]!) {
        resized_w = this.rec_image_shape[2]!;
      }
      imgW = this.rec_image_shape[2]!;
    }

    const resized_img = new this.cv.Mat();
    this.cv.resize(rgb, resized_img, new this.cv.Size(resized_w, imgH), 0, 0);
    rgb.delete();
    const ndarray_img = matToNdArray(resized_img, this.cv);
    resized_img.delete();
    const transposed = ndarray_img.transpose(2, 0, 1); // (3,H,W)

    const transposed_list = (ndArrayToList(transposed) as number[][][]).flat(2);
    const norm_transposed_list = transposed_list.map(
      (v) => (v / 255.0 - 0.5) / 0.5,
    );

    const transposed_ndarray = ndarray(
      Float32Array.from(norm_transposed_list),
      transposed.shape,
    );

    const zeroImg = ndarray(
      Float32Array.from(Array(imgC * imgH * imgW).fill(0)),
      [imgC, imgH, imgW],
    );

    const zeroView = zeroImg.hi(imgC, imgH, resized_w);

    ops.assign(zeroView, transposed_ndarray.hi(imgC, imgH, resized_w));

    const norm_img = ndArrayToList(zeroImg) as number[][][]; // [C,H,W]
    return norm_img;
  }

  // resize_norm_img_vl(img, image_shape) {}

  // resize_norm_img_srn(img, image_shape) {}

  // srn_other_inputs(image_shape, num_heads, max_text_length) {}

  // process_image_srn(img, imgage_shape, num_heads, max_text_length) {}

  // resize_norm_img_sar(img, image_shape, width_downsample_ratio = 0.25) {}

  // resize_norm_img_spin(img) {}

  // resize_norm_img_svtr(img, image_shape) {}

  // resize_norm_img_abinet(img, image_shape) {}

  // norm_img_can(img, image_shape) {}

  async execute(img_list: Mat[]) {
    const img_num = img_list.length;

    const width_list: number[] = img_list.map((img) => img.cols / img.rows);
    // for (const img of img_list) {
    //   width_list.push(img.cols / img.rows);
    // }
    const indices = argsort(width_list);
    const rec_res: [string, number][] = Array(img_num).fill(["", 0]);
    const batch_num = this.rec_batch_num;

    for (let beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num) {
      const end_img_no = Math.min(beg_img_no + batch_num, img_num);
      const norm_img_batch: number[][][][] = [];
      const imgC = this.rec_image_shape[0]!;
      const imgH = this.rec_image_shape[1]!;
      let imgW = this.rec_image_shape[2]!; // ← let に変更
      let max_wh_ratio = imgW / imgH;

      // 最大アスペクト比を計算
      for (let ino = beg_img_no; ino < end_img_no; ino++) {
        const h = img_list[indices[ino]!]?.rows!;
        const w = img_list[indices[ino]!]?.cols!;
        const wh_ratio = w / h;
        max_wh_ratio = Math.max(max_wh_ratio, wh_ratio);
      }

      // imgW を実際の最大幅に再計算
      imgW = Math.trunc(imgH * max_wh_ratio);

      // 画像を正規化してバッチに追加
      for (let ino = beg_img_no; ino < end_img_no; ino++) {
        const norm_img = this.resize_norm_img(
          img_list[indices[ino]!]!,
          max_wh_ratio,
        );
        norm_img_batch.push(norm_img);
      }

      // 正しい imgW を使用してテンソルを作成
      const images_shape = [norm_img_batch.length, imgC, imgH, imgW];
      const img_buffer = Float32Array.from(
        [
          ...[
            norm_img_batch.map((a) => [
              ...a.map((b) => [...b.map((c) => [...c])]),
            ]),
          ],
        ].flat(Infinity),
      );
      const tensor_imgs = new this.ort.Tensor(
        "float32",
        img_buffer,
        images_shape,
      );

      const input_feed = this.get_input_feed(this.rec_input_name, tensor_imgs);
      const ort_run_fetches: ORTRunFetchesType = this.rec_output_name;

      const outputs = await this.rec_onnx_session.run(
        input_feed,
        ort_run_fetches,
      );

      const result_preds = outputs[this.rec_output_name[0]!];

      if (!result_preds) {
        throw new Error("No output from the model");
      }

      const predsNdArray = tensorToNdArray(result_preds);
      const predsList = ndArrayToList(predsNdArray) as number[][][];
      const rec_result = this.postprocess_op.execute(predsList, null);

      for (let rno = 0; rno < rec_result.length; rno++) {
        rec_res[indices[beg_img_no + rno]!] = rec_result[rno]!;
      }
    }
    return rec_res;
  }

  static async get_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU,
    ort: ORT,
  ): Promise<ORTSessionReturnType> {
    const modelHash = this.get_model_hash(modelArrayBuffer);
    if (use_gpu) {
      if (_use_gpu_rec_onnx_session && _use_gpu_session_hash === modelHash) {
        return _use_gpu_rec_onnx_session;
      }
      _use_gpu_rec_onnx_session = await create_onnx_session_fn(
        ort,
        modelArrayBuffer,
        use_gpu,
      );
      _use_gpu_session_hash = modelHash;
      return _use_gpu_rec_onnx_session;
    } else {
      if (_use_cpu_rec_onnx_session && _use_cpu_session_hash === modelHash) {
        return _use_cpu_rec_onnx_session;
      }
      _use_cpu_rec_onnx_session = await create_onnx_session_fn(
        ort,
        modelArrayBuffer,
        use_gpu,
      );
      _use_cpu_session_hash = modelHash;
      return _use_cpu_rec_onnx_session;
    }
  }
}
