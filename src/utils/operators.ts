import type { NdArray } from "ndarray";
import type {
  CV2,
  Data,
  DataKeys,
  DataValues,
  MatData,
  NdArrayData,
} from "../types/type.js";
import { broadcastTo, fillValue, matToLine, matToList } from "./func.js";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import type {
  DET_LIMIT_SIDE_LEN,
  DET_LIMIT_TYPE,
} from "../types/paddle_types.js";
import type { Mat } from "@techstark/opencv-js";

export type NormalizeImageParams = {
  scale: number | null;
  mean: number[] | null;
  std: number[] | null;
  order: "hwc" | "chw" | null;
  cv: CV2;
};

export class NormalizeImage {
  scale: number;
  mean: NdArray<Float32Array>;
  std: NdArray<Float32Array>;
  shape: [number, number, number];
  shapedMean: NdArray<Float32Array>;
  shapedStd: NdArray<Float32Array>;
  cv: CV2;

  constructor(params: NormalizeImageParams) {
    this.cv = params.cv;
    this.scale = params.scale || 1.0 / 255.0;
    this.mean = Array.isArray(params.mean)
      ? ndarray(Float32Array.from(params.mean), [1, 3])
      : ndarray(Float32Array.from([0.485, 0.456, 0.406]), [1, 3]);
    this.std = Array.isArray(params.std)
      ? ndarray(Float32Array.from(params.std), [1, 3])
      : ndarray(Float32Array.from([0.229, 0.224, 0.225]), [1, 3]);
    this.shape = params.order === "hwc" ? [1, 1, 3] : [3, 1, 1];

    this.shapedMean = ndarray(this.mean.data, this.shape);
    this.shapedStd = ndarray(this.std.data, this.shape);
  }

  execute(data: MatData): MatData {
    const img = data.image;
    if (!(img instanceof this.cv.Mat)) {
      throw new Error("ToCHWImage: input image is not Mat");
    }
    const row = img.rows;
    const col = img.cols;
    const channel = img.channels();
    const imgList = matToList(img, this.cv, false) as number[][][];
    if (imgList === null || imgList === undefined) {
      throw new Error("NormalizeImage: unsupported Mat type");
    }
    const shape = [row, col, channel];
    const ndArrayImg = ndarray(Float32Array.from(imgList.flat(2)), shape);
    // img * scale
    const scaledImg = ndarray(new Float32Array(row * col * channel), shape);
    ops.assign(scaledImg, ndArrayImg);
    const result = ops.mulseq(scaledImg, this.scale);
    if (result === false) {
      throw new Error("NormalizeImage: failed to scale image");
    }

    // scaledImg - mean
    const subImg = ndarray(new Float32Array(row * col * channel), shape);

    ops.sub(subImg, scaledImg, broadcastTo(this.shapedMean, shape));

    // (scaledImg - mean) / std
    const divImg = ndarray(new Float32Array(row * col * channel), shape);
    ops.div(divImg, subImg, broadcastTo(this.shapedStd, shape));

    if (channel !== 1 && channel !== 3) {
      throw new Error(
        `NormalizeImage: unsupported channel size ${channel}, only 1 or 3 is supported`,
      );
    }

    const normalizedImg = this.cv.matFromArray(
      row,
      col,
      channel === 1 ? this.cv.CV_32F : this.cv.CV_32FC3,
      divImg.data,
    );
    img.delete();
    return {
      ...data,
      image: normalizedImg,
    };
  }
}

export type DetResizeForTestParams =
  | ((
      | {
          image_shape: [number, number];
          keep_ratio: boolean;
        }
      | {
          limit_side_len: number;
          limit_type: "max" | "min" | "resize_long" | null;
        }
      | {
          resize_long: number | null;
        }
    ) & {
      cv: CV2;
    })
  | {
      cv: CV2;
    };

export class DetResizeForTest {
  resize_type = 0;
  keep_ratio = false;
  limit_type: "max" | "min" | "resize_long" = "max";
  limit_side_len: DET_LIMIT_SIDE_LEN = 960;
  image_shape: [number, number] | null = null;
  resize_long: number | null = null;
  cv: CV2;
  constructor(params: DetResizeForTestParams) {
    this.cv = params.cv;
    if ("image_shape" in params) {
      this.image_shape = params.image_shape;
      this.resize_type = 1;
      if (params.keep_ratio) {
        this.keep_ratio = params.keep_ratio;
      }
    } else if ("limit_side_len" in params) {
      this.limit_side_len = params.limit_side_len;
      this.limit_type = params.limit_type || "min";
    } else if ("resize_long" in params) {
      this.resize_type = 2;
      this.resize_long = params.resize_long || 960;
    } else {
      this.limit_side_len = 736;
      this.limit_type = "min";
    }
  }

  execute(data: MatData): MatData {
    let img = data.image;
    if (!(img instanceof this.cv.Mat)) {
      throw new Error("ToCHWImage: input image is not Mat");
    }
    const imgH = img.rows;
    const imgW = img.cols;
    let ratio_h: number;
    let ratio_w: number;
    if (imgH + imgW < 64) {
      img = this.image_padding(img);
    }

    if (this.resize_type === 0) {
      const result = this.resize_image_type0(img);
      img = result.img;

      ratio_h = result.ratio_h;
      ratio_w = result.ratio_w;
    } else if (this.resize_type === 2) {
      const result = this.resize_image_type2(img);
      img = result.img;
      ratio_h = result.ratio_h;
      ratio_w = result.ratio_w;
    } else {
      const result = this.resize_image_type1(img);
      img = result.img;
      ratio_h = result.ratio_h;
      ratio_w = result.ratio_w;
    }
    data.image = img;
    data.shape = [imgH, imgW, ratio_h, ratio_w];
    return data;
  }

  image_padding(img: Mat, value = 0): Mat {
    const h = img.rows;
    const w = img.cols;
    const c = img.channels();
    const type = img.type();
    const defaultIm = ndarray(Float32Array.from(matToLine(img, this.cv).data), [
      h,
      w,
      c,
    ]);
    const im_pad = fillValue(
      defaultIm,
      [Math.max(32, h), Math.max(32, w), c],
      value,
    );
    const paddedImg = this.cv.matFromArray(
      Math.max(32, h),
      Math.max(32, w),
      type,
      im_pad.data,
    );
    return paddedImg;
  }

  resize_image_type1(img: Mat): { img: Mat; ratio_h: number; ratio_w: number } {
    if (this.image_shape === null) {
      throw new Error("DetResizeForTest: image_shape is null");
    }
    let [resize_h, resize_w] = this.image_shape;
    const ori_h = img.rows;
    const ori_w = img.cols;

    if (this.keep_ratio) {
      resize_w = (ori_w * resize_h) / ori_h;
      const N = Math.ceil(resize_w / 32);
      resize_w = N * 32;
    }
    const ratio_h = resize_h / ori_h;
    const ratio_w = resize_w / ori_w;
    const resizedImg = new this.cv.Mat();
    this.cv.resize(
      img,
      resizedImg,
      new this.cv.Size(Math.trunc(resize_w), Math.trunc(resize_h)),
    );
    img.delete();
    return { img: resizedImg, ratio_h, ratio_w };
  }

  resize_image_type0(img: Mat): { img: Mat; ratio_h: number; ratio_w: number } {
    const h = img.rows;
    const w = img.cols;
    let ratio: number;
    if (this.limit_type === "max") {
      if (Math.max(h, w) > this.limit_side_len) {
        if (h > w) {
          ratio = this.limit_side_len / h;
        } else {
          ratio = this.limit_side_len / w;
        }
      } else {
        ratio = 1.0;
      }
    } else if (this.limit_type === "min") {
      if (Math.min(h, w) < this.limit_side_len) {
        if (h < w) {
          ratio = this.limit_side_len / h;
        } else {
          ratio = this.limit_side_len / w;
        }
      } else {
        ratio = 1.0;
      }
    } else if (this.limit_type === "resize_long") {
      ratio = this.limit_side_len / Math.max(h, w);
    } else {
      throw new Error("DetResizeForTest: unknown limit_type");
    }
    const resize_h_tmp = h * ratio;
    const resize_w_tmp = w * ratio;

    const resize_h = Math.max(
      Math.trunc(Math.round(resize_h_tmp / 32) * 32),
      32,
    );
    const resize_w = Math.max(
      Math.trunc(Math.round(resize_w_tmp / 32) * 32),
      32,
    );

    const resized_img = new this.cv.Mat();
    try {
      this.cv.resize(img, resized_img, new this.cv.Size(resize_w, resize_h));
    } catch {
      resized_img.delete();
      throw new Error(
        `DetResizeForTest: cv.resize failed. resize_w: ${resize_w}, resize_h: ${resize_h}, img.cols: ${
          img.cols
        }, img.rows: ${img.rows}, img.type(): ${img.type()}`,
      );
    }
    const ratio_h = resize_h / h;
    const ratio_w = resize_w / w;
    img.delete();
    return { img: resized_img, ratio_h, ratio_w };
  }

  resize_image_type2(img: Mat): { img: Mat; ratio_h: number; ratio_w: number } {
    const h = img.rows;
    const w = img.cols;
    let ratio: number;
    if (this.resize_long === null) {
      throw new Error("DetResizeForTest: resize_long is null");
    }
    if (h > w) {
      ratio = this.resize_long / h;
    } else {
      ratio = this.resize_long / w;
    }

    const resize_h_tmp = Math.trunc(h * ratio);
    const resize_w_tmp = Math.trunc(w * ratio);

    const max_stride = 128;

    const resize_h =
      Math.trunc((resize_h_tmp + max_stride - 1) / max_stride) * max_stride;
    const resize_w =
      Math.trunc((resize_w_tmp + max_stride - 1) / max_stride) * max_stride;
    const dst_img = new this.cv.Mat();
    this.cv.resize(img, dst_img, new this.cv.Size(resize_w, resize_h));
    const ratio_h = resize_h / h;
    const ratio_w = resize_w / w;
    img.delete();
    return { img: dst_img, ratio_h, ratio_w };
  }
}

export type ToCHWImageParams = {
  cv: CV2;
};

export class ToCHWImage {
  cv: CV2;
  constructor(params: ToCHWImageParams) {
    this.cv = params.cv;
  }
  execute(data: Data): NdArrayData {
    const img = data.image;
    if (!(img instanceof this.cv.Mat)) {
      throw new Error("ToCHWImage: input image is not Mat");
    }
    const imgList = matToLine(img, this.cv);
    let ndArrayImg: NdArray = ndarray(imgList.data, [
      img.rows,
      img.cols,
      img.channels(),
    ]);
    const transposedImg = ndArrayImg.transpose(2, 0, 1);
    img.delete();
    data.image = transposedImg;
    return data as NdArrayData;
  }
}

export type KeepKeysParams = {
  keep_keys: DataKeys[];
};

export class KeepKeys {
  keep_keys: DataKeys[];
  constructor(params: KeepKeysParams) {
    this.keep_keys = params.keep_keys;
  }
  execute(data: Data): DataValues[] {
    let data_list: DataValues[] = [];
    for (const key of this.keep_keys) {
      if (key in data) {
        data_list.push(data[key]);
      }
    }
    return data_list;
  }
}

export type OperatorConfig =
  | {
      type: "NormalizeImage";
      params: NormalizeImageParams;
    }
  | {
      type: "DetResizeForTest";
      params: DetResizeForTestParams;
    }
  | {
      type: "ToCHWImage";
      params: ToCHWImageParams;
    }
  | {
      type: "KeepKeys";
      params: KeepKeysParams;
    };

export function createOperator(config: OperatorConfig) {
  switch (config.type) {
    case "NormalizeImage":
      return new NormalizeImage(config.params);
    case "DetResizeForTest":
      return new DetResizeForTest(config.params);
    case "ToCHWImage":
      return new ToCHWImage(config.params);
    case "KeepKeys":
      return new KeepKeys(config.params);
    default:
      throw new Error(`Unknown operator type`);
  }
}

export type PreProcessOperator =
  | NormalizeImage
  | DetResizeForTest
  | ToCHWImage
  | KeepKeys;

export function transform(
  data: Data,
  ops: PreProcessOperator[],
): Data | DataValues[] | null {
  if (ops.length === 0) {
    return data;
  }
  let op_data: Data = data;
  let op_data_list: DataValues[];
  let isKeepKeysLast = false;
  if (ops[ops.length - 1] instanceof KeepKeys) {
    isKeepKeysLast = true;
  }
  const lastOpIndex = isKeepKeysLast ? ops.length - 1 : ops.length;
  for (let i = 0; i < lastOpIndex; i++) {
    const op = ops[i]!;
    op_data = op.execute(op_data as any) as Data;
    if (op_data === null) {
      return null;
    }
  }
  if (isKeepKeysLast) {
    const keepKeysOp = ops[ops.length - 1] as KeepKeys;
    op_data_list = keepKeysOp.execute(op_data);
    if (op_data_list === null) {
      return null;
    }
    return op_data_list;
  }
  return op_data;
}
