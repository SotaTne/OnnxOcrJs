import type { Mat } from "@techstark/opencv-js";
import type {
  CV2,
  Data,
  DataValues,
  NdArrayData,
  ORT,
  ORTBufferType,
  ORTRunFetchesType,
  ORTSessionReturnType,
  Point,
} from "./types/type.js";
import {
  createOperator,
  transform,
  type PreProcessOperator,
} from "./utils/operators.js";
import {
  DBPostProcess,
  type DBPostProcessParams,
  type OutDict,
} from "./db_postprocess.js";
import type {
  DET_BOX_TYPE,
  DET_DB_BOX_THRESH,
  DET_DB_SCORE_MODE,
  DET_DB_THRESH,
  DET_DB_UNCLIP_RATIO,
  USE_DILATION,
  USE_GCU,
} from "./types/paddle_types.js";
import { PredictBase } from "./predict_base.js";
import { create_onnx_session_fn } from "./onnx_runtime.js";
import {
  cloneNdArray,
  euclideanDistance,
  ndArrayToList,
  tensorToNdArray,
} from "./utils/func.js";

let _use_gpu_det_onnx_session: ORTSessionReturnType | undefined = undefined;
let _use_gpu_session_hash: string | undefined = undefined;
let _use_cpu_det_onnx_session: ORTSessionReturnType | undefined = undefined;
let _use_cpu_session_hash: string | undefined = undefined;

export type TextDetectorParams = {
  drop_score: number | null;
  limit_side_len: number;
  det_limit_type: "max" | "min" | "resize_long" | null;
  det_db_thresh: DET_DB_THRESH;
  det_db_box_thresh: DET_DB_BOX_THRESH;
  det_db_unclip_ratio: DET_DB_UNCLIP_RATIO;
  use_dilation: USE_DILATION;
  det_db_score_mode: DET_DB_SCORE_MODE;
  det_box_type: DET_BOX_TYPE;
  cv: CV2;
  ort: ORT;
  det_model_array_buffer: ORTBufferType;
  use_gpu: USE_GCU;
};

export class TextDetector extends PredictBase {
  cv: CV2;
  postprocess_op: DBPostProcess;
  preprocess_op: PreProcessOperator[];
  ort: ORT;
  det_onnx_session: ORTSessionReturnType;
  det_input_name: string[];
  det_output_name: string[];
  det_box_type: DET_BOX_TYPE;

  constructor(
    params: TextDetectorParams & {
      det_onnx_session: ORTSessionReturnType;
    },
  ) {
    super();
    this.cv = params.cv;
    this.ort = params.ort;
    this.preprocess_op = [
      createOperator({
        type: "DetResizeForTest",
        params: {
          cv: this.cv,
          limit_side_len: params.limit_side_len,
          limit_type: params.det_limit_type,
        },
      }),
      createOperator({
        type: "NormalizeImage",
        params: {
          cv: this.cv,
          std: [0.229, 0.224, 0.225],
          mean: [0.485, 0.456, 0.406],
          scale: 1.0 / 255.0,
          order: "hwc",
        },
      }),
      createOperator({
        type: "ToCHWImage",
        params: {
          cv: this.cv,
        },
      }),
      createOperator({
        type: "KeepKeys",
        params: {
          keep_keys: ["image", "shape"],
        },
      }),
    ];
    const postprocess_params: DBPostProcessParams = {
      name: "DBPostProcess",
      thresh: params.det_db_thresh,
      box_thresh: params.det_db_box_thresh,
      max_candidates: 1000,
      unclip_ratio: params.det_db_unclip_ratio,
      use_dilation: params.use_dilation,
      score_mode: params.det_db_score_mode,
      box_type: params.det_box_type,
      cv: this.cv,
    };
    this.postprocess_op = new DBPostProcess(postprocess_params);

    this.det_box_type = params.det_box_type;

    this.det_onnx_session = params.det_onnx_session;
    this.det_input_name = this.get_input_name(this.det_onnx_session);
    this.det_output_name = this.get_output_name(this.det_onnx_session);
  }

  static async create(params: TextDetectorParams) {
    const det_onnx_session = await TextDetector.get_onnx_session(
      params.det_model_array_buffer,
      params.use_gpu,
      params.ort,
    );
    return new TextDetector({ ...params, det_onnx_session });
  }

  order_points_clockwise(pts: Point[]): Point[] {
    const rect: Point[] = [
      [0, 0], // top-left
      [0, 0], // top-right
      [0, 0], // bottom-right
      [0, 0], // bottom-left
    ];

    const sumPts = pts.map((p) => p[0] + p[1]);

    const minIndex = sumPts.indexOf(Math.min(...sumPts)); // top-left
    const maxIndex = sumPts.indexOf(Math.max(...sumPts)); // bottom-right

    rect[0] = pts[minIndex]!;
    rect[2] = pts[maxIndex]!;

    const tmp = pts.filter((_, i) => i !== minIndex && i !== maxIndex);

    const diff = tmp.map((p) => p[1] - p[0]); // ← ここを変更
    const minDiffIndex = diff.indexOf(Math.min(...diff)); // top-right
    const maxDiffIndex = diff.indexOf(Math.max(...diff)); // bottom-left

    rect[1] = tmp[minDiffIndex]!;
    rect[3] = tmp[maxDiffIndex]!;

    return rect;
  }

  clip_det_res(points: Point[], img_height: number, img_width: number) {
    let new_points = [...points.map((p) => [...p])] as Point[];
    for (let p of new_points) {
      p[0] = Math.round(Math.min(Math.max(p[0], 0), img_width - 1));
      p[1] = Math.round(Math.min(Math.max(p[1], 0), img_height - 1));
    }
    return new_points;
  }

  filter_tag_det_res(dt_boxes: Point[][], image_shape: number[]) {
    const [img_height, img_width] = image_shape;
    const db_boxes_new = [];
    for (const box of dt_boxes) {
      let box_new = this.order_points_clockwise(box);
      box_new = this.clip_det_res(box_new, img_height!, img_width!);
      const rect_width = Math.round(
        euclideanDistance(box_new[0]!, box_new[1]!),
      );
      const rect_height = Math.round(
        euclideanDistance(box_new[0]!, box_new[3]!),
      );
      if (rect_width <= 3 || rect_height <= 3) continue;

      // order_points_clockwiseは [top-left, top-right, bottom-right, bottom-left] を返す
      // しかし期待値は [top-left, bottom-left, bottom-right, top-right] の順序
      // これは実際にはPython版のget_mini_boxesと同じ順序
      db_boxes_new.push(box_new);
    }
    return db_boxes_new;
  }

  filter_tag_det_res_only_clip(dt_boxes: Point[][], image_shape: number[]) {
    const [img_height, img_width] = image_shape;
    const db_boxes_new = [];
    for (const box of dt_boxes) {
      const box_new = this.clip_det_res(box, img_height!, img_width!);
      db_boxes_new.push(box_new);
    }
    return db_boxes_new;
  }

  async execute(_img: Mat) {
    const ori_im = _img.clone();
    const data: Data = { image: ori_im, shape: null };
    const transformed = transform(data, this.preprocess_op) as
      | DataValues[]
      | null;
    if (!transformed || !transformed[0]) {
      ori_im.delete();
      return null;
    }

    const img = transformed[0]! as NdArrayData["image"];
    const shape_list = transformed[1]! as Data["shape"];

    const cloned_img = cloneNdArray(img);
    const img_buffer = Float32Array.from(
      (ndArrayToList(cloned_img) as number[][][]).flat(Infinity),
    );
    const img_shape = [1, ...cloned_img.shape];
    //console.log("img_shape:", img_shape);
    const tensor_img = new this.ort.Tensor("float32", img_buffer, img_shape);
    const input_feed = this.get_input_feed(this.det_input_name, tensor_img);

    const outputs = await this.det_onnx_session.run(
      input_feed,
      this.det_output_name,
    );
    const result_preds = outputs[this.det_output_name[0]!];
    if (!result_preds) throw new Error("No output from the model");

    const preds: OutDict = { maps: tensorToNdArray(result_preds) };
    const post_shape_list = [shape_list!]; // Pythonと同じ形に揃える

    const post_result = await this.postprocess_op.execute(
      preds,
      post_shape_list,
    );
    const dt_boxes = post_result[0]!["points"];

    const dt_boxes_array = (
      Array.isArray(dt_boxes)
        ? dt_boxes
        : (ndArrayToList(dt_boxes) as number[][][])
    ) as Point[][];

    return this.det_box_type === "poly"
      ? this.filter_tag_det_res_only_clip(dt_boxes_array, [
          shape_list![0],
          shape_list![1],
        ])
      : this.filter_tag_det_res(dt_boxes_array, [
          shape_list![0],
          shape_list![1],
        ]);
  }

  static async get_onnx_session(
    modelArrayBuffer: ORTBufferType,
    use_gpu: USE_GCU,
    ort: ORT,
  ): Promise<ORTSessionReturnType> {
    const modelHash = this.get_model_hash(modelArrayBuffer);

    if (use_gpu) {
      if (_use_gpu_det_onnx_session && _use_gpu_session_hash === modelHash) {
        return _use_gpu_det_onnx_session;
      }
      _use_gpu_det_onnx_session = await create_onnx_session_fn(
        ort,
        modelArrayBuffer,
        use_gpu,
      );
      _use_gpu_session_hash = modelHash;
      return _use_gpu_det_onnx_session;
    } else {
      if (_use_cpu_det_onnx_session && _use_cpu_session_hash === modelHash) {
        return _use_cpu_det_onnx_session;
      }
      _use_cpu_det_onnx_session = await create_onnx_session_fn(
        ort,
        modelArrayBuffer,
        use_gpu,
      );
      _use_cpu_session_hash = modelHash;
      return _use_cpu_det_onnx_session;
    }
  }
}
