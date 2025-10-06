import ndarray from "ndarray";
import ops from "ndarray-ops";
import type {
  DET_BOX_TYPE,
  DET_DB_BOX_THRESH,
  DET_DB_SCORE_MODE,
  DET_DB_THRESH,
  DET_DB_UNCLIP_RATIO,
  USE_DILATION,
} from "./types/paddle_types.js";
import type { NdArray } from "ndarray";
import type { Box, CV2, Point } from "./types/type.js";
import type { Mat, Point2f } from "@techstark/opencv-js";
import {
  clip,
  cloneNdArray,
  matToLine,
  matToList,
  matToNdArray,
  ndArrayToList,
  pickAndSet,
  unclip,
  type NdArrayListData,
} from "./utils/func.js";
import type { Paths } from "js-angusj-clipper";
import { ndArrayToMat } from "./utils/funcs/ndarray-to-mat.js";

export type DBPostProcessParams = {
  name: string;
  thresh: DET_DB_THRESH | null;
  box_thresh: DET_DB_BOX_THRESH | null;
  max_candidates: number | null;
  unclip_ratio: DET_DB_UNCLIP_RATIO | null;
  use_dilation: USE_DILATION | null;
  score_mode: DET_DB_SCORE_MODE | null;
  box_type: DET_BOX_TYPE | null;
  cv: CV2;
};

// export type OutDict = {
//   maps:NdArray,
//   rec_pred:NdArray
//   rec_pred_scores:NdArray
//   predict:NdArray
//   align:NdArray
//   ctc:NdArray
//   sar:NdArray
//   points:Point[]
// }

export type OutDict = {
  maps: NdArray;
};

export class DBPostProcess {
  name: string;
  thresh: DET_DB_THRESH;
  box_thresh: DET_DB_BOX_THRESH;
  max_candidates: number;
  unclip_ratio: DET_DB_UNCLIP_RATIO;
  use_dilation: USE_DILATION;
  score_mode: DET_DB_SCORE_MODE;
  box_type: DET_BOX_TYPE;
  min_size: number;
  dilation_kernel: NdArray | null = null;
  cv: CV2;
  constructor(params: DBPostProcessParams) {
    this.name = params.name;
    this.thresh = params.thresh ?? 0.3;
    this.box_thresh = params.box_thresh ?? 0.7;
    this.max_candidates = params.max_candidates ?? 1000;
    this.unclip_ratio = params.unclip_ratio ?? 2.0;
    this.use_dilation = params.use_dilation ?? false;
    this.score_mode = params.score_mode ?? "fast";
    this.box_type = params.box_type ?? "quad";
    this.min_size = 3;
    this.cv = params.cv;
    if (this.use_dilation) {
      this.dilation_kernel = ndarray(new Uint8Array([1, 1, 1, 1]), [2, 2]);
    }
  }

  async execute(
    outs_dict: OutDict,
    shape_list: [number, number, number, number][],
  ) {
    const pred = outs_dict.maps;
    const pickedPred = pred.pick(-1, 0, -1, -1); // X*Y*Z*(1) [[[]]]
    const segmentationPred = ndarray(
      new Float32Array(pickedPred.size),
      pickedPred.shape,
    );
    // console.log("pickedPred:", pickedPred);
    ops.gts(segmentationPred, pickedPred, this.thresh);
    // for batch_index in range(pred.shape[0]):
    if (typeof pred.shape[0] !== "number") {
      throw new Error("pred.shape[0] is not number");
    }
    if (pred.shape[0] !== shape_list.length) {
      throw new Error("pred.shape[0] and shape_list length mismatch");
    }
    if (pred.shape[0] !== segmentationPred.shape[0]) {
      throw new Error("pred.shape[0] and segmentationPred shape[0] mismatch");
    }

    const li = (ndArrayToList(segmentationPred) as number[][][]).flat(
      Infinity,
    ) as number[];
    // console.log("bigger");
    for (let i = 0; i < li.length; i++) {
      if (li[i]! > 0) {
        // console.log(li[i]);
        // console.log(i);
      }
    }
    // console.log("bigger end");

    const boxes_batch: {
      points: NdArray | number[][][];
    }[] = [];
    for (let batch_index = 0; batch_index < pred.shape[0]; batch_index++) {
      const [src_h, src_w, ratio_h, ratio_w] = shape_list[batch_index]!;
      const currentSegment = segmentationPred.pick(batch_index, -1, -1); // segmentationPred[batch_index]
      // console.log("currentSegment", currentSegment);
      const matSegment = this.cv.matFromArray(
        currentSegment.shape[0]!,
        currentSegment.shape[1]!,
        this.cv.CV_8UC1,
        (ndArrayToList(currentSegment) as number[][]).flat(
          Infinity,
        ) as number[],
      );
      // console.log("matSegment:", matSegment);
      const maskSegment = new this.cv.Mat();
      if (this.dilation_kernel !== null) {
        const kernel = this.cv.matFromArray(
          this.dilation_kernel.shape[0]!,
          this.dilation_kernel.shape[1]!,
          this.cv.CV_8UC1,
          this.dilation_kernel.data,
        );
        this.cv.dilate(matSegment, maskSegment, kernel);
        kernel.delete();
      } else {
        matSegment.copyTo(maskSegment);
      }
      matSegment.delete();
      let boxes: Point[][] | NdArray;
      let score: number[];
      if (this.box_type === "poly") {
        [boxes, score] = await this.polygons_from_bitmap(
          currentSegment,
          maskSegment,
          src_w,
          src_h,
        );
      } else if (this.box_type === "quad") {
        [boxes, score] = await this.boxes_from_bitmap(
          currentSegment,
          maskSegment,
          src_w,
          src_h,
        );
      } else {
        throw new Error(`Unknown box_type: ${this.box_type}`);
      }
      // console.log("boxes!:", boxes);
      boxes_batch.push({ points: boxes });
    }
    // console.log("boxes_batch:", boxes_batch);
    return boxes_batch;
  }

  async polygons_from_bitmap(
    pred: NdArray,
    _bitmap: Mat,
    dest_width: number,
    dest_height: number,
  ): Promise<[Point[][], number[]]> {
    const bitmap = _bitmap;
    const height = bitmap.rows;
    const width = bitmap.cols;

    const boxes: Point[][] = [];
    const scores = [];
    const result = matToLine(bitmap, this.cv);
    const bitmapData = result.data.map((v) => (v === 1 ? 255 : 0));
    const scaledBitMap = this.cv.matFromArray(
      result.row,
      result.col,
      this.cv.CV_8UC1,
      bitmapData,
    );
    const contours = new this.cv.MatVector();
    const _hierarchy = new this.cv.Mat();

    this.cv.findContours(
      scaledBitMap,
      contours,
      _hierarchy,
      this.cv.RETR_LIST,
      this.cv.CHAIN_APPROX_SIMPLE,
    );
    for (let i = 0; i < Math.min(contours.size(), this.max_candidates); i++) {
      const contour = contours.get(i);
      const epsilon = 0.002 * this.cv.arcLength(contour, true);
      const approx = new this.cv.Mat();

      this.cv.approxPolyDP(contour, approx, epsilon, true);

      const points = approx;
      if (points.rows < 4) {
        continue;
      }
      const pointsList = matToList(points, this.cv) as
        | number[][]
        | number[][][];
      const flattedList = pointsList.flat(Infinity) as number[];
      const score: number = this.box_score_fast(
        pred,
        ndarray(flattedList, [flattedList.length / 2, 2]),
      );
      if (this.box_thresh > score) {
        continue;
      }
      let box: Mat;
      if (points.rows > 2) {
        let pointsNdArray = matToNdArray(points, this.cv, true); // 可能性: [N,1,2]
        if (
          pointsNdArray.shape.length === 3 &&
          pointsNdArray.shape[1] === 1 &&
          pointsNdArray.shape[2] === 2
        ) {
          pointsNdArray = ndarray(pointsNdArray.data, [
            pointsNdArray.shape[0]!,
            pointsNdArray.shape[2]!,
          ]);
        }
        const unclipedBox = await this.ndArrayUnclip(
          pointsNdArray,
          this.unclip_ratio,
          true,
        ); // (M,2) or null
        if (unclipedBox === null) {
          continue;
        }
        box =
          unclipedBox === null ? points : ndArrayToMat(unclipedBox, this.cv);
      } else {
        continue;
      }
      const [_box, sside] = this.get_mini_boxes(box);
      if (sside < this.min_size + 2) {
        if (box !== points) {
          box.delete();
        }
        continue;
      }
      const ndArrayBox = matToNdArray(box, this.cv, true); // (N,2)
      const ndArrayBoxUpdated01 = pickAndSet(
        ndArrayBox,
        (view) => {
          // 新しい配列を作成（元のviewを変更しない）
          const result = cloneNdArray(view);

          // x座標をスケーリング: x' = round(x / width * dest_width)
          ops.divseq(result, width);
          ops.mulseq(result, dest_width);
          ops.roundeq(result);

          // クリップして返す
          clip(result, result, 0, dest_width);
          return result;
        },
        -1,
        0,
      );
      const ndArrayBoxUpdated = pickAndSet(
        ndArrayBoxUpdated01,
        (view) => {
          const result = cloneNdArray(view);

          // y座標をスケーリング: y' = round(y / height * dest_height)
          ops.divseq(result, height);
          ops.mulseq(result, dest_height);
          ops.roundeq(result);

          clip(result, result, 0, dest_height);
          return result;
        },
        -1,
        1,
      );
      boxes.push(ndArrayToList(ndArrayBoxUpdated) as Point[]);
      scores.push(score);
      if (box !== points) {
        box.delete();
      }
    }
    // メモリクリーンアップ
    scaledBitMap.delete();
    contours.delete();
    _hierarchy.delete();

    return [boxes, scores];
  }

  async boxes_from_bitmap(
    pred: NdArray,
    _bitmap: Mat,
    dest_width: number,
    dest_height: number,
  ): Promise<[NdArray, number[]]> {
    const bitmap = _bitmap;
    const height = bitmap.rows;
    const width = bitmap.cols;

    // バイナリ画像を255スケールに変換
    const result = matToLine(bitmap, this.cv);
    const bitmapData = result.data.map((v) => (v === 1 ? 255 : 0));
    const scaledBitMap = this.cv.matFromArray(
      result.row,
      result.col,
      this.cv.CV_8UC1,
      bitmapData,
    );

    // 輪郭検出
    const contours = new this.cv.MatVector();
    const _hierarchy = new this.cv.Mat();
    this.cv.findContours(
      scaledBitMap,
      contours,
      _hierarchy,
      this.cv.RETR_LIST,
      this.cv.CHAIN_APPROX_SIMPLE,
    );

    const num_contours = Math.min(contours.size(), this.max_candidates);
    const boxes: Box[] = [];
    const scores: number[] = [];

    for (let index = 0; index < num_contours; index++) {
      const contour = contours.get(index);

      // 最小外接矩形の取得
      const [points, sside] = this.get_mini_boxes(contour);
      if (sside < this.min_size) {
        contour.delete();
        continue;
      }

      // points を ndarray に変換
      const flattedPoints = (points as number[][]).flat(Infinity) as number[];
      const pointsArray = ndarray(new Float32Array(flattedPoints), [
        flattedPoints.length / 2,
        2,
      ]);

      // スコア計算
      let score: number;
      if (this.score_mode === "fast") {
        score = this.box_score_fast(pred, pointsArray);
      } else {
        score = this.box_score_slow(pred, contour);
      }

      if (this.box_thresh > score) {
        contour.delete();
        continue;
      }

      // unclip処理
      const unclippedBox = await this.ndArrayUnclip(
        pointsArray,
        this.unclip_ratio,
      );

      if (unclippedBox === null) {
        contour.delete();
        continue;
      }

      // unclippedBoxを適切な形状に変換して再度mini_boxesを取得
      const reshapedUnclippedBox = ndarray(unclippedBox.data, [
        unclippedBox.shape[0]!,
        1,
        2,
      ]);
      const unclippedMat = ndArrayToMat(reshapedUnclippedBox, this.cv);
      const [finalBox, finalSside] = this.get_mini_boxes(unclippedMat);

      if (finalSside < this.min_size + 2) {
        contour.delete();
        unclippedMat.delete();
        continue;
      }

      // 最終的なボックスをndarrayに変換
      const ndArrayBox = ndarray(new Float32Array(finalBox.flat()), [
        finalBox.length,
        2,
      ]);

      // x座標のスケーリング
      const ndArrayBoxUpdated01 = pickAndSet(
        ndArrayBox,
        (view) => {
          const result = cloneNdArray(view);
          ops.divseq(result, width);
          ops.mulseq(result, dest_width);
          ops.roundeq(result);
          clip(result, result, 0, dest_width);
          return result;
        },
        -1,
        0,
      );

      // y座標のスケーリング
      const ndArrayBoxUpdated = pickAndSet(
        ndArrayBoxUpdated01,
        (view) => {
          const result = cloneNdArray(view);
          ops.divseq(result, height);
          ops.mulseq(result, dest_height);
          ops.roundeq(result);
          clip(result, result, 0, dest_height);
          return result;
        },
        -1,
        1,
      );

      // int32に変換
      const intLine = (
        (ndArrayToList(ndArrayBoxUpdated) as number[][]).flat(
          Infinity,
        ) as number[]
      ).map((v) => Math.trunc(v));
      const intBox = ndarray(Int32Array.from(intLine), ndArrayBoxUpdated.shape);

      boxes.push(ndArrayToList(intBox) as Box);
      scores.push(score);

      // メモリクリーンアップ
      contour.delete();
      unclippedMat.delete();
    }

    // メモリクリーンアップ
    scaledBitMap.delete();
    contours.delete();
    _hierarchy.delete();

    // boxes配列をndarrayに変換
    const maxBoxes = boxes.length;
    if (maxBoxes === 0) {
      // 空の場合
      return [ndarray(new Int32Array([]), [0, 4, 2]), []];
    }

    // すべてのボックスが同じ形状であることを前提として、フラット化
    const boxData: number[] = [];
    for (const box of boxes) {
      boxData.push(...(box.flat(Infinity) as number[]));
    }

    // boxes は (num_boxes, 4, 2) の形状にする
    const boxesNdArray = ndarray(Int32Array.from(boxData), [maxBoxes, 4, 2]);

    return [boxesNdArray, scores];
  }

  async ndArrayUnclip(
    box: NdArray,
    unclip_ratio: number,
    onlyOneBox = false,
  ): Promise<NdArray | null> {
    const boxList: [number, number][] = ndArrayToList(box) as [
      number,
      number,
    ][];
    if (boxList.length === 0) {
      return null;
    }
    if (boxList[0]?.length !== 2) {
      throw new Error("ndArrayUnclip: boxList[0] is not [number,number]");
    }
    const clipperBox = boxList.map(([x, y]) => ({ x, y }));
    const result: Paths = await unclip(clipperBox, unclip_ratio);
    if (result.length === 0 || result[0] === undefined) {
      return null;
    }
    if (onlyOneBox && result.length !== 1) {
      return null;
    }
    const resultList = result.map((p) =>
      p.map((v) => {
        return [v.x, v.y];
      }),
    );
    const flatted = resultList.flat(Infinity) as number[];
    const ndArrayResult = ndarray(Float32Array.from(flatted), [
      flatted.length / 2,
      2,
    ]);
    return ndArrayResult;
  }

  box_score_fast(bitmap: NdArray, _box: NdArray) {
    const boxData = ndArrayToList(_box) as number[][];
    const box = ndarray(
      Int32Array.from(boxData.flat(Infinity) as number[]),
      _box.shape,
      _box.stride,
      _box.offset,
    );
    const height = bitmap.shape[0]!;
    const width = bitmap.shape[1]!;
    const flooredN0 = Math.trunc(
      Math.min(
        ...((ndArrayToList(box.pick(-1, 0)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const ceiledN0 = Math.ceil(
      Math.max(
        ...((ndArrayToList(box.pick(-1, 0)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const flooredN1 = Math.trunc(
      Math.min(
        ...((ndArrayToList(box.pick(-1, 1)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const ceiledN1 = Math.ceil(
      Math.max(
        ...((ndArrayToList(box.pick(-1, 1)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );

    const xmin = Math.min(Math.max(flooredN0, 0), width - 1);
    const xmax = Math.min(Math.max(ceiledN0, 0), width - 1);
    const ymin = Math.min(Math.max(flooredN1, 0), height - 1);
    const ymax = Math.min(Math.max(ceiledN1, 0), height - 1);

    const pickedN0 = pickAndSet(
      box,
      (view) => {
        const isSuccess = ops.subseq(view, xmin);
        if (!isSuccess) throw new Error("box_score_fast: ops.subseq failed");
        return view;
      },
      -1,
      0,
    );
    const pickedN1 = pickAndSet(
      pickedN0,
      (view) => {
        const isSuccess = ops.subseq(view, ymin);
        if (!isSuccess) throw new Error("box_score_fast: ops.subseq failed");
        return view;
      },
      -1,
      1,
    );

    const maskCV = this.cv.matFromArray(
      ymax - ymin + 1,
      xmax - xmin + 1,
      this.cv.CV_8UC1,
      Uint8Array.from(Array((ymax - ymin + 1) * (xmax - xmin + 1)).fill(0)),
    );

    const boxCV = ndArrayToMat(pickedN1, this.cv);
    const vectorBoxCV = new this.cv.MatVector();
    vectorBoxCV.push_back(boxCV);
    const color = new this.cv.Scalar(1, 1, 1, 1);
    this.cv.fillPoly(maskCV, vectorBoxCV, color);
    const clonedBitmap = cloneNdArray(bitmap);
    const picked = clonedBitmap.hi(ymax + 1, xmax + 1).lo(ymin, xmin);
    const pickedMat = ndArrayToMat(picked, this.cv);
    const meanValue = this.cv.mean(pickedMat, maskCV)[0]!;

    // メモリクリーンアップ
    boxCV.delete();
    vectorBoxCV.delete();
    maskCV.delete();
    pickedMat.delete();

    return meanValue;
  }

  box_score_slow(bitmap: NdArray, _contour: Mat) {
    const counterData = matToList(_contour, this.cv) as
      | number[][]
      | number[][][];
    const counter = ndarray(Int32Array.from(counterData.flat(Infinity)), [
      _contour.rows,
      _contour.cols,
      _contour.channels(),
    ]);
    let counterShapeTotal = 1;
    for (const i of counter.shape) {
      counterShapeTotal *= i;
    }
    const isCounterEven = counterShapeTotal % 2 === 0;
    if (!isCounterEven) {
      throw new Error("box_score_slow: counter shape is not even");
    }
    const reshapedCounter = ndarray(counter.data, [counterShapeTotal / 2, 2]);
    const height = bitmap.shape[0]!;
    const width = bitmap.shape[1]!;
    const flooredN0 = Math.trunc(
      Math.min(
        ...((ndArrayToList(reshapedCounter.pick(-1, 0)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const ceiledN0 = Math.ceil(
      Math.max(
        ...((ndArrayToList(reshapedCounter.pick(-1, 0)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const flooredN1 = Math.trunc(
      Math.min(
        ...((ndArrayToList(reshapedCounter.pick(-1, 1)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );
    const ceiledN1 = Math.ceil(
      Math.max(
        ...((ndArrayToList(reshapedCounter.pick(-1, 1)) as number[][]).flat(
          Infinity,
        ) as number[]),
      ),
    );

    const xmin = Math.min(Math.max(flooredN0, 0), width - 1);
    const xmax = Math.min(Math.max(ceiledN0, 0), width - 1);
    const ymin = Math.min(Math.max(flooredN1, 0), height - 1);
    const ymax = Math.min(Math.max(ceiledN1, 0), height - 1);

    const maskCV = this.cv.matFromArray(
      ymax - ymin + 1,
      xmax - xmin + 1,
      this.cv.CV_8UC1,
      Uint8Array.from(Array((ymax - ymin + 1) * (xmax - xmin + 1)).fill(0)),
    );

    const pikedN10 = pickAndSet(
      reshapedCounter,
      (view) => {
        const isSuccess = ops.subseq(view, xmin);
        if (!isSuccess) throw new Error("box_score_fast: ops.subseq failed");
        return view;
      },
      -1,
      0,
    );
    const pikedN1 = pickAndSet(
      pikedN10,
      (view) => {
        const isSuccess = ops.subseq(view, ymin);
        if (!isSuccess) throw new Error("box_score_fast: ops.subseq failed");
        return view;
      },
      -1,
      1,
    );
    const pickedArray = (ndArrayToList(pikedN1) as number[][]).flat();
    const counterCV = this.cv.matFromArray(
      pikedN1.shape[0]!,
      1,
      this.cv.CV_32SC2,
      Int32Array.from(pickedArray as number[]),
    ); //(-1,1,2)
    const vectorBoxCV = new this.cv.MatVector();
    vectorBoxCV.push_back(counterCV);
    const color = new this.cv.Scalar(1, 1, 1, 1);
    this.cv.fillPoly(maskCV, vectorBoxCV, color);
    const clonedBitmap = cloneNdArray(bitmap);
    const picked = clonedBitmap.hi(ymax + 1, xmax + 1).lo(ymin, xmin);
    const pickedMat = ndArrayToMat(picked, this.cv);
    const meanValue = this.cv.mean(pickedMat, maskCV)[0]!;

    // メモリクリーンアップ
    counterCV.delete();
    vectorBoxCV.delete();
    maskCV.delete();
    pickedMat.delete();

    return meanValue;
  }

  get_mini_boxes(counter: Mat): [Box, number] {
    const bounding_box = this.cv.minAreaRect(counter);
    // const matPoints: Mat = new this.cv.Mat();
    // this.cv.boxPoints(bounding_box, matPoints);
    const f2Points = this.cv.boxPoints(bounding_box);
    // const beforePoints: Point[] = matToPoints(matPoints, this.cv);
    const beforePoints: Point[] = f2Points.map((p) => [p.x, p.y]);

    if (beforePoints.length !== 4) {
      throw new Error("shape of points must be 4*2");
    }
    const points = beforePoints.sort((a, b) => a[0] - b[0]) as Box;

    let index_a = 0,
      index_b = 1,
      index_c = 2,
      index_d = 3;

    if (points[1][1] > points[0][1]) {
      index_a = 0;
      index_d = 1;
    } else {
      index_a = 1;
      index_d = 0;
    }

    if (points[3][1] > points[2][1]) {
      index_b = 2;
      index_c = 3;
    } else {
      index_b = 3;
      index_c = 2;
    }

    const sortedBox: Box = [
      points[index_a]!,
      points[index_b]!,
      points[index_c]!,
      points[index_d]!,
    ];
    const returnMin = Math.min(
      bounding_box.size.height,
      bounding_box.size.width,
    );
    return [sortedBox, returnMin];
  }
}
