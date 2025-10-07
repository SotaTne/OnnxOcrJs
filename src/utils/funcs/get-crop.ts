import type { Mat } from "@techstark/opencv-js";
import type { Box, CV2, Point } from "../../types/type.js";
import type { DET_BOX_TYPE } from "../../types/paddle_types.js";
import { boxToMat, euclideanDistance } from "../func.js";

/**
 * テキストボックスを読み順（上から下、左から右）にソート
 * Python版のsorted_boxes関数と同等
 */
export function sortedBoxes(dt_boxes: Point[][]): Point[][] {
  const numBoxes = dt_boxes.length;

  // まず (y, x) で大まかにソート
  const sorted = dt_boxes.slice().sort((a, b) => {
    if (a[0]![1] === b[0]![1]) {
      return a[0]![0] - b[0]![0]; // y が同じなら x でソート
    }
    return a[0]![1] - b[0]![1]; // y でソート
  });

  const boxes = [...sorted.map((b) => [...b])] as Point[][];

  // 同じ行内での微調整（y座標の差が10px以内なら左右で並べ替え）
  for (let i = 0; i < numBoxes - 1; i++) {
    for (let j = i; j >= 0; j--) {
      const yDiff = Math.abs(boxes[j + 1]![0]![1] - boxes[j]![0]![1]);
      const xRight = boxes[j + 1]![0]![0];
      const xLeft = boxes[j]![0]![0];

      if (yDiff < 10 && xRight < xLeft) {
        // swap
        const tmp = boxes[j]!;
        boxes[j] = boxes[j + 1]!;
        boxes[j + 1] = tmp;
      } else {
        break;
      }
    }
  }

  return boxes;
}

/**
 * getImgCropListの戻り値型
 */
export type CropResult = {
  crops: Mat[];
  sortedBoxes: Point[][];
};

/**
 * 検出されたボックスをソートし、各ボックスの領域をクロップした画像リストを返す
 *
 * @param ori_img - 元画像
 * @param dt_boxes - 検出されたボックス（未ソート）
 * @param det_box_type - ボックスタイプ（"quad" or "poly"）
 * @param cv - OpenCV.jsインスタンス
 * @returns ソート済みのクロップ画像とボックス
 *
 * @example
 * const { crops, sortedBoxes } = getImgCropList(img, boxes, "quad", cv);
 * // crops[i] と sortedBoxes[i] が対応している
 */
export function getImgCropList(
  ori_img: Mat,
  dt_boxes: Point[][],
  det_box_type: DET_BOX_TYPE,
  cv: CV2,
): CropResult {
  const img_crop_list: Mat[] = [];

  // ボックスをソート（読み順）
  const sorted = sortedBoxes(dt_boxes);

  // 各ボックスをクロップ
  for (const box of sorted) {
    const tmp_box: Box = [...box.map((pt) => [...pt] as Point)] as Box;
    const img_crop: Mat =
      det_box_type === "quad"
        ? get_rotate_crop_image(ori_img, tmp_box, cv)
        : get_minarea_rect_crop(ori_img, tmp_box, cv);
    img_crop_list.push(img_crop);
  }

  return {
    crops: img_crop_list,
    sortedBoxes: sorted,
  };
}

/**
 * 4点の座標から透視変換でクロップ画像を取得
 * 縦長の場合は90度回転
 *
 * @param img - 元画像
 * @param points - 4点の座標（左上、右上、右下、左下の順）
 * @param cv - OpenCV.jsインスタンス
 * @returns クロップ＆回転された画像
 */
export function get_rotate_crop_image(img: Mat, points: Box, cv: CV2): Mat {
  if (points.length !== 4) {
    throw new Error("shape of points must be 4*2");
  }

  // クロップサイズを計算（上辺/下辺の長い方、左辺/右辺の長い方）
  const img_crop_width = Math.floor(
    Math.max(
      euclideanDistance(points[0], points[1]),
      euclideanDistance(points[2], points[3]),
    ),
  );
  const img_crop_height = Math.floor(
    Math.max(
      euclideanDistance(points[0], points[3]),
      euclideanDistance(points[1], points[2]),
    ),
  );

  // 変換先の矩形座標
  const pts_std: Box = [
    [0, 0],
    [img_crop_width, 0],
    [img_crop_width, img_crop_height],
    [0, img_crop_height],
  ];

  const srcTri = boxToMat(points, cv);
  const dstTri = boxToMat(pts_std, cv);

  // 透視変換行列を取得
  const M = cv.getPerspectiveTransform(srcTri, dstTri);
  const dst_img = new cv.Mat();

  // 透視変換を適用
  cv.warpPerspective(
    img,
    dst_img,
    M,
    new cv.Size(img_crop_width, img_crop_height),
    cv.INTER_CUBIC,
    cv.BORDER_REPLICATE,
  );

  // 縦長の場合は90度反時計回りに回転
  const imgSize = dst_img.size();
  if ((imgSize.height * 1.0) / imgSize.width >= 1.5) {
    cv.rotate(dst_img, dst_img, cv.ROTATE_90_COUNTERCLOCKWISE);
  }

  // メモリ解放
  srcTri.delete();
  dstTri.delete();
  M.delete();

  return dst_img;
}

/**
 * 最小外接矩形を使ってクロップ
 *
 * @param img - 元画像
 * @param arg_points - ボックスの点群
 * @param cv - OpenCV.jsインスタンス
 * @returns クロップ＆回転された画像
 */
export function get_minarea_rect_crop(
  img: Mat,
  arg_points: Point[],
  cv: CV2,
): Mat {
  // 点群をOpenCV形式に変換
  const pointsMat = cv.matFromArray(
    arg_points.length,
    1,
    cv.CV_32FC2,
    arg_points.flat(2),
  );

  // 最小外接矩形を取得
  const bounding_box = cv.minAreaRect(pointsMat);

  // 矩形の4点を取得
  const boxPoints = cv.boxPoints(bounding_box);
  const beforePoints: Point[] = boxPoints.map((p) => [p.x, p.y]);

  // x座標でソート
  const points = ([...beforePoints.map((p) => [...p])] as Point[]).sort(
    (a, b) => a[0] - b[0],
  );

  // 左上、右上、右下、左下の順に並べ替え
  let index_a = 0,
    index_b = 1,
    index_c = 2,
    index_d = 3;

  // 左側2点のy座標を比較
  if (points[1]![1] > points[0]![1]) {
    index_a = 0; // 左上
    index_d = 1; // 左下
  } else {
    index_a = 1;
    index_d = 0;
  }

  // 右側2点のy座標を比較
  if (points[3]![1] > points[2]![1]) {
    index_b = 2; // 右上
    index_c = 3; // 右下
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

  // 透視変換でクロップ
  const crop_img = get_rotate_crop_image(img, sortedBox, cv);

  // メモリ解放
  pointsMat.delete();

  return crop_img;
}
