import type { Mat } from "@techstark/opencv-js";
import { TextClassifier, type TextClassifierParams } from "./predict_cls.js";
import { TextDetector, type TextDetectorParams } from "./predict_det.js";
import { TextRecognizer, type TextRecognizerParams } from "./predict_rec.js";
import type {
  DET_BOX_TYPE,
  DROP_SCORE,
  USE_ANGLE_CLS,
} from "./types/paddle_types.js";
import type { Box, CV2 } from "./types/type.js";
import { getImgCropList } from "./utils/funcs/get-crop.js";

export type TextSystemParams = {
  drop_score: DROP_SCORE | null;
  use_angle_cls: USE_ANGLE_CLS;
  det_box_type: DET_BOX_TYPE;
  cv: CV2;
} & TextDetectorParams &
  TextRecognizerParams &
  (TextClassifierParams | undefined);

/**
 * PaddleOCRのTextSystemクラス
 * 検出(Detection) → 分類(Classification) → 認識(Recognition)のパイプラインを実行
 *
 * Python版のTextSystemクラスと同等の機能を提供
 */
export class TextSystem {
  text_detector: TextDetector;
  text_recognizer: TextRecognizer;
  text_classifier: TextClassifier | null = null;
  drop_score: DROP_SCORE;
  use_angle_cls: USE_ANGLE_CLS;
  det_box_type: DET_BOX_TYPE = "quad";
  cv: CV2;

  constructor(
    params: TextSystemParams & {
      text_detector: TextDetector;
      text_recognizer: TextRecognizer;
      text_classifier?: TextClassifier | null;
    },
  ) {
    this.text_detector = params.text_detector;
    this.text_recognizer = params.text_recognizer;
    this.drop_score = params.drop_score ?? 0.5;
    this.use_angle_cls = params.use_angle_cls;
    this.det_box_type = params.det_box_type;
    this.cv = params.cv;

    if (this.use_angle_cls && params.text_classifier) {
      this.text_classifier = params.text_classifier;
    }
  }

  /**
   * TextSystemインスタンスを非同期で作成
   * 各モデルの初期化を並列実行して高速化
   *
   * @param params - TextSystemのパラメータ
   * @returns 初期化済みのTextSystemインスタンス
   */
  static async create(params: TextSystemParams): Promise<TextSystem> {
    // Parallelize model initialization for faster startup
    const [text_detector, text_recognizer, text_classifier] = await Promise.all(
      [
        TextDetector.create(params),
        TextRecognizer.create(params),
        params.use_angle_cls
          ? TextClassifier.create(params)
          : Promise.resolve(null),
      ],
    );

    return new TextSystem({
      ...params,
      text_detector,
      text_recognizer,
      text_classifier,
    });
  }

  /**
   * OCRパイプラインを実行（Python版の__call__メソッドに相当）
   *
   * @param img - 入力画像（OpenCV Mat形式）
   * @param cls - 角度分類を実行するか（default: true）
   * @returns [検出ボックス配列, 認識結果配列] または [null, null]
   *
   * @example
   * const textSystem = await TextSystem.create(params);
   * const [boxes, results] = await textSystem.execute(imageMat);
   *
   * if (boxes && results) {
   *   for (let i = 0; i < boxes.length; i++) {
   *     const box = boxes[i];
   *     const [text, score] = results[i];
   *     console.log(`Text: ${text}, Score: ${score}, Box:`, box);
   *   }
   * }
   */
  async execute(
    img: Mat,
    cls = true,
  ): Promise<[Box[] | null, [string, number][] | null]> {
    const ori_img = img.clone();

    // 1. テキスト検出 (Detection)
    let dt_boxes = (await this.text_detector.execute(ori_img)) as Box[] | null;

    if (!Array.isArray(dt_boxes) || dt_boxes.length === 0) {
      ori_img.delete();
      return [null, null];
    }

    // 2. 画像クロップ（内部で読み順にソートされる）
    const { crops: img_crop_list, sortedBoxes: sortedDtBoxes } = getImgCropList(
      ori_img,
      dt_boxes,
      this.det_box_type,
      this.cv,
    );

    ori_img.delete();

    // 3. 角度分類 (Angle Classification - オプション)
    let crops_to_recognize = img_crop_list;

    if (this.use_angle_cls && cls && this.text_classifier) {
      const [rotated] = await this.text_classifier.execute(img_crop_list);

      // 元のクロップ画像のメモリを解放
      for (const m of img_crop_list) {
        m.delete();
      }

      crops_to_recognize = rotated;
    }

    // 4. テキスト認識 (Recognition)
    const rec_res = await this.text_recognizer.execute(crops_to_recognize);

    // クロップ画像のメモリを解放
    for (const crop_img of crops_to_recognize) {
      crop_img.delete();
    }

    // 5. スコアでフィルタリング
    const filtered_boxes: Box[] = [];
    const filtered_rec_res: [string, number][] = [];

    // 長さ不一致の場合は短い方に合わせる（例外を投げない）
    // Python版のzip(dt_boxes, rec_res)と同等の処理
    const minLen = Math.min(sortedDtBoxes.length, rec_res.length);

    for (let i = 0; i < minLen; i++) {
      const box = sortedDtBoxes[i]!;
      const rec_result = rec_res[i]!;
      const [_text, score] = rec_result;

      // drop_scoreより高いスコアの結果のみ採用
      if (score >= this.drop_score) {
        filtered_boxes.push(box as Box);
        filtered_rec_res.push(rec_result);
      }
    }

    // フィルタ後の結果が空の場合はnullを返す
    if (filtered_boxes.length === 0) {
      return [null, null];
    }

    return [filtered_boxes, filtered_rec_res];
  }
}
