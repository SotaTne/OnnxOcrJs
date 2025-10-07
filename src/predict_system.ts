import type { Mat } from "@techstark/opencv-js";
import { TextClassifier, type TextClassifierParams } from "./predict_cls.js";
import { TextDetector, type TextDetectorParams } from "./predict_det.js";
import { TextRecognizer, type TextRecognizerParams } from "./predict_rec.js";
import type {
  DET_BOX_TYPE,
  DROP_SCORE,
  USE_ANGLE_CLS,
} from "./types/paddle_types.js";
import type { Box, CV2, Point } from "./types/type.js";
import { getImgCropList } from "./utils/funcs/get-crop.js";

export type TextSystemParams = {
  drop_score: DROP_SCORE | null;
  use_angle_cls: USE_ANGLE_CLS;
  det_box_type: DET_BOX_TYPE;
  cv: CV2;
} & TextDetectorParams &
  TextRecognizerParams &
  (TextClassifierParams | undefined);

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

  static async create(params: TextSystemParams): Promise<TextSystem> {
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

  async execute(
    img: Mat,
    cls = true,
  ): Promise<[Box[] | null, [string, number][] | null]> {
    const ori_img = img.clone();

    const dt_boxes = (await this.text_detector.execute(ori_img)) as
      | Box[]
      | null;

    if (!Array.isArray(dt_boxes) || dt_boxes.length === 0) {
      ori_img.delete();
      return [null, null];
    }

    let { crops: img_crop_list, sortedBoxes: sorted_dt_boxes } = getImgCropList(
      ori_img,
      dt_boxes,
      this.det_box_type,
      this.cv,
    );

    ori_img.delete();

    if (this.use_angle_cls && cls && this.text_classifier) {
      [img_crop_list] = await this.text_classifier.execute(img_crop_list);
    }

    const rec_res = await this.text_recognizer.execute(img_crop_list);

    for (const crop_img of img_crop_list) {
      crop_img.delete();
    }

    const filtered_boxes: Box[] = [];
    const filtered_rec_res: [string, number][] = [];

    const minLen = Math.min(sorted_dt_boxes.length, rec_res.length);

    for (let i = 0; i < minLen; i++) {
      const box = sorted_dt_boxes[i]!;
      const [text, score] = rec_res[i]!;

      if (score >= this.drop_score) {
        filtered_boxes.push(box as Box);
        filtered_rec_res.push([text, score]);
      }
    }

    if (filtered_boxes.length === 0) {
      return [null, null];
    }

    return [filtered_boxes, filtered_rec_res];
  }
}
