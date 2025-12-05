import {
  ClipperLibWrapper,
  loadNativeClipperLibInstanceAsync,
  NativeClipperLibRequestedFormat,
  JoinType,
  EndType,
} from "js-angusj-clipper/web";
import type { Paths } from "js-angusj-clipper/web";

let _clipper: ClipperLibWrapper | null = null;

/**
 * Clipperインスタンスを取得（シングルトン）
 * 初回呼び出し時にWASMを読み込み、以降は同じインスタンスを返す
 */
export async function getClipper(): Promise<ClipperLibWrapper> {
  if (_clipper) {
    return _clipper;
  }
  _clipper = await loadNativeClipperLibInstanceAsync(
    NativeClipperLibRequestedFormat.WasmWithAsmJsFallback,
  );
  return _clipper;
}

/**
 * ポリゴンの面積を計算（Shoelace formula）
 */
export function polygonArea(points: { x: number; y: number }[]): number {
  let area = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const { x: x0, y: y0 } = points[i]!;
    const { x: x1, y: y1 } = points[(i + 1) % n]!;
    area += x0 * y1 - x1 * y0;
  }
  return Math.abs(area) / 2;
}

/**
 * ポリゴンの周囲長を計算
 */
export function polygonPerimeter(points: { x: number; y: number }[]): number {
  let perimeter = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const { x: x0, y: y0 } = points[i]!;
    const { x: x1, y: y1 } = points[(i + 1) % n]!;
    perimeter += Math.hypot(x1 - x0, y1 - y0);
  }
  return perimeter;
}

/**
 * ポリゴンを外側に拡大（DB postprocessで使用）
 *
 * @param box - 拡大するポリゴン
 * @param unclipRatio - 拡大率
 * @returns 拡大後のポリゴン配列（Paths形式）
 */
export async function unclip(
  box: { x: number; y: number }[],
  unclipRatio: number,
): Promise<Paths> {
  const clipper = await getClipper();

  const area = polygonArea(box);
  const perimeter = polygonPerimeter(box);
  const distance = (area * unclipRatio) / perimeter;

  if (Math.abs(distance) < 1e-6) {
    return [] as Paths; // 明示的に空を返す
  }

  return (
    clipper.offsetToPaths({
      delta: distance,
      offsetInputs: [
        {
          data: box,
          joinType: JoinType.Round,
          endType: EndType.ClosedPolygon,
        },
      ],
    }) ?? []
  );
}
