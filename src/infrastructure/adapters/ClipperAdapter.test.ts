import { test, expect, beforeAll } from "vitest";
import {
  getClipper,
  polygonArea,
  polygonPerimeter,
  unclip,
} from "./ClipperAdapter.js";

// Clipperの初期化を最初に一度だけ実行
beforeAll(async () => {
  await getClipper();
});

test("getClipper: 初回呼び出しでClipperインスタンスを返す", async () => {
  const clipper = await getClipper();
  expect(clipper).toBeDefined();
  expect(clipper).not.toBeNull();
});

test("getClipper: 2回目以降は同じインスタンスを返す", async () => {
  const clipper1 = await getClipper();
  const clipper2 = await getClipper();
  expect(clipper1).toBe(clipper2);
});

test("polygonArea: 正方形の面積を正しく計算", () => {
  const square = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ];
  expect(polygonArea(square)).toBe(100);
});

test("polygonArea: 三角形の面積を正しく計算", () => {
  const triangle = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 5, y: 10 },
  ];
  expect(polygonArea(triangle)).toBe(50);
});

test("polygonArea: 長方形の面積を正しく計算", () => {
  const rectangle = [
    { x: 0, y: 0 },
    { x: 20, y: 0 },
    { x: 20, y: 10 },
    { x: 0, y: 10 },
  ];
  expect(polygonArea(rectangle)).toBe(200);
});

test("polygonPerimeter: 正方形の周囲長を正しく計算", () => {
  const square = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ];
  expect(polygonPerimeter(square)).toBe(40);
});

test("polygonPerimeter: 三角形の周囲長を正しく計算", () => {
  const triangle = [
    { x: 0, y: 0 },
    { x: 3, y: 0 },
    { x: 0, y: 4 },
  ];
  // 3 + 4 + 5 = 12
  expect(polygonPerimeter(triangle)).toBe(12);
});

test("polygonPerimeter: 長方形の周囲長を正しく計算", () => {
  const rectangle = [
    { x: 0, y: 0 },
    { x: 20, y: 0 },
    { x: 20, y: 10 },
    { x: 0, y: 10 },
  ];
  expect(polygonPerimeter(rectangle)).toBe(60);
});

test("unclip: 正方形のポリゴンを正しく拡大", async () => {
  const polygon = [
    { x: 100, y: 100 },
    { x: 200, y: 100 },
    { x: 200, y: 200 },
    { x: 100, y: 200 },
  ];

  const result = await unclip(polygon, 1.5);

  expect(result).toBeDefined();
  expect(Array.isArray(result)).toBe(true);
  expect(result.length).toBeGreaterThan(0);

  // 拡大後の面積は元より大きい
  if (result.length > 0) {
    const originalArea = polygonArea(polygon);
    // Paths形式から{x,y}配列に変換
    const firstPath = result[0]!.map((pt) => ({ x: pt.x, y: pt.y }));
    const expandedArea = polygonArea(firstPath);
    expect(expandedArea).toBeGreaterThan(originalArea);
  }
});

test("unclip: 距離が極小の場合は空配列を返す", async () => {
  const polygon = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];

  const result = await unclip(polygon, 0.0000001);
  expect(result).toEqual([]);
});

test("unclip: 三角形のポリゴンを正しく拡大", async () => {
  const polygon = [
    { x: 0, y: 0 },
    { x: 100, y: 0 },
    { x: 50, y: 100 },
  ];

  const result = await unclip(polygon, 1.2);

  expect(result).toBeDefined();
  expect(Array.isArray(result)).toBe(true);

  // 拡大後の面積は元より大きい
  if (result.length > 0) {
    const originalArea = polygonArea(polygon);
    // Paths形式から{x,y}配列に変換
    const firstPath = result[0]!.map((pt) => ({ x: pt.x, y: pt.y }));
    const expandedArea = polygonArea(firstPath);
    expect(expandedArea).toBeGreaterThan(originalArea);
  }
});

test("unclip: 複雑な形状のポリゴンを正しく拡大", async () => {
  const polygon = [
    { x: 10, y: 10 },
    { x: 100, y: 10 },
    { x: 100, y: 50 },
    { x: 50, y: 50 },
    { x: 50, y: 100 },
    { x: 10, y: 100 },
  ];

  const result = await unclip(polygon, 1.5);

  expect(result).toBeDefined();
  expect(Array.isArray(result)).toBe(true);

  // 拡大後の面積は元より大きい
  if (result.length > 0) {
    const originalArea = polygonArea(polygon);
    // Paths形式から{x,y}配列に変換
    const firstPath = result[0]!.map((pt) => ({ x: pt.x, y: pt.y }));
    const expandedArea = polygonArea(firstPath);
    expect(expandedArea).toBeGreaterThan(originalArea);
  }
});
