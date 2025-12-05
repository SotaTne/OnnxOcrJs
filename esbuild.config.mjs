import { build } from "esbuild";

const shared = {
  entryPoints: ["src/index.ts"], // ライブラリのエントリポイント
  bundle: true,                  // 依存をまとめる
  sourcemap: true,               // ソースマップ生成
  minify: true,                 // ライブラリ配布用にminify
  target: "esnext",              // 出力ターゲット
  external: [                    // peerDependencies のみ bundle に含めない
    "@techstark/opencv-js",
    "onnxruntime-node",
    "onnxruntime-web",
  ],
};

// ESM ビルド (ブラウザ + Node.js ESM)
await build({
  ...shared,
  format: "esm",
  outfile: "dist/index.js",      // 単一ファイル出力
  platform: "neutral",           // ブラウザとNode.js両対応
  mainFields: ["module", "main"], // モジュール解決の優先順位
  alias: {
    fs: "./stubs/fs.js",
    path: "./stubs/path.js"
  }
});

// CommonJS ビルド (Node.js Legacy)
await build({
  ...shared,
  format: "cjs",
  outfile: "dist/index.cjs",
  platform: "node",
  external: ["fs", "path"],
});