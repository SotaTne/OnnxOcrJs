import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        // ONNXモデルファイルとテキストファイルのキャッシュ設定
        source: "/ppocrv5/:path*.(onnx|txt)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable", // 1年間キャッシュ
          },
        ],
      },
      {
        // 全てのONNXファイルに対するキャッシュ設定（念のため）
        source: "/:path*.onnx",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        // 全てのtxtファイルに対するキャッシュ設定（念のため）
        source: "/:path*.txt",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
