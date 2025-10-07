// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["{src,test}/**/*.{test,spec}.ts?(x)"],
    deps: {
      interopDefault: true, // デフォルトエクスポートを正しく解釈
    },
    testTimeout: 50000,
  },
});
