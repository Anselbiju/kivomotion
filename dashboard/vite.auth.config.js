import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  resolve: {
    alias: { "../../data-app-public.jsx": fileURLToPath(new URL("./src/data-app-public.jsx", import.meta.url)) },
    dedupe: ["react", "react-dom", "recharts"],
  },
  plugins: [viteSingleFile()],
  build: {
    outDir: "auth-dist",
    emptyOutDir: true,
    rollupOptions: { input: "auth-index.html" },
  },
});
