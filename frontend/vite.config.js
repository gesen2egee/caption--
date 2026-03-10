import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var proxyTarget = process.env.CAPTION_RUNTIME_PROXY || "http://127.0.0.1:8765";
export default defineConfig({
    plugins: [react()],
    server: {
        host: "127.0.0.1",
        port: 5173,
        strictPort: true,
        proxy: {
            "/api": {
                target: proxyTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/api/, ""); },
            },
        },
    },
});
