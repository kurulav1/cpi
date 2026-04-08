import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

const DEFAULT_DEV_PORT = 5173;
const DEFAULT_API_PORT = 3001;

function parsePort(rawValue, fallback) {
  const parsed = Number.parseInt(rawValue ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const devPort = parsePort(env.VITE_DEV_PORT, DEFAULT_DEV_PORT);
  const apiOrigin =
    env.VITE_API_ORIGIN?.trim() ||
    `http://localhost:${parsePort(env.VITE_API_PORT, DEFAULT_API_PORT)}`;

  return {
    plugins: [
      react(),
      tailwindcss(),
      VitePWA({
        registerType: "autoUpdate",
        includeAssets: ["icons/*.png", "icons/*.svg"],
        manifest: {
          name: "CPI - CUDA Inference",
          short_name: "CPI",
          description: "Local LLM inference via CUDA",
          theme_color: "#0f172a",
          background_color: "#0f172a",
          display: "standalone",
          orientation: "landscape",
          scope: "/",
          start_url: "/",
          icons: [
            { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
            {
              src: "icons/icon-512.png",
              sizes: "512x512",
              type: "image/png",
              purpose: "any maskable"
            }
          ]
        },
        workbox: {
          // Cache all static assets; skip API routes.
          globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
          navigateFallbackDenylist: [/^\/api/, /^\/v1/]
        }
      })
    ],
    server: {
      port: devPort,
      proxy: {
        "/api": apiOrigin,
        "/v1": apiOrigin
      }
    }
  };
});
