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
      // Service worker disabled: a precaching SW serves nothing useful here and is a common
      // cause of blank/stale pages. selfDestroying unregisters any copy already installed.
      VitePWA({ selfDestroying: true })
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
