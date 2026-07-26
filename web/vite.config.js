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
      // Service worker DISABLED for the public demo. A precaching SW is pure downside here:
      // it serves nothing offline that matters and is a common cause of blank/stale pages on
      // mobile (it only activates over https, which is why the CloudFront URL surfaced it).
      // selfDestroying builds a SW that unregisters itself and clears its caches, so any phone
      // that already installed the old one gets cleaned up on the next visit.
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
