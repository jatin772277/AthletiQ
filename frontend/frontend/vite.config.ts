import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend runs on :8000 (backend/run.py); proxy /api so no CORS setup is needed.
export default defineConfig(({ command }) => {
  const apiBase = process.env.VITE_API_BASE;
  if (command === "build" && process.env.APP_ENV === "production") {
    try {
      const url = new URL(apiBase ?? "");
      if (!/^https?:$/.test(url.protocol) || url.pathname !== "/") throw new Error();
    } catch {
      throw new Error("Production builds require VITE_API_BASE as an absolute http(s) API origin");
    }
  }

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      proxy: {
        "/api": {
          target: process.env.ATHLETIQ_API ?? "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
