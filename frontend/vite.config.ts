import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend runs on :8000 (backend/run.py); proxy /api so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.ATHLETIQ_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
