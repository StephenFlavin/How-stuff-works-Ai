import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Use a relative base so assets resolve correctly no matter what sub-path the
// site is served from (GitHub Pages project URLs are case-sensitive, e.g.
// /How-stuff-works-Ai/). "./" sidesteps any casing/path mismatch.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
