import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site is published at https://stephenflavin.github.io/how-stuff-works-ai/
// so assets must be served from that sub-path.
export default defineConfig({
  base: "/how-stuff-works-ai/",
  plugins: [react()],
});
