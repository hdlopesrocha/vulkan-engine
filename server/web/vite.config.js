import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// `base: './'` keeps asset URLs relative so the bundle works both under the
// C++ WebServer (GET / + /assets/*) and from a plain static file server.
// `npm run dev` proxies /ws + /health to a local bin/server for development.
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 2048,
  },
  server: {
    port: 5173,
    proxy: {
      '/ws': { target: 'ws://localhost:8080', ws: true },
      '/health': 'http://localhost:8080',
    },
  },
});
