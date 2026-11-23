import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // rutas relativas: la carpeta dist/ funciona en cualquier servidor estático
  base: './',
  build: {
    rollupOptions: {
      input: ['index', 'inicio', 'aprendizaje', 'abecedario', 'leccion', 'sobre'].map((p) => `${p}.html`),
    },
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8010' } },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
