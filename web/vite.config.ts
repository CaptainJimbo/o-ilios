import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' keeps asset URLs relative so the build works both at
// captainjimbo.github.io/o-ilios/ (GitHub Pages) and behind any CDN path.
export default defineConfig({
  plugins: [react()],
  base: './',
})
