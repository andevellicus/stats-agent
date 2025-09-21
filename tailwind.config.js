/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./web/templates/**/*.{templ,go}",
    "./web/static/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        'gray-850': '#1e293b',
      },
      animation: {
        'bounce-slow': 'bounce 1.5s infinite',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Monaco', 'Consolas', 'Courier New', 'monospace'],
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [],
}