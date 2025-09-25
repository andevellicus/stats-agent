/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./web/templates/**/*.{templ,go}",
    "./web/static/**/*.html",
    "./web/static/js/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        'primary': '#0088d6',
      },
      fontFamily: {
        'sans': [
          'Inter Variable',
          'Inter',
          'SF Pro Display',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI Variable',
          'Segoe UI',
          'system-ui',
          'sans-serif'
        ],
        'mono': [
          'JetBrains Mono Variable',
          'JetBrains Mono',
          'Fira Code',
          'SF Mono',
          'Monaco',
          'Cascadia Code',
          'Roboto Mono',
          'Consolas',
          'monospace'
        ],
        'display': [
          'Cal Sans',
          'Inter Variable',
          'Inter',
          'SF Pro Display',
          'system-ui',
          'sans-serif'
        ]
      },
      animation: {
        'bounce-slow': 'bounce 2s infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        }
      }
    },
  },
  // Add the plugins section
  plugins: [
    require('@tailwindcss/typography'),
  ],
}