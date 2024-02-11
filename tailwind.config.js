// tailwind.config.js
module.exports = {
  content: ["./web_app/templates/**/*.html"],
  theme: {
    screens: {
      sm: '480px',
      md: '768px',
      lg: '976px',
      xl: '1440px',
    },
    colors: {
      'purple': '#a78bfa',
      'gray-dark': '#334155',
      'gray': '#475569',
      'gray-light': '#64748b',
      'white': '#e2e8f0'
    },
    fontFamily: {
      sans: ['Graphik', 'sans-serif'],
      serif: ['Merriweather', 'serif'],
    },
    extend: {
      spacing: {
        '1': '8px',
        '2': '12px',
        '3': '16px',
        '4': '24px',
        '5': '32px',
        '6': '48px',
      }
    },
  },
  plugins: [],
}