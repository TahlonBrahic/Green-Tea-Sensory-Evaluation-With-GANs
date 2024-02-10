/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./web_app/templates]",
  theme: {
    extend: {
      animation: {
        typewriter: "typewriter 2s steps(11) forwards"
      },
      keyframes: {
        typewriter: {
          to: {
            left: "100%"
          }
        }
      }
    },
  },
  plugins: [],
}