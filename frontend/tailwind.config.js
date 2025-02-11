/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        play: ['Play', 'sans-serif'],
      },
      animation: {
        "fade-in": "fadeIn 2s ease-in-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "0.3" }, // Adjust opacity as needed
        },
      },
    },
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: ['luxury'],
  },
};