import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        dark: {
          950: "#050811",
          900: "#090e1c",
          850: "#0d152a",
          800: "#131e3a",
          750: "#1a284c",
          700: "#223561",
        },
        brand: {
          400: "#2dd4bf",
          500: "#14b8a6",
          600: "#0d9488",
        },
        cobot: {
          cyan: "#00f2fe",
          blue: "#4facfe",
          amber: "#f59e0b",
          rose: "#f43f5e",
        },
      },
      fontFamily: {
        display: ["var(--font-orbitron)", "sans-serif"],
        sans: ["var(--font-space-grotesk)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "laser-breathe": "laserBreathe 3s ease-in-out infinite",
      },
      keyframes: {
        laserBreathe: {
          "0%, 100%": { opacity: "0.2" },
          "50%": { opacity: "0.4" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
