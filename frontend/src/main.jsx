import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

// apply theme synchronously before react renders
(function applyTheme() {
  const saved = localStorage.getItem("theme") || "system";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = saved === "dark" || (saved === "system" && prefersDark);
  document.documentElement.classList.toggle("dark", isDark);
})();

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);