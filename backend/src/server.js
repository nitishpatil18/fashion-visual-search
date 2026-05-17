import express from "express";
import cors from "cors";
import morgan from "morgan";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const IMAGE_DIR = path.join(PROJECT_ROOT, "ml-service", "data", "raw", "fashion-dataset", "images");

const ML_SERVICE = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";
const PORT = process.env.PORT || 5050;

const app = express();
app.use(cors());
app.use(morgan("dev"));
app.use(express.json());

// serve product images from the ml-service dataset
app.use("/images", express.static(IMAGE_DIR, { maxAge: "1d" }));

// health
app.get("/api/health", async (req, res) => {
  try {
    const r = await axios.get(`${ML_SERVICE}/health`, { timeout: 5000 });
    res.json({ backend: "ok", ml_service: r.data });
  } catch (e) {
    res.status(503).json({ backend: "ok", ml_service: "unreachable", error: e.message });
  }
});

// text search
app.get("/api/search/text", async (req, res) => {
  const { q, top_k = 20, rerank = "true" } = req.query;
  if (!q) return res.status(400).json({ error: "missing query param q" });
  try {
    const r = await axios.get(`${ML_SERVICE}/search/text`, {
      params: { q, top_k, rerank },
      timeout: 30000,
    });
    res.json(r.data);
  } catch (e) {
    console.error("text search failed:", e.message);
    res.status(502).json({ error: "ml service error", detail: e.message });
  }
});

// image search
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

app.post("/api/search/image", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "missing image file" });
  const { top_k = 20, rerank = "true" } = req.body;
  const form = new FormData();
  form.append("image", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });
  form.append("top_k", String(top_k));
  form.append("rerank", String(rerank));
  try {
    const r = await axios.post(`${ML_SERVICE}/search/image`, form, {
      headers: form.getHeaders(),
      timeout: 60000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    res.json(r.data);
  } catch (e) {
    console.error("image search failed:", e.message);
    res.status(502).json({ error: "ml service error", detail: e.message });
  }
});

app.get("/api/search/explain", async (req, res) => {
  const { q } = req.query;
  if (!q) return res.status(400).json({ error: "missing query param q" });
  try {
    const r = await axios.get(`${ML_SERVICE}/search/explain`, {
      params: { q },
      timeout: 30000,
    });
    res.json(r.data);
  } catch (e) {
    console.error("explain failed:", e.message);
    res.status(502).json({ error: "ml service error", detail: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`[backend] listening on http://localhost:${PORT}`);
  console.log(`[backend] proxying ml-service at ${ML_SERVICE}`);
  console.log(`[backend] serving images from ${IMAGE_DIR}`);
});