import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "navy blue formal shirt for men",
  "red dress for women",
  "black running shoes",
  "silver watch for women",
  "brown leather wallet",
  "kids cartoon t-shirt",
];

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "system");

  useEffect(() => {
    const apply = () => {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const isDark = theme === "dark" || (theme === "system" && prefersDark);
      document.documentElement.classList.toggle("dark", isDark);
    };
    apply();
    localStorage.setItem("theme", theme);
    if (theme === "system") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      mq.addEventListener("change", apply);
      return () => mq.removeEventListener("change", apply);
    }
  }, [theme]);

  return [theme, setTheme];
}

function ThemeToggle({ theme, setTheme }) {
  const options = [
    { v: "light", label: "light" },
    { v: "dark", label: "dark" },
    { v: "system", label: "auto" },
  ];
  return (
    <div className="inline-flex border border-app rounded-md overflow-hidden text-[11px]">
      {options.map((o) => (
        <button
          key={o.v}
          onClick={() => setTheme(o.v)}
          className={`px-2.5 py-1 transition ${
            theme === o.v
              ? "bg-accent text-accent-fg"
              : "text-muted hover:text-app"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function RankDelta({ pre, post }) {
  if (pre == null || post == null) return null;
  const delta = pre - post; // positive = moved up
  if (delta === 0) {
    return <span className="text-[10px] text-subtle">rank {post}</span>;
  }
  const moved = delta > 0;
  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
        moved ? "bg-positive-soft" : "bg-negative-soft"
      }`}
      title={moved ? `moved up ${delta} positions` : `moved down ${-delta} positions`}
    >
      {moved ? "↑" : "↓"} {Math.abs(delta)}
      <span className="text-subtle">·</span>
      <span>{pre}→{post}</span>
    </span>
  );
}

function ResultCard({ r, index }) {
  return (
    <article
      className="bg-card border border-app rounded-lg overflow-hidden hover:border-strong transition-all card-fade-in"
      style={{ animationDelay: `${index * 18}ms` }}
    >
      <div className="aspect-[3/4] bg-elev overflow-hidden">
        <img
          src={`/${r.image_path}`}
          alt={r.productDisplayName}
          loading="lazy"
          className="w-full h-full object-cover transition-transform duration-500 hover:scale-[1.03]"
        />
      </div>
      <div className="p-3 space-y-1.5">
        <div
          className="text-xs font-medium text-app truncate"
          title={r.productDisplayName}
        >
          {r.productDisplayName}
        </div>
        <div className="text-[10px] text-muted truncate">
          {r.articleType} · {r.baseColour} · {r.gender}
        </div>
        <div className="flex items-center justify-between pt-1">
          <RankDelta pre={r.pre_rank} post={index + 1} />
          <span className="text-[10px] text-subtle tabular-nums">
            {r.score >= 0 ? r.score.toFixed(3) : r.score.toFixed(2)}
          </span>
        </div>
      </div>
    </article>
  );
}

function SkeletonCard({ index }) {
  return (
    <div
      className="bg-card border border-app rounded-lg overflow-hidden card-fade-in"
      style={{ animationDelay: `${index * 18}ms` }}
    >
      <div className="aspect-[3/4] shimmer" />
      <div className="p-3 space-y-2">
        <div className="h-3 w-3/4 rounded shimmer" />
        <div className="h-2.5 w-1/2 rounded shimmer" />
        <div className="h-2.5 w-1/3 rounded shimmer" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="border border-dashed border-app rounded-xl py-20 text-center">
      <div className="inline-block w-10 h-10 rounded-full border-2 border-app flex items-center justify-center mb-4 text-app">
        ⌕
      </div>
      <div className="text-sm text-muted">type a query or upload an image to begin.</div>
      <div className="text-[11px] text-subtle mt-1">
        try the suggestions above, or "ethnic wear for women", "winter jacket for men".
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [mode, setMode] = useState(null);
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [activeQuery, setActiveQuery] = useState("");
  const [uploadedPreview, setUploadedPreview] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef(null);

  async function runTextSearch(q) {
    if (!q || !q.trim()) return;
    setLoading(true);
    setError(null);
    setUploadedPreview(null);
    try {
      const r = await fetch(
        `/api/search/text?q=${encodeURIComponent(q)}&top_k=${topK}&rerank=${rerank}`
      );
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = await r.json();
      setResults(data.results || []);
      setMode("text");
      setActiveQuery(q);
      setLatency(data.latency_ms);
    } catch (e) {
      setError(e.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function runImageSearch(file) {
    if (!file) return;
    setLoading(true);
    setError(null);
    setUploadedPreview(URL.createObjectURL(file));
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("top_k", String(topK));
      form.append("rerank", String(rerank));
      const r = await fetch("/api/search/image", { method: "POST", body: form });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data = await r.json();
      setResults(data.results || []);
      setMode("image");
      setActiveQuery(file.name);
      setLatency(data.latency_ms);
    } catch (e) {
      setError(e.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-app text-app flex flex-col">
      {/* top nav */}
      <header className="border-b border-app">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded bg-accent text-accent-fg flex items-center justify-center text-[11px] font-semibold">
              fs
            </div>
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight">fashion search</div>
              <div className="text-[10px] text-subtle">clip · projection heads · lightgbm rerank</div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/nitishpatil18/fashion-visual-search"
              target="_blank"
              rel="noreferrer"
              className="text-xs text-muted hover:text-app transition"
            >
              github
            </a>
            <ThemeToggle theme={theme} setTheme={setTheme} />
          </div>
        </div>
      </header>

      {/* hero + search */}
      <section className="border-b border-app bg-elev">
        <div className="max-w-3xl mx-auto px-6 py-12 text-center space-y-6">
          <div className="inline-flex items-center gap-2 text-[11px] text-muted border border-app rounded-full px-3 py-1 bg-card">
            <span className="w-1.5 h-1.5 rounded-full bg-positive" />
            evaluated on 3,878 held-out queries · recall@10 ↑ 3.2× over baseline
          </div>
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
            visual search over 44,419 fashion products
          </h1>
          <p className="text-sm text-muted max-w-xl mx-auto">
            type what you're looking for or upload a photo. retrieval is two-stage:
            faiss over fine-tuned clip embeddings, reranked by a learning-to-rank model.
          </p>

          {/* search bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              runTextSearch(query);
            }}
            className="flex flex-col sm:flex-row gap-2 max-w-2xl mx-auto"
          >
            <div className="flex-1 relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="navy blue formal shirt for men"
                className="w-full bg-card border border-app rounded-md px-4 py-3 text-sm placeholder:text-subtle focus:outline-none focus:border-strong"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-3 bg-accent text-accent-fg rounded-md text-sm font-medium hover:opacity-90 disabled:opacity-50 transition"
            >
              search
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="px-5 py-3 bg-card border border-app rounded-md text-sm font-medium hover:border-strong disabled:opacity-50 transition"
            >
              upload image
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) runImageSearch(f);
              }}
            />
          </form>

          <div className="flex flex-wrap gap-1.5 justify-center">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => {
                  setQuery(ex);
                  runTextSearch(ex);
                }}
                className="px-2.5 py-1 text-[11px] border border-app rounded-full hover:border-strong text-muted hover:text-app transition bg-card"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* status bar */}
      <section className="border-b border-app">
        <div className="max-w-6xl mx-auto px-6 py-2.5 flex items-center justify-between text-[11px] text-muted">
          <div className="flex items-center gap-4">
            {activeQuery ? (
              <>
                <span>
                  query: <span className="text-app font-medium">{activeQuery}</span>
                  <span className="text-subtle ml-1">({mode})</span>
                </span>
                {latency !== null && (
                  <span>latency: <span className="text-app tabular-nums">{latency}ms</span></span>
                )}
                <span>
                  rerank: <span className="text-app">{rerank ? "on" : "off"}</span>
                </span>
              </>
            ) : (
              <span className="text-subtle">no search yet</span>
            )}
          </div>
          <button
            onClick={() => setShowAdvanced((s) => !s)}
            className="text-muted hover:text-app transition"
          >
            {showAdvanced ? "hide advanced" : "advanced"}
          </button>
        </div>
        {showAdvanced && (
          <div className="border-t border-app bg-elev">
            <div className="max-w-6xl mx-auto px-6 py-3 flex flex-wrap items-center gap-6 text-[11px]">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={rerank}
                  onChange={(e) => setRerank(e.target.checked)}
                />
                <span className="text-app">rerank with lightgbm</span>
                <span className="text-subtle">(top-100 candidates from faiss → reranked)</span>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-muted">top-k</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="bg-card border border-app rounded px-2 py-1 text-app"
                >
                  {[10, 20, 30, 50].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
              <span className="text-subtle">
                rank deltas show pre-rerank → post-rerank position
              </span>
            </div>
          </div>
        )}
      </section>

      {/* results */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        {uploadedPreview && (
          <div className="mb-6 flex items-center gap-4">
            <img
              src={uploadedPreview}
              alt="query"
              className="w-20 h-20 object-cover rounded-md border border-app"
            />
            <div className="text-xs text-muted">
              <div className="text-app font-medium">your uploaded image</div>
              <div>finding visually similar products</div>
            </div>
          </div>
        )}

        {error && (
          <div className="text-xs bg-negative-soft border border-app rounded-md px-3 py-2">
            error: {error}
          </div>
        )}

        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {Array.from({ length: topK }).map((_, i) => (
              <SkeletonCard key={i} index={i} />
            ))}
          </div>
        )}

        {!loading && !error && results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {results.map((r, i) => (
              <ResultCard key={`${r.id}-${i}`} r={r} index={i} />
            ))}
          </div>
        )}

        {!loading && !error && !mode && <EmptyState />}

        {!loading && !error && mode && results.length === 0 && (
          <div className="text-sm text-muted">no results.</div>
        )}
      </main>

      {/* footer */}
      <footer className="border-t border-app">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-[11px] text-subtle">
          <div>
            built with clip vit-b/32 · faiss flat ip index · lightgbm lambdarank · evaluated on 3,878 queries
          </div>
          <div>
            <a
              href="https://github.com/nitishpatil18/fashion-visual-search"
              target="_blank"
              rel="noreferrer"
              className="hover:text-app transition"
            >
              source
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}