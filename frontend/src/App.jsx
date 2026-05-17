import { useState, useRef } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [mode, setMode] = useState(null); // "text" | "image" | null
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [activeQuery, setActiveQuery] = useState("");
  const fileInputRef = useRef(null);
  const [uploadedPreview, setUploadedPreview] = useState(null);

  async function runTextSearch(q) {
    if (!q || !q.trim()) return;
    setLoading(true); setError(null); setUploadedPreview(null);
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
    setLoading(true); setError(null);
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

  function onSubmit(e) {
    e.preventDefault();
    runTextSearch(query);
  }

  function onFileChange(e) {
    const f = e.target.files?.[0];
    if (f) runImageSearch(f);
  }

  const examples = [
    "navy blue formal shirt for men",
    "red dress for women",
    "black running shoes",
    "silver watch for women",
    "kids cartoon t-shirt",
    "brown leather wallet",
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* header */}
      <header className="border-b border-neutral-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">fashion visual search</h1>
            <p className="text-xs text-neutral-500">clip + projection heads + lightgbm reranker · 44k products</p>
          </div>
          <a
            href="https://github.com/nitishpatil18"
            target="_blank"
            rel="noreferrer"
            className="text-xs text-neutral-600 hover:text-neutral-900 underline"
          >
            github
          </a>
        </div>
      </header>

      {/* search bar */}
      <section className="border-b border-neutral-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <form onSubmit={onSubmit} className="flex flex-col gap-3 md:flex-row md:items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="describe what you're looking for..."
              className="flex-1 border border-neutral-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-3 bg-neutral-900 text-white rounded-lg text-sm font-medium hover:bg-neutral-700 disabled:opacity-50"
            >
              search
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="px-5 py-3 border border-neutral-300 rounded-lg text-sm font-medium hover:bg-neutral-100 disabled:opacity-50"
            >
              upload image
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={onFileChange}
            />
          </form>

          {/* controls */}
          <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-neutral-600">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
              />
              <span>rerank (lightgbm)</span>
            </label>
            <label className="flex items-center gap-2">
              top-k
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="border border-neutral-300 rounded px-2 py-1"
              >
                {[10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            {latency !== null && (
              <span>latency: <span className="font-medium">{latency} ms</span></span>
            )}
            {activeQuery && (
              <span>query: <span className="font-medium">{activeQuery}</span> ({mode})</span>
            )}
          </div>

          {/* examples */}
          {!mode && (
            <div className="mt-4 flex flex-wrap gap-2">
              {examples.map((ex) => (
                <button
                  key={ex}
                  onClick={() => { setQuery(ex); runTextSearch(ex); }}
                  className="px-3 py-1.5 text-xs border border-neutral-200 rounded-full hover:bg-neutral-100 text-neutral-700"
                >
                  {ex}
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* results */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        {uploadedPreview && (
          <div className="mb-6 flex items-center gap-4">
            <img src={uploadedPreview} alt="query" className="w-24 h-24 object-cover rounded-lg border border-neutral-200" />
            <div className="text-sm text-neutral-600">
              <div className="font-medium text-neutral-900">your uploaded image</div>
              <div>finding visually similar products...</div>
            </div>
          </div>
        )}

        {loading && (
          <div className="text-sm text-neutral-500">searching...</div>
        )}

        {error && (
          <div className="text-sm text-red-600 border border-red-200 bg-red-50 rounded-lg px-4 py-3">
            error: {error}
          </div>
        )}

        {!loading && !error && results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {results.map((r, i) => (
              <article
                key={`${r.id}-${i}`}
                className="bg-white border border-neutral-200 rounded-lg overflow-hidden hover:shadow-sm transition"
              >
                <div className="aspect-square bg-neutral-100 overflow-hidden">
                  <img
                    src={`/${r.image_path}`}
                    alt={r.productDisplayName}
                    loading="lazy"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-3">
                  <div className="text-xs font-medium text-neutral-900 truncate" title={r.productDisplayName}>
                    {r.productDisplayName}
                  </div>
                  <div className="text-xs text-neutral-500 mt-1">
                    {r.articleType} · {r.baseColour}
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        {!loading && !error && mode && results.length === 0 && (
          <div className="text-sm text-neutral-500">no results.</div>
        )}
      </main>

      <footer className="border-t border-neutral-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-3 text-xs text-neutral-500">
          built with clip vit-b/32, faiss flat index, lightgbm ranker. evaluated with recall@k, map@k, ndcg@k on a held-out test set of 3,878 queries.
        </div>
      </footer>
    </div>
  );
}