import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { fetchHealth, streamChat } from "./lib/chatStream";

const TEMPLATES = [
  { value: "tinyllama",       label: "TinyLlama" },
  { value: "tinyllama-chatml",label: "TinyLlama ChatML" },
  { value: "llama2",          label: "Llama 2" },
  { value: "llama3",          label: "Llama 3" },
  { value: "llama4",          label: "Llama 4" },
  { value: "mistral",         label: "Mistral" },
  { value: "phi3",            label: "Phi-3" },
  { value: "qwen2",           label: "Qwen 2" },
  { value: "qwen3_5",         label: "Qwen 3.5" },
  { value: "plain",           label: "Plain" },
];

const HF_FAMILIES = ["llama2", "llama3", "mistral", "mixtral", "phi3", "phimoe", "qwen2"];

const STARTERS = [
  "Explain something fascinating in simple terms.",
  "Help me draft a message to a friend.",
  "Give me ideas for a weekend project.",
];

const API_ROUTES = Object.freeze({
  models: "/api/models",
  quantSelect: "/api/quant/select",
  quantState: "/api/quant/state",
  quantConvert: "/api/quant/convert",
  quantJobs: "/api/quant/jobs",
  quantStatus: "/api/quant/status",
  pickFolder: "/api/system/pick-folder",
  modelDir: "/api/system/model-dir",
  hubSearch: "/api/hub/search",
  hubDownload: "/api/hub/download",
  hubJob: "/api/hub/jobs"
});

const STORAGE_KEYS = Object.freeze({
  hubToken: "hub_token",
  hubOutputDir: "hub_output_dir",
  perfMode: "cpi_perf_mode",
  theme: "cpi_theme",
  autoMaxTokens: "cpi_auto_max_tokens",
  longFormMode: "cpi_long_form_mode",
  thinking: "cpi_thinking",
  chats: "cpi_chats",
  activeChat: "cpi_active_chat"
});

const MAX_STORED_CHATS = 100;

function safeStorageGet(key, fallback = "") {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function safeStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore storage write failures in private mode / strict browser settings
  }
}

function sanitizeStoredOutputDir(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  // Cleanup legacy/dev default path accidentally persisted in prior builds.
  if (/[/\\]users[/\\]newuser([/\\]|$)/i.test(raw)) return "";
  return raw;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function createMsg(role, content, extra = {}) {
  return { id: crypto.randomUUID(), role, content, ...extra };
}

function seedMessages() {
  return [{ id: "seed", role: "assistant", content: "Ready.", seed: true }];
}

//  Chat storage (localStorage)

function loadStoredChats() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEYS.chats) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (c) => c && typeof c.id === "string" && Array.isArray(c.messages) && c.messages.length > 0
    );
  } catch {
    return [];
  }
}

function persistStoredChats(chats) {
  try {
    localStorage.setItem(STORAGE_KEYS.chats, JSON.stringify(chats));
  } catch {
    // quota exceeded / private mode: chat history just won't persist
  }
}

function chatTitleFrom(messages) {
  const firstUser = messages.find((m) => m.role === "user" && String(m.content || "").trim());
  if (!firstUser) return "New chat";
  const t = String(firstUser.content).trim().replace(/\s+/g, " ");
  return t.length > 48 ? `${t.slice(0, 48)}...` : t;
}

function fmtChatStamp(ts) {
  if (!Number.isFinite(ts)) return "";
  const d = new Date(ts);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function tailPath(v) {
  if (!v) return "-";
  const parts = v.split(/[/\\]/).filter(Boolean);
  return parts.length <= 3 ? v : `.../${parts.slice(-3).join("/")}`;
}

function fmtMs(ms) {
  if (!ms) return null;
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function fmtDl(n) {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function fmtBytes(n) {
  if (n >= 1073741824) return `${(n / 1073741824).toFixed(2)} GB`;
  if (n >= 1048576)    return `${(n / 1048576).toFixed(1)} MB`;
  if (n >= 1024)       return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function quantLabel(mode) {
  if (mode === "int8") return "INT8";
  if (mode === "int4") return "INT4";
  return "FP16";
}

// Split a raw model label (from its filename) into a clean base name and a
// human-readable format tag, so "Qwen2.5-Coder-32B-Instruct-streaming-int4"
// shows as "Qwen2.5-Coder-32B-Instruct" + tag "int4 · streaming" instead of
// cramming the storage format into the name (which collides with the runtime
// quant selector and confuses people).
function modelDisplay(profile) {
  const raw = String(profile?.label || "");
  const lower = raw.toLowerCase();
  const isStreaming = lower.includes("streaming");
  const fmt = lower.includes("int4") ? "int4" : lower.includes("int8") ? "int8" : "";
  const base = raw
    .replace(/[-_]?streaming/gi, "")
    .replace(/[-_]?int[48]/gi, "")
    .replace(/[-_]{2,}/g, "-")
    .replace(/[-_]+$/g, "")
    .trim() || raw;
  const tag = [fmt, isStreaming ? "streaming" : ""].filter(Boolean).join(" · ");
  return { base, tag, isStreaming, fmt, pinned: Boolean(fmt || isStreaming) };
}

// "Streaming" = the model is larger than VRAM, so its (usually int4/int8) weights
// are streamed from disk/RAM layer-by-layer during inference. Lets you run huge
// models (e.g. 32B) at the cost of speed.
const STREAMING_HELP =
  "Streaming model: too large to fit in VRAM, so its weights are streamed from disk during inference — runs big models, but slower.";

// Group ready profiles by base model name and collapse each model's precision
// options — runtime FP16/INT8/INT4 on the full-precision file, plus any pre-packed
// streaming builds — into a single ordered variant list. So the UI can offer one
// model picker + one variant picker instead of scattering formats across names,
// tags and a separate quant menu. Each variant maps to a (profileId, quantMode).
function buildModelGroups(profiles) {
  const byBase = new Map();
  for (const p of profiles || []) {
    const d = modelDisplay(p);
    if (!byBase.has(d.base)) byBase.set(d.base, { base: d.base, variants: [] });
    const g = byBase.get(d.base);
    if (d.pinned) {
      // Pre-packed / streaming file → one fixed variant (its baked format).
      const fmtRank = d.fmt === "int4" ? 2 : d.fmt === "int8" ? 1 : 0;
      g.variants.push({
        key: `${p.id}|none`,
        label: `${(d.fmt || "packed").toUpperCase()}${d.isStreaming ? " · streaming" : ""}`,
        profileId: p.id, quantMode: "none", template: p.template,
        order: 10 + fmtRank
      });
    } else {
      // Full-precision file → its selectable runtime precisions.
      for (const m of (p.quant?.selectableModes ?? ["none"])) {
        g.variants.push({
          key: `${p.id}|${m}`,
          label: m === "none" ? "FP16" : `${quantLabel(m)} · runtime`,
          profileId: p.id, quantMode: m, template: p.template,
          order: m === "none" ? 0 : m === "int8" ? 1 : 2
        });
      }
    }
  }
  const groups = [...byBase.values()];
  for (const g of groups) {
    // de-dup identical keys, then order fp16 < int8 < int4 < streaming
    const seen = new Set();
    g.variants = g.variants.filter((v) => (seen.has(v.key) ? false : seen.add(v.key)));
    // "Streaming" is a fit-necessity, not a user choice: if a full-precision build
    // exists (order < 10), hide the pre-packed streaming builds — the engine keeps
    // what fits VRAM resident and streams any overflow automatically. Streaming
    // only stays visible when it's the ONLY way to run the model (e.g. a 32B with
    // no fp16 build). Runtime INT8/INT4 remain as a genuine quality/speed choice.
    const hasFullPrecision = g.variants.some((v) => v.order < 10);
    if (hasFullPrecision) g.variants = g.variants.filter((v) => v.order < 10);
    g.variants.sort((a, b) => a.order - b.order);
  }
  groups.sort((a, b) => a.base.localeCompare(b.base));
  return groups;
}

// Turn low-level engine errors into something a user can act on.
function friendlyEngineError(msg) {
  const m = String(msg || "");
  if (/gpu-cache-all|fully resident|not supported by the batched|INT8\/INT4-quantized/i.test(m)) {
    return "This model can't run with fast batching — it's a streaming/quantized model that doesn't fit fully in VRAM. Pick a full-precision (FP16) model, or turn batching off.";
  }
  if (/pool exhausted|max_context budget/i.test(m)) {
    return "Ran out of KV-cache space for this many concurrent requests. Lower the context size or reduce concurrency.";
  }
  return m;
}

function defaultQuantForProfile(profile) {
  const selectable = profile?.quant?.selectableModes ?? ["none"];
  const pickIfAllowed = (mode) => (selectable.includes(mode) ? mode : "");
  const recommended = pickIfAllowed(profile?.quant?.recommendedMode);
  if (recommended) return recommended;

  if (profile?.quant?.packed?.int4 && !profile?.quant?.packed?.int8) return pickIfAllowed("int4") || "none";
  if (profile?.quant?.packed?.int8 && !profile?.quant?.packed?.int4) return pickIfAllowed("int8") || "none";

  const label = String(profile?.label || "").toLowerCase();
  if (label.includes("int4")) return pickIfAllowed("int4") || "none";
  if (label.includes("int8")) return pickIfAllowed("int8") || "none";

  return (
    profile?.quant?.effectiveMode ||
    profile?.quant?.configuredMode ||
    profile?.quant?.selectableModes?.[0] ||
    "none"
  );
}

// Recursively flatten React children (incl. rehype-highlight's nested token
// spans) back to plain text — used for the copy-to-clipboard button.
function nodeText(node) {
  if (node == null || node === false) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (node.props && node.props.children != null) return nodeText(node.props.children);
  return "";
}

// Fenced code block: syntax-highlighted body (rehype-highlight has already
// wrapped `children` in hljs token spans) plus a header with the language label
// and a copy button.
function CodeBlock({ language, children }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try {
      navigator.clipboard.writeText(nodeText(children).replace(/\n$/, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // clipboard blocked (insecure context / permissions) — ignore
    }
  };
  return (
    <div className="msg-code-wrap">
      <div className="msg-code-head">
        <span className="msg-code-lang">{language || "code"}</span>
        <button type="button" className="msg-code-copy" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="msg-code">
        <code className={language ? `hljs language-${language}` : "hljs"}>{children}</code>
      </pre>
    </div>
  );
}

// Collapsible reasoning ("thinking") block shown above a reasoning model's
// answer. Auto-expanded while the model is still thinking (no answer yet),
// auto-collapses once the answer starts; the user can toggle either way.
function ReasoningBlock({ text, active }) {
  const [open, setOpen] = useState(true);
  const userToggled = useRef(false);
  useEffect(() => {
    if (!active && !userToggled.current) setOpen(false);
  }, [active]);
  if (!text) return null;
  return (
    <div className={`reasoning ${active ? "reasoning-active" : ""}`}>
      <button
        type="button"
        className="reasoning-toggle"
        onClick={() => { userToggled.current = true; setOpen((o) => !o); }}
      >
        <span className="reasoning-caret">{open ? "▾" : "▸"}</span>
        {active ? "Thinking…" : "Thought process"}
      </button>
      {open && <div className="reasoning-body">{text}</div>}
    </div>
  );
}

function MsgContent({ text }) {
  const clean = formatModelMarkdown(text);

  return (
    <div className="msg-rich">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          p: ({ children }) => <p className="msg-paragraph">{children}</p>,
          h1: ({ children }) => <h1 className="msg-heading">{children}</h1>,
          h2: ({ children }) => <h2 className="msg-heading">{children}</h2>,
          h3: ({ children }) => <h3 className="msg-heading msg-heading-sm">{children}</h3>,
          ul: ({ children }) => <ul className="msg-list">{children}</ul>,
          ol: ({ children }) => <ol className="msg-list msg-list-ordered">{children}</ol>,
          blockquote: ({ children }) => <blockquote className="msg-blockquote">{children}</blockquote>,
          // react-markdown v10 dropped the `inline` prop: distinguish fenced
          // blocks (rehype-highlight adds an hljs/language- class) from inline code.
          code: ({ className, children }) => {
            const isBlock = /\blanguage-|\bhljs\b/.test(className || "");
            if (!isBlock) {
              return <code className="msg-inline-code">{children}</code>;
            }
            const match = /language-(\w+)/.exec(className || "");
            return <CodeBlock language={match ? match[1] : ""}>{children}</CodeBlock>;
          },
          // Block code's <code> already renders its own <pre> via CodeBlock; unwrap
          // the default <pre> to avoid double nesting.
          pre: ({ children }) => <>{children}</>,
          table: ({ children }) => (
            <div className="msg-table-wrap">
              <table className="msg-table">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="msg-table-head">{children}</thead>,
          th: ({ children }) => <th className="msg-table-th">{children}</th>,
          td: ({ children }) => <td className="msg-table-td">{children}</td>,
          a: ({ href, children }) => (
            <a
              className="msg-link"
              href={href}
              target="_blank"
              rel="noreferrer"
            >
              {children}
            </a>
          ),
          hr: () => <hr className="msg-rule" />
        }}
      >
        {clean}
      </ReactMarkdown>
    </div>
  );
}

function formatModelMarkdown(text) {
  let clean = String(text || "")
    .replace(/\r/g, "")
    .replace(/^\n+/, "")
    .replace(/\n{3,}/g, "\n\n");

  // Many models emit list markers inline in one paragraph. Split those into
  // real markdown blocks so the final render preserves the intended layout.
  clean = clean.replace(/([:!?\.])\s+(\d+\.\s+)/g, "$1\n\n$2");
  clean = clean.replace(/([:!?\.])\s+([*-]\s+)/g, "$1\n\n$2");

  // If numbered items are packed together on one line, give each item its own line.
  clean = clean.replace(/(\d+\.\s+[^\n]+?)\s+(?=\d+\.\s+)/g, "$1\n");

  // Same for packed bullet lists.
  clean = clean.replace(/([*-]\s+[^\n]+?)\s+(?=[*-]\s+)/g, "$1\n");

  return clean.trim();
}

function normalise(msgs) {
  return msgs.filter((m) => !m.seed).map((m) => ({ role: m.role, content: m.content }));
}

//  Job row 

function JobRow({ jobId, repoId, onDone }) {
  const [lines,    setLines]    = useState([]);
  const [status,   setStatus]   = useState("running");
  const [ll2cPath, setLl2cPath] = useState("");
  const [progress, setProgress] = useState(null);
  const lastErrorRef = useRef("");

  useEffect(() => {
    const es = new EventSource(`/api/hub/status/${jobId}`);
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data);
        if (ev.type === "log" || ev.type === "error") {
          setLines((p) => [...p, { kind: ev.type, text: ev.msg }]);
          if (ev.type === "log") setProgress(null);
          if (ev.type === "error") lastErrorRef.current = ev.msg || "";
        }
        if (ev.type === "progress") setProgress(ev);
        if (ev.type === "done")     { setLl2cPath(ev.path); setProgress(null); }
        if (ev.type === "status")   {
          setStatus(ev.status);
          if (ev.status === "done" || ev.status === "error") {
            es.close();
            onDone?.(repoId, ev.status, lastErrorRef.current);
          }
        }
      } catch { /* ignore */ }
    };
    es.onerror = () => {
      setStatus((s) => s === "running" ? "error" : s);
      onDone?.(repoId, "error", lastErrorRef.current || "SSE connection lost");
      es.close();
    };
    return () => es.close();
  }, [jobId, onDone, repoId]);

  const badgeCls =
    status === "done"        ? "badge badge-green"
    : status === "error"     ? "badge badge-red"
    : status === "cancelled" ? "badge badge-neutral"
    :                          "badge badge-blue";

  return (
    <div className="hub-job">
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:"0.6rem" }}>
        <span style={{ fontFamily:"var(--mono)", fontSize:"0.72rem", color:"var(--text-2)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          {repoId}
        </span>
        <div style={{ display:"flex", alignItems:"center", gap:"0.4rem", flexShrink:0 }}>
          {status === "running" && (
            <button className="btn btn-ghost" style={{ fontSize:"0.7rem", padding:"0.2rem 0.55rem" }}
              onClick={() => fetch(`${API_ROUTES.hubJob}/${jobId}`, { method:"DELETE" })}>
              Cancel
            </button>
          )}
          <span className={badgeCls}>{status}</span>
        </div>
      </div>

      {ll2cPath && (
        <p style={{ marginTop:"0.2rem", fontSize:"0.68rem", color:"var(--text-3)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          Path: {ll2cPath}
        </p>
      )}

      {progress && (
        <div style={{ marginTop:"0.5rem" }}>
          <div style={{ display:"flex", justifyContent:"space-between", fontSize:"0.67rem", color:"var(--text-2)", marginBottom:"0.25rem", gap:"0.5rem" }}>
            <span style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
              [{progress.file_idx}/{progress.file_total}] {progress.file}
            </span>
            <span style={{ flexShrink:0, fontFamily:"var(--mono)" }}>
              {progress.pct >= 0 ? `${progress.pct}%` : ""}
              {progress.total > 0
                ? ` - ${fmtBytes(progress.downloaded)} / ${fmtBytes(progress.total)}`
                : ` - ${fmtBytes(progress.downloaded)}`}
              {progress.speed > 0 ? ` - ${fmtBytes(progress.speed)}/s` : ""}
              {progress.eta   > 0 ? ` - ${progress.eta < 60 ? `${progress.eta}s` : `${Math.round(progress.eta/60)}m`}` : ""}
            </span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width:`${Math.max(1, progress.pct >= 0 ? progress.pct : 0)}%` }} />
          </div>
        </div>
      )}

      <div className="log-box">
        {lines.length === 0
          ? <span style={{ color:"var(--text-3)", fontStyle:"italic" }}>Starting...</span>
          : lines.map((l, i) => (
            <div key={i} style={{ color: l.kind === "error" ? "#fca5a5" : undefined }}>{l.text}</div>
          ))
        }
      </div>
    </div>
  );
}

function QuantJobRow({ jobId, label, initialStatus = "running", initialOutPath = "", initialProgress = null, onDone }) {
  const [lines, setLines] = useState([]);
  const [status, setStatus] = useState(initialStatus);
  const [outPath, setOutPath] = useState(initialOutPath);
  const [progress, setProgress] = useState(initialProgress);

  useEffect(() => {
    const es = new EventSource(`${API_ROUTES.quantStatus}/${jobId}`);
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data);
        if (ev.type === "log" || ev.type === "error") {
          setLines((p) => [...p, { kind: ev.type, text: ev.msg }]);
        }
        if (ev.type === "progress") {
          setProgress(ev);
          setLines((p) => [...p, { kind: "log", text: `[${ev.done}/${ev.total}] ${ev.tensor}` }]);
        }
        if (ev.type === "done") {
          setOutPath(ev.path || "");
          setProgress({ done: 1, total: 1, pct: 100, tensor: "" });
        }
        if (ev.type === "status") {
          setStatus(ev.status);
          if (["done", "error", "cancelled"].includes(ev.status)) {
            es.close();
            onDone?.();
          }
        }
      } catch {
        // ignore
      }
    };
    es.onerror = () => {
      setStatus((s) => (s === "running" ? "error" : s));
      es.close();
    };
    return () => es.close();
  }, [jobId, onDone]);

  const badgeCls =
    status === "done"
      ? "badge badge-green"
      : status === "error"
        ? "badge badge-red"
        : status === "cancelled"
          ? "badge badge-neutral"
          : "badge badge-blue";

  return (
    <div className="hub-job">
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:"0.6rem" }}>
        <span style={{ fontFamily:"var(--mono)", fontSize:"0.72rem", color:"var(--text-2)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          {label}
        </span>
        <div style={{ display:"flex", alignItems:"center", gap:"0.4rem", flexShrink:0 }}>
          {status === "running" && (
            <button className="btn btn-ghost" style={{ fontSize:"0.7rem", padding:"0.2rem 0.55rem" }}
              onClick={() => fetch(`${API_ROUTES.quantJobs}/${jobId}`, { method:"DELETE" })}>
              Cancel
            </button>
          )}
          <span className={badgeCls}>{status}</span>
        </div>
      </div>

      {outPath && (
        <p style={{ marginTop:"0.2rem", fontSize:"0.68rem", color:"var(--text-3)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          Path: {outPath}
        </p>
      )}

      {progress && (
        <div style={{ marginTop:"0.45rem" }}>
          <div style={{ display:"flex", justifyContent:"space-between", fontSize:"0.67rem", color:"var(--text-2)", marginBottom:"0.2rem", gap:"0.5rem" }}>
            <span style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
              {progress.tensor || "finishing"}
            </span>
            <span style={{ flexShrink:0, fontFamily:"var(--mono)" }}>
              {Number.isFinite(progress.pct) ? `${progress.pct}%` : ""}
            </span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width:`${Math.max(1, Number(progress.pct) || 0)}%` }} />
          </div>
        </div>
      )}

      <div className="log-box">
        {lines.length === 0
          ? <span style={{ color:"var(--text-3)", fontStyle:"italic" }}>Starting...</span>
          : lines.slice(-50).map((l, i) => (
            <div key={i} style={{ color: l.kind === "error" ? "#fca5a5" : undefined }}>{l.text}</div>
          ))
        }
      </div>
    </div>
  );
}

//  Hub panel 

function HubPanel() {
  const [query,      setQuery]      = useState("");
  const [results,    setResults]    = useState([]);
  const [searching,  setSearching]  = useState(false);
  const [searchErr,  setSearchErr]  = useState("");
  const [hfToken,    setHfToken]    = useState(() => safeStorageGet(STORAGE_KEYS.hubToken));
  const [family,     setFamily]     = useState("");
  const [outputDir,  setOutputDir]  = useState(() => sanitizeStoredOutputDir(safeStorageGet(STORAGE_KEYS.hubOutputDir)));
  const [jobs,       setJobs]       = useState([]);
  const [local,      setLocal]      = useState([]);
  const [dlSet,      setDlSet]      = useState(new Set());
  const [dlStatus,   setDlStatus]   = useState({});
  const [quantJobs,  setQuantJobs]  = useState([]);
  const [quantSet,   setQuantSet]   = useState(new Set());
  const [pickingDir, setPickingDir] = useState(false);
  const tokenApplied = hfToken.trim().length > 0;

  useEffect(() => {
    const cleaned = sanitizeStoredOutputDir(outputDir);
    if (cleaned !== outputDir) {
      setOutputDir(cleaned);
      safeStorageSet(STORAGE_KEYS.hubOutputDir, cleaned);
    }
  }, [outputDir]);

  const refreshLocalModels = useCallback(() => {
    fetchJson(API_ROUTES.models)
      .then((payload) => setLocal(payload.models ?? []))
      .catch(() => {});
  }, []);

  const syncModelDir = useCallback(async (dir) => {
    const payload = await fetchJson(API_ROUTES.modelDir, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ dir })
    });
    if (Array.isArray(payload.models)) {
      setLocal(payload.models);
    } else {
      refreshLocalModels();
    }
  }, [refreshLocalModels]);

  useEffect(() => {
    void syncModelDir(outputDir.trim());
  }, [outputDir, syncModelDir]);

  useEffect(() => {
    refreshLocalModels();
  }, [refreshLocalModels]);

  useEffect(() => {
    fetchJson(API_ROUTES.quantJobs)
      .then((payload) => {
        const jobsFromApi = payload.jobs ?? [];
        setQuantJobs(
          jobsFromApi.map((j) => ({
            jobId: j.jobId,
            label: j.modelLabel ? `${j.modelLabel} - ${quantLabel(j.quantMode)}` : (j.path ? tailPath(j.path) : j.jobId),
            profileId: j.profileId || "",
            mode: j.quantMode || "",
            status: j.status || "running",
            path: j.path || "",
            progress: j.progress || null
          }))
        );
      })
      .catch(() => {});
  }, []);

  async function doSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSearching(true); setSearchErr(""); setResults([]);
    try {
      const params = new URLSearchParams({ q, limit: "12" });
      if (hfToken) params.set("token", hfToken);
      const data = await fetchJson(`${API_ROUTES.hubSearch}?${params}`);
      setResults(data.models ?? []);
    } catch (err) {
      setSearchErr(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function doDownload(repoId) {
    if (dlSet.has(repoId)) return;
    setDlSet((p) => new Set([...p, repoId]));
    setDlStatus((s) => ({ ...s, [repoId]: { state: "running", error: "" } }));
    const body = { repoId };
    if (hfToken)           body.hfToken   = hfToken;
    if (family)            body.family    = family;
    if (outputDir.trim())  body.outputDir = outputDir.trim() + "/" + repoId.replace("/", "__");
    try {
      const data = await fetchJson(API_ROUTES.hubDownload, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      setJobs((p) => [{ jobId: data.jobId, repoId }, ...p]);
    } catch (err) {
      setDlSet((p) => { const s = new Set(p); s.delete(repoId); return s; });
      setDlStatus((s) => ({ ...s, [repoId]: { state: "error", error: err.message || "Download request failed" } }));
      alert(`Download error: ${err.message}`);
    }
  }

  async function pickOutputDir() {
    if (pickingDir) return;
    setPickingDir(true);
    try {
      const payload = await fetchJson(API_ROUTES.pickFolder, {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({
          initialDir: outputDir.trim() || undefined
        })
      });
      if (!payload?.cancelled && payload?.path) {
        setOutputDir(payload.path);
        safeStorageSet(STORAGE_KEYS.hubOutputDir, payload.path);
      }
    } catch (err) {
      alert(`Folder picker error: ${err.message}`);
    } finally {
      setPickingDir(false);
    }
  }

  const onJobDone = useCallback((repoId, status = "done", errorMsg = "") => {
    setDlSet((p) => { const s = new Set(p); s.delete(repoId); return s; });
    setDlStatus((s) => ({
      ...s,
      [repoId]: { state: status === "done" ? "done" : "error", error: errorMsg || "" }
    }));
    refreshLocalModels();
  }, [refreshLocalModels]);

  async function doQuantConvert(profileId, mode, label) {
    const key = `${profileId}|${mode}`;
    if (quantSet.has(key)) return;
    setQuantSet((p) => new Set([...p, key]));
    try {
      const data = await fetchJson(API_ROUTES.quantConvert, {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ profileId, quantMode: mode })
      });
      const entryLabel = `${label} - ${quantLabel(mode)}`;
      setQuantJobs((prev) => [{
        jobId: data.jobId,
        label: entryLabel,
        profileId,
        mode,
        status: "running",
        path: data.outputPath || "",
        progress: null
      }, ...prev]);
    } catch (err) {
      alert(`Quant conversion error: ${err.message}`);
      setQuantSet((p) => {
        const s = new Set(p);
        s.delete(key);
        return s;
      });
    }
  }

  const onQuantJobDone = useCallback((profileId, mode) => {
    if (profileId && mode) {
      setQuantSet((p) => {
        const s = new Set(p);
        s.delete(`${profileId}|${mode}`);
        return s;
      });
    }
    refreshLocalModels();
  }, [refreshLocalModels]);

  return (
    <div className="hub-scroll">
      <div className="hub-inner">

        {/* Jobs */}
        {jobs.length > 0 && (
          <div className="hub-card">
            <p className="hub-card-label">Download Jobs</p>
            {jobs.map((j) => <JobRow key={j.jobId} jobId={j.jobId} repoId={j.repoId} onDone={onJobDone} />)}
          </div>
        )}

        {quantJobs.length > 0 && (
          <div className="hub-card">
            <p className="hub-card-label">Quantization Jobs</p>
            {quantJobs.map((j) => (
              <QuantJobRow
                key={j.jobId}
                jobId={j.jobId}
                label={j.label}
                initialStatus={j.status}
                initialOutPath={j.path}
                initialProgress={j.progress}
                onDone={() => onQuantJobDone(j.profileId, j.mode)}
              />
            ))}
          </div>
        )}

        {/* Search */}
        <div className="hub-card">
          <p className="hub-card-label">Search Hugging Face</p>
          <form style={{ display:"flex", gap:"0.5rem" }} onSubmit={doSearch}>
            <input className="field-ctrl" type="text" placeholder="Qwen2-1.5B, Mistral-7B, TinyLlama..."
              value={query} onChange={(e) => setQuery(e.target.value)} />
            <button type="submit" className="btn btn-primary" style={{ flexShrink:0 }} disabled={searching || !query.trim()}>
              {searching ? "..." : "Search"}
            </button>
          </form>

          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"0.5rem", marginTop:"0.6rem" }}>
            <div className="field">
              <label className="field-label">Output dir</label>
              <div style={{ display:"flex", gap:"0.4rem", alignItems:"center" }}>
                <input className="field-ctrl" type="text" placeholder="D:\\models (or leave blank)"
                  style={{ flex:1, minWidth:0 }}
                  value={outputDir}
                  onChange={(e) => { setOutputDir(e.target.value); safeStorageSet(STORAGE_KEYS.hubOutputDir, e.target.value); }} />
                <button
                  type="button"
                  className="btn btn-ghost"
                  style={{ flexShrink:0, fontSize:"0.72rem", padding:"0.32rem 0.55rem" }}
                  disabled={pickingDir}
                  onClick={pickOutputDir}
                >
                  {pickingDir ? "Browsing..." : "Browse..."}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost"
                  style={{ flexShrink:0, fontSize:"0.72rem", padding:"0.32rem 0.55rem" }}
                  disabled={!outputDir.trim()}
                  onClick={() => { setOutputDir(""); safeStorageSet(STORAGE_KEYS.hubOutputDir, ""); }}
                >
                  Clear
                </button>
              </div>
              <p style={{ fontSize:"0.68rem", marginTop:"0.22rem", color:"var(--text-3)" }}>
                Leave blank to use the default artifacts/hub path.
              </p>
            </div>
            <div className="field">
              <label className="field-label">Family override</label>
              <select className="field-ctrl" value={family} onChange={(e) => setFamily(e.target.value)}>
                <option value="">Auto-detect</option>
                {HF_FAMILIES.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
          </div>

          <div className="field" style={{ marginTop:"0.5rem" }}>
            <label className="field-label">HF Token (optional)</label>
            <input className="field-ctrl" type="password" placeholder="hf_..."
              style={{ fontFamily:"var(--mono)", fontSize:"0.76rem" }}
              value={hfToken}
              onChange={(e) => { setHfToken(e.target.value); safeStorageSet(STORAGE_KEYS.hubToken, e.target.value); }} />
            <p style={{ fontSize:"0.68rem", marginTop:"0.2rem", color: tokenApplied ? "var(--green)" : "var(--amber)" }}>
              {tokenApplied ? "Token applied to Hugging Face requests." : "No token applied (public access only)."}
            </p>
          </div>

          {searchErr && (
            <p style={{ marginTop:"0.6rem", fontSize:"0.77rem", color:"var(--red)", background:"var(--red-bg)", padding:"0.4rem 0.65rem", borderRadius:"0.35rem" }}>
              {searchErr}
            </p>
          )}

          {results.length > 0 && (
            <div style={{ marginTop:"0.75rem" }}>
              {results.map((m) => (
                <div key={m.id} className="hub-row">
                  <div style={{ minWidth:0, flex:1 }}>
                    <p style={{ fontFamily:"var(--mono)", fontSize:"0.78rem", color:"var(--text)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{m.id}</p>
                    <p style={{ fontSize:"0.68rem", color:"var(--text-3)", marginTop:"0.1rem" }}>
                      Downloads {fmtDl(m.downloads)}{m.likes ? ` - Likes ${m.likes}` : ""}{m.pipelineTag ? ` - ${m.pipelineTag}` : ""}{m.gated ? " - gated" : ""}
                    </p>
                    {dlStatus[m.id]?.state === "error" && dlStatus[m.id]?.error && (
                      <p style={{ fontSize:"0.66rem", color:"var(--red)", marginTop:"0.15rem", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                        {dlStatus[m.id].error}
                      </p>
                    )}
                  </div>
                  <button
                    className={dlStatus[m.id]?.state === "error" ? "btn btn-ghost" : "btn btn-primary"}
                    style={{ fontSize:"0.74rem", padding:"0.3rem 0.7rem" }}
                    disabled={dlSet.has(m.id)}
                    onClick={() => doDownload(m.id)}
                    title={dlStatus[m.id]?.error || ""}
                  >
                    {dlSet.has(m.id)
                      ? "Queued"
                      : dlStatus[m.id]?.state === "done"
                        ? "Downloaded"
                        : dlStatus[m.id]?.state === "error"
                          ? "Retry"
                          : "Download"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Local models */}
        <div className="hub-card">
          <p className="hub-card-label">Local Models</p>
          {local.length === 0 ? (
            <p style={{ fontSize:"0.8rem", color:"var(--text-3)" }}>No models found in artifacts/.</p>
          ) : (
            local.map((m) => (
              <div key={m.id} className="hub-row">
                <div style={{ minWidth:0, flex:1 }}>
                  <p style={{ fontSize:"0.8rem", color:"var(--text)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{m.label}</p>
                  <p style={{ fontSize:"0.67rem", color:"var(--text-3)", fontFamily:"var(--mono)", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", marginTop:"0.1rem" }}>{m.modelPath}</p>
                  {m.quant && (
                    <p style={{ fontSize:"0.66rem", color:"var(--text-2)", marginTop:"0.2rem" }}>
                      q: default {quantLabel(m.quant.effectiveMode)} - packed {m.quant.packed.int8 ? "INT8" : "-"}/{m.quant.packed.int4 ? "INT4" : "-"}{m.quant.packed.tq3 ? " - TQ3" : ""}
                    </p>
                  )}
                </div>
                <div style={{ display:"flex", alignItems:"center", gap:"0.35rem", flexShrink:0 }}>
                  {m.quant?.conversion?.int8?.state === "available" && (
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize:"0.68rem", padding:"0.2rem 0.45rem" }}
                      disabled={quantSet.has(`${m.id}|int8`)}
                      onClick={() => doQuantConvert(m.id, "int8", m.label)}
                    >
                      {quantSet.has(`${m.id}|int8`) ? "INT8..." : "Pack INT8"}
                    </button>
                  )}
                  {m.quant?.conversion?.int4?.state === "available" && (
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize:"0.68rem", padding:"0.2rem 0.45rem" }}
                      disabled={quantSet.has(`${m.id}|int4`)}
                      onClick={() => doQuantConvert(m.id, "int4", m.label)}
                    >
                      {quantSet.has(`${m.id}|int4`) ? "INT4..." : "Pack INT4"}
                    </button>
                  )}
                  <span
                    className={`badge ${m.ready ? "badge-green" : "badge-amber"}`}
                    title={m.unsupportedReason || ""}
                  >
                    {m.ready ? "ready" : m.status}
                  </span>
                </div>
                {!m.ready && m.unsupportedReason && (
                  <div style={{ fontSize:"0.72rem", color:"var(--muted)", marginTop:"0.25rem" }}>
                    {m.unsupportedReason}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

      </div>
    </div>
  );
}

//  App 

export default function App() {
  const [view,     setView]     = useState("chat");
  const [showCfg,  setShowCfg]  = useState(false);
  const [theme,    setTheme]    = useState(() => safeStorageGet(STORAGE_KEYS.theme, "dark") || "dark");
  // Restore stored chats and the previously active conversation once.
  const [initialChatState] = useState(() => {
    const storedChats = loadStoredChats();
    const storedActiveId = safeStorageGet(STORAGE_KEYS.activeChat, "");
    const active = storedChats.find((c) => c.id === storedActiveId) || null;
    return {
      chats: storedChats,
      activeChatId: active ? active.id : crypto.randomUUID(),
      messages: active
        ? [...seedMessages(), ...active.messages.map((m) => createMsg(m.role, m.content))]
        : seedMessages()
    };
  });
  const [chats,        setChats]        = useState(initialChatState.chats);
  const [activeChatId, setActiveChatId] = useState(initialChatState.activeChatId);
  const [messages,     setMessages]     = useState(initialChatState.messages);
  const [draft,    setDraft]    = useState("");
  const [health,   setHealth]   = useState({ ready:false, busy:false, activeKind:null, config:null });
  const [settings, setSettings] = useState(() => ({
    profileId: "",
    template: "tinyllama",
    systemPrompt: "",
    temperature: 0.7,
    maxNewTokens: 320,
    maxContext: 2048,
    quantMode: "none",
    autoMaxTokens: safeStorageGet(STORAGE_KEYS.autoMaxTokens, "1") !== "0",
    longFormMode: safeStorageGet(STORAGE_KEYS.longFormMode, "0") === "1",
    thinking: safeStorageGet(STORAGE_KEYS.thinking, "0") === "1",
    performanceMode: (() => {
      return safeStorageGet(STORAGE_KEYS.perfMode) === "1";
    })()
  }));
  const [hydrated, setHydrated] = useState(false);
  const [streaming,setStreaming]= useState(false);
  const [error,    setError]    = useState("");
  const [runMeta,  setRunMeta]  = useState(null);
  const [warmup,   setWarmup]   = useState({ state:"idle", workerKey:"", error:"" });
  const [stamp,    setStamp]    = useState("...");
  const [quantState, setQuantState] = useState(null);

  const scrollRef   = useRef(null);
  const taRef       = useRef(null);
  const abortRef    = useRef(null);
  const messagesRef = useRef(messages);
  const warmupAbortRef = useRef(null);
  const warmupSeqRef = useRef(0);
  const deltaQueueRef = useRef("");
  const deltaTargetRef = useRef("");
  const deltaRafRef = useRef(0);

  const profiles     = health.config?.availableProfiles ?? [];
  const readyProfiles= profiles.filter((p) => p.ready);
  const selProfile   =
    readyProfiles.find((p) => p.id === settings.profileId) ||
    profiles.find((p) => p.id === settings.profileId) ||
    health.config?.selectedProfile || null;
  const selDisplay   = selProfile ? modelDisplay(selProfile) : { base:"", tag:"", isStreaming:false, fmt:"", pinned:false };
  // Consolidated model + precision/variant selection.
  const modelGroups  = buildModelGroups(readyProfiles);
  const currentVarKey = `${settings.profileId}|${settings.quantMode || "none"}`;
  const selectedGroup = modelGroups.find((g) => g.variants.some((v) => v.profileId === settings.profileId))
    || modelGroups[0] || null;
  const selectedVariant = selectedGroup?.variants.find((v) => v.key === currentVarKey)
    || selectedGroup?.variants.find((v) => v.profileId === settings.profileId)
    || selectedGroup?.variants[0] || null;
  const applyVariant = (v) => {
    if (!v) return;
    setSettings((cur) => ({ ...cur, profileId: v.profileId, quantMode: v.quantMode, template: v.template || cur.template }));
    fetchJson(API_ROUTES.quantSelect, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profileId: v.profileId, quantMode: v.quantMode })
    }).catch(() => {});
  };
  const maxContextLimit = Math.max(
    128,
    Number(selProfile?.maxPositionEmbeddings) || 0,
    Number(health.config?.maxContext) || 0,
    2048
  );
  const selectedWorkerKey =
    `${settings.profileId || ""}|ctx:${settings.maxContext || 0}|perf:${settings.performanceMode ? 1 : 0}|q:${settings.quantMode || "none"}`;
  const warmupForSelected = warmup.workerKey === selectedWorkerKey ? warmup.state : "idle";
  const warmupDoneForSelected = warmupForSelected === "ready";
  const backendBusyForWarmup = health.busy && health.activeKind === "warmup";
  const selectedQuantConversionState =
    quantState?.quant?.conversion?.[settings.quantMode]?.state ??
    selProfile?.quant?.conversion?.[settings.quantMode]?.state ??
    "unavailable";
  const selectedQuantJob = quantState?.latestJobForSelectedQuant || null;
  const selectedQuantJobRunning = selectedQuantJob?.status === "running";
  const selectedQuantJobPct =
    Number.isFinite(Number(selectedQuantJob?.progress?.pct))
      ? Number(selectedQuantJob.progress.pct)
      : null;
  const activeWorkerQuantMode = quantState?.activeWorker?.quantMode || "";
  const selectedQuantNeedsPacking =
    Boolean(selProfile) &&
    (settings.quantMode === "int8" || settings.quantMode === "int4") &&
    selectedQuantConversionState === "available";
  const engineState = (() => {
    if (!health.ready) return { label:"Not configured", dot:"dot-amber", badge:"badge-amber" };
    if (!settings.profileId) return { label:"No model", dot:"dot-amber", badge:"badge-amber" };
    if (streaming) return { label:"Streaming", dot:"dot-blue", badge:"badge-blue" };
    if (backendBusyForWarmup) return { label:"Preparing", dot:"dot-blue", badge:"badge-blue" };
    if (health.busy) return { label:"Streaming", dot:"dot-blue", badge:"badge-blue" };
    if (warmupForSelected === "preparing") return { label:"Preparing", dot:"dot-blue", badge:"badge-blue" };
    if (warmupForSelected === "lazy") return { label:"Loads on first message", dot:"dot-amber", badge:"badge-amber" };
    if (warmupForSelected === "error") return { label:"Preparation failed", dot:"dot-red", badge:"badge-red" };
    if (warmupDoneForSelected) return { label:"Ready", dot:"dot-green", badge:"badge-green" };
    return { label:"Idle", dot:"dot-amber", badge:"badge-amber" };
  })();

  // Health polling
  useEffect(() => {
    async function poll() {
      try {
        const next = await fetchHealth();
        setHealth(next);
        setError("");
        setStamp(new Date().toLocaleTimeString());
      } catch (e) {
        setError(e.message);
        setStamp("unreachable");
      }
    }
    poll();
    const t = setInterval(poll, 10000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!settings.profileId) {
      setQuantState(null);
      return;
    }

    let cancelled = false;
    const pollQuant = async () => {
      try {
        const params = new URLSearchParams({
          profileId: settings.profileId,
          quantMode: settings.quantMode || "none"
        });
        const next = await fetchJson(`${API_ROUTES.quantState}?${params.toString()}`);
        if (!cancelled) {
          setQuantState(next);
        }
      } catch {
        if (!cancelled) {
          setQuantState(null);
        }
      }
    };

    pollQuant();
    const t = setInterval(pollQuant, 2500);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [settings.profileId, settings.quantMode]);

  // Hydrate settings from server config once
  useEffect(() => {
    if (!health.config) return;
    const profs   = health.config.availableProfiles ?? [];
    const fallback = profs.find((p) => p.id === health.config.selectedProfileId) || profs[0] || null;
    setSettings((cur) => {
      let changed = false;
      const next  = { ...cur };
      if (!hydrated) {
        if (health.config.systemPrompt && next.systemPrompt !== health.config.systemPrompt)  { next.systemPrompt  = health.config.systemPrompt;  changed = true; }
        if (health.config.temperature  != null && next.temperature  !== health.config.temperature)  { next.temperature  = health.config.temperature;  changed = true; }
        if (health.config.maxNewTokens != null && next.maxNewTokens !== health.config.maxNewTokens) { next.maxNewTokens = health.config.maxNewTokens; changed = true; }
        if (health.config.maxContext   != null && next.maxContext   !== health.config.maxContext)   { next.maxContext   = health.config.maxContext;   changed = true; }
      }
      const alive = next.profileId && profs.some((p) => p.id === next.profileId && p.ready);
      if ((!next.profileId || !alive) && fallback) {
        if (next.profileId !== fallback.id)       { next.profileId = fallback.id;       changed = true; }
        if (next.template  !== fallback.template) { next.template  = fallback.template; changed = true; }
        const fallbackQuant = defaultQuantForProfile(fallback);
        if (next.quantMode !== fallbackQuant)     { next.quantMode = fallbackQuant;     changed = true; }
      }
      const activeProfile = profs.find((p) => p.id === next.profileId) || fallback;
      if (activeProfile) {
        const allowed = activeProfile.quant?.selectableModes ?? ["none"];
        if (!allowed.includes(next.quantMode)) {
          next.quantMode = allowed[0] || "none";
          changed = true;
        }
        const contextCap = Math.max(
          128,
          Number(activeProfile.maxPositionEmbeddings) || 0,
          Number(health.config?.maxContext) || 0
        );
        if (!Number.isFinite(next.maxContext) || next.maxContext < 128) {
          next.maxContext = Math.min(Number(health.config?.maxContext) || 2048, contextCap);
          changed = true;
        } else if (next.maxContext > contextCap) {
          next.maxContext = contextCap;
          changed = true;
        }
      }
      return changed ? next : cur;
    });
    if (!hydrated) setHydrated(true);
  }, [hydrated, health.config]);

  useEffect(() => {
    const nextTheme = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    safeStorageSet(STORAGE_KEYS.theme, nextTheme);
  }, [theme]);

  useEffect(() => {
    safeStorageSet(STORAGE_KEYS.autoMaxTokens, settings.autoMaxTokens ? "1" : "0");
  }, [settings.autoMaxTokens]);

  useEffect(() => {
    safeStorageSet(STORAGE_KEYS.longFormMode, settings.longFormMode ? "1" : "0");
  }, [settings.longFormMode]);

  useEffect(() => {
    safeStorageSet(STORAGE_KEYS.thinking, settings.thinking ? "1" : "0");
  }, [settings.thinking]);

  // Warm selected model immediately so first response avoids a full cold start.
  useEffect(() => {
    const profileId = settings.profileId;
    if (!profileId || !selProfile?.ready || streaming) return;
    // Streaming models load slowly and lazily (weights stream from disk, often
    // gigabytes). A blocking pre-warm just times out and looks like a failure, so
    // skip it — they load on the first message instead.
    if (selDisplay.isStreaming) { setWarmup({ state:"lazy", workerKey: selectedWorkerKey, error:"" }); return; }
    const workerKey = selectedWorkerKey;

    warmupAbortRef.current?.abort();
    const ctrl = new AbortController();
    warmupAbortRef.current = ctrl;
    const seq = ++warmupSeqRef.current;
    setWarmup({ state:"preparing", workerKey, error:"" });

    const pollWarmup = async () => {
      const maxPolls = 240; // ~5 minutes at 1.25s
      for (let i = 0; i < maxPolls; i += 1) {
        const res = await fetch("/api/warmup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            profileId,
            performanceMode: settings.performanceMode,
            quantMode: settings.quantMode,
            maxContext: settings.maxContext
          }),
          signal: ctrl.signal
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.error || `Preparation failed (${res.status})`);
        }
        if (seq !== warmupSeqRef.current || ctrl.signal.aborted) return;
        if (data?.ready === true || data?.pending === false) {
          setWarmup({ state:"ready", workerKey, error:"" });
          return;
        }
        setWarmup({ state:"preparing", workerKey, error:"" });
        await new Promise((resolve) => setTimeout(resolve, 1250));
      }
      throw new Error("Preparation timed out. Try sending a prompt to start the model.");
    };

    pollWarmup().catch((e) => {
      if (seq !== warmupSeqRef.current || ctrl.signal.aborted) return;
      setWarmup({ state:"error", workerKey, error: e.message || "preparation failed" });
    });

    return () => ctrl.abort();
  }, [selectedWorkerKey, selProfile?.ready, settings.profileId, settings.performanceMode, settings.quantMode, settings.maxContext, streaming]);

  useEffect(() => () => warmupAbortRef.current?.abort(), []);
  useEffect(() => () => stopDeltaPump(), []);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Sync the current conversation into the chat list. Skipped while streaming
  // so delta frames don't churn state; the final message lands when streaming
  // flips back to false.
  useEffect(() => {
    if (streaming) return;
    const real = messages
      .filter((m) => !m.seed && String(m.content || "").trim())
      .map(({ role, content }) => ({ role, content }));
    if (real.length === 0) return;
    setChats((cur) => {
      const existing = cur.find((c) => c.id === activeChatId);
      if (existing && JSON.stringify(existing.messages) === JSON.stringify(real)) {
        return cur;
      }
      const entry = {
        id: activeChatId,
        title: chatTitleFrom(real),
        createdAt: existing?.createdAt ?? Date.now(),
        updatedAt: Date.now(),
        messages: real
      };
      return [entry, ...cur.filter((c) => c.id !== activeChatId)].slice(0, MAX_STORED_CHATS);
    });
  }, [messages, streaming, activeChatId]);

  useEffect(() => {
    persistStoredChats(chats);
  }, [chats]);

  useEffect(() => {
    safeStorageSet(STORAGE_KEYS.activeChat, activeChatId);
  }, [activeChatId]);

  function startNewChat() {
    abortRef.current?.abort();
    setActiveChatId(crypto.randomUUID());
    updateMessages(seedMessages());
    setRunMeta(null);
    setError("");
  }

  function openChat(id) {
    if (id === activeChatId) return;
    const chat = chats.find((c) => c.id === id);
    if (!chat) return;
    abortRef.current?.abort();
    setActiveChatId(id);
    updateMessages([...seedMessages(), ...chat.messages.map((m) => createMsg(m.role, m.content))]);
    setRunMeta(null);
    setError("");
  }

  function deleteChat(id) {
    setChats((cur) => cur.filter((c) => c.id !== id));
    if (id === activeChatId) {
      startNewChat();
    }
  }

  // Auto-scroll
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // During streaming, avoid re-starting smooth scroll on every delta frame.
    if (streaming) {
      el.scrollTop = el.scrollHeight;
      return;
    }
    el.scrollTo({ top: el.scrollHeight, behavior:"smooth" });
  }, [messages, streaming]);

  function growTa(val) {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "0";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    if (!val) { el.style.height = "0"; el.style.height = "50px"; }
  }

  function applyStarter(text) {
    setDraft(text);
    requestAnimationFrame(() => { growTa(text); taRef.current?.focus(); });
  }

  function updateMessages(updater) {
    setMessages((cur) => {
      const next = typeof updater === "function" ? updater(cur) : updater;
      messagesRef.current = next;
      return next;
    });
  }

  function snapshotMessages() {
    const queued = deltaQueueRef.current;
    const targetId = deltaTargetRef.current;
    if (!queued || !targetId) {
      return messagesRef.current;
    }
    return messagesRef.current.map((m) =>
      m.id === targetId ? { ...m, content: `${m.content}${queued}` } : m
    );
  }

  function stopDeltaPump() {
    if (deltaRafRef.current) {
      cancelAnimationFrame(deltaRafRef.current);
      deltaRafRef.current = 0;
    }
  }

  function flushDeltaQueue() {
    const rest = deltaQueueRef.current;
    const targetId = deltaTargetRef.current;
    deltaQueueRef.current = "";
    deltaTargetRef.current = "";
    stopDeltaPump();
    if (!rest || !targetId) return;
    updateMessages((cur) =>
      cur.map((m) =>
        m.id === targetId ? { ...m, content: `${m.content}${rest}`, streaming: true } : m
      )
    );
  }

  function startDeltaPump() {
    if (deltaRafRef.current) return;
    const pump = () => {
      const targetId = deltaTargetRef.current;
      const queued = deltaQueueRef.current;
      if (!targetId || !queued) {
        deltaRafRef.current = 0;
        return;
      }

      // Natural chunking keeps cadence smooth without large "jumps".
      let maxTake = 3;
      if (queued.length > 768) maxTake = 18;
      else if (queued.length > 384) maxTake = 14;
      else if (queued.length > 192) maxTake = 10;
      else if (queued.length > 96) maxTake = 8;
      else if (queued.length > 48) maxTake = 6;

      let take = Math.min(maxTake, queued.length);
      const boundary = queued.slice(0, take + 2).search(/[\s,.;:!?]/);
      if (boundary >= 2 && boundary <= maxTake + 1) {
        take = boundary + 1;
      }

      const chunk = queued.slice(0, take);
      deltaQueueRef.current = queued.slice(take);
      updateMessages((cur) =>
        cur.map((m) =>
          m.id === targetId ? { ...m, content: `${m.content}${chunk}`, streaming: true } : m
        )
      );
      deltaRafRef.current = requestAnimationFrame(pump);
    };
    deltaRafRef.current = requestAnimationFrame(pump);
  }

  function enqueueDelta(targetId, deltaText) {
    if (!deltaText) return;
    if (deltaTargetRef.current && deltaTargetRef.current !== targetId) {
      flushDeltaQueue();
    }
    deltaTargetRef.current = targetId;
    deltaQueueRef.current += deltaText;
    startDeltaPump();
  }

  async function submit(override) {
    const text = (override ?? draft).trim();
    if (!text || streaming) return;

    const userMsg = createMsg("user", text);
    const asstMsg = createMsg("assistant", "", { streaming:true });
    const thread  = normalise([...snapshotMessages(), userMsg]);
    const ctrl    = new AbortController();

    setError(""); setRunMeta({ state:"starting" }); setStreaming(true);
    abortRef.current = ctrl;
    updateMessages((cur) => [...cur, userMsg, asstMsg]);
    setDraft(""); requestAnimationFrame(() => growTa(""));

    try {
      const done = await streamChat({
        messages: thread, settings,
        signal: ctrl.signal,
        onEvent: (ev) => {
          if (ev.type === "start") {
            setRunMeta({
              state:"streaming",
              modelLabel: ev.modelLabel,
              maxNewTokens: ev.maxNewTokens ?? null,
              autoMaxTokens: ev.autoMaxTokens ?? settings.autoMaxTokens,
              longFormMode: ev.longFormMode ?? settings.longFormMode
            }); return;
          }
          if (ev.type === "metrics" && ev.metrics) {
            setRunMeta((prev) => ({ ...(prev || {}), state:"streaming", metrics: ev.metrics }));
            return;
          }
          if (ev.type === "reasoning") {
            // Reasoning (<think>) streams before the answer; accumulate directly
            // into the message's reasoning field (rendered in a collapsible block).
            updateMessages((cur) => cur.map((m) =>
              m.id === asstMsg.id
                ? { ...m, reasoning: `${m.reasoning || ""}${ev.delta}`, reasoningStreaming: true }
                : m
            ));
            return;
          }
          if (ev.type === "delta") {
            enqueueDelta(asstMsg.id, ev.delta);
          }
        },
      });
      flushDeltaQueue();
      const finalMessage = typeof done?.message === "string" ? done.message : "";
      const finalReasoning = typeof done?.reasoning === "string" ? done.reasoning : "";
      updateMessages((cur) => cur.map((m) =>
        m.id === asstMsg.id
          ? { ...m, content: finalMessage || m.content,
              reasoning: finalReasoning || m.reasoning || "",
              streaming:false, reasoningStreaming:false }
          : m
      ));
      setRunMeta((prev) => ({
        ...(prev || {}),
        state: done?.type === "aborted" ? "stopped" : "done",
        elapsedMs: done?.elapsedMs ?? 0,
        generatedTokens: done?.generatedTokens ?? null,
        tokPerS: done?.tokPerS ?? null,
        decodeMs: done?.decodeMs ?? null,
        decodeTokPerS: done?.decodeTokPerS ?? null,
        metrics: done?.metrics ?? null,
        maxNewTokens: done?.maxNewTokens ?? prev?.maxNewTokens ?? null,
        autoMaxTokens: done?.autoMaxTokens ?? prev?.autoMaxTokens ?? settings.autoMaxTokens,
        longFormMode: done?.longFormMode ?? prev?.longFormMode ?? settings.longFormMode
      }));
    } catch (e) {
      flushDeltaQueue();
      const aborted = e.name === "AbortError";
      const rawMsg = (e?.message || "No response from engine.").trim();
      const errMsg = friendlyEngineError(rawMsg);
      const briefErr = errMsg.length > 220 ? `${errMsg.slice(0, 220)}...` : errMsg;
      updateMessages((cur) => cur.map((m) =>
        m.id === asstMsg.id
          ? { ...m, content: m.content || (aborted ? "Stopped." : briefErr), streaming:false }
          : m
      ));
      if (!aborted) setError(errMsg);
      setRunMeta({ state: aborted ? "stopped" : "error" });
    } finally {
      stopDeltaPump();
      abortRef.current = null; setStreaming(false);
    }
  }

  const empty = messages.filter((m) => !m.seed).length === 0;

  return (
    <div className={`shell shell-${view}`}>

      {/*  Topbar  */}
      <header className="topbar">
        <span className="topbar-brand">CPI</span>
        <span className="topbar-sep" />

        {/* Model selector (base name) */}
        <select
          className="topbar-select"
          value={selectedGroup?.base || ""}
          disabled={streaming}
          onChange={(e) => {
            const g = modelGroups.find((x) => x.base === e.target.value);
            // Default to FP16 (non-streaming) if available, else the first variant.
            const def = g?.variants.find((v) => v.quantMode === "none" && !v.label.includes("streaming"))
              || g?.variants[0];
            applyVariant(def);
          }}
          title="Model"
        >
          {modelGroups.length === 0 && (
            <option value="">{health.config ? "No models found" : "Loading"}</option>
          )}
          {modelGroups.map((g) => (
            <option key={g.base} value={g.base}>{g.base}</option>
          ))}
        </select>

        {/* Variant / precision selector (FP16 · INT8 · INT4 · streaming) */}
        <select
          className="topbar-select"
          value={selectedVariant?.key || ""}
          disabled={streaming || !selectedGroup || (selectedGroup.variants.length <= 1)}
          onChange={(e) => applyVariant(selectedGroup?.variants.find((v) => v.key === e.target.value))}
          title="Precision / variant — FP16: full quality. INT8 / INT4: smaller & faster (slight quality trade-off). 'streaming' builds run models too large for VRAM by streaming weights from disk."
        >
          {(selectedGroup?.variants ?? []).map((v) => (
            <option key={v.key} value={v.key}>{v.label}</option>
          ))}
        </select>

        <span className="topbar-gap" />

        {/* Status */}
        <span className="topbar-status">
          <span className={`dot ${engineState.dot}`} />
          <span>Engine</span>
          <span className={`badge ${engineState.badge}`}>{engineState.label}</span>
          {selectedQuantJobRunning && (
            <span className="badge badge-blue">
              Converting{selectedQuantJobPct != null ? ` ${selectedQuantJobPct}%` : ""}
            </span>
          )}
          {activeWorkerQuantMode && activeWorkerQuantMode !== settings.quantMode && (
            <span className="badge badge-red">Worker Q {quantLabel(activeWorkerQuantMode)}</span>
          )}
          {settings.performanceMode && <span className="badge badge-blue">Perf</span>}
          <span className="topbar-model" title={selProfile?.label || ""}>
            {selDisplay.base || "-"}{selectedVariant && selectedVariant.label !== "FP16" ? ` · ${selectedVariant.label}` : ""}
          </span>
        </span>

        <span style={{ fontSize:"0.7rem", color:"var(--text-3)", marginLeft:"0.4rem", flexShrink:0 }}>{stamp}</span>
        <span className="topbar-sep" />

        {/* Hub toggle */}
        <button
          type="button"
          className={`topbar-btn ${view === "hub" ? "topbar-btn-on" : ""}`}
          onClick={() => setView((v) => v === "hub" ? "chat" : "hub")}
        >
          Model Hub
        </button>

        <button
          type="button"
          className="topbar-btn"
          onClick={() => setTheme((cur) => cur === "light" ? "dark" : "light")}
          title="Toggle color theme"
        >
          {theme === "light" ? "Dark" : "Light"}
        </button>

        {/* Settings */}
        <button type="button" className="topbar-icon" title="Settings" onClick={() => setShowCfg(true)}>
          Cfg
        </button>
      </header>

      {/*  Content  */}
      <div className="content">

        {view === "hub" ? (
          <HubPanel />
        ) : (
          <div className="chat-layout">

            {/* Chat sidebar */}
            <aside className="chat-sidebar">
              <button
                type="button"
                className="btn btn-primary chat-new-btn"
                onClick={startNewChat}
              >
                + New chat
              </button>
              <div className="chat-list">
                {chats.length === 0 ? (
                  <p className="chat-list-empty">No previous chats yet.</p>
                ) : (
                  chats.map((c) => (
                    <div
                      key={c.id}
                      className={`chat-item ${c.id === activeChatId ? "chat-item-on" : ""}`}
                      onClick={() => openChat(c.id)}
                      title={c.title}
                    >
                      <div className="chat-item-main">
                        <span className="chat-item-title">{c.title}</span>
                        <span className="chat-item-stamp">{fmtChatStamp(c.updatedAt)}</span>
                      </div>
                      <button
                        type="button"
                        className="chat-item-del"
                        title="Delete chat"
                        onClick={(e) => { e.stopPropagation(); deleteChat(c.id); }}
                      >
                        ×
                      </button>
                    </div>
                  ))
                )}
              </div>
            </aside>

            <div className="chat-main">
            {/* Notices */}
            {error && <div className="notice notice-warn">{error}</div>}
            {warmup.state === "error" && warmup.workerKey === selectedWorkerKey && (
              <div className="notice notice-warn">Model preparation failed: {warmup.error}</div>
            )}
            {selectedQuantJobRunning && (
              <div className="notice notice-info">
                Converting {selProfile?.label || "model"} to {quantLabel(settings.quantMode)}{selectedQuantJobPct != null ? ` (${selectedQuantJobPct}%)` : ""}.
              </div>
            )}
            {selectedQuantNeedsPacking && (
              <div className="notice notice-info">
                Selected {quantLabel(settings.quantMode)} mode will quantize from fp16 at runtime. Open Model Hub to pack this mode for faster startup.
              </div>
            )}
            {!selectedQuantJobRunning && selectedQuantJob?.status === "done" && (
              <div className="notice notice-info">
                Latest {quantLabel(settings.quantMode)} conversion completed.
              </div>
            )}
            {!selectedQuantJobRunning && selectedQuantJob?.status === "error" && (
              <div className="notice notice-warn">
                Latest {quantLabel(settings.quantMode)} conversion failed. Check Model Hub - Quantization Jobs logs.
              </div>
            )}
            {!health.ready && (
              <div className="notice notice-info">
                Set <code>LLAMA_INFER_BIN</code>, <code>LLAMA_MODEL_DIRS</code>, and <code>LLAMA_TOKENIZER_PATH</code> in <code>web/.env</code>.
              </div>
            )}

            {/* Messages */}
            <div className="msgs-scroll" ref={scrollRef}>
              <div className="msgs-inner">

                {empty && (
                  <div className="empty">
                    <p className="empty-title">{selProfile?.label || "CPI"}</p>
                    <div className="starters">
                      {STARTERS.map((s) => (
                        <button key={s} type="button" className="starter" onClick={() => applyStarter(s)}>{s}</button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.filter((m) => !m.seed || !empty).map((m, idx) => {
                  const isLast = idx === messages.length - 1;
                  if (m.role === "user") {
                    return (
                      <div key={m.id} className="msg msg-user">
                        <div className="msg-user-bubble">{m.content}</div>
                      </div>
                    );
                  }
                  return (
                    <div key={m.id} className="msg msg-asst">
                      {!m.seed && <span className="msg-who">{selProfile?.label || "Assistant"}</span>}
                      <div className="msg-asst-text">
                        {/* Reasoning models: collapsible <think> block above the answer. */}
                        {m.reasoning && (
                          <ReasoningBlock text={m.reasoning} active={Boolean(m.reasoningStreaming) && !m.content} />
                        )}
                        {/* Render markdown live while streaming too, so code
                            blocks / lists / tables format as they arrive rather
                            than only on completion. Delta updates are throttled
                            via requestAnimationFrame, and ReactMarkdown tolerates
                            partially-streamed (e.g. unclosed ```) content. */}
                        <MsgContent text={m.content} />
                        {m.streaming && <span className="cursor" />}
                      </div>
                      {!m.streaming && isLast && runMeta?.elapsedMs > 0 && (
                        <div className="run-meta">{fmtMs(runMeta.elapsedMs)}</div>
                      )}
                    </div>
                  );
                })}

              </div>
            </div>

            {/* Composer */}
            <div className="composer-wrap">
              <div className="composer-inner">
                <div className="composer-box">
                  <textarea
                    ref={taRef}
                    className="composer-ta"
                    value={draft}
                    rows={1}
                    placeholder="Message..."
                    onChange={(e) => { setDraft(e.target.value); growTa(e.target.value); }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
                    }}
                  />
                  <div className="composer-foot">
                    <span className="composer-hint">Enter to send - Shift+Enter for newline</span>
                    <div className="composer-acts">
                      {streaming && (
                        <button type="button" className="btn btn-ghost" onClick={() => abortRef.current?.abort()}>Stop</button>
                      )}
                      <button
                        type="button"
                        className="btn btn-primary"
                        disabled={!draft.trim() || streaming || !health.ready}
                        onClick={() => submit()}
                      >
                        Send
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            </div>
          </div>
        )}

        {/*  Settings drawer  */}
        {showCfg && (
          <div className="overlay" onClick={() => setShowCfg(false)}>
            <div className="drawer" onClick={(e) => e.stopPropagation()}>
              <div className="drawer-head">
                <span className="drawer-title">Settings</span>
                <button type="button" className="drawer-close" onClick={() => setShowCfg(false)}>X</button>
              </div>

              <div className="drawer-body">

                {/* Generation */}
                <div className="drawer-group">
                  <p className="drawer-group-label">Generation</p>

                  <div className="field">
                    <label className="field-label">
                      Temperature
                      <span className="field-mono">{settings.temperature.toFixed(2)}</span>
                    </label>
                    <input type="range" min="0" max="1.5" step="0.05" value={settings.temperature}
                      onChange={(e) => setSettings((c) => ({ ...c, temperature: Number(e.target.value) }))} />
                  </div>

                  <div className="field">
                    <label className="field-label">Output budget</label>
                    <label className="field-check" style={{ marginBottom:"0.55rem" }}>
                      <input
                        type="checkbox"
                        checked={Boolean(settings.autoMaxTokens)}
                        onChange={(e) => setSettings((c) => ({ ...c, autoMaxTokens: e.target.checked }))}
                      />
                      <span>Auto-select max new tokens per model and prompt.</span>
                    </label>
                    <input
                      className="field-ctrl"
                      type="number"
                      min="32"
                      max="4096"
                      value={settings.maxNewTokens}
                      disabled={Boolean(settings.autoMaxTokens)}
                      onChange={(e) => setSettings((c) => ({ ...c, maxNewTokens: Number(e.target.value) }))}
                    />
                    <p className="field-help">
                      {settings.autoMaxTokens
                        ? "Manual cap is stored as a fallback. The server chooses the actual budget."
                        : "Manual upper bound for generated tokens."}
                    </p>
                  </div>

                  {settings.template === "qwen3_5" && (
                    <div className="field">
                      <label className="field-label">Reasoning</label>
                      <label className="field-check">
                        <input
                          type="checkbox"
                          checked={Boolean(settings.thinking)}
                          onChange={(e) => setSettings((c) => ({ ...c, thinking: e.target.checked }))}
                        />
                        <span>Let the model think before answering.</span>
                      </label>
                      <p className="field-help">
                        Streams a collapsible chain-of-thought above the answer. Slower, but
                        stronger on math and multi-step questions.
                      </p>
                    </div>
                  )}

                  <div className="field">
                    <label className="field-label">Long-form mode</label>
                    <label className="field-check">
                      <input
                        type="checkbox"
                        checked={Boolean(settings.longFormMode)}
                        onChange={(e) => setSettings((c) => ({ ...c, longFormMode: e.target.checked }))}
                      />
                      <span>Allow much larger detailed answers when the prompt calls for them.</span>
                    </label>
                  </div>

                  <div className="field">
                    <label className="field-label">
                      Context size
                      <span className="field-mono">{settings.maxContext}</span>
                    </label>
                    <input
                      className="field-ctrl"
                      type="number"
                      min="128"
                      max={maxContextLimit}
                      step="128"
                      value={settings.maxContext}
                      onChange={(e) =>
                        setSettings((c) => ({
                          ...c,
                          maxContext: Math.max(128, Math.min(maxContextLimit, Number(e.target.value) || 128))
                        }))
                      }
                    />
                  </div>

                  <div className="field">
                    <label className="field-label">Performance mode</label>
                    <label className="field-check">
                      <input
                        type="checkbox"
                        checked={Boolean(settings.performanceMode)}
                        onChange={(e) => {
                          const enabled = e.target.checked;
                          setSettings((c) => ({ ...c, performanceMode: enabled }));
                          safeStorageSet(STORAGE_KEYS.perfMode, enabled ? "1" : "0");
                        }}
                      />
                      <span>Disable CPU/RAM throttling for higher throughput.</span>
                    </label>
                  </div>

                  <div className="field">
                    <label className="field-label">System prompt</label>
                    <textarea className="field-ctrl" value={settings.systemPrompt}
                      onChange={(e) => setSettings((c) => ({ ...c, systemPrompt: e.target.value }))} />
                  </div>

                  <div className="field">
                    <label className="field-label">
                      Prompt format
                      <span className="field-mono" style={{ fontSize:"0.7rem", color:"var(--text-3)", marginLeft:"0.4rem" }}>auto-set by model</span>
                    </label>
                    <select className="field-ctrl" value={settings.template}
                      onChange={(e) => setSettings((c) => ({ ...c, template: e.target.value }))}>
                      {TEMPLATES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                    </select>
                  </div>
                </div>

                {/* Runtime */}
                <div className="drawer-group">
                  <p className="drawer-group-label">Runtime</p>

                  <div className="kv">
                    <span className="kv-key">Executable</span>
                    <span className="kv-val">{tailPath(health.config?.inferBin)}</span>
                  </div>
                  <div className="kv">
                    <span className="kv-key">Model</span>
                    <span className="kv-val">{tailPath(selProfile?.modelPath || health.config?.modelPath)}</span>
                  </div>
                  <div className="kv">
                    <span className="kv-key">Tokenizer</span>
                    <span className="kv-val">{tailPath(health.config?.tokenizerPath)}</span>
                  </div>
                  <div className="kv">
                    <span className="kv-key">Quant Mode</span>
                    <span className="kv-val">{quantLabel(settings.quantMode)}</span>
                  </div>
                  <div className="kv">
                    <span className="kv-key">Quant Source</span>
                    <span className="kv-val">
                      {settings.quantMode === "none"
                        ? "FP16 tensors"
                        : (selectedQuantConversionState === "ready"
                            ? `${quantLabel(settings.quantMode)} packed tensors`
                            : "Runtime quantization from fp16")}
                    </span>
                  </div>
                  {selDisplay.isStreaming && (
                    <div className="kv">
                      <span className="kv-key">Loading</span>
                      <span className="kv-val" title={STREAMING_HELP}>Streaming (weights from disk)</span>
                    </div>
                  )}
                  <p className="field-help" style={{ marginTop:"0.5rem" }}>
                    <strong>Quant</strong> = weight precision: FP16 (full quality) vs INT8/INT4 (smaller &amp; faster,
                    slight quality loss). It only applies to full-precision models — models whose name already lists a
                    format (e.g. <em>int4</em>) are pre-packed and fixed. <strong>Streaming</strong> models are too big for
                    VRAM, so their weights stream from disk (runs huge models, but slower).
                  </p>
                  {selectedQuantJob && (
                    <div className="kv">
                      <span className="kv-key">Conversion Job</span>
                      <span className="kv-val">
                        {selectedQuantJob.status}
                        {selectedQuantJob?.progress?.pct != null ? ` - ${selectedQuantJob.progress.pct}%` : ""}
                      </span>
                    </div>
                  )}
                  {selProfile?.quant && (
                    <div className="kv">
                      <span className="kv-key">Packed</span>
                      <span className="kv-val">
                        INT8 {selProfile.quant.packed?.int8 ? "yes" : "no"} - INT4 {selProfile.quant.packed?.int4 ? "yes" : "no"}{selProfile.quant.packed?.tq3 ? " - TQ3 yes" : ""}
                      </span>
                    </div>
                  )}
                  {selProfile?.moe && (
                    <div className="kv">
                      <span className="kv-key">MoE</span>
                      <span className="kv-val">
                        experts {selProfile.moe.numLocalExperts} - top-k {selProfile.moe.numExpertsPerTok || 2}
                        {selProfile.moe.expertIntermediateSize ? ` - inter ${selProfile.moe.expertIntermediateSize}` : ""}
                      </span>
                    </div>
                  )}

                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"0.5rem", marginTop:"0.2rem" }}>
                    {[
                      ["Context",  health.config?.maxContext ?? "-"],
                      ["Chosen ctx", settings.maxContext ?? "-"],
                      ["Models",   readyProfiles.length],
                    ].map(([k, v]) => (
                      <div key={k} style={{ background:"var(--surface-2)", border:"1px solid var(--border)", borderRadius:"0.4rem", padding:"0.5rem 0.65rem" }}>
                        <p style={{ fontSize:"0.6rem", textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-3)" }}>{k}</p>
                        <p style={{ fontSize:"0.95rem", fontWeight:600, color:"var(--text)", marginTop:"0.2rem" }}>{v}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Last run */}
                {runMeta && (
                  <div className="drawer-group">
                    <p className="drawer-group-label">Last run</p>
                    <div className="stat-line"><span>State</span><span className="stat-line-val">{runMeta.state}</span></div>
                    {runMeta.elapsedMs > 0 && (
                      <div className="stat-line"><span>Duration</span><span className="stat-line-val">{fmtMs(runMeta.elapsedMs)}</span></div>
                    )}
                    {runMeta.maxNewTokens > 0 && (
                      <div className="stat-line"><span>Output budget</span><span className="stat-line-val">{runMeta.maxNewTokens}</span></div>
                    )}
                    <div className="stat-line">
                      <span>Budget mode</span>
                      <span className="stat-line-val">{runMeta.autoMaxTokens === false ? "manual" : "auto"}</span>
                    </div>
                    <div className="stat-line">
                      <span>Long-form</span>
                      <span className="stat-line-val">{runMeta.longFormMode ? "on" : "off"}</span>
                    </div>
                    {runMeta.generatedTokens > 0 && (
                      <div className="stat-line"><span>Tokens</span><span className="stat-line-val">{runMeta.generatedTokens}</span></div>
                    )}
                    {runMeta.tokPerS > 0 && (
                      <div className="stat-line"><span>Tok/s</span><span className="stat-line-val">{Number(runMeta.tokPerS).toFixed(2)}</span></div>
                    )}
                    {runMeta.decodeTokPerS > 0 && (
                      <div className="stat-line"><span>Decode tok/s</span><span className="stat-line-val">{Number(runMeta.decodeTokPerS).toFixed(2)}</span></div>
                    )}
                    {runMeta.decodeMs > 0 && (
                      <div className="stat-line"><span>Decode time</span><span className="stat-line-val">{fmtMs(runMeta.decodeMs)}</span></div>
                    )}
                    {runMeta.modelLabel && (
                      <div className="stat-line"><span>Model</span><span className="stat-line-val" style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", maxWidth:"140px" }}>{runMeta.modelLabel}</span></div>
                    )}
                    {runMeta.metrics?.moe_quant_mode && runMeta.metrics.moe_quant_mode !== "none" && (
                      <div className="stat-line"><span>MoE quant</span><span className="stat-line-val">{String(runMeta.metrics.moe_quant_mode).toUpperCase()}</span></div>
                    )}
                    {Number(runMeta.metrics?.moe_router_ms) > 0 && (
                      <div className="stat-line"><span>MoE router</span><span className="stat-line-val">{Number(runMeta.metrics.moe_router_ms).toFixed(2)} ms</span></div>
                    )}
                    {Number(runMeta.metrics?.moe_expert_ms) > 0 && (
                      <div className="stat-line"><span>MoE experts</span><span className="stat-line-val">{Number(runMeta.metrics.moe_expert_ms).toFixed(2)} ms</span></div>
                    )}
                    {Number(runMeta.metrics?.moe_merge_ms) > 0 && (
                      <div className="stat-line"><span>MoE merge</span><span className="stat-line-val">{Number(runMeta.metrics.moe_merge_ms).toFixed(2)} ms</span></div>
                    )}
                    {Array.isArray(runMeta.metrics?.moe_selected) && runMeta.metrics.moe_selected.length > 0 && (
                      <div className="stat-line">
                        <span>MoE top-k</span>
                        <span className="stat-line-val">
                          L0 {runMeta.metrics.moe_selected[0]?.experts?.join?.("/") || "n/a"}
                        </span>
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
