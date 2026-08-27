// Drives the cpi_embed worker (CUDA BERT-family embedding model) over its
// line-delimited JSON protocol. One persistent process, requests serialized
// through a queue. Independent of the chat cpi worker (separate process,
// shares the GPU; bge-small is tiny).

import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));      // web/server
const REPO_ROOT = path.resolve(HERE, "..", "..");              // repo root

let proc = null;
let buffer = "";
let current = null;
const queue = [];
let seq = 0;
let readyDim = null;
let lastError = null;

// cpi_embed is a CUDA-only target, so it lands in whichever CUDA build dir the
// user configured (build-run, build-cuda, …), never the plain CPU build/. We
// can't assume a single location, so try, in order: an explicit override, the
// dir holding the resolved cpi (they're built together), then every
// known build dir; returning the first that exists. Returns a path that may
// not exist (the best guess) only when nothing is found, so the caller's
// existence check produces a clear "set EMBED_BIN" error.
function embedBinCandidates(config) {
  const exe = process.platform === "win32" ? "cpi_embed.exe" : "cpi_embed";
  const sub = process.platform === "win32" ? "Release" : "";
  const out = [];
  if (process.env.EMBED_BIN) out.push(process.env.EMBED_BIN);
  if (config?.inferBin) out.push(path.join(path.dirname(config.inferBin), exe));
  for (const d of ["build-run", "build-cuda", "build"]) {
    out.push(path.join(REPO_ROOT, d, sub, exe));
  }
  return out;
}

function resolveEmbedBin(config) {
  const candidates = embedBinCandidates(config);
  return candidates.find((p) => fs.existsSync(p)) || candidates[0];
}

function resolveEmbedModel() {
  if (process.env.EMBED_MODEL_PATH) return process.env.EMBED_MODEL_PATH;
  return path.join(REPO_ROOT, "artifacts", "hub", "BAAI__bge-small-en-v1.5");
}

// Cheap readiness probe (no process spawn): does the binary + model exist?
// Lets the server log embeddings enabled/disabled at startup and lets clients
// preflight instead of discovering a 500 mid-index.
export function embedStatus(config) {
  const binary = resolveEmbedBin(config);
  const model = resolveEmbedModel();
  const available = fs.existsSync(binary) && fs.existsSync(model);
  return { available, binary, model, dim: readyDim, running: Boolean(proc) };
}

function ensureWorker(config) {
  if (proc) return;
  const bin = resolveEmbedBin(config);
  const model = resolveEmbedModel();
  if (!fs.existsSync(bin)) throw new Error(`embed binary not found: ${bin} (set EMBED_BIN)`);
  if (!fs.existsSync(model)) throw new Error(`embed model not found: ${model} (set EMBED_MODEL_PATH)`);
  proc = spawn(bin, [model], { stdio: ["pipe", "pipe", "pipe"] });
  buffer = "";
  proc.stdout.on("data", onData);
  proc.stderr.on("data", (d) => {
    const s = String(d);
    const m = s.match(/dim=(\d+)/);
    if (m) readyDim = Number(m[1]);
  });
  proc.on("exit", (code) => {
    lastError = `embed worker exited (code ${code})`;
    proc = null;
    if (current) { current.reject(new Error(lastError)); current = null; }
    while (queue.length) queue.shift().reject(new Error(lastError));
  });
  proc.on("error", (err) => {
    lastError = `embed worker spawn error: ${err.message}`;
    if (current) { current.reject(err); current = null; }
    while (queue.length) queue.shift().reject(err);
    proc = null;
  });
}

function onData(chunk) {
  buffer += String(chunk);
  let nl;
  while ((nl = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line.startsWith("{")) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    if (current) {
      const c = current;
      current = null;
      if (msg.error) c.reject(new Error(msg.error));
      else c.resolve(msg);
      pump();
    }
  }
}

function pump() {
  if (current || queue.length === 0 || !proc) return;
  current = queue.shift();
  proc.stdin.write(JSON.stringify(current.payload) + "\n");
}

// Embeds an array of strings. input_type is "query" | "document".
// Resolves to { embeddings: number[][], tokens: number[], dim: number }.
export function embed(config, inputs, inputType) {
  return new Promise((resolve, reject) => {
    try { ensureWorker(config); } catch (e) { reject(e); return; }
    queue.push({
      payload: { id: String(++seq), input: inputs, input_type: inputType },
      resolve,
      reject,
    });
    pump();
  });
}

export function embedReadyDim() {
  return readyDim;
}
