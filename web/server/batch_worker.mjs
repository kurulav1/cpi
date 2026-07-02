// Continuous-batching client for the llama_infer --interactive-batch worker.
//
// The default chat path (index.mjs) is single-flight: one request occupies the
// worker at a time. This module instead multiplexes many concurrent requests
// onto ONE worker process that batches them internally (the engine's streaming
// scheduler), demuxing the worker's NDJSON events back to each request by id.
//
// It is intentionally self-contained (no coupling to the single-flight worker
// lifecycle) so it can be adopted opt-in and smoke-tested in isolation:
//   node web/server/batch_worker.mjs <model.ll2c> <tokenizer.json>
//
// Event protocol (worker -> here), one JSON object per line, each tagged {id}:
//   {type:"start",id}                      request admitted
//   {type:"delta",id,delta}                incremental decoded text
//   {type:"done",id,finish_reason,text}    request finished
//   {type:"error",id,error}                request failed
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { pathToFileURL } from "node:url";

// Turn the single-flight interactive launch args into batch args: swap
// --interactive for --interactive-batch (and require paging + resident cache,
// which the batched scheduler needs). Callers may pass fully-formed args instead.
export function toBatchArgs(interactiveArgs) {
  const args = interactiveArgs.filter((a) => a !== "--interactive");
  if (!args.includes("--interactive-batch")) args.push("--interactive-batch");
  if (!args.includes("--paged-blocks")) args.push("--paged-blocks");
  if (!args.includes("--gpu-cache-all")) args.push("--gpu-cache-all");
  return args;
}

// Create a batch worker around a spawned llama_infer --interactive-batch process.
// Returns { submit, activeCount, close }.
export function createBatchWorker({ bin, args, env, cwd, onReadyError }) {
  const child = spawn(bin, args, { stdio: ["pipe", "pipe", "pipe"], env: env ?? process.env, cwd });
  const pending = new Map(); // id -> { onStart, onDelta, onDone, onError, text }
  const decoder = new StringDecoder("utf8");
  let stdoutBuffer = "";

  const routeLine = (line) => {
    let ev = null;
    try {
      ev = JSON.parse(line);
    } catch {
      return;
    }
    if (!ev || typeof ev !== "object") return;
    const id = ev.id;
    if (!id) return;
    const req = pending.get(id);
    if (!req) return;
    switch (ev.type) {
      case "start":
        req.onStart?.();
        break;
      case "delta":
        if (typeof ev.delta === "string" && ev.delta.length) {
          req.text += ev.delta;
          req.onDelta?.(ev.delta);
        }
        break;
      case "done":
        pending.delete(id);
        req.onDone?.({
          text: typeof ev.text === "string" ? ev.text : req.text,
          finishReason: ev.finish_reason || "stop"
        });
        break;
      case "error":
        pending.delete(id);
        req.onError?.(new Error(ev.error || "batch worker error"));
        break;
      default:
        break;
    }
  };

  child.stdout.on("data", (chunk) => {
    stdoutBuffer += decoder.write(chunk);
    let nl = stdoutBuffer.indexOf("\n");
    while (nl !== -1) {
      const line = stdoutBuffer.slice(0, nl).trim();
      stdoutBuffer = stdoutBuffer.slice(nl + 1);
      if (line) routeLine(line);
      nl = stdoutBuffer.indexOf("\n");
    }
  });

  const failAll = (err) => {
    for (const [, req] of pending) req.onError?.(err);
    pending.clear();
  };
  // Resolves once the child process has fully exited (its single-instance mutex
  // and GPU memory are released) — used to serialize teardown before respawning.
  let markExited;
  const exited = new Promise((resolve) => { markExited = resolve; });
  child.on("exit", (code) => { failAll(new Error(`batch worker exited (code ${code})`)); markExited(); });
  child.on("error", (err) => {
    onReadyError?.(err);
    failAll(err);
  });

  // Submit one request. `id` must be unique among in-flight requests.
  const submit = ({ id, prompt, maxNew, temp, minNew, stopTexts, addBos, onStart, onDelta, onDone, onError }) => {
    if (pending.has(id)) throw new Error(`duplicate in-flight request id: ${id}`);
    pending.set(id, { onStart, onDelta, onDone, onError, text: "" });
    const payload = {
      id,
      prompt,
      max_new: maxNew ?? 256,
      ...(minNew != null ? { min_new: minNew } : {}),
      ...(temp != null ? { temp } : {}),
      ...(stopTexts ? { stop_texts: stopTexts } : {}),
      ...(addBos != null ? { add_bos: addBos } : {})
    };
    child.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
      if (err && pending.has(id)) {
        pending.delete(id);
        onError?.(new Error(`failed to write request: ${err.message}`));
      }
    });
  };

  const close = () => {
    try {
      child.stdin.write(`${JSON.stringify({ shutdown: true })}\n`);
      child.stdin.end();
    } catch {
      // ignore
    }
  };

  // Force-terminate immediately (used when respawning for a different model, so
  // the old process releases its mutex + VRAM before the new one loads).
  const kill = () => {
    try { child.kill("SIGKILL"); } catch { /* ignore */ }
  };

  return {
    submit,
    activeCount: () => pending.size,
    close,
    kill,
    exited,
    child
  };
}

// --- Standalone smoke test -------------------------------------------------
// node web/server/batch_worker.mjs <model.ll2c> <tokenizer.json>
// Submits several concurrent requests and prints each demuxed result.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [model, tokenizer] = process.argv.slice(2);
  if (!model || !tokenizer) {
    console.error("usage: node batch_worker.mjs <model.ll2c> <tokenizer.json>");
    process.exit(2);
  }
  const bin = process.platform === "win32"
    ? "build/Release/llama_infer.exe"
    : "build/llama_infer";
  const args = toBatchArgs([
    model, "--tokenizer", tokenizer, "--max-context", "512", "--interactive", "--web"
  ]);
  const env = { ...process.env, LLAMA_INFER_INSTANCE_MUTEX: "Local\\llama_infer_batch_selftest" };
  const worker = createBatchWorker({ bin, args, env });

  const prompts = [
    { id: "req-1", prompt: "Count to five:", maxNew: 20 },
    { id: "req-2", prompt: "Name three colors:", maxNew: 20 },
    { id: "req-3", prompt: "Say hello:", maxNew: 20 }
  ];
  let remaining = prompts.length;
  const results = new Map();
  for (const p of prompts) {
    worker.submit({
      ...p,
      onStart: () => console.log(`[start] ${p.id}`),
      onDone: ({ text, finishReason }) => {
        results.set(p.id, text);
        console.log(`[done ] ${p.id} (${finishReason}): ${JSON.stringify(text)}`);
        if (--remaining === 0) {
          const ok = [...results.values()].every((t) => t && t.length > 0);
          console.log(`\nbatch_worker self-test: ${ok ? "PASS" : "FAIL"} (${results.size} concurrent requests demuxed)`);
          worker.close();
          setTimeout(() => process.exit(ok ? 0 : 1), 200);
        }
      },
      onError: (err) => {
        console.error(`[error] ${p.id}: ${err.message}`);
        if (--remaining === 0) { worker.close(); process.exit(1); }
      }
    });
  }
}
