// Parity gate for device-side top-k sampling.
//
// Drives llama_infer --interactive twice with the SAME seed and sampling knobs —
// once with the GPU candidate selection (default) and once forced onto the host
// path (LLAMA_INFER_PLAN_NO_DEVICE_TOPK=1) — and requires an IDENTICAL token
// sequence. Both paths feed the same shared candidate math and the same RNG, so
// a seeded run must agree exactly; any divergence means the GPU selected a
// different candidate set (a silent distribution bug).
//
//   node tools/device_topk_parity.mjs
import { spawn } from "node:child_process";

const BIN = "build/Release/llama_infer.exe";
const MODEL = "artifacts/hub/google__gemma-4-E2B-it/gemma4-e2b.cpi";
const TOK = "artifacts/hub/google__gemma-4-E2B-it/hf/tokenizer.json";

function run(label, env) {
  return new Promise((resolve, reject) => {
    const p = spawn(BIN, [MODEL, "--tokenizer", TOK, "--interactive",
                          "--temp", "0.8", "--top-k", "40", "--top-p", "0.95"],
      { env: { ...process.env, ...env, LLAMA_INFER_INSTANCE_MUTEX: "Local\\cpi_topk_parity_" + label } });
    let buf = "", text = "", done = false, err = "";
    p.stdout.on("data", (d) => {
      buf += d.toString();
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line.startsWith("{")) continue;
        let e; try { e = JSON.parse(line); } catch { continue; }
        if (e.type === "delta") text += e.delta ?? "";
        if (e.type === "done") { done = true; p.stdin.write(JSON.stringify({ shutdown: true }) + "\n"); }
        if (e.type === "error") { err = e.error || "engine error"; }
      }
    });
    p.stderr.on("data", () => {});
    p.on("exit", () => (err ? reject(new Error(err)) : resolve(done ? text : null)));
    // one seeded, sampled request
    p.stdin.write(JSON.stringify({
      id: "p1",
      prompt: "<|turn>user\nWrite one sentence about the ocean.<turn|>\n<|turn>model\n",
      max_new: 40, temp: 0.8, seed: 12345, add_bos: true
    }) + "\n");
  });
}

const dev = await run("dev", {});
const host = await run("host", { LLAMA_INFER_PLAN_NO_DEVICE_TOPK: "1" });
console.log("device top-k:", JSON.stringify(dev));
console.log("host   top-k:", JSON.stringify(host));
const ok = Boolean(dev) && dev === host;
console.log(ok
  ? "\nPARITY OK — seeded sampled generation is identical on the device and host paths"
  : "\nPARITY FAILED — the GPU selected a different candidate set");
process.exit(ok ? 0 : 1);
