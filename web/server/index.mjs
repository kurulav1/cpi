import dotenv from "dotenv";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { randomUUID } from "node:crypto";

import { getRuntimeConfig, publicRuntimeSummary } from "./config.mjs";
import { buildPromptPackage } from "./prompting.mjs";

dotenv.config();

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "4mb" }));

// Only one generation runs at a time (single GPU). The worker process is kept
// warm across requests and restarted only when the selected profile changes.
let activeRequest = null;
let interactiveWorker = null;
const quantOverrides = new Map(); // profileId -> "none" | "int8" | "int4"

// â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function writeNdjson(response, payload) {
  response.write(`${JSON.stringify(payload)}\n`);
}

function writeSse(response, data) {
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

function sleepMs(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return;
  const sab = new SharedArrayBuffer(4);
  const arr = new Int32Array(sab);
  Atomics.wait(arr, 0, 0, ms);
}

function isTruthyFlag(value) {
  return value === true || value === 1 || value === "1" || value === "true";
}

function requestBody(req) {
  return req.body && typeof req.body === "object" ? req.body : {};
}

function findProfileByIdOrLabel(config, profileId) {
  if (!profileId) return config.selectedProfile;
  return (
    config.availableProfiles.find(
      (profile) => profile.id === profileId || profile.label === profileId
    ) ?? null
  );
}

function stripWeightQuantArgs(args) {
  const cleaned = [];
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--weight-quant") {
      i += 1;
      continue;
    }
    if (arg === "--int8-streaming" || arg === "--int4-streaming") {
      continue;
    }
    cleaned.push(arg);
  }
  return cleaned;
}

function normalizeQuantMode(value) {
  const raw = String(value || "").toLowerCase();
  if (raw === "int8" || raw === "q8") return "int8";
  if (raw === "int4" || raw === "q4") return "int4";
  return "none";
}

function resolveQuantModeForProfile(profile, requestedMode = "") {
  const allowedModes = profile?.quant?.selectableModes ?? ["none"];
  const requested = requestedMode ? normalizeQuantMode(requestedMode) : "";
  const override = quantOverrides.has(profile.id) ? quantOverrides.get(profile.id) : "";
  const preferred =
    requested ||
    override ||
    profile?.quant?.effectiveMode ||
    profile?.quant?.configuredMode ||
    "none";
  return allowedModes.includes(preferred) ? preferred : allowedModes[0] || "none";
}

function buildProfileExtraArgs(profile, quantMode) {
  const mode = normalizeQuantMode(quantMode);
  const args = stripWeightQuantArgs(profile.extraArgs || []);
  if (mode === "int8" || mode === "int4") {
    args.push("--weight-quant", mode);
  }
  return args;
}

function setStreamingHeaders(res, contentType) {
  res.status(200);
  res.setHeader("Content-Type", contentType);
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.socket?.setNoDelay?.(true);
  res.flushHeaders?.();
}

function escapePowerShellSingleQuoted(value) {
  return String(value || "").replace(/'/g, "''");
}

async function pickFolderNative(initialDir = "") {
  if (process.platform !== "win32") {
    throw new Error("Native folder picker is currently supported on Windows only.");
  }

  const safeInitialDir = escapePowerShellSingleQuoted(initialDir);
  const script = [
    "Add-Type -AssemblyName System.Windows.Forms | Out-Null",
    "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog",
    "$dlg.Description = 'Choose output folder'",
    "$dlg.ShowNewFolderButton = $true",
    `$initial = '${safeInitialDir}'`,
    "if ($initial -and (Test-Path -LiteralPath $initial)) { $dlg.SelectedPath = $initial }",
    "$res = $dlg.ShowDialog()",
    "if ($res -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dlg.SelectedPath }"
  ].join("; ");

  return new Promise((resolve, reject) => {
    const child = spawn("powershell.exe", ["-NoProfile", "-STA", "-Command", script], {
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => {
      reject(new Error(`Failed to open folder dialog: ${err.message}`));
    });
    child.on("close", (code) => {
      const selectedPath = stdout.trim();
      if (code === 0) {
        resolve(selectedPath);
        return;
      }
      const detail = stderr.trim() || `exit code ${code}`;
      reject(new Error(`Folder dialog failed: ${detail}`));
    });
  });
}

function sampleHostUsage(previousCpu) {
  const cpus = os.cpus();
  const totals = cpus.reduce(
    (acc, cpu) => {
      const t = cpu.times;
      const total = t.user + t.nice + t.sys + t.idle + t.irq;
      acc.total += total;
      acc.idle += t.idle;
      return acc;
    },
    { total: 0, idle: 0 }
  );

  let cpuPercent = -1;
  if (
    previousCpu &&
    totals.total > previousCpu.total &&
    totals.idle >= previousCpu.idle
  ) {
    const totalDiff = totals.total - previousCpu.total;
    const idleDiff = totals.idle - previousCpu.idle;
    cpuPercent = Math.max(0, Math.min(100, 100 * (1 - idleDiff / totalDiff)));
  }

  const totalMem = os.totalmem();
  const usedMem = totalMem - os.freemem();
  const memoryPercent = totalMem > 0 ? (100 * usedMem) / totalMem : -1;

  return {
    cpuPercent,
    memoryPercent,
    cpuSnapshot: totals
  };
}

// Build the llama_infer argument list from a request body and server config.
// Accepts both internal fields (maxNewTokens, profileId) and OpenAI fields
// (max_tokens, model) so this function works for all endpoints.
function buildCliArgs(config, body) {
  const profileId = body.profileId ?? body.model;
  const selectedProfile = findProfileByIdOrLabel(config, profileId);

  if (!selectedProfile) {
    throw new Error(
      "No model available. Set modelPath/modelDirs in config.json or LLAMA_MODEL_PATH."
    );
  }
  if (!selectedProfile.ready) {
    throw new Error(`Model is not runnable: ${selectedProfile.label}.`);
  }

  const performanceMode = isTruthyFlag(body.performanceMode);
  const quantMode = resolveQuantModeForProfile(selectedProfile, body.quantMode);
  const profileExtraArgs = buildProfileExtraArgs(selectedProfile, quantMode);

  const promptPackage = buildPromptPackage(body.messages, {
    template: body.template || selectedProfile.template || config.template,
    systemPrompt: body.systemPrompt || config.systemPrompt,
    // Reduce prefill in perf mode so token-1 starts sooner in longer chats.
    maxTurns: performanceMode ? 4 : 8,
    maxChars: performanceMode ? 3500 : 9000
  });

  if (!promptPackage.prompt) {
    throw new Error("Provide at least one user message.");
  }

  const maxNewTokens = Math.round(
    clampNumber(body.maxNewTokens ?? body.max_tokens, 32, 4096, config.maxNewTokens)
  );
  const temperature = clampNumber(body.temperature, 0, 2, config.temperature);
  const noResourceLimits =
    performanceMode || profileExtraArgs.includes("--no-resource-limits");

  return {
    profile: selectedProfile,
    profileExtraArgs,
    quantMode,
    prompt: promptPackage.prompt,
    addBos: promptPackage.addBos,
    stopTexts: promptPackage.stopTexts,
    noResourceLimits,
    performanceMode,
    workerKey: `${selectedProfile.id}|perf:${performanceMode ? 1 : 0}|q:${quantMode}`,
    meta: {
      profileId: selectedProfile.id,
      modelLabel: selectedProfile.label,
      template: promptPackage.template,
      messageCount: promptPackage.messages.length,
      maxNewTokens,
      temperature,
      performanceMode,
      quantMode
    }
  };
}

function buildWorkerCliConfig(profile, options = {}) {
  const performanceMode = isTruthyFlag(options.performanceMode);
  const quantMode = resolveQuantModeForProfile(profile, options.quantMode);
  const profileExtraArgs = buildProfileExtraArgs(profile, quantMode);
  const noResourceLimits =
    performanceMode || profileExtraArgs.includes("--no-resource-limits");
  return {
    profile,
    profileExtraArgs,
    quantMode,
    noResourceLimits,
    performanceMode,
    workerKey: `${profile.id}|perf:${performanceMode ? 1 : 0}|q:${quantMode}`,
    meta: {
      profileId: profile.id,
      modelLabel: profile.label,
      template: profile.template,
      messageCount: 0,
      maxNewTokens: 0,
      temperature: 0,
      performanceMode,
      quantMode
    }
  };
}

// Core generation engine
//
// Keeps a warm interactive llama_infer worker per selected model profile.
// Each request is sent as one NDJSON line over stdin and streamed back as
// NDJSON events on stdout.

function buildInteractiveLaunchArgs(config, cliConfig) {
  const selectedProfile = cliConfig.profile;
  const profileExtraArgs = cliConfig.profileExtraArgs || selectedProfile.extraArgs;
  const noResourceLimits =
    cliConfig.noResourceLimits ||
    profileExtraArgs.includes("--no-resource-limits");
  return [
    selectedProfile.modelPath,
    "--tokenizer", selectedProfile.tokenizerPath,
    "--max-context", String(config.maxContext),
    "--top-k", String(config.topK),
    "--top-p", String(config.topP),
    "--repeat-penalty", String(config.repeatPenalty),
    "--interactive",
    "--web",
    "--runtime-metrics",
    ...(noResourceLimits
      ? ["--no-resource-limits"]
      : [
          "--max-cpu-percent", String(config.maxCpuPercent),
          "--max-memory-percent", String(config.maxMemoryPercent),
          "--resource-sample-ms", String(config.resourceSampleMs),
          "--resource-sustain-ms", String(config.resourceSustainMs),
          "--resource-throttle-ms", String(config.resourceThrottleMs)
        ]),
    ...profileExtraArgs
  ];
}

function killWorker(worker, force = false) {
  if (!worker?.child) return;
  try {
    if (force) {
      worker.child.kill("SIGKILL");
      return;
    }
    worker.child.kill("SIGTERM");
    setTimeout(() => worker.child.kill("SIGKILL"), 1000).unref();
  } catch {
    // ignore
  }
}

function parseWorkerStdout(worker) {
  let nl = worker.stdoutBuffer.indexOf("\n");
  while (nl !== -1) {
    const line = worker.stdoutBuffer.slice(0, nl).trim();
    worker.stdoutBuffer = worker.stdoutBuffer.slice(nl + 1);

    if (line) {
      let event = null;
      try {
        event = JSON.parse(line);
      } catch {
        event = null;
      }

      const pending = worker.pending;
      if (event && pending) {
        if (event.id && event.id !== pending.id) {
          // Ignore stale output from a previous request.
        } else if (event.type === "delta") {
          if (event.delta) pending.onDelta?.(event.delta);
        } else if (event.type === "metrics") {
          const metrics =
            event.metrics && typeof event.metrics === "object" ? event.metrics : null;
          if (metrics) {
            worker.lastMetrics = metrics;
            pending.onMetrics?.(metrics);
          }
        } else if (event.type === "done") {
          const text = typeof event.text === "string" ? event.text : "";
          const elapsedMs = Number.isFinite(Number(event.elapsed_ms))
            ? Number(event.elapsed_ms)
            : Date.now() - pending.startedAt;
          const generatedTokens = Number.isFinite(Number(event.generated_tokens))
            ? Number(event.generated_tokens)
            : null;
          const tokPerSFromEvent = Number(event.tok_per_s);
          const tokPerS = Number.isFinite(tokPerSFromEvent)
            ? tokPerSFromEvent
            : (generatedTokens != null && elapsedMs > 0
                ? (1000.0 * generatedTokens) / elapsedMs
                : null);
          const metrics =
            event.metrics && typeof event.metrics === "object"
              ? event.metrics
              : worker.lastMetrics || null;
          if (metrics) {
            worker.lastMetrics = metrics;
          }
          worker.ready = true;
          worker.pending = null;
          pending.resolve({ text, elapsedMs, generatedTokens, tokPerS, metrics, ...pending.meta });
        } else if (event.type === "error") {
          worker.pending = null;
          pending.reject(event.error || "interactive worker error");
        }
      }
    }

    nl = worker.stdoutBuffer.indexOf("\n");
  }
}

function spawnInteractiveWorker(config, cliConfig) {
  const args = buildInteractiveLaunchArgs(config, cliConfig);
  const child = spawn(config.inferBin, args, {
    cwd: config.repoRoot,
    env: process.env
  });

  const worker = {
    workerKey: cliConfig.workerKey ?? `${cliConfig.meta.profileId}|perf:0|q:${cliConfig.meta.quantMode || "none"}`,
    child,
    ready: false,
    lastMetrics: null,
    stdoutDecoder: new StringDecoder("utf8"),
    stderrDecoder: new StringDecoder("utf8"),
    stdoutBuffer: "",
    stderrText: "",
    pending: null
  };

  child.stdout.on("data", (chunk) => {
    worker.stdoutBuffer += worker.stdoutDecoder.write(chunk);
    parseWorkerStdout(worker);
  });

  child.stderr.on("data", (chunk) => {
    worker.stderrText += worker.stderrDecoder.write(chunk);
  });

  child.stdin.on("error", (err) => {
    if (!worker.pending) return;
    const pending = worker.pending;
    worker.pending = null;
    pending.reject(`Failed to send request to worker: ${err.message}`);
  });

  child.on("close", (code, signal) => {
    worker.stdoutBuffer += worker.stdoutDecoder.end();
    parseWorkerStdout(worker);
    worker.stderrText += worker.stderrDecoder.end();

    if (worker.pending) {
      const pending = worker.pending;
      worker.pending = null;
      pending.reject(
        worker.stderrText.trim() ||
          `interactive worker exited with code ${code}${signal ? ` (${signal})` : ""}.`
      );
    }

    if (interactiveWorker === worker) {
      interactiveWorker = null;
    }
  });

  child.on("error", (err) => {
    if (worker.pending) {
      const pending = worker.pending;
      worker.pending = null;
      pending.reject(`Failed to launch llama_infer: ${err.message}`);
    }

    if (interactiveWorker === worker) {
      interactiveWorker = null;
    }
  });

  return worker;
}

function ensureInteractiveWorker(config, cliConfig) {
  const targetWorkerKey =
    cliConfig.workerKey ?? `${cliConfig.meta.profileId}|perf:0|q:${cliConfig.meta.quantMode || "none"}`;
  if (
    interactiveWorker &&
    interactiveWorker.workerKey === targetWorkerKey &&
    !interactiveWorker.child.killed &&
    interactiveWorker.child.exitCode == null
  ) {
    return interactiveWorker;
  }

  if (interactiveWorker) {
    killWorker(interactiveWorker, true);
    interactiveWorker = null;
    // llama_infer enforces single instance; give OS a moment to release lock.
    sleepMs(120);
  }

  interactiveWorker = spawnInteractiveWorker(config, cliConfig);
  return interactiveWorker;
}

function resolveWarmupProfile(config, profileId) {
  const profile = findProfileByIdOrLabel(config, profileId);
  if (!profile) {
    throw new Error("No model available for warmup.");
  }
  if (!profile.ready) {
    throw new Error(`Model is not runnable: ${profile.label}.`);
  }
  return profile;
}

async function warmupProfileWorker(config, profileId, options = {}) {
  const profile = resolveWarmupProfile(config, profileId);

  const workerCliConfig = buildWorkerCliConfig(profile, options);
  const worker = ensureInteractiveWorker(config, workerCliConfig);
  if (worker.ready) {
    return profile;
  }

  // Warmup should mean "first token ready soon", not just "process spawned".
  await new Promise((resolve, reject) => {
    const warmCliConfig = {
      ...workerCliConfig,
      prompt: "warmup",
      addBos: true,
      stopTexts: [],
      meta: {
        ...workerCliConfig.meta,
        maxNewTokens: 1,
        temperature: 0
      }
    };
    runGeneration(config, warmCliConfig, {
      requestKind: "warmup",
      onDelta: () => {},
      onDone: () => resolve(),
      onError: (msg) => reject(new Error(msg))
    });
  });

  return profile;
}

function runGeneration(
  config,
  cliConfig,
  { onStart, onDelta, onMetrics, onDone, onError, requestKind = "generation" }
) {
  const requestId = randomUUID();
  let settled = false;
  let resourceTimer = null;
  let previousCpu = null;
  let overLimitSince = 0;
  let worker = null;

  const stopResourceMonitor = () => {
    if (resourceTimer) {
      clearInterval(resourceTimer);
      resourceTimer = null;
    }
  };

  const settle = (fn) => {
    if (settled) return;
    settled = true;
    stopResourceMonitor();
    activeRequest = null;
    fn();
  };

  try {
    worker = ensureInteractiveWorker(config, cliConfig);
  } catch (err) {
    return {
      cancel() {
        settle(() => onError(err.message || String(err)));
      }
    };
  }

  onStart?.(cliConfig.meta);
  activeRequest = {
    kind: requestKind,
    cancel: () => {
      killWorker(worker);
      settle(() => onError("generation cancelled"));
    }
  };

  worker.pending = {
    id: requestId,
    startedAt: Date.now(),
    meta: cliConfig.meta,
    onDelta,
    onMetrics,
    resolve: (result) => settle(() => onDone(result)),
    reject: (msg) => settle(() => onError(msg))
  };

  const payload = {
    id: requestId,
    prompt: cliConfig.prompt,
    max_new: cliConfig.meta.maxNewTokens,
    temp: cliConfig.meta.temperature,
    stop_texts: cliConfig.stopTexts,
    add_bos: cliConfig.addBos
  };

  const sampleMs = Math.max(0, config.resourceSampleMs || 250);
  const sustainMs = Math.max(1, config.resourceSustainMs || 5000);
  if (sampleMs > 0 && !cliConfig.noResourceLimits) {
    resourceTimer = setInterval(() => {
      if (settled) return;
      const usage = sampleHostUsage(previousCpu);
      previousCpu = usage.cpuSnapshot;
      const cpuOver = usage.cpuPercent >= 0 && usage.cpuPercent > config.maxCpuPercent;
      const memOver = usage.memoryPercent >= 0 && usage.memoryPercent > config.maxMemoryPercent;

      if (!cpuOver && !memOver) {
        overLimitSince = 0;
        return;
      }
      if (!overLimitSince) {
        overLimitSince = Date.now();
        return;
      }
      if (Date.now() - overLimitSince < sustainMs) {
        return;
      }

      const message =
        `resource_limit_exceeded: cpu=${usage.cpuPercent.toFixed(1)}% ` +
        `(limit=${config.maxCpuPercent}%) memory=${usage.memoryPercent.toFixed(1)}% ` +
        `(limit=${config.maxMemoryPercent}%) sustain_ms=${Date.now() - overLimitSince}`;
      killWorker(worker);
      settle(() => onError(message));
    }, sampleMs);
    resourceTimer.unref?.();
  }

  try {
    if (
      !worker.child.stdin ||
      worker.child.stdin.destroyed ||
      worker.child.stdin.writableEnded
    ) {
      if (worker.pending?.id === requestId) {
        worker.pending = null;
      }
      settle(() => onError("Failed to send request to worker: stdin is closed."));
    } else {
      worker.child.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
        if (!err) return;
        if (worker.pending?.id === requestId) {
          worker.pending = null;
        }
        settle(() => onError(`Failed to send request to worker: ${err.message}`));
      });
    }
  } catch (err) {
    if (worker.pending?.id === requestId) {
      worker.pending = null;
    }
    settle(() => onError(`Failed to send request to worker: ${err.message}`));
  }

  return {
    cancel() {
      if (!activeRequest?.cancel) return;
      activeRequest.cancel();
    }
  };
}

// â”€â”€ request guards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function guardReady(config, res) {
  if (!config.ready) {
    res.status(503).json({
      error:
        "Runtime not ready. Check config.json (or .env): inferBin, modelPath, tokenizerPath."
    });
    return false;
  }
  return true;
}

async function guardIdle(
  res,
  { waitForWarmup = false, warmupWaitMs = 180000, pollMs = 25 } = {}
) {
  if (waitForWarmup && activeRequest?.kind === "warmup") {
    const deadline = Date.now() + Math.max(0, Number(warmupWaitMs) || 0);
    const intervalMs = Math.max(10, Number(pollMs) || 25);
    while (activeRequest?.kind === "warmup" && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }

  if (activeRequest) {
    const reason =
      activeRequest.kind === "warmup"
        ? "Engine is still warming up. Please retry in a moment."
        : "Engine is busy with another generation.";
    res.status(409).json({ error: reason });
    return false;
  }
  return true;
}

function onDisconnect(req, res, cancel) {
  const abort = () => cancel();
  req.on("aborted", abort);
  res.on("close", () => {
    if (!res.writableEnded) abort();
  });
}

function parseWorkerKey(workerKey) {
  if (!workerKey || typeof workerKey !== "string") {
    return { profileId: "", performanceMode: false, quantMode: "none" };
  }
  const parts = workerKey.split("|");
  const profileId = parts[0] || "";
  let performanceMode = false;
  let quantMode = "none";
  for (let i = 1; i < parts.length; i += 1) {
    const part = parts[i];
    if (part.startsWith("perf:")) {
      performanceMode = part.slice(5) === "1";
      continue;
    }
    if (part.startsWith("q:")) {
      quantMode = normalizeQuantMode(part.slice(2));
    }
  }
  return { profileId, performanceMode, quantMode };
}

function findLatestQuantJobForProfile(profileId, quantMode = "") {
  if (!profileId) return null;
  let latest = null;
  for (const [jobId, job] of quantJobs.entries()) {
    if (job.profileId !== profileId) continue;
    if (quantMode && job.quantMode !== quantMode) continue;
    const candidate = { jobId, ...job };
    if (!latest || candidate.createdAt > latest.createdAt) {
      latest = candidate;
    }
  }
  return latest;
}

function summarizeQuantJob(job) {
  if (!job) return null;
  return {
    jobId: job.jobId,
    status: job.status,
    profileId: job.profileId || "",
    quantMode: job.quantMode || "",
    modelLabel: job.modelLabel || "",
    path: job.path || "",
    createdAt: job.createdAt || 0,
    updatedAt: job.updatedAt || 0,
    progress: job.progress || null
  };
}

// â”€â”€ routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// GET /api/health
// Returns server status, busy flag, and full runtime config summary.
app.get("/api/health", (_req, res) => {
  const config = getRuntimeConfig();
  const summary = publicRuntimeSummary(config);
  const selectedProfile = summary.selectedProfile;
  const selectedQuantMode = selectedProfile
    ? resolveQuantModeForProfile(selectedProfile)
    : "none";
  const activeQuantJob = summarizeQuantJob(
    findLatestQuantJobForProfile(selectedProfile?.id || "", selectedQuantMode)
  );
  const activeWorker = interactiveWorker
    ? {
        ready: Boolean(interactiveWorker.ready),
        alive: Boolean(
          interactiveWorker.child &&
          !interactiveWorker.child.killed &&
          interactiveWorker.child.exitCode == null
        ),
        metrics: interactiveWorker.lastMetrics || null,
        ...parseWorkerKey(interactiveWorker.workerKey)
      }
    : null;
  res.json({
    ok: true,
    ready: config.ready,
    busy: Boolean(activeRequest),
    activeKind: activeRequest?.kind ?? null,
    config: summary,
    quantOverrides: Object.fromEntries(quantOverrides),
    activeWorker,
    activeQuantJob
  });
});

// GET /api/models
// Returns the list of discovered model profiles.
app.get("/api/models", (_req, res) => {
  const config = getRuntimeConfig();
  res.json({
    models: publicRuntimeSummary(config).availableProfiles,
    quantOverrides: Object.fromEntries(quantOverrides)
  });
});

// POST /api/quant/select
// Sets the default runtime quant mode override for a profile.
// Body: { profileId: string, quantMode: "none" | "int8" | "int4" }
app.post("/api/quant/select", (req, res) => {
  const config = getRuntimeConfig();
  const body = requestBody(req);
  const profile = findProfileByIdOrLabel(config, body.profileId);
  if (!profile) {
    res.status(404).json({ error: "Model profile not found." });
    return;
  }

  const quantMode = normalizeQuantMode(body.quantMode);
  const allowedModes = profile.quant?.selectableModes ?? ["none"];
  if (!allowedModes.includes(quantMode)) {
    res.status(400).json({
      error: `Quant mode '${quantMode}' is not supported for ${profile.label}.`,
      allowed: allowedModes
    });
    return;
  }

  const configured = profile.quant?.configuredMode || "none";
  if (quantMode === configured) {
    quantOverrides.delete(profile.id);
  } else {
    quantOverrides.set(profile.id, quantMode);
  }

  res.json({
    ok: true,
    profileId: profile.id,
    modelLabel: profile.label,
    quantMode,
    override: quantOverrides.get(profile.id) || "",
    allowed: allowedModes
  });
});

// GET /api/quant/state?profileId=&quantMode=
// Returns effective quant state for a profile plus latest conversion job.
app.get("/api/quant/state", (req, res) => {
  const config = getRuntimeConfig();
  const profile = findProfileByIdOrLabel(config, req.query.profileId);
  if (!profile) {
    res.status(404).json({ error: "Model profile not found." });
    return;
  }

  const requestedMode = req.query.quantMode || "";
  const effectiveQuantMode = resolveQuantModeForProfile(profile, requestedMode);
  const latestForMode = summarizeQuantJob(
    findLatestQuantJobForProfile(profile.id, effectiveQuantMode)
  );
  const latestAny = summarizeQuantJob(findLatestQuantJobForProfile(profile.id));
  const worker = interactiveWorker
    ? {
        ready: Boolean(interactiveWorker.ready),
        alive: Boolean(
          interactiveWorker.child &&
          !interactiveWorker.child.killed &&
          interactiveWorker.child.exitCode == null
        ),
        metrics: interactiveWorker.lastMetrics || null,
        ...parseWorkerKey(interactiveWorker.workerKey)
      }
    : null;

  res.json({
    ok: true,
    profileId: profile.id,
    modelLabel: profile.label,
    quant: profile.quant,
    selectedQuantMode: effectiveQuantMode,
    configuredQuantMode: profile.quant?.configuredMode || "none",
    overrideQuantMode: quantOverrides.get(profile.id) || "",
    latestJobForSelectedQuant: latestForMode,
    latestJobAnyQuant: latestAny,
    activeWorker: worker
  });
});

// POST /api/warmup
// Preloads/warms the persistent interactive worker for the selected profile.
// Body: { profileId?: string }
app.post("/api/warmup", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardReady(config, res)) return;

  try {
    const body = requestBody(req);
    const performanceMode = isTruthyFlag(body.performanceMode);
    const quantMode = body.quantMode;
    const profile = resolveWarmupProfile(config, body.profileId);
    const targetWorkerKey = buildWorkerCliConfig(profile, { performanceMode, quantMode }).workerKey;

    const ready =
      Boolean(interactiveWorker) &&
      interactiveWorker.workerKey === targetWorkerKey &&
      interactiveWorker.ready === true;

    if (ready) {
      res.json({
        ok: true,
        warmed: profile.id,
        modelLabel: profile.label,
        pending: false,
        ready: true,
        busy: Boolean(activeRequest)
      });
      return;
    }

    // If a generation is already running, report warmup as pending instead of
    // failing with 409; the UI can keep showing a non-error warming state.
    if (activeRequest) {
      res.json({
        ok: true,
        warmed: profile.id,
        modelLabel: profile.label,
        pending: true,
        ready: false,
        busy: true
      });
      return;
    }

    // Kick warmup in background so this endpoint stays responsive even when
    // first-token warmup takes tens of seconds on larger models.
    void warmupProfileWorker(config, profile.id, { performanceMode, quantMode }).catch((err) => {
      console.warn(
        `[cpi] warmup failed for ${profile.label}: ${err?.message || String(err)}`
      );
    });

    res.json({
      ok: true,
      warmed: profile.id,
      modelLabel: profile.label,
      pending: true,
      ready: false,
      busy: true
    });
  } catch (err) {
    res.status(400).json({ error: err.message || String(err) });
  }
});

// POST /api/generate
// Blocking (non-streaming) inference. Returns the full response as JSON once
// generation is complete.
//
// Request body:
//   messages       array of { role: "user"|"assistant"|"system", content: string }
//   profileId?     model profile id (defaults to the server's selected profile)
//   template?      chat template override
//   systemPrompt?  system prompt override
//   maxNewTokens?  max tokens to generate (clamped 32â€“4096)
//   temperature?   sampling temperature (clamped 0â€“2)
//
// Response:
//   { text, elapsedMs, profileId, modelLabel, template, messageCount, maxNewTokens, temperature }
app.post("/api/generate", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardReady(config, res)) return;
  if (!(await guardIdle(res, { waitForWarmup: true }))) return;

  let cliConfig;
  try {
    cliConfig = buildCliArgs(config, requestBody(req));
  } catch (err) {
    res.status(400).json({ error: err.message });
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onDelta: () => {},
    onDone: (result) => res.json(result),
    onError: (msg) => {
      if (!res.headersSent) res.status(500).json({ error: msg });
    }
  });

  onDisconnect(req, res, cancel);
});

// POST /api/chat/stream
// Streaming inference. Returns newline-delimited JSON events:
//   { type: "start", profileId, modelLabel, ... }
//   { type: "delta", delta: "..." }         (one or more)
//   { type: "metrics", metrics: {...} }     (optional, per-token runtime metrics)
//   { type: "done",  elapsedMs, message }
//   { type: "error", error: "..." }         (on failure)
//
// Request body: same as /api/generate
app.post("/api/chat/stream", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardReady(config, res)) return;
  if (!(await guardIdle(res, { waitForWarmup: true }))) return;

  let cliConfig;
  try {
    cliConfig = buildCliArgs(config, requestBody(req));
  } catch (err) {
    res.status(400).json({ error: err.message });
    return;
  }

  setStreamingHeaders(res, "application/x-ndjson; charset=utf-8");

  const { cancel } = runGeneration(config, cliConfig, {
    onStart: (meta) => writeNdjson(res, { type: "start", ...meta }),
    onDelta: (delta) => writeNdjson(res, { type: "delta", delta }),
    onMetrics: (metrics) => writeNdjson(res, { type: "metrics", metrics }),
    onDone: ({ text, elapsedMs, generatedTokens, tokPerS, metrics }) => {
      writeNdjson(res, {
        type: "done",
        elapsedMs,
        generatedTokens,
        tokPerS,
        metrics: metrics || null,
        message: text
      });
      res.end();
    },
    onError: (msg) => {
      writeNdjson(res, { type: "error", error: msg });
      res.end();
    }
  });

  onDisconnect(req, res, cancel);
});

// POST /v1/chat/completions
// OpenAI-compatible chat completions endpoint.
//
// Supports both streaming (stream: true â†’ SSE) and non-streaming (stream: false â†’ JSON).
// Request body follows the OpenAI API spec:
//   model?       maps to profileId / model label
//   messages     array of { role, content }
//   temperature? max_tokens? stream?
//
// Non-streaming response:
//   { id, object, created, model, choices: [{ message: { role, content }, finish_reason }], usage: null }
//
// Streaming response (SSE):
//   data: { id, object, created, model, choices: [{ delta: { role|content }, finish_reason }] }
//   data: [DONE]
app.post("/v1/chat/completions", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardReady(config, res)) return;
  if (!(await guardIdle(res, { waitForWarmup: true }))) return;

  const body = requestBody(req);
  const stream = Boolean(body.stream);
  const completionId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  let cliConfig;
  try {
    cliConfig = buildCliArgs(config, body);
  } catch (err) {
    res.status(400).json({
      error: { message: err.message, type: "invalid_request_error" }
    });
    return;
  }

  const modelName = cliConfig.meta.modelLabel;

  if (stream) {
    setStreamingHeaders(res, "text/event-stream; charset=utf-8");

    // First chunk: role declaration
    writeSse(res, {
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model: modelName,
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
    });

    const { cancel } = runGeneration(config, cliConfig, {
      onDelta: (delta) =>
        writeSse(res, {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          choices: [{ index: 0, delta: { content: delta }, finish_reason: null }]
        }),
      onDone: () => {
        writeSse(res, {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
        });
        res.write("data: [DONE]\n\n");
        res.end();
      },
      onError: (msg) => {
        writeSse(res, { error: { message: msg, type: "server_error" } });
        res.end();
      }
    });

    onDisconnect(req, res, cancel);
  } else {
    const { cancel } = runGeneration(config, cliConfig, {
      onDelta: () => {},
      onDone: ({ text }) =>
        res.json({
          id: completionId,
          object: "chat.completion",
          created,
          model: modelName,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: text },
              finish_reason: "stop"
            }
          ],
          usage: null
        }),
      onError: (msg) => {
        if (!res.headersSent)
          res.status(500).json({
            error: { message: msg, type: "server_error" }
          });
      }
    });

    onDisconnect(req, res, cancel);
  }
});

// quantization conversion jobs
const quantJobs = new Map(); // jobId -> { status, path, log[], child, subscribers, metadata }

function newQuantJob(meta = {}) {
  const now = Date.now();
  return {
    status: "pending",
    path: "",
    log: [],
    child: null,
    subscribers: new Set(),
    profileId: meta.profileId || "",
    modelLabel: meta.modelLabel || "",
    quantMode: meta.quantMode || "",
    progress: null,
    createdAt: now,
    updatedAt: now
  };
}

function quantEmit(job, event) {
  job.updatedAt = Date.now();
  if (event?.type === "progress") {
    const pct = Number(event.pct);
    job.progress = {
      done: Number(event.done) || 0,
      total: Number(event.total) || 0,
      pct: Number.isFinite(pct) ? pct : null,
      tensor: event.tensor || "",
      updatedAt: job.updatedAt
    };
  }
  if (event?.type === "done") {
    job.progress = {
      done: 1,
      total: 1,
      pct: 100,
      tensor: "",
      updatedAt: job.updatedAt
    };
  }
  job.log.push(event);
  for (const subscriber of job.subscribers) {
    subscriber.write(`data: ${JSON.stringify(event)}\n\n`);
  }
}

function closeQuantSubscribers(job) {
  for (const subscriber of job.subscribers) subscriber.end();
  job.subscribers.clear();
}

function defaultQuantOutputPath(modelPath, quantMode) {
  const parsed = path.parse(modelPath);
  return path.join(parsed.dir, `${parsed.name}-streaming-${quantMode}${parsed.ext || ".ll2c"}`);
}

function resolveQuantProfile(config, profileId) {
  const profile = findProfileByIdOrLabel(config, profileId);
  if (!profile) {
    throw new Error("Model profile not found.");
  }
  if (!profile.modelPath?.toLowerCase().endsWith(".ll2c")) {
    throw new Error("Only .ll2c models support streaming quant conversion.");
  }
  if (!fs.existsSync(profile.modelPath)) {
    throw new Error("Model file does not exist.");
  }
  return profile;
}

// POST /api/quant/convert
// Body: { profileId, quantMode: "int8"|"int4", outputPath?, overwrite? }
app.post("/api/quant/convert", (req, res) => {
  const config = getRuntimeConfig();
  const body = requestBody(req);
  const quantMode = normalizeQuantMode(body.quantMode);
  if (quantMode !== "int8" && quantMode !== "int4") {
    res.status(400).json({ error: "quantMode must be int8 or int4." });
    return;
  }

  let profile;
  try {
    profile = resolveQuantProfile(config, body.profileId);
  } catch (err) {
    res.status(400).json({ error: err.message || String(err) });
    return;
  }

  const allowedModes = profile.quant?.selectableModes ?? ["none"];
  if (!allowedModes.includes(quantMode)) {
    res.status(400).json({
      error: `Quant mode ${quantMode} is not supported by ${profile.label}.`,
      allowed: allowedModes
    });
    return;
  }

  const conversionState = profile.quant?.conversion?.[quantMode]?.state || "unavailable";
  if (conversionState === "ready") {
    res.status(409).json({
      error: `${quantMode} packed tensors are already present for this model.`
    });
    return;
  }
  if (conversionState !== "available") {
    res.status(400).json({
      error:
        profile.quant?.conversion?.[quantMode]?.reason ||
        "This model cannot be converted to the requested quant mode."
    });
    return;
  }

  const existingRunning = findLatestQuantJobForProfile(profile.id, quantMode);
  if (existingRunning && existingRunning.status === "running") {
    res.status(409).json({
      error: `A ${quantMode} conversion is already running for ${profile.label}.`,
      jobId: existingRunning.jobId
    });
    return;
  }

  const outputPath = body.outputPath
    ? path.resolve(String(body.outputPath))
    : defaultQuantOutputPath(profile.modelPath, quantMode);
  const overwrite = isTruthyFlag(body.overwrite);
  if (!overwrite && fs.existsSync(outputPath)) {
    res.status(409).json({
      error: `Output already exists: ${outputPath}`,
      outputPath
    });
    return;
  }

  const scriptPath = path.resolve(config.repoRoot, "tools", "quantize_ll2c_streaming.py");
  if (!fs.existsSync(scriptPath)) {
    res.status(500).json({ error: "quantize_ll2c_streaming.py not found." });
    return;
  }

  const jobId = randomUUID();
  const job = newQuantJob({
    profileId: profile.id,
    modelLabel: profile.label,
    quantMode
  });
  quantJobs.set(jobId, job);
  job.status = "running";

  const args = [scriptPath, "--input", profile.modelPath, "--output", outputPath, "--mode", quantMode];
  if (overwrite) args.push("--overwrite");

  const child = spawn(pythonBin(), args, { cwd: config.repoRoot, env: process.env });
  job.child = child;

  const decoder = new StringDecoder("utf8");
  let partial = "";

  child.stdout.on("data", (chunk) => {
    partial += decoder.write(chunk);
    const lines = partial.split("\n");
    partial = lines.pop();
    for (const line of lines) {
      const text = line.trim();
      if (!text) continue;
      try {
        const event = JSON.parse(text);
        quantEmit(job, event);
        if (event.type === "done" && event.path) {
          job.path = String(event.path);
          job.status = "done";
        }
        if (event.type === "error") {
          job.status = "error";
        }
      } catch {
        quantEmit(job, { type: "log", msg: text });
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) quantEmit(job, { type: "log", msg: text });
  });

  child.on("error", (err) => {
    job.status = "error";
    quantEmit(job, { type: "error", msg: `Failed to spawn converter: ${err.message}` });
    quantEmit(job, { type: "status", status: "error" });
    closeQuantSubscribers(job);
  });

  child.on("close", (code) => {
    partial += decoder.end();
    if (partial.trim()) {
      quantEmit(job, { type: "log", msg: partial.trim() });
    }
    job.child = null;
    if (job.status === "running") {
      job.status = code === 0 ? "done" : "error";
      quantEmit(job, { type: "status", status: job.status });
    }
    closeQuantSubscribers(job);
  });

  res.json({
    ok: true,
    jobId,
    profileId: profile.id,
    modelLabel: profile.label,
    quantMode,
    outputPath
  });
});

// GET /api/quant/status/:jobId - SSE stream of conversion events
app.get("/api/quant/status/:jobId", (req, res) => {
  const job = quantJobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  setStreamingHeaders(res, "text/event-stream; charset=utf-8");
  for (const event of job.log) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
  if (job.status !== "running" && job.status !== "pending") {
    res.end();
    return;
  }
  job.subscribers.add(res);
  req.on("close", () => job.subscribers.delete(res));
});

// GET /api/quant/jobs
app.get("/api/quant/jobs", (_req, res) => {
  const jobs = [];
  for (const [jobId, job] of quantJobs.entries()) {
    jobs.push({
      jobId,
      status: job.status,
      path: job.path,
      profileId: job.profileId || "",
      modelLabel: job.modelLabel || "",
      quantMode: job.quantMode || "",
      progress: job.progress || null,
      createdAt: job.createdAt || 0,
      updatedAt: job.updatedAt || 0
    });
  }
  jobs.sort((a, b) => b.createdAt - a.createdAt);
  res.json({ jobs });
});

// DELETE /api/quant/jobs/:jobId - cancel a running conversion
app.delete("/api/quant/jobs/:jobId", (req, res) => {
  const job = quantJobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  if (job.child) {
    job.child.kill("SIGTERM");
    setTimeout(() => job.child?.kill("SIGKILL"), 2000).unref();
  }
  job.status = "cancelled";
  quantEmit(job, { type: "status", status: "cancelled" });
  closeQuantSubscribers(job);
  res.json({ ok: true });
});

// â”€â”€ model hub â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Download jobs are tracked in-memory. Each job spawns hf_download.py and
// streams its JSON-line output to any connected SSE subscribers.
//
// Routes:
//   GET  /api/hub/search?q=&limit=&token=
//   POST /api/hub/download  { repoId, outputDir?, hfToken? }  â†’ { jobId }
//   GET  /api/hub/status/:jobId   (SSE)
//   GET  /api/hub/jobs

const hubJobs = new Map(); // jobId â†’ { status, path, log[], child, subscribers }

function newHubJob() {
  return { status: "pending", path: "", log: [], child: null, subscribers: new Set() };
}

function hubEmit(job, event) {
  job.log.push(event);
  for (const res of job.subscribers) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
}

function closeJobSubscribers(job) {
  for (const subscriber of job.subscribers) subscriber.end();
  job.subscribers.clear();
}

function pythonBin() {
  return process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
}

function forEachJsonLine(rawText, onJson, onText) {
  for (const line of rawText.split("\n")) {
    const text = line.trim();
    if (!text) continue;
    try {
      const parsed = JSON.parse(text);
      onJson(parsed);
    } catch {
      onText?.(text);
    }
  }
}

// POST /api/system/pick-folder
app.post("/api/system/pick-folder", async (req, res) => {
  try {
    const { initialDir } = requestBody(req);
    const selectedPath = await pickFolderNative(
      typeof initialDir === "string" ? initialDir.trim() : ""
    );
    if (!selectedPath) {
      res.json({ cancelled: true, path: "" });
      return;
    }
    res.json({ cancelled: false, path: selectedPath });
  } catch (err) {
    res.status(500).json({ error: err.message || "Unable to open folder picker." });
  }
});

// GET /api/hub/search
app.get("/api/hub/search", (req, res) => {
  const q = (req.query.q || "").trim();
  const limit = Math.min(50, Math.max(1, parseInt(req.query.limit, 10) || 10));
  const token = (req.query.token || "").trim();

  if (!q) {
    res.status(400).json({ error: "q is required" });
    return;
  }

  const config = getRuntimeConfig();
  const scriptPath = path.resolve(config.repoRoot, "tools", "hf_download.py");
  const args = [scriptPath, "search", q, "--limit", String(limit)];
  if (token) args.push("--token", token);

  let raw = "";
  const child = spawn(pythonBin(), args, { env: process.env });
  child.stdout.on("data", (chunk) => { raw += chunk.toString(); });
  child.stderr.on("data", () => {});
  child.on("error", (err) => {
    if (!res.headersSent) res.status(500).json({ error: `Failed to spawn hf_download: ${err.message}` });
  });
  child.on("close", () => {
    forEachJsonLine(raw, (parsed) => {
      if (res.headersSent) return;
      if (parsed.type === "results") {
        res.json(parsed);
        return;
      }
      if (parsed.type === "error") {
        res.status(500).json({ error: parsed.msg });
      }
    });
    if (!res.headersSent) res.status(500).json({ error: "No results from hf_download" });
  });
});

// POST /api/hub/download
app.post("/api/hub/download", (req, res) => {
  const { repoId, outputDir, hfToken, family } = requestBody(req);
  if (!repoId || typeof repoId !== "string" || !repoId.trim()) {
    res.status(400).json({ error: "repoId is required" });
    return;
  }

  const jobId = randomUUID();
  const job = newHubJob();
  hubJobs.set(jobId, job);
  job.status = "running";

  const config = getRuntimeConfig();
  const scriptPath = path.resolve(config.repoRoot, "tools", "hf_download.py");
  const args = [scriptPath, "download", repoId.trim()];
  if (outputDir) args.push("--output-dir", outputDir);
  if (family) args.push("--family", family);
  if (hfToken) args.push("--token", hfToken);

  const child = spawn(pythonBin(), args, { cwd: config.repoRoot, env: process.env });
  job.child = child;

  const decoder = new StringDecoder("utf8");
  let partial = "";

  child.stdout.on("data", (chunk) => {
    partial += decoder.write(chunk);
    const lines = partial.split("\n");
    partial = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t) continue;
      try {
        const ev = JSON.parse(t);
        hubEmit(job, ev);
        if (ev.type === "done") { job.path = ev.path; job.status = "done"; }
        if (ev.type === "error") { job.status = "error"; }
      } catch {
        hubEmit(job, { type: "log", msg: t });
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) hubEmit(job, { type: "log", msg: text });
  });

  child.on("error", (err) => {
    job.status = "error";
    hubEmit(job, { type: "error", msg: `Failed to spawn hf_download: ${err.message}` });
    hubEmit(job, { type: "status", status: "error" });
    closeJobSubscribers(job);
  });

  child.on("close", (code) => {
    partial += decoder.end();
    if (partial.trim()) hubEmit(job, { type: "log", msg: partial.trim() });
    job.child = null;
    if (job.status === "running") {
      job.status = code === 0 ? "done" : "error";
      hubEmit(job, { type: "status", status: job.status });
    }
    closeJobSubscribers(job);
  });

  res.json({ jobId });
});

// GET /api/hub/status/:jobId  â€” SSE stream of job events
app.get("/api/hub/status/:jobId", (req, res) => {
  const job = hubJobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }

  setStreamingHeaders(res, "text/event-stream; charset=utf-8");

  // Replay buffered events
  for (const ev of job.log) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }

  if (job.status !== "running" && job.status !== "pending") {
    res.end();
    return;
  }

  job.subscribers.add(res);
  req.on("close", () => job.subscribers.delete(res));
});

// DELETE /api/hub/jobs/:jobId â€” cancel a running job
app.delete("/api/hub/jobs/:jobId", (req, res) => {
  const job = hubJobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  if (job.child) {
    job.child.kill("SIGTERM");
    setTimeout(() => job.child?.kill("SIGKILL"), 2000).unref();
  }
  job.status = "cancelled";
  hubEmit(job, { type: "status", status: "cancelled" });
  closeJobSubscribers(job);
  res.json({ ok: true });
});

// GET /api/hub/jobs â€” list all known jobs
app.get("/api/hub/jobs", (_req, res) => {
  const jobs = [];
  for (const [id, job] of hubJobs) {
    jobs.push({ jobId: id, status: job.status, path: job.path });
  }
  res.json({ jobs });
});

// â”€â”€ static UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const runtimeConfig = getRuntimeConfig();
const distDir = path.resolve(runtimeConfig.webRoot, "dist");
const indexFile = path.resolve(distDir, "index.html");

if (fs.existsSync(indexFile)) {
  app.use(express.static(distDir));
  app.get(/^(?!\/api|\/v1).*/, (_req, res) => res.sendFile(indexFile));
}

app.listen(runtimeConfig.port, () => {
  const s = publicRuntimeSummary(runtimeConfig);
  const modelInfo = s.selectedProfile
    ? `model=${s.selectedProfile.label} template=${s.selectedProfile.template}`
    : "no model configured";
  console.log(
    `[cpi] http://localhost:${runtimeConfig.port}  ready=${s.ready}  ${modelInfo}`
  );
  if (!s.ready) {
    console.log(
      "[cpi] Set modelPath and tokenizerPath in web/config.json (or LLAMA_MODEL_PATH / LLAMA_TOKENIZER_PATH)."
    );
  }
});
