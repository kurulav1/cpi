import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(webRoot, "..");
const artifactsRoot = path.resolve(repoRoot, "artifacts");
const runtimeStatePath = path.resolve(artifactsRoot, "runtime-state.json");
const legacyRuntimeStatePath = path.resolve(webRoot, ".runtime-state.json");

// ── file config (web/config.json) ─────────────────────────────────────────────

function loadFileConfig() {
  const configPath = path.resolve(webRoot, "config.json");
  if (!fs.existsSync(configPath)) {
    return {};
  }
  try {
    const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
    return typeof raw === "object" && raw !== null ? raw : {};
  } catch (err) {
    console.warn(`[config] Failed to parse config.json: ${err.message}`);
    return {};
  }
}

const FILE_CONFIG = loadFileConfig();

function loadRuntimeState() {
  const sourcePath = fs.existsSync(runtimeStatePath)
    ? runtimeStatePath
    : legacyRuntimeStatePath;
  if (!fs.existsSync(sourcePath)) {
    return {};
  }
  try {
    const raw = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
    return typeof raw === "object" && raw !== null ? raw : {};
  } catch (err) {
    console.warn(`[config] Failed to parse runtime state: ${err.message}`);
    return {};
  }
}

function saveRuntimeState(nextState) {
  const safeState =
    typeof nextState === "object" && nextState !== null ? nextState : {};
  fs.mkdirSync(path.dirname(runtimeStatePath), { recursive: true });
  fs.writeFileSync(runtimeStatePath, JSON.stringify(safeState, null, 2));
}

function sameStoredPath(leftPath, rightPath) {
  if (!leftPath && !rightPath) {
    return true;
  }
  if (!leftPath || !rightPath) {
    return false;
  }
  return normalizeKey(leftPath) === normalizeKey(rightPath);
}

export function setPreferredModelDir(inputPath) {
  const trimmed = String(inputPath || "").trim();
  const state = loadRuntimeState();
  if (!trimmed) {
    if (!state.preferredModelDir) {
      return "";
    }
    delete state.preferredModelDir;
    saveRuntimeState(state);
    return "";
  }
  const resolved = resolveExistingPath(trimmed);
  if (sameStoredPath(state.preferredModelDir, resolved)) {
    return resolved;
  }
  state.preferredModelDir = resolved;
  saveRuntimeState(state);
  return resolved;
}

export function getPreferredModelDir() {
  const state = loadRuntimeState();
  return typeof state.preferredModelDir === "string"
    ? resolveExistingPath(state.preferredModelDir)
    : "";
}

// env var wins over config.json wins over hardcoded default
function pick(envKey, fileKey, fallback) {
  if (process.env[envKey] !== undefined && process.env[envKey] !== "") {
    return process.env[envKey];
  }
  if (FILE_CONFIG[fileKey] !== undefined && FILE_CONFIG[fileKey] !== "") {
    return String(FILE_CONFIG[fileKey]);
  }
  return fallback !== undefined ? String(fallback) : "";
}

function pickRaw(envKey, fileKey) {
  if (process.env[envKey] !== undefined) {
    return String(process.env[envKey]);
  }
  if (FILE_CONFIG[fileKey] !== undefined) {
    return String(FILE_CONFIG[fileKey]);
  }
  return undefined;
}

const DEFAULT_RUNTIME = Object.freeze({
  port: 3001,
  template: "tinyllama",
  systemPrompt: "You are a helpful assistant.",
  forceCpu: false,
  maxNewTokens: 256,
  maxContext: 2048,
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repeatPenalty: 1.05,
  maxCpuPercent: 85,
  maxMemoryPercent: 85,
  resourceSampleMs: 250,
  resourceSustainMs: 5000,
  resourceThrottleMs: 50,
  extraArgs: ""
});

// ── helpers ───────────────────────────────────────────────────────────────────

function resolveExistingPath(inputPath, fallback = "") {
  if (!inputPath) {
    return fallback;
  }
  if (path.isAbsolute(inputPath)) {
    return inputPath;
  }
  return path.resolve(webRoot, inputPath);
}

function parseInteger(rawValue, fallback) {
  const parsed = Number.parseInt(rawValue ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseFloatValue(rawValue, fallback) {
  const parsed = Number.parseFloat(rawValue ?? "");
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBooleanValue(rawValue, fallback) {
  if (typeof rawValue === "boolean") {
    return rawValue;
  }
  const normalized = String(rawValue ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function readIntSetting(envKey, fileKey, defaultValue, min, max) {
  const parsed = parseInteger(
    pick(envKey, fileKey, String(defaultValue)),
    defaultValue
  );
  const minValue = Number.isFinite(min) ? min : Number.NEGATIVE_INFINITY;
  const maxValue = Number.isFinite(max) ? max : Number.POSITIVE_INFINITY;
  return Math.min(maxValue, Math.max(minValue, parsed));
}

function readFloatSetting(envKey, fileKey, defaultValue, min, max) {
  const parsed = parseFloatValue(
    pick(envKey, fileKey, String(defaultValue)),
    defaultValue
  );
  const minValue = Number.isFinite(min) ? min : Number.NEGATIVE_INFINITY;
  const maxValue = Number.isFinite(max) ? max : Number.POSITIVE_INFINITY;
  return Math.min(maxValue, Math.max(minValue, parsed));
}

function readBoolSetting(envKey, fileKey, defaultValue) {
  const envValue = process.env[envKey];
  if (envValue !== undefined && envValue !== "") {
    return parseBooleanValue(envValue, defaultValue);
  }
  if (FILE_CONFIG[fileKey] !== undefined && FILE_CONFIG[fileKey] !== "") {
    return parseBooleanValue(FILE_CONFIG[fileKey], defaultValue);
  }
  return defaultValue;
}

function splitArgs(rawValue) {
  if (!rawValue?.trim()) {
    return [];
  }

  const args = [];
  let current = "";
  let quote = "";

  for (const char of rawValue.trim()) {
    if (quote) {
      if (char === quote) {
        quote = "";
      } else {
        current += char;
      }
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        args.push(current);
        current = "";
      }
      continue;
    }
    current += char;
  }

  if (current) {
    args.push(current);
  }

  return args;
}

function splitPathList(rawValue) {
  if (!rawValue) {
    return [];
  }
  if (Array.isArray(rawValue) && rawValue.length === 0) {
    return [];
  }
  if (!Array.isArray(rawValue) && !String(rawValue).trim()) {
    return [];
  }
  // Support both array (from config.json) and path-delimiter string (from env)
  const raw = Array.isArray(rawValue)
    ? rawValue.join(path.delimiter)
    : rawValue;
  return raw
    .split(path.delimiter)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => resolveExistingPath(entry));
}

function defaultBinaryPath() {
  if (process.platform === "win32") {
    return path.resolve(repoRoot, "build", "Release", "llama_infer.exe");
  }
  return path.resolve(repoRoot, "build", "llama_infer");
}

function normalizeKey(value) {
  return path.resolve(value).replace(/\\/g, "/").toLowerCase();
}

function uniquePaths(values) {
  const seen = new Set();
  const result = [];

  for (const value of values) {
    if (!value) {
      continue;
    }
    const normalized = normalizeKey(value);
    if (seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(path.resolve(value));
  }

  return result;
}

function ensureDirectory(candidatePath) {
  if (!candidatePath || !fs.existsSync(candidatePath)) {
    return "";
  }
  const stat = fs.statSync(candidatePath);
  return stat.isDirectory() ? candidatePath : path.dirname(candidatePath);
}

function walkFiles(rootDir, matcher, maxDepth = 4) {
  if (!rootDir || !fs.existsSync(rootDir)) {
    return [];
  }

  const rootStat = fs.statSync(rootDir);
  if (!rootStat.isDirectory()) {
    return matcher(rootDir) ? [rootDir] : [];
  }

  const results = [];
  const stack = [{ dir: rootDir, depth: 0 }];

  while (stack.length > 0) {
    const current = stack.pop();
    let entries = [];

    try {
      entries = fs.readdirSync(current.dir, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(current.dir, entry.name);
      if (entry.isDirectory()) {
        if (current.depth < maxDepth) {
          stack.push({ dir: fullPath, depth: current.depth + 1 });
        }
        continue;
      }
      if (matcher(fullPath)) {
        results.push(fullPath);
      }
    }
  }

  return results;
}

function isSafetensorsModelDir(candidatePath) {
  if (!candidatePath || !fs.existsSync(candidatePath)) {
    return false;
  }
  let stat;
  try {
    stat = fs.statSync(candidatePath);
  } catch {
    return false;
  }
  if (!stat.isDirectory()) {
    return false;
  }
  return (
    walkFiles(
      candidatePath,
      (p) => p.toLowerCase().endsWith(".safetensors"),
      1
    ).length > 0
  );
}

const LL2C_MAGIC = "LL2CUDA\0";
const LL2C_ENTRY_BYTES = 80;
const LL2C_HEADER_MIN_BYTES = 52;
const LL2C_HEADER_V4_BYTES = 88;
const LL2C_MAX_TABLE_BYTES = 64 * 1024 * 1024;
const ll2cQuantCache = new Map();

function parseConfiguredQuantMode(extraArgs) {
  let mode = "none";
  for (let i = 0; i < extraArgs.length; i += 1) {
    const arg = extraArgs[i];
    if (arg === "--weight-quant") {
      const value = (extraArgs[i + 1] || "").toLowerCase();
      if (value === "int4") {
        mode = "int4";
      } else if (value === "int8") {
        mode = "int8";
      } else {
        mode = "none";
      }
      i += 1;
      continue;
    }
    if (arg === "--int4-streaming") {
      mode = "int4";
      continue;
    }
    if (arg === "--int8-streaming" && mode !== "int4") {
      mode = "int8";
    }
  }
  return mode;
}

function inspectLl2cQuantInfo(modelPath) {
  if (!modelPath || !fs.existsSync(modelPath) || !modelPath.toLowerCase().endsWith(".ll2c")) {
    return null;
  }

  const stat = fs.statSync(modelPath);
  const cacheKey = normalizeKey(modelPath);
  const cached = ll2cQuantCache.get(cacheKey);
  if (cached && cached.size === stat.size && cached.mtimeMs === stat.mtimeMs) {
    return cached.data;
  }

  let fd;
  try {
    fd = fs.openSync(modelPath, "r");
    const header = Buffer.alloc(96);
    const headerRead = fs.readSync(fd, header, 0, header.length, 0);
    if (headerRead < LL2C_HEADER_MIN_BYTES) {
      throw new Error("file too small");
    }
    if (header.subarray(0, 8).toString("binary") !== LL2C_MAGIC) {
      throw new Error("invalid magic");
    }

    const version = header.readInt32LE(8);
    const hidden = header.readInt32LE(16);
    const inter = header.readInt32LE(20);
    const layers = header.readInt32LE(24);
    const numLocalExperts = (version >= 4 && headerRead >= LL2C_HEADER_V4_BYTES)
      ? header.readInt32LE(76)
      : 0;
    const numExpertsPerTok = (version >= 4 && headerRead >= LL2C_HEADER_V4_BYTES)
      ? header.readInt32LE(80)
      : 0;
    const expertIntermediateSize = (version >= 4 && headerRead >= LL2C_HEADER_V4_BYTES)
      ? header.readInt32LE(84)
      : 0;
    const tensorCount = version >= 2 ? header.readInt32LE(44) : header.readInt32LE(40);
    const tableOffset = Number(version >= 2 ? header.readBigInt64LE(48) : header.readBigInt64LE(44));
    if (hidden <= 0 || inter <= 0 || layers <= 0 || tensorCount <= 0 || tableOffset < 0) {
      throw new Error("invalid header values");
    }

    const tableBytes = tensorCount * LL2C_ENTRY_BYTES;
    if (tableBytes <= 0 || tableBytes > LL2C_MAX_TABLE_BYTES) {
      throw new Error("tensor table too large");
    }
    if (tableOffset + tableBytes > stat.size) {
      throw new Error("tensor table out of bounds");
    }

    const table = Buffer.alloc(tableBytes);
    const tableRead = fs.readSync(fd, table, 0, tableBytes, tableOffset);
    if (tableRead !== tableBytes) {
      throw new Error("failed to read full tensor table");
    }

    const names = new Set();
    for (let i = 0; i < tensorCount; i += 1) {
      const base = i * LL2C_ENTRY_BYTES;
      const raw = table.subarray(base, base + 64);
      const nul = raw.indexOf(0);
      const name = raw.subarray(0, nul >= 0 ? nul : raw.length).toString("utf8");
      if (name) {
        names.add(name);
      }
    }

    const totalMlp = numLocalExperts > 0
      ? layers * Math.max(0, numLocalExperts) * 3
      : layers * 3;
    let fp16Mlp = 0;
    let int8Mlp = 0;
    let int4Mlp = 0;
    for (let layer = 0; layer < layers; layer += 1) {
      if (numLocalExperts > 0) {
        for (let expert = 0; expert < numLocalExperts; expert += 1) {
          const p = `layers.${layer}.feed_forward.experts.${expert}`;
          for (const suffix of ["w1", "w2", "w3"]) {
            const baseName = `${p}.${suffix}`;
            if (names.has(baseName)) {
              fp16Mlp += 1;
            }
            if (names.has(`${baseName}.int8`) && names.has(`${baseName}.scale`)) {
              int8Mlp += 1;
            }
            if (names.has(`${baseName}.int4`) && names.has(`${baseName}.scale`)) {
              int4Mlp += 1;
            }
          }
        }
      } else {
        const p = `layers.${layer}.feed_forward`;
        for (const suffix of ["w1", "w2", "w3"]) {
          const baseName = `${p}.${suffix}`;
          if (names.has(baseName)) {
            fp16Mlp += 1;
          }
          if (names.has(`${baseName}.int8`) && names.has(`${baseName}.scale`)) {
            int8Mlp += 1;
          }
          if (names.has(`${baseName}.int4`) && names.has(`${baseName}.scale`)) {
            int4Mlp += 1;
          }
        }
      }
    }

    const data = {
      version,
      hidden,
      inter,
      layers,
      numLocalExperts,
      numExpertsPerTok,
      expertIntermediateSize,
      totalMlpTensors: totalMlp,
      fp16MlpTensors: fp16Mlp,
      int8MlpTensors: int8Mlp,
      int4MlpTensors: int4Mlp,
      fp16MlpComplete: fp16Mlp === totalMlp,
      packedInt8Complete: int8Mlp === totalMlp,
      packedInt4Complete: int4Mlp === totalMlp,
      hasTq3: names.has("tq3_codebook") && names.has("tq3_signs_hidden")
    };
    ll2cQuantCache.set(cacheKey, {
      size: stat.size,
      mtimeMs: stat.mtimeMs,
      data
    });
    return data;
  } catch {
    return null;
  } finally {
    if (fd != null) {
      try {
        fs.closeSync(fd);
      } catch {
        // ignore
      }
    }
  }
}

function buildQuantState(modelPath, extraArgs, { isSafetensorsDir = false, family = "" } = {}) {
  const configuredMode = parseConfiguredQuantMode(extraArgs);
  const normalizedFamily = String(family || "").toLowerCase();
  const qwen35Safetensors = isSafetensorsDir && normalizedFamily === "qwen3_5";
  const fallback = {
    format: isSafetensorsDir ? "safetensors" : "unknown",
    configuredMode,
    recommendedMode: configuredMode,
    effectiveMode: qwen35Safetensors
      ? (configuredMode === "int8" || configuredMode === "int4" ? configuredMode : "none")
      : configuredMode,
    selectableModes: qwen35Safetensors ? ["none", "int8", "int4"] : ["none"],
    packed: { int8: false, int4: false, tq3: false },
    mlpCoverage: {
      totalTensors: 0,
      fp16: 0,
      int8: 0,
      int4: 0,
      numLocalExperts: 0,
      numExpertsPerTok: 0,
      expertIntermediateSize: 0
    },
    conversion: {
      int8: qwen35Safetensors
        ? { state: "native-runtime", reason: "Native Qwen3.5 safetensors runtime supports explicit int8 execution without LL2C conversion." }
        : { state: "unavailable", reason: "LL2C model required for streaming quant conversion." },
      int4: qwen35Safetensors
        ? { state: "native-runtime", reason: "Native Qwen3.5 safetensors runtime supports explicit int4 execution without LL2C conversion." }
        : { state: "unavailable", reason: "LL2C model required for streaming quant conversion." }
    }
  };

  if (!modelPath?.toLowerCase().endsWith(".ll2c")) {
    return fallback;
  }

  const info = inspectLl2cQuantInfo(modelPath);
  if (!info) {
    return {
      ...fallback,
      format: "ll2c",
      conversion: {
        int8: { state: "unavailable", reason: "Failed to inspect LL2C tensor table." },
        int4: { state: "unavailable", reason: "Failed to inspect LL2C tensor table." }
      }
    };
  }

  const supportsNone = info.fp16MlpComplete;
  const supportsInt8 = info.fp16MlpComplete || info.packedInt8Complete || info.packedInt4Complete;
  const supportsInt4 = info.fp16MlpComplete || info.packedInt8Complete || info.packedInt4Complete;

  const selectableModes = [];
  if (supportsNone) selectableModes.push("none");
  if (supportsInt8) selectableModes.push("int8");
  if (supportsInt4) selectableModes.push("int4");
  if (selectableModes.length === 0) selectableModes.push("none");

  let recommendedMode = configuredMode;
  if (info.packedInt4Complete && !info.packedInt8Complete) {
    recommendedMode = "int4";
  } else if (info.packedInt8Complete && !info.packedInt4Complete) {
    recommendedMode = "int8";
  }

  const effectiveMode = selectableModes.includes(recommendedMode)
    ? recommendedMode
    : (selectableModes.includes(configuredMode) ? configuredMode : selectableModes[0]);

  const conversionState = (target) => {
    if (target === "int8" && info.packedInt8Complete) {
      return { state: "ready", reason: "" };
    }
    if (target === "int4" && info.packedInt4Complete) {
      return { state: "ready", reason: "" };
    }
    if (info.fp16MlpComplete) {
      return { state: "available", reason: "" };
    }
    return {
      state: "unavailable",
      reason: "Missing fp16 MLP source tensors required for offline repacking."
    };
  };

  return {
    format: "ll2c",
    configuredMode,
    recommendedMode,
    effectiveMode,
    selectableModes,
    packed: {
      int8: info.packedInt8Complete,
      int4: info.packedInt4Complete,
      tq3: info.hasTq3
    },
    mlpCoverage: {
      totalTensors: info.totalMlpTensors,
      fp16: info.fp16MlpTensors,
      int8: info.int8MlpTensors,
      int4: info.int4MlpTensors,
      numLocalExperts: info.numLocalExperts || 0,
      numExpertsPerTok: info.numExpertsPerTok || 0,
      expertIntermediateSize: info.expertIntermediateSize || 0
    },
    conversion: {
      int8: conversionState("int8"),
      int4: conversionState("int4")
    }
  };
}

// ── per-model auto-configuration ─────────────────────────────────────────────

// Reads a HuggingFace config.json from a model directory and extracts the
// parameters that influence inference settings.
function readHfModelConfig(modelPath) {
  const modelDir = isSafetensorsModelDir(modelPath)
    ? modelPath
    : path.dirname(modelPath);

  let raw;
  const candidates = [
    path.join(modelDir, "config.json"),
    path.join(modelDir, "hf", "config.json")
  ];
  for (const configPath of candidates) {
    if (!fs.existsSync(configPath)) continue;
    try {
      raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
      if (raw && typeof raw === "object") {
        break;
      }
    } catch {
      raw = null;
    }
  }
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const result = {};
  const textConfig =
    raw.text_config && typeof raw.text_config === "object" ? raw.text_config : raw;

  // Model type / architecture
  const rootModelType = (raw.model_type || "").toLowerCase();
  const modelType = (textConfig.model_type || rootModelType || "").toLowerCase();
  const arch = ((textConfig.architectures || raw.architectures || [])[0] || "").toLowerCase();
  result.rootModelType = rootModelType || null;
  result.modelType = modelType || arch || null;

  // RoPE theta
  if (typeof textConfig.rope_theta === "number") {
    result.ropeTheta = textConfig.rope_theta;
  } else if (
    textConfig.rope_scaling &&
    typeof textConfig.rope_scaling.rope_type === "string" &&
    textConfig.rope_scaling.rope_type !== "default"
  ) {
    // Llama3 uses rope_scaling with no explicit rope_theta field in some checkpoints
    result.ropeTheta = 500000.0;
  }

  // Context length
  if (typeof textConfig.max_position_embeddings === "number") {
    result.maxPositionEmbeddings = textConfig.max_position_embeddings;
  }

  // Vocab size (useful for sanity checks)
  if (typeof textConfig.vocab_size === "number") {
    result.vocabSize = textConfig.vocab_size;
  }
  if (typeof textConfig.num_local_experts === "number") {
    result.numLocalExperts = textConfig.num_local_experts;
  }
  if (typeof textConfig.num_experts_per_tok === "number") {
    result.numExpertsPerTok = textConfig.num_experts_per_tok;
  }
  if (typeof textConfig.expert_intermediate_size === "number") {
    result.expertIntermediateSize = textConfig.expert_intermediate_size;
  }
  if (Array.isArray(textConfig.layer_types)) {
    result.layerTypes = textConfig.layer_types.map((value) => String(value));
  }
  if (typeof textConfig.linear_num_key_heads === "number") {
    result.linearNumKeyHeads = textConfig.linear_num_key_heads;
  }
  if (typeof textConfig.linear_num_value_heads === "number") {
    result.linearNumValueHeads = textConfig.linear_num_value_heads;
  }
  if (typeof textConfig.partial_rotary_factor === "number") {
    result.partialRotaryFactor = textConfig.partial_rotary_factor;
  }
  if (result.modelType === "mixtral" && !result.numLocalExperts) {
    result.numLocalExperts = 8;
    if (!result.numExpertsPerTok) result.numExpertsPerTok = 2;
  }

  return result;
}

function ll2cFamilyFromId(modelFamilyId) {
  switch (Number(modelFamilyId) || 0) {
    case 1: return "llama2";
    case 2: return "llama3";
    case 3: return "mistral";
    case 4: return "phi3";
    case 5: return "qwen2";
    case 6: return "mixtral";
    case 7: return "qwen3_5";
    default: return "";
  }
}

function ll2cLayerTypeName(kindId) {
  switch (Number(kindId) || 0) {
    case 1: return "sliding_window_attention";
    case 2: return "linear_attention";
    default: return "full_attention";
  }
}

function readLl2cModelConfig(modelPath) {
  if (!modelPath || path.extname(modelPath).toLowerCase() !== ".ll2c") {
    return null;
  }
  try {
    const fd = fs.openSync(modelPath, "r");
    try {
      const header = Buffer.alloc(112);
      const bytesRead = fs.readSync(fd, header, 0, header.length, 0);
      if (bytesRead < 12) return null;
      if (header.subarray(0, 7).toString("ascii") !== "LL2CUDA") {
        return null;
      }

      const version = header.readInt32LE(8);
      if (version < 3) {
        return null;
      }

      const result = {
        maxPositionEmbeddings: header.readInt32LE(36)
      };
      const family = ll2cFamilyFromId(version >= 3 ? header.readInt32LE(72) : 0);
      if (family) {
        result.modelType = family;
        result.rootModelType = family;
      }
      const ropeTheta = header.readFloatLE(56);
      if (Number.isFinite(ropeTheta) && ropeTheta > 0) {
        result.ropeTheta = ropeTheta;
      }
      if (version >= 4) {
        result.numLocalExperts = header.readInt32LE(76);
        result.numExpertsPerTok = header.readInt32LE(80);
        result.expertIntermediateSize = header.readInt32LE(84);
      }
      if (version >= 5 && bytesRead >= 112) {
        result.partialRotaryFactor = header.readFloatLE(88);
        result.linearNumKeyHeads = header.readInt32LE(92);
        result.linearNumValueHeads = header.readInt32LE(96);
        const attentionTypeCount = header.readInt32LE(100);
        const attentionTypeOffset = Number(header.readBigInt64LE(104));
        if (attentionTypeCount > 0 && attentionTypeOffset > 0) {
          const meta = Buffer.alloc(attentionTypeCount * 4);
          const metaRead = fs.readSync(fd, meta, 0, meta.length, attentionTypeOffset);
          if (metaRead === meta.length) {
            result.layerTypes = [];
            for (let i = 0; i < attentionTypeCount; i += 1) {
              result.layerTypes.push(ll2cLayerTypeName(meta.readInt32LE(i * 4)));
            }
          }
        }
      }
      return result;
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return null;
  }
}

function mergeModelConfigs(...configs) {
  const merged = {};
  for (const config of configs) {
    if (!config || typeof config !== "object") continue;
    for (const [key, value] of Object.entries(config)) {
      if (value === undefined || value === null || value === "") continue;
      merged[key] = value;
    }
  }
  return Object.keys(merged).length > 0 ? merged : null;
}

function tokenizerConfigCandidates(modelPath, tokenizerPath) {
  const candidates = [];
  const push = (value) => {
    if (!value) return;
    const resolved = path.resolve(value);
    if (!candidates.some((item) => normalizeKey(item) === normalizeKey(resolved))) {
      candidates.push(resolved);
    }
  };

  if (tokenizerPath) {
    const tokenizerDir = fs.existsSync(tokenizerPath) && fs.statSync(tokenizerPath).isDirectory()
      ? tokenizerPath
      : path.dirname(tokenizerPath);
    push(path.join(tokenizerDir, "tokenizer_config.json"));
  }

  if (modelPath) {
    if (fs.existsSync(modelPath) && fs.statSync(modelPath).isDirectory()) {
      push(path.join(modelPath, "tokenizer_config.json"));
    } else {
      const modelDir = path.dirname(modelPath);
      push(path.join(modelDir, "tokenizer_config.json"));
      push(path.join(modelDir, "hf", "tokenizer_config.json"));
    }
  }

  return candidates;
}

function readHfTokenizerConfig(modelPath, tokenizerPath) {
  for (const candidate of tokenizerConfigCandidates(modelPath, tokenizerPath)) {
    if (!fs.existsSync(candidate)) continue;
    try {
      const raw = JSON.parse(fs.readFileSync(candidate, "utf8"));
      if (!raw || typeof raw !== "object") {
        continue;
      }
      return {
        path: candidate,
        chatTemplate:
          typeof raw.chat_template === "string" ? raw.chat_template : "",
        useDefaultSystemPrompt:
          typeof raw.use_default_system_prompt === "boolean"
            ? raw.use_default_system_prompt
            : null
      };
    } catch (err) {
      console.warn(`[config] Failed to parse tokenizer config ${candidate}: ${err.message}`);
    }
  }
  return null;
}

function inferTemplateFromChatTemplate(chatTemplate) {
  const raw = String(chatTemplate || "").trim();
  if (!raw) return "";

  if (raw.includes("<|im_start|>") && raw.includes("<|im_end|>") && raw.includes("<think>")) {
    return "qwen3_5";
  }
  if (raw.includes("<|start_header_id|>") && raw.includes("<|eot_id|>")) {
    return raw.includes("<|begin_of_text|>") ? "llama4" : "llama3";
  }
  if (raw.includes("<|im_start|>") && raw.includes("<|im_end|>")) {
    return "qwen2";
  }
  if (raw.includes("<|system|>") && raw.includes("<|user|>") && raw.includes("<|assistant|>")) {
    return "tinyllama-chatml";
  }
  if (raw.includes("<|user|>") && raw.includes("<|assistant|>") && raw.includes("<|end|>")) {
    return "phi3";
  }
  if (raw.includes("[INST]") && raw.includes("[/INST]")) {
    return raw.includes("<<SYS>>") ? "llama2" : "mistral";
  }

  return "";
}

// Infer RoPE theta from model path and optional HF config.
function inferRopeTheta(modelPath, hfConfig) {
  // Explicit from HF config
  if (hfConfig?.ropeTheta) {
    return hfConfig.ropeTheta;
  }

  // Filename heuristics
  const lower = path.basename(modelPath).toLowerCase();
  if (
    lower.includes("llama-3") ||
    lower.includes("llama3") ||
    lower.includes("llama4") ||
    lower.includes("llama-4")
  ) {
    return 500000.0;
  }

  return 10000.0; // Llama 2 / Mistral / most others
}

function modelFamilyName(value) {
  const lower = value.toLowerCase();
  if (lower.includes("tinyllama")) return "tinyllama";
  if (lower.includes("llama-4") || lower.includes("llama4")) return "llama4";
  if (lower.includes("llama-3") || lower.includes("llama3")) return "llama3";
  if (lower.includes("llama-2") || lower.includes("llama2")) return "llama2";
  if (lower.includes("mixtral")) return "mixtral";
  if (lower.includes("mistral")) return "mistral";
  if (lower.includes("phi")) return "phi";
  if (lower.includes("gemma")) return "gemma";
  if (lower.includes("qwen")) return "qwen";
  return "generic";
}

function inferProfileFamily(modelPath, hfConfig) {
  const modelType = String(hfConfig?.modelType || "").toLowerCase();
  const rootModelType = String(hfConfig?.rootModelType || "").toLowerCase();

  if (modelType.includes("qwen3_5") || rootModelType.includes("qwen3_5")) {
    return "qwen3_5";
  }
  if (modelType.includes("mixtral")) {
    return "mixtral";
  }
  if (modelType.includes("qwen2")) {
    return "qwen";
  }
  return modelFamilyName(path.basename(modelPath || ""));
}

function inferUnsupportedArchitecture(hfConfig = {}) {
  if (!hfConfig || typeof hfConfig !== "object") {
    return "";
  }
  const modelType = String(hfConfig.modelType || "").toLowerCase();
  const rootModelType = String(hfConfig.rootModelType || "").toLowerCase();
  const layerTypes = Array.isArray(hfConfig.layerTypes)
    ? hfConfig.layerTypes.map((value) => String(value).toLowerCase())
    : [];
  const hasLinearAttention =
    layerTypes.some((value) => value.includes("linear_attention")) ||
    Number(hfConfig.linearNumKeyHeads) > 0 ||
    Number(hfConfig.linearNumValueHeads) > 0;

  if (modelType.includes("qwen3_5") || rootModelType.includes("qwen3_5")) {
    return "";
  }

  if (hasLinearAttention) {
    return "This model uses linear attention layers, which the native engine does not support yet.";
  }

  return "";
}

function inferTemplate(modelPath, tokenizerPath, fallbackTemplate, hfConfig, hfTokenizerConfig) {
  const family = modelFamilyName(path.basename(modelPath));
  const tokenizerExt = path.extname(tokenizerPath || "").toLowerCase();
  const templateFromTokenizerConfig = inferTemplateFromChatTemplate(
    hfTokenizerConfig?.chatTemplate
  );

  if (templateFromTokenizerConfig) {
    return templateFromTokenizerConfig;
  }

  // HF config model_type override
  const modelType = hfConfig?.modelType || "";
  if (modelType.includes("llama3") || modelType.includes("llama-3")) {
    return "llama3";
  }
  if (modelType.includes("qwen3_5")) {
    return "qwen3_5";
  }
  if (modelType.includes("phimoe") || modelType.includes("phi")) {
    return "phi3";
  }
  if (modelType.includes("mixtral")) {
    return "mistral";
  }

  if (family === "tinyllama") {
    return "tinyllama";
  }
  if (family === "llama2") return "llama2";
  if (family === "llama3") return "llama3";
  if (family === "llama4") return "llama4";
  if (family === "mixtral") return "mistral";
  if (family === "phi") return "phi3";
  if (family === "qwen") return "qwen2";
  if (family === "qwen3_5") return "qwen3_5";

  if (tokenizerExt === ".json" && isSafetensorsModelDir(modelPath)) {
    return "llama4";
  }

  if (fallbackTemplate) return fallbackTemplate;

  return tokenizerExt === ".json" ? "tinyllama" : "plain";
}

function pathDistance(leftPath, rightPath) {
  const leftParts = path.resolve(leftPath).split(path.sep).filter(Boolean);
  const rightParts = path.resolve(rightPath).split(path.sep).filter(Boolean);
  let prefix = 0;
  while (
    prefix < leftParts.length &&
    prefix < rightParts.length &&
    leftParts[prefix].toLowerCase() === rightParts[prefix].toLowerCase()
  ) {
    prefix += 1;
  }
  return leftParts.length + rightParts.length - prefix * 2;
}

function scoreTokenizerCandidate(modelPath, tokenizerPath) {
  const modelName = path.basename(modelPath).toLowerCase();
  const tokenizerName = path.basename(tokenizerPath).toLowerCase();
  const tokenizerExt = path.extname(tokenizerPath).toLowerCase();
  const family = modelFamilyName(modelName);
  let score = 0;

  const modelDir = isSafetensorsModelDir(modelPath)
    ? modelPath
    : path.dirname(modelPath);
  const tokenizerDir = path.dirname(tokenizerPath);
  const distance = pathDistance(modelDir, tokenizerDir);

  if (normalizeKey(modelDir) === normalizeKey(tokenizerDir)) score += 60;
  score += Math.max(0, 26 - distance * 4);

  if (family === "tinyllama") {
    if (tokenizerExt === ".json") score += 28;
    if (tokenizerPath.toLowerCase().includes("tinyllama")) score += 36;
    if (tokenizerExt === ".model") score -= 10;
  } else if (family === "llama2") {
    // LLaMA-2 uses SentencePiece (.model)
    if (tokenizerExt === ".model") score += 22;
    if (
      tokenizerPath.toLowerCase().includes("llama2") ||
      tokenizerPath.toLowerCase().includes("llama-2")
    ) {
      score += 32;
    }
  } else if (family === "llama3") {
    // LLaMA-3 uses tiktoken/BPE (tokenizer.json) — .model files are invalid
    if (tokenizerExt === ".json") score += 40;
    if (tokenizerExt === ".model") score -= 50;
    if (
      tokenizerPath.toLowerCase().includes("llama3") ||
      tokenizerPath.toLowerCase().includes("llama-3")
    ) {
      score += 32;
    }
  } else if (family === "llama4") {
    if (tokenizerExt === ".json") score += 28;
    if (
      tokenizerPath.toLowerCase().includes("llama4") ||
      tokenizerPath.toLowerCase().includes("llama-4") ||
      tokenizerPath.toLowerCase().includes("llama3") ||
      tokenizerPath.toLowerCase().includes("llama-3")
    ) {
      score += 28;
    }
  } else if (tokenizerExt === ".json") {
    score += 6;
  }

  if (tokenizerName === "tokenizer.json" || tokenizerName === "tokenizer.model") {
    score += 8;
  }

  return score;
}

function chooseTokenizer(modelPath, tokenizerCandidates, preferredTokenizerPath) {
  if (preferredTokenizerPath && fs.existsSync(preferredTokenizerPath)) {
    return preferredTokenizerPath;
  }
  if (tokenizerCandidates.length === 0) return "";
  if (tokenizerCandidates.length === 1) return tokenizerCandidates[0];

  const ranked = tokenizerCandidates
    .map((p) => ({ tokenizerPath: p, score: scoreTokenizerCandidate(modelPath, p) }))
    .sort((a, b) => b.score - a.score);

  return ranked[0].score > 0 ? ranked[0].tokenizerPath : "";
}

// Build extra args that are auto-determined from model properties.
function buildAutoExtraArgs(modelPath, hfConfig, globalExtraArgs) {
  const args = [...globalExtraArgs];
  const lower = path.basename(modelPath).toLowerCase();
  const looksStreaming = lower.includes("streaming") || lower.includes("packed");

  const looksTq3 = lower.includes("tq3") || lower.includes("turbo");

  // Keep streaming-quant toggles only for packed/streaming models;
  // keep --enable-tq-cached only for TQ3 models.
  const filtered = args.filter(
    (arg) => (arg !== "--int8-streaming" || looksStreaming) &&
             (arg !== "--int4-streaming" || looksStreaming) &&
             (arg !== "--enable-tq-cached" || looksTq3)
  );

  // Auto-inject --rope-theta for non-default theta values
  const ropeTheta = inferRopeTheta(modelPath, hfConfig);
  if (ropeTheta !== 10000.0 && !filtered.includes("--rope-theta")) {
    filtered.push("--rope-theta", String(ropeTheta));
  }

  return filtered;
}

function discoverSafetensorsModelDirs(scanRoots) {
  return uniquePaths(
    scanRoots.flatMap((root) =>
      walkFiles(
        root,
        (p) => p.toLowerCase().endsWith(".safetensors"),
        6
      ).map((p) => path.dirname(p))
    )
  );
}

function buildProfile(modelPath, tokenizerPath, baseConfig, source = "discovered") {
  const ll2cConfig = readLl2cModelConfig(modelPath);
  const hfConfig = readHfModelConfig(modelPath);
  const modelConfig = mergeModelConfigs(ll2cConfig, hfConfig);
  const hfTokenizerConfig = readHfTokenizerConfig(modelPath, tokenizerPath);
  const template = inferTemplate(
    modelPath,
    tokenizerPath,
    baseConfig.template,
    modelConfig,
    hfTokenizerConfig
  );
  const family = inferProfileFamily(modelPath, modelConfig);
  const extraArgs = modelPath
    ? buildAutoExtraArgs(modelPath, modelConfig, baseConfig.extraArgs)
    : [...baseConfig.extraArgs];
  const tokenizerFormat = path.extname(tokenizerPath || "").toLowerCase();
  const ropeTheta = inferRopeTheta(modelPath, modelConfig);
  const unsupportedReason = inferUnsupportedArchitecture(modelConfig);

  // Safetensors model directories are only supported by the Llama4 engine.
  // All other families must be converted to .ll2c before use.
  const isSafetensorsDir = isSafetensorsModelDir(modelPath);
  const supportsNativeSafetensors =
    isSafetensorsDir && (family === "llama4" || family === "qwen3_5");
  const filesExist =
    Boolean(modelPath) &&
    fs.existsSync(modelPath) &&
    Boolean(tokenizerPath) &&
    fs.existsSync(tokenizerPath);
  const ready =
    filesExist &&
    !unsupportedReason &&
    (!isSafetensorsDir || supportsNativeSafetensors);

  let status;
  if (unsupportedReason) {
    status = "unsupported-architecture";
  } else if (ready) {
    status = "ready";
  } else if (isSafetensorsDir && !supportsNativeSafetensors) {
    status = "needs-conversion";
  } else if (!tokenizerPath || !fs.existsSync(tokenizerPath)) {
    status = "tokenizer-missing";
  } else {
    status = "model-missing";
  }
  const quant = buildQuantState(modelPath, extraArgs, { isSafetensorsDir, family });
  const moeFromQuant = quant?.mlpCoverage?.numLocalExperts > 0
    ? {
        numLocalExperts: quant.mlpCoverage.numLocalExperts,
        numExpertsPerTok: quant.mlpCoverage.numExpertsPerTok,
        expertIntermediateSize: quant.mlpCoverage.expertIntermediateSize
      }
    : null;
  const moeFromHf = (modelConfig?.numLocalExperts > 0)
    ? {
        numLocalExperts: modelConfig.numLocalExperts,
        numExpertsPerTok: modelConfig.numExpertsPerTok || 2,
        expertIntermediateSize: modelConfig.expertIntermediateSize || 0
      }
    : null;
  const moe = moeFromQuant || moeFromHf;
  const baseLabel = path.basename(modelPath || "Unconfigured model", ".ll2c");
  const label =
    baseLabel.toLowerCase() === "hf" && modelPath
      ? path.basename(path.dirname(modelPath))
      : baseLabel;

  return {
    id: modelPath,
    source,
    label,
    family,
    modelPath,
    tokenizerPath,
    tokenizerFormat,
    template,
    tokenizerChatTemplatePath: hfTokenizerConfig?.path || "",
    tokenizerUsesDefaultSystemPrompt:
      hfTokenizerConfig?.useDefaultSystemPrompt,
    extraArgs,
    ropeTheta,
    maxPositionEmbeddings: modelConfig?.maxPositionEmbeddings ?? null,
    unsupportedReason,
    quant,
    moe,
    ready,
    status
  };
}

function publicModelProfile(profile) {
  return {
    id: profile.id,
    source: profile.source,
    label: profile.label,
    family: profile.family,
    modelPath: profile.modelPath,
    tokenizerPath: profile.tokenizerPath,
    tokenizerFormat: profile.tokenizerFormat,
    template: profile.template,
    tokenizerChatTemplatePath: profile.tokenizerChatTemplatePath,
    tokenizerUsesDefaultSystemPrompt: profile.tokenizerUsesDefaultSystemPrompt,
    extraArgs: profile.extraArgs,
    ropeTheta: profile.ropeTheta,
    maxPositionEmbeddings: profile.maxPositionEmbeddings,
    unsupportedReason: profile.unsupportedReason || "",
    quant: profile.quant,
    moe: profile.moe,
    ready: profile.ready,
    status: profile.status
  };
}

function discoverModelProfiles(baseConfig) {
  const preferredModelDir = getPreferredModelDir();
  const scanRoots = uniquePaths([
    ...splitPathList(process.env.LLAMA_MODEL_DIRS || FILE_CONFIG.modelDirs || ""),
    ensureDirectory(preferredModelDir),
    ensureDirectory(baseConfig.modelPath),
    ensureDirectory(baseConfig.tokenizerPath),
    artifactsRoot
  ]);

  const modelCandidates = uniquePaths([
    baseConfig.modelPath,
    ...scanRoots.flatMap((root) =>
      walkFiles(root, (p) => p.toLowerCase().endsWith(".ll2c"))
    ),
    ...discoverSafetensorsModelDirs(scanRoots)
  ]);

  const tokenizerCandidates = uniquePaths([
    baseConfig.tokenizerPath,
    ...scanRoots.flatMap((root) =>
      walkFiles(root, (p) => {
        const lower = p.toLowerCase();
        return lower.endsWith("tokenizer.json") || lower.endsWith("tokenizer.model");
      })
    )
  ]);

  const profiles = modelCandidates.map((modelPath) => {
    const preferredTokenizer =
      baseConfig.modelPath &&
      normalizeKey(modelPath) === normalizeKey(baseConfig.modelPath)
        ? baseConfig.tokenizerPath
        : "";
    const tokenizerPath = chooseTokenizer(
      modelPath,
      tokenizerCandidates,
      preferredTokenizer
    );
    return buildProfile(
      modelPath,
      tokenizerPath,
      baseConfig,
      preferredTokenizer ? "configured" : "discovered"
    );
  });

  if (
    baseConfig.modelPath &&
    !profiles.some(
      (p) => normalizeKey(p.modelPath) === normalizeKey(baseConfig.modelPath)
    )
  ) {
    profiles.push(
      buildProfile(
        baseConfig.modelPath,
        baseConfig.tokenizerPath,
        baseConfig,
        "configured"
      )
    );
  }

  return profiles.sort((a, b) => {
    if (a.ready !== b.ready) return a.ready ? -1 : 1;
    if (a.source !== b.source) return a.source === "configured" ? -1 : 1;
    return a.label.localeCompare(b.label);
  });
}

function chooseSelectedProfile(profiles, configuredModelPath) {
  if (configuredModelPath) {
    const configured = profiles.find(
      (p) => normalizeKey(p.modelPath) === normalizeKey(configuredModelPath)
    );
    if (configured?.ready) return configured;
  }
  return profiles.find((p) => p.ready) ?? profiles[0] ?? null;
}

// ── public API ────────────────────────────────────────────────────────────────

export function getRuntimeConfig() {
  const inferBin = resolveExistingPath(
    pick("LLAMA_INFER_BIN", "inferBin", ""),
    defaultBinaryPath()
  );
  const configuredModelPath = resolveExistingPath(
    pick("LLAMA_MODEL_PATH", "modelPath", "")
  );
  const configuredTokenizerPath = resolveExistingPath(
    pick("LLAMA_TOKENIZER_PATH", "tokenizerPath", "")
  );
  const explicitTemplateRaw = pickRaw("LLAMA_CHAT_TEMPLATE", "chatTemplate");
  const hasExplicitTemplate = explicitTemplateRaw !== undefined && explicitTemplateRaw !== "";
  const explicitTemplate = hasExplicitTemplate ? explicitTemplateRaw : "";
  const explicitSystemPromptRaw = pickRaw("LLAMA_SYSTEM_PROMPT", "systemPrompt");
  const hasExplicitSystemPrompt = explicitSystemPromptRaw !== undefined;
  const explicitSystemPrompt = hasExplicitSystemPrompt ? explicitSystemPromptRaw : "";
  const template = hasExplicitTemplate
    ? explicitTemplate
    : DEFAULT_RUNTIME.template;

  const baseConfig = {
    port: readIntSetting("PORT", "port", DEFAULT_RUNTIME.port, 1, 65535),
    repoRoot,
    webRoot,
    inferBin,
    modelPath: configuredModelPath,
    tokenizerPath: configuredTokenizerPath,
    template,
    forceCpu: readBoolSetting(
      "LLAMA_FORCE_CPU",
      "forceCpu",
      DEFAULT_RUNTIME.forceCpu
    ),
    systemPrompt: hasExplicitSystemPrompt
      ? explicitSystemPrompt
      : DEFAULT_RUNTIME.systemPrompt,
    maxNewTokens: readIntSetting(
      "LLAMA_MAX_NEW",
      "maxNewTokens",
      DEFAULT_RUNTIME.maxNewTokens,
      32
    ),
    maxContext: readIntSetting(
      "LLAMA_MAX_CONTEXT",
      "maxContext",
      DEFAULT_RUNTIME.maxContext,
      128
    ),
    temperature: readFloatSetting(
      "LLAMA_TEMPERATURE",
      "temperature",
      DEFAULT_RUNTIME.temperature,
      0
    ),
    topK: readIntSetting("LLAMA_TOP_K", "topK", DEFAULT_RUNTIME.topK, 0),
    topP: readFloatSetting("LLAMA_TOP_P", "topP", DEFAULT_RUNTIME.topP, 0, 1),
    repeatPenalty: readFloatSetting(
      "LLAMA_REPEAT_PENALTY",
      "repeatPenalty",
      DEFAULT_RUNTIME.repeatPenalty,
      1
    ),
    maxCpuPercent: readFloatSetting(
      "LLAMA_MAX_CPU_PERCENT",
      "maxCpuPercent",
      DEFAULT_RUNTIME.maxCpuPercent,
      1,
      100
    ),
    maxMemoryPercent: readFloatSetting(
      "LLAMA_MAX_MEMORY_PERCENT",
      "maxMemoryPercent",
      DEFAULT_RUNTIME.maxMemoryPercent,
      1,
      100
    ),
    resourceSampleMs: readIntSetting(
      "LLAMA_RESOURCE_SAMPLE_MS",
      "resourceSampleMs",
      DEFAULT_RUNTIME.resourceSampleMs,
      0
    ),
    resourceSustainMs: readIntSetting(
      "LLAMA_RESOURCE_SUSTAIN_MS",
      "resourceSustainMs",
      DEFAULT_RUNTIME.resourceSustainMs,
      1
    ),
    resourceThrottleMs: readIntSetting(
      "LLAMA_RESOURCE_THROTTLE_MS",
      "resourceThrottleMs",
      DEFAULT_RUNTIME.resourceThrottleMs,
      0
    ),
    extraArgs: splitArgs(
      pick("LLAMA_EXTRA_ARGS", "extraArgs", DEFAULT_RUNTIME.extraArgs)
    )
  };

  const availableProfiles = discoverModelProfiles(baseConfig);
  const preferredModelDir = getPreferredModelDir();
  const selectedProfile = chooseSelectedProfile(
    availableProfiles,
    configuredModelPath
  );
  const systemPrompt = hasExplicitSystemPrompt
    ? explicitSystemPrompt
    : (
        selectedProfile?.tokenizerUsesDefaultSystemPrompt === false &&
        !hasExplicitSystemPrompt
      )
        ? ""
        : baseConfig.systemPrompt;

  return {
    ...baseConfig,
    preferredModelDir,
    availableProfiles,
    selectedProfileId: selectedProfile?.id ?? "",
    selectedProfile,
    modelPath: selectedProfile?.modelPath ?? configuredModelPath,
    tokenizerPath: selectedProfile?.tokenizerPath ?? configuredTokenizerPath,
    template: hasExplicitTemplate ? template : (selectedProfile?.template ?? template),
    systemPrompt,
    ready: fs.existsSync(inferBin) && Boolean(selectedProfile?.ready)
  };
}

export function publicRuntimeSummary(config) {
  return {
    ready: config.ready,
    inferBin: config.inferBin,
    modelPath: config.modelPath,
    tokenizerPath: config.tokenizerPath,
    template: config.template,
    systemPrompt: config.systemPrompt,
    forceCpu: config.forceCpu,
    maxNewTokens: config.maxNewTokens,
    maxContext: config.maxContext,
    temperature: config.temperature,
    topK: config.topK,
    topP: config.topP,
    repeatPenalty: config.repeatPenalty,
    preferredModelDir: config.preferredModelDir,
    maxCpuPercent: config.maxCpuPercent,
    maxMemoryPercent: config.maxMemoryPercent,
    resourceSampleMs: config.resourceSampleMs,
    resourceSustainMs: config.resourceSustainMs,
    resourceThrottleMs: config.resourceThrottleMs,
    extraArgs: config.extraArgs,
    selectedProfileId: config.selectedProfileId,
    selectedProfile: config.selectedProfile
      ? publicModelProfile(config.selectedProfile)
      : null,
    availableProfiles: config.availableProfiles.map(publicModelProfile)
  };
}
