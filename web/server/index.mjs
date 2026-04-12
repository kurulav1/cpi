import dotenv from "dotenv";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { randomUUID } from "node:crypto";

import { getRuntimeConfig, publicRuntimeSummary, setPreferredModelDir } from "./config.mjs";
import {
  buildInternalBodyFromChatRequest,
  buildInternalBodyFromCompletionRequest,
  buildInternalBodyFromResponsesRequest,
  buildOpenAiChatCompletion,
  buildOpenAiCompletion,
  buildOpenAiResponseObject,
  buildOpenAiUsage,
  normalizeOpenAiStop,
  openAiErrorPayload,
  publicOpenAiModel,
  publicOpenAiModelId,
  sendOpenAiError
} from "./openai_compat.mjs";
import { buildPromptPackage } from "./prompting.mjs";

dotenv.config();

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "4mb" }));

// Only one generation runs at a time (single GPU). The worker process is kept
// warm across requests and restarted only when the selected profile changes.
let activeRequest = null;
let interactiveWorker = null;
let preferredHfWorker = null;
const quantOverrides = new Map(); // profileId -> "none" | "int8" | "int4"

// â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function writeNdjson(response, payload) {
  response.write(`${JSON.stringify(payload)}\n`);
}

function writeSse(response, data) {
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}

function generationPolicy(profile, template = "") {
  const family = String(profile?.family || "").toLowerCase();
  const normalizedTemplate = String(template || "").toLowerCase();

  if (family === "llama2" || normalizedTemplate === "llama2") {
    return {
      streamTextDeltas: true,
      historyStrategy: "llama2-summary",
      finalResponseExtractor: "llama2-answer-only",
      streamingExtractor: "llama2-answer-live"
    };
  }

  if (family === "tinyllama" || normalizedTemplate === "tinyllama" || normalizedTemplate === "tinyllama-chatml") {
    return {
      streamTextDeltas: true,
      historyStrategy: "tinyllama-focused",
      finalResponseExtractor: "default",
      streamingExtractor: "default"
    };
  }

  return {
    streamTextDeltas: true,
    historyStrategy: "default",
    finalResponseExtractor: "default",
    streamingExtractor: "default"
  };
}

function wordCount(text = "") {
  return String(text || "").trim().split(/\s+/).filter(Boolean).length;
}

function lastUserMessage(messages = []) {
  if (!Array.isArray(messages)) return "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === "user" && messages[i]?.content) {
      return String(messages[i].content);
    }
  }
  return "";
}

function instructionText(messages = []) {
  if (!Array.isArray(messages)) return "";
  return messages
    .filter((message) => message?.role === "system" || message?.role === "developer")
    .map((message) => String(message.content || ""))
    .join("\n");
}

function familyOutputBudget(profile) {
  const family = String(profile?.family || "").toLowerCase();
  if (family === "llama3" || family === "llama4") {
    return {
      brief: 96,
      factual: 192,
      structured: 320,
      code: 640,
      normal: 384,
      continuation: 512,
      long: 1536,
      longCode: 2048,
      longContinuation: 4096,
      cautiousLongContext: 384,
      hardCap: 4096
    };
  }
  if (family === "llama2") {
    return {
      brief: 96,
      factual: 160,
      structured: 256,
      code: 512,
      normal: 320,
      continuation: 384,
      long: 1024,
      longCode: 1024,
      longContinuation: 1536,
      cautiousLongContext: 320,
      hardCap: 2048
    };
  }
  if (family === "qwen3_5" || family === "qwen2") {
    return {
      brief: 96,
      factual: 160,
      structured: 256,
      code: 448,
      normal: 256,
      continuation: 320,
      long: 768,
      longCode: 768,
      longContinuation: 1024,
      cautiousLongContext: 256,
      hardCap: 1536
    };
  }
  return {
    brief: 96,
    factual: 192,
    structured: 320,
    code: 512,
    normal: 320,
    continuation: 384,
    long: 1024,
    longCode: 1024,
    longContinuation: 1536,
    cautiousLongContext: 320,
    hardCap: 2048
  };
}

function computeDynamicMaxNewTokens(body, profile, maxContext) {
  const latestUser = lastUserMessage(body.messages);
  const instructions = instructionText(body.messages);
  const combined = `${instructions}\n${latestUser}`.toLowerCase();
  const latestLower = latestUser.toLowerCase();
  const latestWords = wordCount(latestUser);
  const longFormMode = isTruthyFlag(body.longFormMode);
  const brief =
    /(one short sentence|short sentence|briefly|brief answer|concise|few words|one line|single word|only the answer|just the answer|exactly two bullet|two bullet)/i.test(combined);
  const structured =
    /(bullet|bullets|list|steps|outline|table|compare|pros and cons|json|markdown)/i.test(combined);
  const codeLike =
    /(code|function|class|script|bug|debug|stack|trace|error|regex|sql|python|javascript|typescript|c\+\+|implement|refactor)/i.test(combined);
  const continuation =
    /^(go on|continue|keep going|tell me more|elaborate|expand|more details|deeper|walk me through|carry on)\b/i.test(latestLower.trim());
  const explicitLong =
    longFormMode ||
    /(detailed|in depth|deep dive|thorough|comprehensive|full explanation|essay|long-form|long form|step by step)/i.test(combined);
  const shortFactual =
    latestWords > 0 &&
    latestWords <= 14 &&
    /^(who|what|when|where|why|how|is|are|can|could|does|do|did|explain)\b/i.test(latestLower.trim());
  const budget = familyOutputBudget(profile);
  let recommended = budget.normal;

  if (brief) {
    recommended = budget.brief;
  } else if (explicitLong) {
    if (continuation) {
      recommended = budget.longContinuation;
    } else if (codeLike) {
      recommended = budget.longCode;
    } else {
      recommended = budget.long;
    }
  } else if (codeLike) {
    recommended = budget.code;
  } else if (structured) {
    recommended = budget.structured;
  } else if (shortFactual) {
    recommended = budget.factual;
  } else if (continuation) {
    recommended = budget.continuation;
  }

  if (Number(maxContext) >= 8192 && !explicitLong) {
    recommended = Math.min(recommended, budget.cautiousLongContext);
  }

  return Math.round(clampNumber(recommended, 32, budget.hardCap, budget.normal));
}

function expandStopTexts(template, stopTexts = []) {
  const expanded = new Set(stopTexts);
  if (
    template === "tinyllama" ||
    template === "tinyllama-chatml" ||
    template === "plain"
  ) {
    for (const marker of [
      "\nUser:",
      "\nuser:",
      "\nUSER:",
      "\nBot:",
      "\nbot:",
      "\nassistant|>",
      "\nAssistant|>",
      "\nAssistant:",
      "\nassistant:",
      "\nASSISTANT:",
      "Bot:",
      "bot:",
      "assistant|>",
      "Assistant|>",
      "user:",
      "User:",
      "assistant:",
      "Assistant:"
    ]) {
      expanded.add(marker);
    }
  }
  return [...expanded];
}

function normalizeGeneratedChatText(text, template) {
  let cleaned = String(text || "").replace(/\r/g, "");

  if (
    template === "tinyllama" ||
    template === "tinyllama-chatml" ||
    template === "plain"
  ) {
    cleaned = cleaned.replace(
      /^\s*(?:>\s*)?(?:<\|assistant\|>\s*|assistant\|+\>\s*|assistant\|>\s*|assistant\s*:)?\s*/i,
      ""
    );

    const roleMarker =
      /(?:^|\n)\s*(?:>\s*)?(?:<\|user\|>|<\|system\|>|<\|assistant\|>|assistant\|+\>|assistant\|>|user\s*:|assistant\s*:|system\s*:|bot\s*:)/i;
    const match = cleaned.match(roleMarker);
    if (match && typeof match.index === "number") {
      cleaned = cleaned.slice(0, match.index);
    }
  }

  if (template === "llama2") {
    const answerMatches = [...cleaned.matchAll(/(?:^|\n)\s*Answer:\s*/gi)];
    if (answerMatches.length > 0) {
      const lastAnswer = answerMatches[answerMatches.length - 1];
      const answerStart = (lastAnswer.index ?? 0) + lastAnswer[0].length;
      cleaned = cleaned.slice(answerStart);
    } else if (/(?:^|\n)\s*Question:\s*/i.test(cleaned)) {
      cleaned = "";
    }

    const followupMarker = /(?:^|\n)\s*(?:Question:|User:|System:)/i;
    const match = cleaned.match(followupMarker);
    if (match && typeof match.index === "number" && match.index >= 0) {
      cleaned = cleaned.slice(0, match.index);
    }

    cleaned = cleaned.replace(
      /\n\s*(?:Q(?:u(?:e(?:s(?:t(?:i(?:o(?:n?)?)?)?)?)?)?)?|U(?:s(?:e(?:r?)?)?)?|S(?:y(?:s(?:t(?:e(?:m?)?)?)?)?)?)\s*$/i,
      ""
    );
  }

  // Tiny models often drift into a second sentence and then get cut mid-phrase.
  // If we have at least one complete sentence, drop any trailing incomplete tail.
  const lastTerminal = Math.max(
    cleaned.lastIndexOf("."),
    cleaned.lastIndexOf("!"),
    cleaned.lastIndexOf("?")
  );
  if (lastTerminal >= 0 && lastTerminal + 1 < cleaned.length) {
    const trailing = cleaned.slice(lastTerminal + 1).trim();
    if (trailing && !/[.!?]$/.test(trailing)) {
      cleaned = cleaned.slice(0, lastTerminal + 1);
    }
  }

  return cleaned.trim();
}

function cleanupRepeatedWords(text) {
  let cleaned = String(text || "");
  for (let i = 0; i < 4; i += 1) {
    const next = cleaned.replace(/\b([A-Za-z][A-Za-z'-]*)\b(?:\s+\1\b)+/gi, "$1");
    if (next === cleaned) break;
    cleaned = next;
  }
  for (let i = 0; i < 4; i += 1) {
    const next = cleaned.replace(/\b([A-Za-z]{3,})\1\b/gi, "$1");
    if (next === cleaned) break;
    cleaned = next;
  }
  cleaned = cleaned.replace(/([A-Za-z])\1{4,}/g, "$1");
  cleaned = cleaned.replace(/([!?.,])\1+/g, "$1");
  cleaned = cleaned.replace(/\s{2,}/g, " ").trim();
  cleaned = cleaned.replace(/^["'`]+|["'`]+$/g, "").trim();
  return cleaned;
}

function cleanupRepeatedSentences(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return "";

  const sentences = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [cleaned];
  const deduped = [];
  let previousKey = "";

  for (const sentence of sentences) {
    const normalized = sentence.replace(/\s+/g, " ").trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (key === previousKey) {
      continue;
    }
    deduped.push(normalized);
    previousKey = key;
  }

  return deduped.join(" ").trim();
}

function normalizeWhitespace(text = "") {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function collapseAdjacentNearDuplicateWords(text = "") {
  const tokens = String(text || "").trim().split(/\s+/).filter(Boolean);
  const normalizedAlpha = (token) => token.toLowerCase().replace(/[^a-z]/g, "");
  const deduped = [];

  for (const token of tokens) {
    if (deduped.length === 0) {
      deduped.push(token);
      continue;
    }

    const prev = deduped[deduped.length - 1];
    const prevNorm = normalizedAlpha(prev);
    const tokenNorm = normalizedAlpha(token);
    const prefixLike =
      prevNorm.length >= 3 &&
      tokenNorm.length >= 3 &&
      (prevNorm === tokenNorm ||
        prevNorm.startsWith(tokenNorm) ||
        tokenNorm.startsWith(prevNorm));

    if (!prefixLike) {
      deduped.push(token);
      continue;
    }

    if (tokenNorm.length >= prevNorm.length) {
      deduped[deduped.length - 1] = token;
    }
  }

  return deduped.join(" ").trim();
}

function trimIncompleteTrailingTail(text = "") {
  const cleaned = String(text || "").trim();
  const lastTerminal = Math.max(
    cleaned.lastIndexOf("."),
    cleaned.lastIndexOf("!"),
    cleaned.lastIndexOf("?")
  );
  if (lastTerminal < 0 || lastTerminal + 1 >= cleaned.length) {
    return cleaned;
  }

  const trailing = cleaned.slice(lastTerminal + 1).trim();
  if (trailing && !/[.!?]$/.test(trailing)) {
    return cleaned.slice(0, lastTerminal + 1).trim();
  }
  return cleaned;
}

function extractLlama2AnswerOnly(text = "") {
  let cleaned = String(text || "").replace(/\r/g, "").trim();
  if (!cleaned) return "";

  const instMarker = cleaned.lastIndexOf("[/INST]");
  if (instMarker >= 0) {
    cleaned = cleaned.slice(instMarker + "[/INST]".length).trim();
  }

  const promptEcho = cleaned.match(
    /(?:^|\n)\s*(?:Relevant context:|Current user message:|Current question:|Question:|Instructions:|\[INST\]|<<SYS>>)/i
  );
  if (promptEcho && typeof promptEcho.index === "number" && promptEcho.index === 0) {
    cleaned = "";
  }

  cleaned = cleaned.replace(/^\s*(?:Answer|Assistant)\s*:?\s*/i, "");
  cleaned = cleaned.replace(/^\s*(?:<\/?s>|<<SYS>>|<<\/SYS>>|\[INST\]|\[\/INST\])\s*/gi, "");
  cleaned = cleaned.replace(/^[\s:;,.!?-]+/, "");

  const followupMarker = cleaned.match(
    /(?:^|\n)\s*(?:Relevant context:|Current user message:|Current question:|Question:|User:|System:|Instructions:|Explanation\s*:|Answer\s*:|\[INST\]|<<SYS>>)/i
  );
  if (followupMarker && typeof followupMarker.index === "number" && followupMarker.index > 0) {
    cleaned = cleaned.slice(0, followupMarker.index).trim();
  }

  cleaned = cleaned.replace(
    /\n\s*(?:Relevant|Current|Question|User|System|Instructions|Explanation|Answer|Assistant)\s*:?.*$/i,
    ""
  );
  cleaned = cleanupRepeatedWords(cleaned);
  cleaned = collapseAdjacentNearDuplicateWords(cleaned);
  cleaned = cleanupRepeatedSentences(cleaned);
  cleaned = trimIncompleteTrailingTail(cleaned);
  return normalizeWhitespace(cleaned);
}

function extractLlama2StreamingAnswer(text = "") {
  let cleaned = String(text || "").replace(/\r/g, "");
  if (!cleaned) return "";

  const instMarker = cleaned.lastIndexOf("[/INST]");
  if (instMarker >= 0) {
    cleaned = cleaned.slice(instMarker + "[/INST]".length);
  }

  const answerMatches = [...cleaned.matchAll(/(?:^|\n)\s*Answer\s*:*/gi)];
  if (answerMatches.length > 0) {
    const lastAnswer = answerMatches[answerMatches.length - 1];
    const answerStart = (lastAnswer.index ?? 0) + lastAnswer[0].length;
    cleaned = cleaned.slice(answerStart);
  } else {
    const promptEcho = cleaned.match(
      /(?:^|\n)\s*(?:Relevant context:|Current user message:|Current question:|Question:|Instructions:|\[INST\]|<<SYS>>)/i
    );
    if (promptEcho && typeof promptEcho.index === "number" && promptEcho.index === 0) {
      return "";
    }
  }

  cleaned = cleaned.replace(/^\s*(?:Answer|Assistant)\s*:?\s*/i, "");
  cleaned = cleaned.replace(/^\s*(?:<\/?s>|<<SYS>>|<<\/SYS>>|\[INST\]|\[\/INST\])\s*/gi, "");
  cleaned = cleaned.replace(/^[\s:;,-]+/, "");

  const followupMarker = cleaned.match(
    /(?:^|\n)\s*(?:Relevant context:|Current user message:|Current question:|Question:|User:|System:|Instructions:|\[INST\]|<<SYS>>)/i
  );
  if (followupMarker && typeof followupMarker.index === "number" && followupMarker.index > 0) {
    cleaned = cleaned.slice(0, followupMarker.index);
  }

  cleaned = cleaned.replace(
    /\n\s*(?:Relevant|Current|Question|User|System|Instructions|Answer|Assistant)\s*:?\s*$/i,
    ""
  );
  cleaned = cleanupRepeatedWords(cleaned);
  cleaned = collapseAdjacentNearDuplicateWords(cleaned);
  return cleaned.replace(/[ \t]{2,}/g, " ").replace(/^\s+/, "");
}

function normalizeStreamingResponseText(text, template, policy = null) {
  const effectivePolicy = policy ?? generationPolicy(null, template);
  if (effectivePolicy.streamingExtractor === "llama2-answer-live") {
    return extractLlama2StreamingAnswer(text);
  }
  return normalizeGeneratedChatText(text, template);
}

function normalizeFinalResponseText(text, template, policy = null) {
  const effectivePolicy = policy ?? generationPolicy(null, template);
  if (effectivePolicy.finalResponseExtractor === "llama2-answer-only") {
    return extractLlama2AnswerOnly(text);
  }
  return cleanupRepeatedSentences(
    cleanupRepeatedWords(normalizeGeneratedChatText(text, template))
  );
}

function isTinyLlamaFamily(cliConfig) {
  return cliConfig?.profile?.family === "tinyllama";
}

function prefersHfChatBackend(target) {
  void target;
  return false;
}

function shouldStreamTextDeltas(cliConfigOrTemplate) {
  if (typeof cliConfigOrTemplate === "string") {
    return generationPolicy(null, cliConfigOrTemplate).streamTextDeltas;
  }
  if (
    cliConfigOrTemplate &&
    typeof cliConfigOrTemplate === "object" &&
    Object.prototype.hasOwnProperty.call(cliConfigOrTemplate, "streamTextDeltas")
  ) {
    return cliConfigOrTemplate.streamTextDeltas !== false;
  }
  return generationPolicy(
    cliConfigOrTemplate?.profile,
    cliConfigOrTemplate?.meta?.template || cliConfigOrTemplate?.template || ""
  ).streamTextDeltas;
}

function isGreetingPrompt(text = "") {
  return /^(?:hey|hi|hello|hullo|yo)\b/i.test(String(text || "").trim());
}

function looksLikeGreetingReply(text = "") {
  return /\b(?:hey|hi|hello|greetings)\b/i.test(String(text || "").trim());
}

function hasRoleLeakage(text = "") {
  return /<\|(?:user|assistant|system)\|>|(?:^|\n)\s*(?:user|assistant|system|bot)\s*:/i.test(
    String(text || "")
  );
}

function looksMalformedTinyLlamaReply(text = "") {
  const cleaned = String(text || "").trim();
  if (!cleaned) return true;
  if (/[|<>]{2,}/.test(cleaned)) return true;
  if (/[A-Za-z]\|[A-Za-z]/.test(cleaned)) return true;
  if (/^(?:[:;>|.\-]{2,}|\W{3,})/.test(cleaned)) return true;
  return false;
}

function tinyLlamaRetrySystemPrompt(userMessage = "") {
  if (isGreetingPrompt(userMessage)) {
    return "Reply to the user's greeting with exactly one short greeting sentence.";
  }
  return "Answer the user's last message in one short factual sentence. Focus only on the latest user question. Do not use role labels.";
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

function computePromptBudget(maxContext, performanceMode = false) {
  const resolvedContext = Math.max(128, Number(maxContext) || 0);
  const contextRatio = Math.max(1, resolvedContext / 2048);
  const maxTurns = performanceMode
    ? Math.max(4, Math.min(12, Math.round(4 * Math.sqrt(contextRatio))))
    : Math.max(8, Math.min(24, Math.round(8 * Math.sqrt(contextRatio))));
  const maxChars = performanceMode
    ? Math.max(3500, Math.round(resolvedContext * 4.5))
    : Math.max(9000, Math.round(resolvedContext * 5.5));
  return { maxTurns, maxChars };
}

function shouldAutoLiftResourceLimits(profile, maxContext) {
  const family = String(profile?.family || "").toLowerCase();
  const requestedContext = Math.max(0, Number(maxContext) || 0);
  const maxPositionEmbeddings = Math.max(0, Number(profile?.maxPositionEmbeddings) || 0);
  return (
    requestedContext >= 8192 &&
    maxPositionEmbeddings >= 8192 &&
    (family === "llama3" || family === "llama4" || family === "qwen3_5")
  );
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
    const detail = selectedProfile.unsupportedReason
      ? ` ${selectedProfile.unsupportedReason}`
      : "";
    throw new Error(`Model is not runnable: ${selectedProfile.label}.${detail}`);
  }

  const performanceMode = isTruthyFlag(body.performanceMode);
  const quantMode = resolveQuantModeForProfile(selectedProfile, body.quantMode);
  const profileExtraArgs = buildProfileExtraArgs(selectedProfile, quantMode);
  const forceCpu = isTruthyFlag(body.forceCpu) || config.forceCpu;
  const requestTemplate = body.template || selectedProfile.template || config.template;
  const requestSystemPrompt = body.systemPrompt !== undefined
    ? body.systemPrompt
    : (requestTemplate === "tinyllama" ? "" : config.systemPrompt);
  const policy = generationPolicy(selectedProfile, requestTemplate);
  const maxSupportedContext = Math.max(
    128,
    Number(selectedProfile.maxPositionEmbeddings) || 0,
    Number(config.maxContext) || 0
  );
  const maxContext = Math.round(
    clampNumber(body.maxContext, 128, maxSupportedContext, config.maxContext)
  );
  const autoMaxTokens = body.autoMaxTokens === undefined
    ? true
    : isTruthyFlag(body.autoMaxTokens);
  const longFormMode = isTruthyFlag(body.longFormMode);
  const promptBudget = computePromptBudget(maxContext, performanceMode);

  const promptPackage = buildPromptPackage(body.messages, {
    template: requestTemplate,
    systemPrompt: requestSystemPrompt,
    historyStrategy: policy.historyStrategy,
    maxTurns: promptBudget.maxTurns,
    maxChars: promptBudget.maxChars
  });

  if (!promptPackage.prompt) {
    throw new Error("Provide at least one user message.");
  }

  const maxNewTokens = autoMaxTokens
    ? computeDynamicMaxNewTokens(body, selectedProfile, maxContext)
    : Math.round(
        clampNumber(body.maxNewTokens ?? body.max_tokens, 32, 4096, config.maxNewTokens)
      );
  const temperature = clampNumber(body.temperature, 0, 2, config.temperature);
  const noResourceLimits =
    performanceMode ||
    shouldAutoLiftResourceLimits(selectedProfile, maxContext) ||
    profileExtraArgs.includes("--no-resource-limits");

  return {
    profile: selectedProfile,
    profileExtraArgs,
    quantMode,
    messages: promptPackage.messages,
    prompt: promptPackage.prompt,
    addBos: promptPackage.addBos,
    stopTexts: expandStopTexts(
      promptPackage.template,
      [...promptPackage.stopTexts, ...normalizeOpenAiStop(body.stop ?? body.stopTexts)]
    ),
    policy,
    maxContext,
    noResourceLimits,
    performanceMode,
    forceCpu,
    workerKey: `${selectedProfile.id}|ctx:${maxContext}|perf:${performanceMode ? 1 : 0}|q:${quantMode}|cpu:${forceCpu ? 1 : 0}`,
    meta: {
      profileId: selectedProfile.id,
      modelLabel: selectedProfile.label,
      template: promptPackage.template,
      messageCount: promptPackage.messages.length,
      maxNewTokens,
      maxContext,
      autoMaxTokens,
      longFormMode,
      temperature,
      performanceMode,
      quantMode,
      forceCpu
    }
  };
}

function buildWorkerCliConfig(profile, options = {}) {
  const performanceMode = isTruthyFlag(options.performanceMode);
  const quantMode = resolveQuantModeForProfile(profile, options.quantMode);
  const profileExtraArgs = buildProfileExtraArgs(profile, quantMode);
  const policy = generationPolicy(profile, profile?.template || "");
  const maxSupportedContext = Math.max(
    128,
    Number(profile?.maxPositionEmbeddings) || 0,
    Number(options.maxContext) || 0
  );
  const maxContext = Math.round(
    clampNumber(options.maxContext, 128, maxSupportedContext, maxSupportedContext)
  );
  const noResourceLimits =
    performanceMode ||
    shouldAutoLiftResourceLimits(profile, maxContext) ||
    profileExtraArgs.includes("--no-resource-limits");
  const forceCpu = isTruthyFlag(options.forceCpu);
  return {
    profile,
    profileExtraArgs,
    quantMode,
    policy,
    maxContext,
    noResourceLimits,
    performanceMode,
    forceCpu,
    workerKey: `${profile.id}|ctx:${maxContext}|perf:${performanceMode ? 1 : 0}|q:${quantMode}|cpu:${forceCpu ? 1 : 0}`,
    meta: {
      profileId: profile.id,
      modelLabel: profile.label,
      template: profile.template,
      messageCount: 0,
      maxNewTokens: 0,
      maxContext,
      temperature: 0,
      performanceMode,
      quantMode,
      forceCpu
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
  const forceCpu = cliConfig.forceCpu || config.forceCpu;
  return [
    selectedProfile.modelPath,
    "--tokenizer", selectedProfile.tokenizerPath,
    "--max-context", String(cliConfig.maxContext ?? config.maxContext),
    "--top-k", String(config.topK),
    "--top-p", String(config.topP),
    "--repeat-penalty", String(config.repeatPenalty),
    "--interactive",
    "--web",
    "--runtime-metrics",
    ...(forceCpu ? ["--cpu"] : []),
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

function resolvePreferredHfModelDir(profile) {
  const modelPath = profile?.modelPath;
  if (!modelPath) return "";

  const candidate = path.resolve(path.dirname(modelPath), "hf");
  const hasConfig = fs.existsSync(path.join(candidate, "config.json"));
  const hasTokenizer = ["tokenizer.json", "tokenizer.model"]
    .some((file) => fs.existsSync(path.join(candidate, file)));
  const hasWeights = [
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json"
  ].some((file) => fs.existsSync(path.join(candidate, file)));

  return hasConfig && hasTokenizer && hasWeights
    ? candidate
    : "";
}

function parsePreferredHfStdout(worker) {
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

      if (event?.type === "ready") {
        worker.ready = true;
        worker.readyResolve?.();
      } else if (worker.pending && event) {
        const pending = worker.pending;
        if (event.id && event.id !== pending.id) {
          // Ignore stale output from a previous request.
        } else if (event.type === "done") {
          worker.pending = null;
          pending.resolve({
            text: typeof event.text === "string" ? event.text : "",
            elapsedMs: Number.isFinite(Number(event.elapsed_ms))
              ? Number(event.elapsed_ms)
              : Date.now() - pending.startedAt,
            generatedTokens: Number.isFinite(Number(event.generated_tokens))
              ? Number(event.generated_tokens)
              : null,
            tokPerS: Number.isFinite(Number(event.tok_per_s))
              ? Number(event.tok_per_s)
              : null,
            metrics: null,
            ...pending.meta
          });
        } else if (event.type === "error") {
          worker.pending = null;
          pending.reject(event.error || "hf worker error");
        }
      }
    }

    nl = worker.stdoutBuffer.indexOf("\n");
  }
}

function spawnPreferredHfWorker(config, cliConfig) {
  const modelDir = resolvePreferredHfModelDir(cliConfig.profile);
  const scriptPath = path.resolve(config.repoRoot, "tools", "hf_chat_worker.py");
  const child = spawn(pythonBin(), [scriptPath, "--model-dir", modelDir], {
    cwd: config.repoRoot,
    env: {
      ...process.env,
      PYTHONUTF8: "1"
    }
  });

  let readyResolve = null;
  let readyReject = null;
  const readyPromise = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });

  const worker = {
    modelDir,
    child,
    ready: false,
    readyPromise,
    readyResolve,
    readyReject,
    stdoutDecoder: new StringDecoder("utf8"),
    stderrDecoder: new StringDecoder("utf8"),
    stdoutBuffer: "",
    stderrText: "",
    pending: null
  };

  child.stdout.on("data", (chunk) => {
    worker.stdoutBuffer += worker.stdoutDecoder.write(chunk);
      parsePreferredHfStdout(worker);
  });

  child.stderr.on("data", (chunk) => {
    worker.stderrText += worker.stderrDecoder.write(chunk);
  });

  child.stdin.on("error", (err) => {
    if (!worker.pending) return;
    const pending = worker.pending;
    worker.pending = null;
    pending.reject(`Failed to send request to HF worker: ${err.message}`);
  });

  child.on("close", (code, signal) => {
    worker.stdoutBuffer += worker.stdoutDecoder.end();
    parsePreferredHfStdout(worker);
    worker.stderrText += worker.stderrDecoder.end();

    if (!worker.ready) {
      worker.readyReject?.(
        worker.stderrText.trim() ||
        `HF worker exited with code ${code}${signal ? ` (${signal})` : ""}.`
      );
    }

    if (worker.pending) {
      const pending = worker.pending;
      worker.pending = null;
      pending.reject(
        worker.stderrText.trim() ||
        `HF worker exited with code ${code}${signal ? ` (${signal})` : ""}.`
      );
    }

      if (preferredHfWorker === worker) {
        preferredHfWorker = null;
      }
  });

  child.on("error", (err) => {
    if (!worker.ready) {
      worker.readyReject?.(`Failed to launch HF worker: ${err.message}`);
    }
    if (worker.pending) {
      const pending = worker.pending;
      worker.pending = null;
      pending.reject(`Failed to launch HF worker: ${err.message}`);
    }
      if (preferredHfWorker === worker) {
        preferredHfWorker = null;
      }
  });

  return worker;
}

async function ensurePreferredHfWorker(config, cliConfig) {
  const modelDir = resolvePreferredHfModelDir(cliConfig.profile);
  if (!modelDir) {
    return null;
  }

  if (!preferredHfWorker || preferredHfWorker.modelDir !== modelDir) {
    killWorker(preferredHfWorker, true);
    preferredHfWorker = spawnPreferredHfWorker(config, cliConfig);
  }

  await preferredHfWorker.readyPromise;
  return preferredHfWorker;
}

async function runPreferredHfGeneration(config, cliConfig) {
  const worker = await ensurePreferredHfWorker(config, cliConfig);
  if (!worker) {
    throw new Error("Preferred HF backend is not available.");
  }
  if (worker.pending) {
    throw new Error("Preferred HF worker is busy.");
  }

  const requestId = randomUUID();
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    worker.pending = {
      id: requestId,
      startedAt,
      meta: cliConfig.meta,
      resolve,
      reject
    };

    const payload = {
      id: requestId,
      messages: cliConfig.messages,
      max_new: cliConfig.meta.maxNewTokens,
      temperature: cliConfig.meta.temperature
    };

    worker.child.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
      if (!err) return;
      if (worker.pending?.id === requestId) {
        worker.pending = null;
      }
      reject(new Error(`Failed to send request to HF worker: ${err.message}`));
    });
  });
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
          if (event.delta) {
            pending.rawText = `${pending.rawText || ""}${event.delta}`;
            if (!shouldStreamTextDeltas(pending.policy || pending.meta?.template)) {
              nl = worker.stdoutBuffer.indexOf("\n");
              continue;
            }
            const cleaned = normalizeStreamingResponseText(
              pending.rawText,
              pending.meta?.template,
              pending.policy
            );
            const emitted = pending.emittedText || "";
            if (cleaned.startsWith(emitted)) {
              const nextDelta = cleaned.slice(emitted.length);
              pending.emittedText = cleaned;
              if (nextDelta) {
                pending.onDelta?.(nextDelta);
              }
            } else if (emitted.startsWith(cleaned)) {
              // Wait for the normalized prefix to stabilize again.
            } else if (!cleaned) {
              // Ignore transient scaffolding until we have stable answer text.
            }
          }
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
          const decodeMs = Number.isFinite(Number(event.decode_ms))
            ? Number(event.decode_ms)
            : null;
          const tokPerSFromEvent = Number(event.tok_per_s);
          const tokPerS = Number.isFinite(tokPerSFromEvent)
            ? tokPerSFromEvent
            : (generatedTokens != null && elapsedMs > 0
                ? (1000.0 * generatedTokens) / elapsedMs
                : null);
          const decodeTokPerSFromEvent = Number(event.decode_tok_per_s);
          const decodeTokPerS = Number.isFinite(decodeTokPerSFromEvent)
            ? decodeTokPerSFromEvent
            : (generatedTokens != null && decodeMs != null && decodeMs > 0
                ? (1000.0 * generatedTokens) / decodeMs
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
          pending.resolve({
            text,
            elapsedMs,
            generatedTokens,
            tokPerS,
            decodeMs,
            decodeTokPerS,
            metrics,
            ...pending.meta
          });
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
    workerKey:
      cliConfig.workerKey ??
      `${cliConfig.meta.profileId}|perf:0|q:${cliConfig.meta.quantMode || "none"}|cpu:${cliConfig.meta.forceCpu ? 1 : 0}`,
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
      const exitMessage =
        worker.stderrText.trim() ||
        `interactive worker exited with code ${code}${signal ? ` (${signal})` : ""}.`;
      if (pending.handleWorkerExit?.(exitMessage)) {
        worker.pending = null;
      } else {
        worker.pending = null;
        pending.reject(exitMessage);
      }
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
    cliConfig.workerKey ??
    `${cliConfig.meta.profileId}|perf:0|q:${cliConfig.meta.quantMode || "none"}|cpu:${cliConfig.meta.forceCpu ? 1 : 0}`;
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
    throw new Error("No model available for preparation.");
  }
  if (!profile.ready) {
    const detail = profile.unsupportedReason ? ` ${profile.unsupportedReason}` : "";
    throw new Error(`Model is not runnable: ${profile.label}.${detail}`);
  }
  return profile;
}

async function warmupProfileWorker(config, profileId, options = {}) {
  const profile = resolveWarmupProfile(config, profileId);

  const workerCliConfig = buildWorkerCliConfig(profile, options);
  if (prefersHfChatBackend(profile) && resolvePreferredHfModelDir(profile)) {
    activeRequest = { kind: "warmup", cancel: () => {} };
    try {
      await ensurePreferredHfWorker(config, workerCliConfig);
      return profile;
    } finally {
      activeRequest = null;
    }
  }

  const worker = ensureInteractiveWorker(config, workerCliConfig);
  if (worker.ready) {
    return profile;
  }

  // Preparation should mean "first token ready soon", not just "process spawned".
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

async function generatePreferredFamilyResponse(config, cliConfig) {
  if (isTinyLlamaFamily(cliConfig)) {
    return generateWithTinyLlamaRetry(config, cliConfig);
  }

  const result = await runPreferredFamilyGenerationOnce(config, cliConfig, "generation");
  return {
    ...result,
    text: normalizeFinalResponseText(result.text, cliConfig.meta.template, cliConfig.policy)
  };
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

  const payload = {
    id: requestId,
    prompt: cliConfig.prompt,
    max_new: cliConfig.meta.maxNewTokens,
    temp: cliConfig.meta.temperature,
    stop_texts: cliConfig.stopTexts,
    add_bos: cliConfig.addBos,
    // TinyLlama often starts with a short lead-in sentence before the real
    // answer. Cutting on the first sentence boundary turns sane answers into
    // fragments like "Sure, who is Elon Musk?".
    sentence_stop: false
  };

  let requestAttempt = 0;
  const maxAutoRetries = 1;

  const canRetryWorkerExit = (pending) =>
    requestKind === "generation" &&
    requestAttempt < maxAutoRetries &&
    !settled &&
    pending &&
    !pending.rawText &&
    !pending.emittedText;

  const sendPayload = (targetWorker) => {
    if (
      !targetWorker.child.stdin ||
      targetWorker.child.stdin.destroyed ||
      targetWorker.child.stdin.writableEnded
    ) {
      if (targetWorker.pending?.id === requestId) {
        targetWorker.pending = null;
      }
      settle(() => onError("Failed to send request to worker: stdin is closed."));
      return;
    }

    targetWorker.child.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
      if (!err) return;
      if (targetWorker.pending?.id === requestId) {
        targetWorker.pending = null;
      }
      settle(() => onError(`Failed to send request to worker: ${err.message}`));
    });
  };

  worker.pending = {
    id: requestId,
    startedAt: Date.now(),
    meta: cliConfig.meta,
    policy: cliConfig.policy,
    rawText: "",
    emittedText: "",
    onDelta,
    onMetrics,
    handleWorkerExit: (message) => {
      const pending = worker?.pending;
      if (!canRetryWorkerExit(pending)) {
        return false;
      }

      requestAttempt += 1;
      console.warn(
        `[cpi] worker exited before producing output for ${cliConfig.meta.modelLabel}; retrying once (${message})`
      );

      try {
        const replacementWorker = ensureInteractiveWorker(config, cliConfig);
        worker = replacementWorker;
      } catch (err) {
        settle(() => onError(err.message || String(err)));
        return true;
      }

      worker.pending = {
        ...pending,
        startedAt: Date.now(),
        handleWorkerExit: pending.handleWorkerExit
      };
      sendPayload(worker);
      return true;
    },
    resolve: (result) =>
      settle(() =>
        onDone({
          ...result,
          text: normalizeFinalResponseText(
            result.text,
            cliConfig.meta.template,
            cliConfig.policy
          )
        })
      ),
    reject: (msg) => settle(() => onError(msg))
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
    sendPayload(worker);
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

function runGenerationOnce(config, cliConfig, requestKind = "generation") {
  return new Promise((resolve, reject) => {
    runGeneration(config, cliConfig, {
      requestKind,
      onStart: () => {},
      onDelta: () => {},
      onMetrics: () => {},
      onDone: (result) => resolve(result),
      onError: (msg) => reject(new Error(msg))
    });
  });
}

async function runPreferredFamilyGenerationOnce(config, cliConfig, requestKind = "generation") {
  const hfModelDir =
    prefersHfChatBackend(cliConfig) ? resolvePreferredHfModelDir(cliConfig.profile) : "";
  if (hfModelDir) {
    try {
      return await runPreferredHfGeneration(config, cliConfig);
    } catch (err) {
      const family = String(cliConfig?.profile?.family || cliConfig?.meta?.template || "model");
      console.warn(`[${family}] HF backend failed, using llama_infer: ${err.message}`);
    }
  }

  return runGenerationOnce(config, cliConfig, requestKind);
}

async function generateWithTinyLlamaRetry(config, cliConfig) {
  const lastUserMessage =
    [...(cliConfig.messages || [])].reverse().find((message) => message.role === "user")?.content || "";
  const greetingPrompt = isGreetingPrompt(lastUserMessage);
  const firstResult = await runPreferredFamilyGenerationOnce(config, cliConfig, "generation");
  const firstText = normalizeFinalResponseText(firstResult.text, cliConfig.meta.template);
  const needsRetry =
    isTinyLlamaFamily(cliConfig) &&
    (
      !firstText ||
      hasRoleLeakage(firstText) ||
      looksMalformedTinyLlamaReply(firstText) ||
      (greetingPrompt && (!looksLikeGreetingReply(firstText) || wordCount(firstText) > 12))
    );

  if (!needsRetry) {
    return { ...firstResult, text: firstText };
  }

  if (interactiveWorker) {
    killWorker(interactiveWorker, true);
    interactiveWorker = null;
    sleepMs(120);
  }

  const retryPackage = buildPromptPackage(cliConfig.messages || [], {
    template: cliConfig.meta.template,
    systemPrompt: tinyLlamaRetrySystemPrompt(lastUserMessage),
    historyStrategy: "tinyllama-focused",
    maxTurns: cliConfig.performanceMode ? 4 : 8,
    maxChars: cliConfig.performanceMode ? 3500 : 9000
  });
  const retryCliConfig = {
    ...cliConfig,
    prompt: retryPackage.prompt,
    addBos: retryPackage.addBos,
    stopTexts: expandStopTexts(retryPackage.template, retryPackage.stopTexts),
    messages: retryPackage.messages,
    meta: {
      ...cliConfig.meta,
      maxNewTokens: Math.min(
        cliConfig.meta.maxNewTokens,
        greetingPrompt ? 24 : 64
      )
    }
  };
  const retryResult = await runPreferredFamilyGenerationOnce(config, retryCliConfig, "generation");
  const retryText = normalizeFinalResponseText(retryResult.text, retryCliConfig.meta.template);
  if (greetingPrompt && (!looksLikeGreetingReply(retryText) || wordCount(retryText) > 12)) {
    return { ...retryResult, text: "Hello!" };
  }
  return {
    ...retryResult,
    text: retryText
  };
}

// â”€â”€ request guards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function guardReady(config, res) {
  if (!config.ready) {
    const detail = config.selectedProfile?.unsupportedReason
      ? ` ${config.selectedProfile.unsupportedReason}`
      : "";
    res.status(503).json({
      error:
        `Runtime not ready. Check config.json (or .env): inferBin, modelPath, tokenizerPath.${detail}`
    });
    return false;
  }
  return true;
}

function guardOpenAiReady(config, res) {
  if (!config.ready) {
    const detail = config.selectedProfile?.unsupportedReason
      ? ` ${config.selectedProfile.unsupportedReason}`
      : "";
    sendOpenAiError(
      res,
      503,
      `Runtime not ready. Check config.json (or .env): inferBin, modelPath, tokenizerPath.${detail}`,
      { type: "server_error", code: "runtime_not_ready" }
    );
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
        ? "Engine is still preparing. Please retry in a moment."
        : "Engine is busy with another generation.";
    res.status(409).json({ error: reason });
    return false;
  }
  return true;
}

async function guardOpenAiIdle(
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
        ? "Engine is still preparing. Please retry in a moment."
        : "Engine is busy with another generation.";
    sendOpenAiError(res, 409, reason, {
      type: "server_error",
      code: activeRequest.kind === "warmup" ? "engine_preparing" : "engine_busy"
    });
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
    return { profileId: "", maxContext: 0, performanceMode: false, quantMode: "none", forceCpu: false };
  }
  const parts = workerKey.split("|");
  const profileId = parts[0] || "";
  let maxContext = 0;
  let performanceMode = false;
  let quantMode = "none";
  let forceCpu = false;
  for (let i = 1; i < parts.length; i += 1) {
    const part = parts[i];
    if (part.startsWith("ctx:")) {
      maxContext = Number(part.slice(4)) || 0;
      continue;
    }
    if (part.startsWith("perf:")) {
      performanceMode = part.slice(5) === "1";
      continue;
    }
    if (part.startsWith("q:")) {
      quantMode = normalizeQuantMode(part.slice(2));
      continue;
    }
    if (part.startsWith("cpu:")) {
      forceCpu = part.slice(4) === "1";
    }
  }
  return { profileId, maxContext, performanceMode, quantMode, forceCpu };
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
    const maxContext = Number(body.maxContext) || config.maxContext;
    const profile = resolveWarmupProfile(config, body.profileId);
    const targetWorkerKey = buildWorkerCliConfig(profile, { performanceMode, quantMode, maxContext }).workerKey;

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

    // If a generation is already running, report preparation as pending instead of
    // failing with 409; the UI can keep showing a non-error preparing state.
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

    // Kick preparation in background so this endpoint stays responsive even when
    // first-token preparation takes tens of seconds on larger models.
    void warmupProfileWorker(config, profile.id, { performanceMode, quantMode, maxContext }).catch((err) => {
      console.warn(
        `[cpi] preparation failed for ${profile.label}: ${err?.message || String(err)}`
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

  if (prefersHfChatBackend(cliConfig)) {
    try {
      const result = await generatePreferredFamilyResponse(config, cliConfig);
      res.json({ ...result, ...cliConfig.meta });
    } catch (err) {
      if (!res.headersSent) res.status(500).json({ error: err.message || String(err) });
    }
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onDelta: () => {},
    onDone: (result) => res.json({ ...result, ...cliConfig.meta }),
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

  if (prefersHfChatBackend(cliConfig)) {
    writeNdjson(res, { type: "start", ...cliConfig.meta });
    try {
      const result = await generatePreferredFamilyResponse(config, cliConfig);
      if (shouldStreamTextDeltas(cliConfig.policy) && result.text) {
        writeNdjson(res, { type: "delta", delta: result.text });
      }
      writeNdjson(res, {
        type: "done",
        elapsedMs: result.elapsedMs,
        generatedTokens: result.generatedTokens,
        tokPerS: result.tokPerS,
        decodeMs: result.decodeMs ?? null,
        decodeTokPerS: result.decodeTokPerS ?? null,
        metrics: result.metrics || null,
        message: result.text
      });
    } catch (err) {
      writeNdjson(res, { type: "error", error: err.message || String(err) });
    }
    res.end();
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onStart: (meta) => writeNdjson(res, { type: "start", ...meta }),
    onDelta: (delta) => writeNdjson(res, { type: "delta", delta }),
    onMetrics: (metrics) => writeNdjson(res, { type: "metrics", metrics }),
    onDone: ({ text, elapsedMs, generatedTokens, tokPerS, decodeMs, decodeTokPerS, metrics }) => {
      writeNdjson(res, {
        type: "done",
        elapsedMs,
        generatedTokens,
        tokPerS,
        decodeMs,
        decodeTokPerS,
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

// GET /v1/models
// OpenAI-compatible model listing.
app.get("/v1/models", (_req, res) => {
  const config = getRuntimeConfig();
  res.json({
    object: "list",
    data: publicRuntimeSummary(config).availableProfiles
      .filter((profile) => profile.ready)
      .map(publicOpenAiModel)
  });
});

app.get("/v1/models/:model", (req, res) => {
  const config = getRuntimeConfig();
  const profile = findProfileByIdOrLabel(config, req.params.model);
  if (!profile || !profile.ready) {
    sendOpenAiError(res, 404, `Model '${req.params.model}' was not found.`, {
      type: "invalid_request_error",
      code: "model_not_found"
    });
    return;
  }
  res.json(publicOpenAiModel(profile));
});

// POST /v1/completions
// OpenAI-compatible legacy completions endpoint.
app.post("/v1/completions", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardOpenAiReady(config, res)) return;
  if (!(await guardOpenAiIdle(res, { waitForWarmup: true }))) return;

  const body = requestBody(req);
  const stream = Boolean(body.stream);
  const includeUsage = Boolean(body.stream_options?.include_usage);
  const completionId = `cmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  let cliConfig;
  let internalBody;
  try {
    internalBody = buildInternalBodyFromCompletionRequest(body);
    cliConfig = buildCliArgs(config, internalBody);
  } catch (err) {
    sendOpenAiError(res, 400, err.message, { type: "invalid_request_error" });
    return;
  }

  const modelName = publicOpenAiModelId(cliConfig.profile);

  if (stream) {
    setStreamingHeaders(res, "text/event-stream; charset=utf-8");

    if (internalBody.echo && internalBody.promptText) {
      writeSse(res, {
        id: completionId,
        object: "text_completion",
        created,
        model: modelName,
        system_fingerprint: openAiSystemFingerprint(cliConfig),
        choices: [{ index: 0, text: internalBody.promptText, logprobs: null, finish_reason: null }]
      });
    }

    if (prefersHfChatBackend(cliConfig)) {
      try {
        const result = await generatePreferredFamilyResponse(config, cliConfig);
        const text = result.text;
        if (shouldStreamTextDeltas(cliConfig.policy) && text) {
          writeSse(res, {
            id: completionId,
            object: "text_completion",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [{ index: 0, text, logprobs: null, finish_reason: null }]
          });
        }
        if (includeUsage) {
          writeSse(res, {
            id: completionId,
            object: "text_completion",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [],
            usage: buildOpenAiUsage(result, cliConfig)
          });
        }
        writeSse(res, {
          id: completionId,
          object: "text_completion",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, text: "", logprobs: null, finish_reason: "stop" }]
        });
        res.write("data: [DONE]\n\n");
        res.end();
      } catch (err) {
        writeSse(res, openAiErrorPayload(err.message || String(err), "server_error"));
        res.end();
      }
      return;
    }

    const { cancel } = runGeneration(config, cliConfig, {
      onDelta: (delta) =>
        writeSse(res, {
          id: completionId,
          object: "text_completion",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, text: delta, logprobs: null, finish_reason: null }]
        }),
      onDone: (result) => {
        if (includeUsage) {
          writeSse(res, {
            id: completionId,
            object: "text_completion",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [],
            usage: buildOpenAiUsage(result, cliConfig)
          });
        }
        writeSse(res, {
          id: completionId,
          object: "text_completion",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, text: "", logprobs: null, finish_reason: "stop" }]
        });
        res.write("data: [DONE]\n\n");
        res.end();
      },
      onError: (msg) => {
        writeSse(res, openAiErrorPayload(msg, "server_error"));
        res.end();
      }
    });

    onDisconnect(req, res, cancel);
    return;
  }

  if (prefersHfChatBackend(cliConfig)) {
    try {
      const result = await generatePreferredFamilyResponse(config, cliConfig);
      res.json(
        buildOpenAiCompletion(completionId, created, modelName, cliConfig, result, internalBody)
      );
    } catch (err) {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, err.message || String(err), { type: "server_error" });
      }
    }
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onDelta: () => {},
    onDone: (result) =>
      res.json(
        buildOpenAiCompletion(completionId, created, modelName, cliConfig, result, internalBody)
      ),
    onError: (msg) => {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, msg, { type: "server_error" });
      }
    }
  });

  onDisconnect(req, res, cancel);
});

// POST /v1/chat/completions
// OpenAI-compatible chat completions endpoint.
app.post("/v1/chat/completions", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardOpenAiReady(config, res)) return;
  if (!(await guardOpenAiIdle(res, { waitForWarmup: true }))) return;

  const body = requestBody(req);
  const stream = Boolean(body.stream);
  const includeUsage = Boolean(body.stream_options?.include_usage);
  const completionId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  let cliConfig;
  let internalBody;
  try {
    internalBody = buildInternalBodyFromChatRequest(body);
    cliConfig = buildCliArgs(config, internalBody);
  } catch (err) {
    sendOpenAiError(res, 400, err.message, { type: "invalid_request_error" });
    return;
  }

  const modelName = publicOpenAiModelId(cliConfig.profile);

  if (stream) {
    setStreamingHeaders(res, "text/event-stream; charset=utf-8");

    writeSse(res, {
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model: modelName,
      system_fingerprint: openAiSystemFingerprint(cliConfig),
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
    });

    if (prefersHfChatBackend(cliConfig)) {
      try {
        const result = await generatePreferredFamilyResponse(config, cliConfig);
        if (shouldStreamTextDeltas(cliConfig.policy) && result.text) {
          writeSse(res, {
            id: completionId,
            object: "chat.completion.chunk",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }]
          });
        }
        if (includeUsage) {
          writeSse(res, {
            id: completionId,
            object: "chat.completion.chunk",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [],
            usage: buildOpenAiUsage(result, cliConfig)
          });
        }
        writeSse(res, {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
        });
        res.write("data: [DONE]\n\n");
        res.end();
      } catch (err) {
        writeSse(res, openAiErrorPayload(err.message || String(err), "server_error"));
        res.end();
      }
      return;
    }

    const { cancel } = runGeneration(config, cliConfig, {
      onDelta: (delta) =>
        writeSse(res, {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, delta: { content: delta }, finish_reason: null }]
        }),
      onDone: (result) => {
        if (includeUsage) {
          writeSse(res, {
            id: completionId,
            object: "chat.completion.chunk",
            created,
            model: modelName,
            system_fingerprint: openAiSystemFingerprint(cliConfig),
            choices: [],
            usage: buildOpenAiUsage(result, cliConfig)
          });
        }
        writeSse(res, {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          system_fingerprint: openAiSystemFingerprint(cliConfig),
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
        });
        res.write("data: [DONE]\n\n");
        res.end();
      },
      onError: (msg) => {
        writeSse(res, openAiErrorPayload(msg, "server_error"));
        res.end();
      }
    });

    onDisconnect(req, res, cancel);
    return;
  }

  if (prefersHfChatBackend(cliConfig)) {
    try {
      const result = await generatePreferredFamilyResponse(config, cliConfig);
      res.json(buildOpenAiChatCompletion(completionId, created, modelName, cliConfig, result));
    } catch (err) {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, err.message || String(err), { type: "server_error" });
      }
    }
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onDelta: () => {},
    onDone: (result) =>
      res.json(buildOpenAiChatCompletion(completionId, created, modelName, cliConfig, result)),
    onError: (msg) => {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, msg, { type: "server_error" });
      }
    }
  });

  onDisconnect(req, res, cancel);
});

// POST /v1/responses
// Minimal OpenAI-compatible responses endpoint for text generation clients.
app.post("/v1/responses", async (req, res) => {
  const config = getRuntimeConfig();
  if (!guardOpenAiReady(config, res)) return;
  if (!(await guardOpenAiIdle(res, { waitForWarmup: true }))) return;

  const body = requestBody(req);
  const stream = Boolean(body.stream);
  const responseId = `resp_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  let cliConfig;
  let internalBody;
  try {
    internalBody = buildInternalBodyFromResponsesRequest(body);
    cliConfig = buildCliArgs(config, internalBody);
  } catch (err) {
    sendOpenAiError(res, 400, err.message, { type: "invalid_request_error" });
    return;
  }

  const modelName = publicOpenAiModelId(cliConfig.profile);

  if (stream) {
    setStreamingHeaders(res, "text/event-stream; charset=utf-8");
    const baseResponse = {
      id: responseId,
      object: "response",
      created_at: created,
      status: "in_progress",
      model: modelName
    };
    writeSse(res, { type: "response.created", response: baseResponse });

    if (prefersHfChatBackend(cliConfig)) {
      try {
        const result = await generatePreferredFamilyResponse(config, cliConfig);
        const itemId = `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
        if (shouldStreamTextDeltas(cliConfig.policy) && result.text) {
          writeSse(res, {
            type: "response.output_text.delta",
            item_id: itemId,
            output_index: 0,
            content_index: 0,
            delta: result.text
          });
        }
        writeSse(res, {
          type: "response.output_text.done",
          item_id: itemId,
          output_index: 0,
          content_index: 0,
          text: result.text
        });
        writeSse(res, {
          type: "response.completed",
          response: buildOpenAiResponseObject(responseId, created, modelName, cliConfig, result, internalBody)
        });
        res.write("data: [DONE]\n\n");
        res.end();
      } catch (err) {
        writeSse(res, openAiErrorPayload(err.message || String(err), "server_error"));
        res.end();
      }
      return;
    }

    const itemId = `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
    const { cancel } = runGeneration(config, cliConfig, {
      onDelta: (delta) =>
        writeSse(res, {
          type: "response.output_text.delta",
          item_id: itemId,
          output_index: 0,
          content_index: 0,
          delta
        }),
      onDone: (result) => {
        writeSse(res, {
          type: "response.output_text.done",
          item_id: itemId,
          output_index: 0,
          content_index: 0,
          text: result.text
        });
        writeSse(res, {
          type: "response.completed",
          response: buildOpenAiResponseObject(responseId, created, modelName, cliConfig, result, internalBody)
        });
        res.write("data: [DONE]\n\n");
        res.end();
      },
      onError: (msg) => {
        writeSse(res, openAiErrorPayload(msg, "server_error"));
        res.end();
      }
    });

    onDisconnect(req, res, cancel);
    return;
  }

  if (prefersHfChatBackend(cliConfig)) {
    try {
      const result = await generatePreferredFamilyResponse(config, cliConfig);
      res.json(
        buildOpenAiResponseObject(responseId, created, modelName, cliConfig, result, internalBody)
      );
    } catch (err) {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, err.message || String(err), { type: "server_error" });
      }
    }
    return;
  }

  const { cancel } = runGeneration(config, cliConfig, {
    onDelta: () => {},
    onDone: (result) =>
      res.json(
        buildOpenAiResponseObject(responseId, created, modelName, cliConfig, result, internalBody)
      ),
    onError: (msg) => {
      if (!res.headersSent) {
        sendOpenAiError(res, 500, msg, { type: "server_error" });
      }
    }
  });

  onDisconnect(req, res, cancel);
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

// POST /api/system/model-dir
app.post("/api/system/model-dir", (req, res) => {
  try {
    const { dir } = requestBody(req);
    const preferredModelDir =
      typeof dir === "string" ? setPreferredModelDir(dir) : setPreferredModelDir("");
    const config = getRuntimeConfig();
    res.json({
      ok: true,
      preferredModelDir,
      models: publicRuntimeSummary(config).availableProfiles
    });
  } catch (err) {
    res.status(500).json({ error: err.message || "Unable to update model directory." });
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
  if (outputDir) {
    setPreferredModelDir(outputDir);
    args.push("--output-dir", outputDir);
  }
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
