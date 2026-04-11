const STOP_SEQUENCES = {
  tinyllama:        ["</s>", "\nUser:"],
  "tinyllama-chatml": ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\nUser:"],
  llama2:   ["</s>", "[INST]", "<<SYS>>", "<</SYS>>", "\nRelevant context:", "\nCurrent user message:", "\nUser:", "\nAssistant:", "<|"],
  llama3:   ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
  llama4:   ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"],
  mistral:  ["</s>", "[INST]"],
  phi3:     ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>"],
  qwen2:    ["<|im_end|>", "<|im_start|>"],
  qwen3_5:  ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
  plain:    []
};

function sanitizeContent(value) {
  return String(value ?? "")
    .replace(/\r/g, "")
    .trim();
}

function normalizeUserContent(value, options = {}) {
  let text = sanitizeContent(value);
  if (options.template === "tinyllama" || options.template === "tinyllama-chatml") {
    text = text.replace(/^\s*whos\b/i, "Who is");
    text = text.replace(/^\s*whats\b/i, "What is");
    if (/^\s*(?:who|what)\s+is\b/i.test(text) && /\.\s*$/.test(text)) {
      text = text.replace(/\.\s*$/, "?");
    }
  }
  return text;
}

function trimConversation(messages, maxChars = 9000) {
  const safeMessages = [];
  let totalChars = 0;

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    totalChars += message.content.length;
    if (totalChars > maxChars && safeMessages.length > 0) {
      break;
    }
    safeMessages.unshift(message);
  }

  return safeMessages;
}

function wordCount(text = "") {
  return String(text || "").trim().split(/\s+/).filter(Boolean).length;
}

function significantTokens(text = "") {
  const stopwords = new Set([
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "whos",
    "why",
    "you"
  ]);

  return String(text || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .map((token) => token.trim())
    .filter((token) => token.length >= 2 && !stopwords.has(token));
}

function isIdentityQuestion(text = "") {
  return /^\s*(?:who(?:'s|\sis)?|whos)\b/i.test(String(text || "").trim());
}

function isDirectAssistantQuestion(text = "") {
  return /^\s*(?:who|what)\s+are\s+you\b/i.test(String(text || "").trim());
}

function shouldKeepTinyLlamaContext(text = "") {
  const raw = String(text || "").trim();
  if (!raw) return false;
  if (isIdentityQuestion(raw) || isDirectAssistantQuestion(raw)) return false;
  if (
    /^\s*(?:and|also|then|so|but|continue|go on|elaborate|tell me more|what about|how about)\b/i.test(raw)
  ) {
    return true;
  }
  if (/^\s*(?:why|how|when|where)\b/i.test(raw)) {
    return true;
  }

  const tokens = significantTokens(raw);
  return wordCount(raw) <= 2 || tokens.length === 0;
}

function compactAssistantHistory(text = "", maxChars = 220) {
  const clean = sanitizeContent(text).replace(/\s+/g, " ");
  if (!clean) return "";

  const sentences = clean.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [clean];
  let compact = "";

  for (const sentence of sentences) {
    const normalized = sentence.replace(/\s+/g, " ").trim();
    if (!normalized) continue;
    const next = compact ? `${compact} ${normalized}` : normalized;
    if (next.length > maxChars && compact) {
      break;
    }
    compact = next;
    if (compact.length >= Math.min(140, maxChars)) {
      break;
    }
  }

  if (!compact) {
    compact = clean;
  }

  if (compact.length <= maxChars) {
    return compact;
  }

  return compact.slice(0, maxChars).replace(/\s+\S*$/, "").trim();
}

function compactUserHistory(text = "", maxChars = 120) {
  const clean = sanitizeContent(text).replace(/\s+/g, " ");
  if (!clean) return "";
  if (clean.length <= maxChars) return clean;

  const sentence = (clean.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [clean])[0]
    .replace(/\s+/g, " ")
    .trim();
  if (sentence.length <= maxChars) {
    return sentence;
  }

  return clean.slice(0, maxChars).replace(/\s+\S*$/, "").trim();
}

function isGreetingOnly(text = "") {
  return /^(?:hey|hi|hello|hullo|yo|good\s+(?:morning|afternoon|evening))\b[!.?,\s]*$/i.test(
    String(text || "").trim()
  );
}

function looksGreetingReply(text = "") {
  return /^\s*(?:hey|hi|hello|greetings|good\s+(?:morning|afternoon|evening))\b/i.test(
    String(text || "").trim()
  );
}

function selectLlama2Messages(messages) {
  const turns = toTurns(messages, { maxTurns: 12 });
  if (turns.length === 0) {
    return messages;
  }

  const latestTurn = turns[turns.length - 1];
  const completedTurns = turns
    .slice(0, -1)
    .filter((turn) => turn.assistant)
    .filter(
      (turn) => !(isGreetingOnly(turn.user) && looksGreetingReply(turn.assistant))
    )
    .slice(-3);

  const focusedMessages = messages.filter((message) => message.role === "system").slice(-1);
  for (const turn of completedTurns) {
    focusedMessages.push({ role: "user", content: turn.user });
    focusedMessages.push({ role: "assistant", content: turn.assistant });
  }
  focusedMessages.push({ role: "user", content: latestTurn.user });
  if (latestTurn.assistant) {
    focusedMessages.push({ role: "assistant", content: latestTurn.assistant });
  }

  return focusedMessages;
}

function selectTinyLlamaMessages(messages) {
  const turns = toTurns(messages, { maxTurns: 32 });
  if (turns.length === 0) {
    return messages;
  }

  const latestTurn = turns[turns.length - 1];
  const includeContext = shouldKeepTinyLlamaContext(latestTurn.user);
  const selectedTurns = includeContext ? turns.slice(-2) : [latestTurn];
  const focusedMessages = messages.filter((message) => message.role === "system").slice(-1);

  for (const [index, turn] of selectedTurns.entries()) {
    focusedMessages.push({ role: "user", content: turn.user });
    if (turn.assistant) {
      focusedMessages.push({
        role: "assistant",
        content:
          index === selectedTurns.length - 1
            ? turn.assistant
            : compactAssistantHistory(turn.assistant)
      });
    }
  }

  return focusedMessages;
}

function selectHistory(messages, options = {}) {
  if (options.historyStrategy === "tinyllama-focused") {
    return selectTinyLlamaMessages(messages);
  }
  if (options.historyStrategy === "llama2-summary") {
    return selectLlama2Messages(messages);
  }
  return messages;
}

function normalizeMessages(messages, options = {}) {
  if (!Array.isArray(messages)) {
    return [];
  }
  const maxChars = Number.isFinite(Number(options.maxChars))
    ? Math.max(512, Number(options.maxChars))
    : 9000;

  const allowedRoles = new Set(["system", "user", "assistant"]);
  const cleaned = messages
    .map((message) => ({
      role: allowedRoles.has(message?.role) ? message.role : "user",
      content:
        (allowedRoles.has(message?.role) ? message.role : "user") === "user"
          ? normalizeUserContent(message?.content, options)
          : sanitizeContent(message?.content)
    }))
    .filter((message) => message.content.length > 0);

  return trimConversation(selectHistory(cleaned, options), maxChars);
}

function toTurns(messages, options = {}) {
  const maxTurns = Number.isFinite(Number(options.maxTurns))
    ? Math.max(1, Number(options.maxTurns))
    : 8;
  const turns = [];
  let pendingUser = "";

  for (const message of messages) {
    if (message.role === "system") {
      continue;
    }

    if (message.role === "user") {
      if (pendingUser) {
        turns.push({ user: pendingUser, assistant: "" });
      }
      pendingUser = message.content;
      continue;
    }

    if (!pendingUser) {
      continue;
    }

    turns.push({
      user: pendingUser,
      assistant: message.content
    });
    pendingUser = "";
  }

  if (pendingUser) {
    turns.push({ user: pendingUser, assistant: "" });
  }

  return turns.slice(-maxTurns);
}

function formatTinyLlama(turns, systemPrompt) {
  const lines = [];

  if (systemPrompt) {
    lines.push(systemPrompt);
  }

  for (const turn of turns) {
    lines.push(`User: ${turn.user}`);
    if (turn.assistant) {
      lines.push(`Assistant: ${turn.assistant}`);
    } else {
      lines.push("Assistant:");
    }
  }

  return lines.join("\n\n").trim();
}

function formatChatMl(turns, systemPrompt) {
  const blocks = [];

  if (systemPrompt) {
    blocks.push(`<|system|>\n${systemPrompt}</s>`);
  }

  for (const turn of turns) {
    blocks.push(`<|user|>\n${turn.user}</s>`);
    if (turn.assistant) {
      blocks.push(`<|assistant|>\n${turn.assistant}</s>`);
    } else {
      blocks.push("<|assistant|>");
    }
  }

  return blocks.join("\n");
}

function formatLlama2(turns, systemPrompt) {
  const defaultSystemPrompt =
    "You are a helpful assistant. Reply naturally to the user's latest message only. Use the recent context if it helps answer the latest message. Do not continue the transcript or add meta explanation unless the user asks for it.";

  const effectiveSystemPrompt = systemPrompt || defaultSystemPrompt;
  const currentTurn = turns[turns.length - 1] || { user: "Hello" };
  const contextTurns = turns
    .slice(0, -1)
    .filter((turn) => turn.assistant)
    .slice(-3);

  const userBlocks = [];
  if (contextTurns.length > 0) {
    userBlocks.push("Relevant context:");
    for (const turn of contextTurns) {
      userBlocks.push(`- User: ${compactUserHistory(turn.user)}`);
      userBlocks.push(`- Assistant: ${compactAssistantHistory(turn.assistant, 160)}`);
    }
  }
  userBlocks.push(`Current user message: ${compactUserHistory(currentTurn.user, 220)}`);
  userBlocks.push("Respond with only the assistant's answer.");

  return `[INST] <<SYS>>\n${effectiveSystemPrompt}\n<</SYS>>\n\n${userBlocks.join("\n")}\n[/INST]`;
}

function formatLlama4(turns, systemPrompt) {
  const messages = [{ role: "system", content: systemPrompt }];

  for (const turn of turns) {
    messages.push({ role: "user", content: turn.user });
    if (turn.assistant) {
      messages.push({ role: "assistant", content: turn.assistant });
    }
  }

  let prompt = "<|begin_of_text|>";
  for (const message of messages) {
    prompt += `<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content}<|eot_id|>`;
  }
  prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return prompt;
}

// Llama3 header format: identical structure to Llama4 but only used with
// the llama3 template name and <|eot_id|> stop token.
function formatLlama3(turns, systemPrompt) {
  return formatLlama4(turns, systemPrompt);
}

// Mistral instruction format: [INST] … [/INST]. No dedicated system-prompt
// token exists; the system prompt is prepended to the first user turn.
function formatMistral(turns, systemPrompt) {
  const blocks = [];
  for (const [i, turn] of turns.entries()) {
    const userContent = i === 0 && systemPrompt
      ? `${systemPrompt}\n\n${turn.user}`
      : turn.user;
    blocks.push(`[INST] ${userContent} [/INST]`);
    if (turn.assistant) {
      blocks.push(`${turn.assistant}</s>`);
    }
  }
  return blocks.join(" ").trim();
}

// Phi-3 instruction format.
function formatPhi3(turns, systemPrompt) {
  const blocks = [`<|system|>\n${systemPrompt}<|end|>`];
  for (const turn of turns) {
    blocks.push(`<|user|>\n${turn.user}<|end|>`);
    if (turn.assistant) {
      blocks.push(`<|assistant|>\n${turn.assistant}<|end|>`);
    } else {
      blocks.push("<|assistant|>\n");
    }
  }
  return blocks.join("\n");
}

// Qwen2 ChatML format.
function formatQwen2(turns, systemPrompt) {
  const blocks = [`<|im_start|>system\n${systemPrompt}<|im_end|>`];
  for (const turn of turns) {
    blocks.push(`<|im_start|>user\n${turn.user}<|im_end|>`);
    if (turn.assistant) {
      blocks.push(`<|im_start|>assistant\n${turn.assistant}<|im_end|>`);
    } else {
      blocks.push("<|im_start|>assistant\n");
    }
  }
  return blocks.join("\n");
}

function formatQwen35(turns, systemPrompt) {
  const blocks = [`<|im_start|>system\n${systemPrompt}<|im_end|>`];
  for (const turn of turns) {
    blocks.push(`<|im_start|>user\n${turn.user}<|im_end|>`);
    if (turn.assistant) {
      blocks.push(`<|im_start|>assistant\n${turn.assistant}<|im_end|>`);
    } else {
      blocks.push("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
  }
  return blocks.join("\n");
}

function formatPlain(turns, systemPrompt) {
  const lines = [`System: ${systemPrompt}`];

  for (const turn of turns) {
    lines.push(`User: ${turn.user}`);
    if (turn.assistant) {
      lines.push(`Assistant: ${turn.assistant}`);
    } else {
      lines.push("Assistant:");
    }
  }

  return lines.join("\n\n").trim();
}

export function buildPromptPackage(messages, options = {}) {
  const normalized = normalizeMessages(messages, options);
  const turns = toTurns(normalized, options);
  const template = options.template || "tinyllama";
  const hasExplicitSystemPrompt =
    Object.prototype.hasOwnProperty.call(options, "systemPrompt");
  const systemPrompt = sanitizeContent(options.systemPrompt);
  const effectiveSystemPrompt = hasExplicitSystemPrompt
    ? systemPrompt
    : (template === "tinyllama" ? "" : "You are a helpful assistant.");

  if (turns.length === 0) {
    return {
      messages: normalized,
      prompt: "",
      template,
      stopTexts: STOP_SEQUENCES[template] ?? STOP_SEQUENCES.plain,
      addBos: template === "plain"
    };
  }

  if (template === "tinyllama-chatml") {
    return {
      messages: normalized,
      prompt: formatChatMl(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: true
    };
  }

  if (template === "llama2") {
    return {
      messages: normalized,
      prompt: formatLlama2(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: true
    };
  }

  if (template === "llama3") {
    return {
      messages: normalized,
      prompt: formatLlama3(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.llama3,
      addBos: false
    };
  }

  if (template === "llama4") {
    return {
      messages: normalized,
      prompt: formatLlama4(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.llama4,
      addBos: false
    };
  }

  if (template === "mistral") {
    return {
      messages: normalized,
      prompt: formatMistral(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.mistral,
      addBos: false
    };
  }

  if (template === "phi3") {
    return {
      messages: normalized,
      prompt: formatPhi3(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.phi3,
      addBos: false
    };
  }

  if (template === "qwen2") {
    return {
      messages: normalized,
      prompt: formatQwen2(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.qwen2,
      addBos: false
    };
  }

  if (template === "qwen3_5") {
    return {
      messages: normalized,
      prompt: formatQwen35(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.qwen3_5,
      addBos: false
    };
  }

  if (template === "plain") {
    return {
      messages: normalized,
      prompt: formatPlain(turns, effectiveSystemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: true
    };
  }

  return {
    messages: normalized,
    prompt: formatTinyLlama(turns, effectiveSystemPrompt),
    template: "tinyllama",
    stopTexts: STOP_SEQUENCES.tinyllama,
    addBos: false
  };
}
