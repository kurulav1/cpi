const STOP_SEQUENCES = {
  tinyllama:        ["</s>", "\nUser:"],
  "tinyllama-chatml": ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\nUser:"],
  llama2:   ["</s>", "[INST]", "<<SYS>>", "</", "<|"],
  llama3:   ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
  llama4:   ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"],
  mistral:  ["</s>", "[INST]"],
  phi3:     ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>"],
  qwen2:    ["<|im_end|>", "<|im_start|>"],
  plain:    []
};

function sanitizeContent(value) {
  return String(value ?? "")
    .replace(/\r/g, "")
    .trim();
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
      content: sanitizeContent(message?.content)
    }))
    .filter((message) => message.content.length > 0);

  return trimConversation(cleaned, maxChars);
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
  const lines = [systemPrompt];

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
  const blocks = [`<|system|>\n${systemPrompt}</s>`];

  for (const turn of turns) {
    blocks.push(`<|user|>\n${turn.user}</s>`);
    if (turn.assistant) {
      blocks.push(`<|assistant|>\n${turn.assistant}</s>`);
    } else {
      blocks.push("<|assistant|>\n");
    }
  }

  return blocks.join("\n");
}

function formatLlama2(turns, systemPrompt) {
  if (turns.length === 0) {
    return `[INST] <<SYS>>\n${systemPrompt}\n<</SYS>>\n\nHello [/INST]`;
  }

  const [firstTurn, ...rest] = turns;
  let prompt = `<s>[INST] <<SYS>>\n${systemPrompt}\n<</SYS>>\n\n${firstTurn.user} [/INST]`;

  if (firstTurn.assistant) {
    prompt += ` ${firstTurn.assistant} </s>`;
  }

  for (const turn of rest) {
    prompt += ` <s>[INST] ${turn.user} [/INST]`;
    if (turn.assistant) {
      prompt += ` ${turn.assistant} </s>`;
    }
  }

  return prompt.trim();
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
  const systemPrompt =
    sanitizeContent(options.systemPrompt) ||
    "You are a helpful assistant.";

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
      prompt: formatChatMl(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: false
    };
  }

  if (template === "llama2") {
    return {
      messages: normalized,
      prompt: formatLlama2(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: false
    };
  }

  if (template === "llama3") {
    return {
      messages: normalized,
      prompt: formatLlama3(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.llama3,
      addBos: false
    };
  }

  if (template === "llama4") {
    return {
      messages: normalized,
      prompt: formatLlama4(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.llama4,
      addBos: false
    };
  }

  if (template === "mistral") {
    return {
      messages: normalized,
      prompt: formatMistral(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.mistral,
      addBos: false
    };
  }

  if (template === "phi3") {
    return {
      messages: normalized,
      prompt: formatPhi3(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.phi3,
      addBos: false
    };
  }

  if (template === "qwen2") {
    return {
      messages: normalized,
      prompt: formatQwen2(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES.qwen2,
      addBos: false
    };
  }

  if (template === "plain") {
    return {
      messages: normalized,
      prompt: formatPlain(turns, systemPrompt),
      template,
      stopTexts: STOP_SEQUENCES[template],
      addBos: true
    };
  }

  return {
    messages: normalized,
    prompt: formatTinyLlama(turns, systemPrompt),
    template: "tinyllama",
    stopTexts: STOP_SEQUENCES.tinyllama,
    addBos: false
  };
}
