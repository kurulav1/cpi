import { randomUUID } from "node:crypto";

function toFiniteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function estimateTokenCount(text) {
  const clean = String(text || "").trim();
  if (!clean) return 0;
  return Math.max(1, Math.ceil(clean.length / 4));
}

export function openAiErrorPayload(message, type = "invalid_request_error", param = null, code = null) {
  return {
    error: {
      message: String(message || "Request failed."),
      type,
      ...(param ? { param } : {}),
      ...(code ? { code } : {})
    }
  };
}

export function sendOpenAiError(res, status, message, options = {}) {
  res.status(status).json(
    openAiErrorPayload(message, options.type, options.param, options.code)
  );
}

export function normalizeOpenAiContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return content == null ? "" : String(content);
  }

  const parts = [];
  for (const item of content) {
    if (typeof item === "string") {
      if (item.trim()) parts.push(item);
      continue;
    }
    if (!item || typeof item !== "object") {
      continue;
    }

    const type = String(item.type || "").toLowerCase();
    if (
      type === "text" ||
      type === "input_text" ||
      type === "output_text"
    ) {
      const text = typeof item.text === "string"
        ? item.text
        : (typeof item.content === "string" ? item.content : "");
      if (text.trim()) parts.push(text);
      continue;
    }

    if (type === "refusal" && typeof item.refusal === "string" && item.refusal.trim()) {
      parts.push(item.refusal);
    }
  }

  return parts.join("\n\n");
}

export function normalizeOpenAiMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages
    .map((message) => {
      const rawRole = String(message?.role || "user").toLowerCase();
      const role = rawRole === "developer"
        ? "developer"
        : (rawRole === "system" || rawRole === "assistant" || rawRole === "user"
            ? rawRole
            : "user");
      return {
        role,
        content: normalizeOpenAiContent(message?.content)
      };
    })
    .filter((message) => message.content.trim().length > 0);
}

export function normalizeOpenAiPrompt(prompt) {
  if (Array.isArray(prompt)) {
    if (prompt.length === 0) return "";
    if (prompt.length > 1) {
      throw new Error("Only a single prompt is supported per request.");
    }
    return normalizeOpenAiContent(prompt[0]);
  }
  return normalizeOpenAiContent(prompt);
}

export function normalizeOpenAiStop(stop) {
  if (Array.isArray(stop)) {
    return stop
      .map((value) => String(value ?? "").trim())
      .filter(Boolean);
  }
  if (stop == null || stop === "") {
    return [];
  }
  return [String(stop)];
}

function responseFormatInstruction(responseFormat) {
  if (!responseFormat || typeof responseFormat !== "object") {
    return "";
  }

  const type = String(responseFormat.type || "").toLowerCase();
  if (!type || type === "text") {
    return "";
  }
  if (type === "json_object") {
    return "Return a single valid JSON object and no surrounding prose.";
  }
  if (type === "json_schema") {
    const schema = responseFormat.json_schema?.schema;
    const schemaText = schema ? JSON.stringify(schema) : "";
    return schemaText
      ? `Return JSON that matches this schema: ${schemaText}`
      : "Return valid JSON that matches the requested schema.";
  }
  throw new Error(`Unsupported response_format type '${responseFormat.type}'.`);
}

function validateOpenAiRequestShape(body, mode = "chat") {
  const n = Number(body?.n);
  if (Number.isFinite(n) && n > 1) {
    throw new Error("Only n=1 is supported.");
  }

  if (body?.tools && Array.isArray(body.tools) && body.tools.length > 0) {
    throw new Error("Tool calling is not supported yet.");
  }
  if (body?.functions && Array.isArray(body.functions) && body.functions.length > 0) {
    throw new Error("Function calling is not supported yet.");
  }
  if (body?.tool_choice && body.tool_choice !== "none" && body.tool_choice !== "auto") {
    throw new Error("tool_choice is not supported yet.");
  }
  if (body?.function_call && body.function_call !== "none" && body.function_call !== "auto") {
    throw new Error("function_call is not supported yet.");
  }
  if (body?.audio || (Array.isArray(body?.modalities) && body.modalities.some((value) => String(value).toLowerCase() === "audio"))) {
    throw new Error("Audio output is not supported.");
  }
  if (mode === "completion" && body?.suffix) {
    throw new Error("suffix is not supported.");
  }
}

export function buildInternalBodyFromChatRequest(body) {
  validateOpenAiRequestShape(body, "chat");
  const messages = normalizeOpenAiMessages(body.messages);
  const extraInstruction = responseFormatInstruction(body.response_format);
  const internalMessages = extraInstruction
    ? [{ role: "developer", content: extraInstruction }, ...messages]
    : messages;

  return {
    profileId: body.model,
    messages: internalMessages,
    systemPrompt: "",
    temperature: body.temperature,
    max_tokens: body.max_tokens,
    stop: body.stop,
    maxContext: body.max_context ?? body.maxContext,
    autoMaxTokens: body.autoMaxTokens,
    longFormMode: body.longFormMode,
    performanceMode: body.performanceMode,
    quantMode: body.quantMode,
    forceCpu: body.forceCpu
  };
}

export function buildInternalBodyFromCompletionRequest(body) {
  validateOpenAiRequestShape(body, "completion");
  const prompt = normalizeOpenAiPrompt(body.prompt);
  const extraInstruction = responseFormatInstruction(body.response_format);
  const messages = [];
  if (extraInstruction) {
    messages.push({ role: "developer", content: extraInstruction });
  }
  if (prompt.trim()) {
    messages.push({ role: "user", content: prompt });
  }

  return {
    profileId: body.model,
    messages,
    systemPrompt: "",
    temperature: body.temperature,
    max_tokens: body.max_tokens,
    stop: body.stop,
    maxContext: body.max_context ?? body.maxContext,
    autoMaxTokens: body.autoMaxTokens,
    longFormMode: body.longFormMode,
    performanceMode: body.performanceMode,
    quantMode: body.quantMode,
    forceCpu: body.forceCpu,
    echo: Boolean(body.echo),
    promptText: prompt
  };
}

function normalizeResponsesInput(input) {
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }
  if (!Array.isArray(input)) {
    return input && typeof input === "object"
      ? normalizeOpenAiMessages([input])
      : [];
  }

  const messages = [];
  for (const item of input) {
    if (typeof item === "string") {
      if (item.trim()) messages.push({ role: "user", content: item });
      continue;
    }
    if (!item || typeof item !== "object") {
      continue;
    }

    if (item.type === "message") {
      messages.push(...normalizeOpenAiMessages([item]));
      continue;
    }

    if (
      item.type === "input_text" ||
      item.type === "text" ||
      typeof item.text === "string"
    ) {
      const text = normalizeOpenAiContent(item);
      if (text.trim()) messages.push({ role: "user", content: text });
      continue;
    }

    if (item.role || item.content) {
      messages.push(...normalizeOpenAiMessages([item]));
    }
  }

  return messages;
}

export function buildInternalBodyFromResponsesRequest(body) {
  validateOpenAiRequestShape(body, "responses");
  const instructions = normalizeOpenAiContent(body.instructions);
  const inputMessages = normalizeResponsesInput(body.input);
  const extraInstruction = responseFormatInstruction(body.text?.format || body.response_format);
  const messages = [];
  if (instructions.trim()) {
    messages.push({ role: "developer", content: instructions });
  }
  if (extraInstruction) {
    messages.push({ role: "developer", content: extraInstruction });
  }
  messages.push(...inputMessages);

  return {
    profileId: body.model,
    messages,
    systemPrompt: "",
    temperature: body.temperature,
    max_tokens: body.max_output_tokens ?? body.max_tokens,
    stop: body.stop,
    maxContext: body.max_context ?? body.maxContext,
    autoMaxTokens: body.autoMaxTokens,
    longFormMode: body.longFormMode,
    performanceMode: body.performanceMode,
    quantMode: body.quantMode,
    forceCpu: body.forceCpu,
    instructions
  };
}

export function publicOpenAiModelId(profile) {
  return String(profile?.label || profile?.id || "model");
}

export function publicOpenAiModel(profile) {
  const id = publicOpenAiModelId(profile);
  return {
    id,
    object: "model",
    created: 0,
    owned_by: "cpi",
    permission: [],
    root: id,
    parent: null
  };
}

export function buildOpenAiUsage(result, cliConfig) {
  const promptTokens =
    toFiniteNumber(result?.metrics?.prompt_tokens) ??
    estimateTokenCount(cliConfig?.prompt);
  const completionTokens =
    toFiniteNumber(result?.generatedTokens) ??
    toFiniteNumber(result?.metrics?.generated_tokens) ??
    estimateTokenCount(result?.text);
  const totalTokens = (promptTokens || 0) + (completionTokens || 0);
  return {
    prompt_tokens: promptTokens || 0,
    completion_tokens: completionTokens || 0,
    total_tokens: totalTokens
  };
}

function buildOpenAiResponsesUsage(result, cliConfig) {
  const usage = buildOpenAiUsage(result, cliConfig);
  return {
    input_tokens: usage.prompt_tokens,
    input_tokens_details: {
      cached_tokens: 0
    },
    output_tokens: usage.completion_tokens,
    output_tokens_details: {
      reasoning_tokens: 0
    },
    total_tokens: usage.total_tokens
  };
}

function openAiSystemFingerprint(cliConfig) {
  const raw = String(cliConfig?.workerKey || cliConfig?.profile?.modelPath || "cpi");
  return `fp_${Buffer.from(raw).toString("base64url").slice(0, 12)}`;
}

export function buildOpenAiChatCompletion(completionId, created, modelName, cliConfig, result) {
  return {
    id: completionId,
    object: "chat.completion",
    created,
    model: modelName,
    system_fingerprint: openAiSystemFingerprint(cliConfig),
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        finish_reason: "stop"
      }
    ],
    usage: buildOpenAiUsage(result, cliConfig)
  };
}

export function buildOpenAiCompletion(completionId, created, modelName, cliConfig, result, options = {}) {
  const echoedText = options.echo && options.promptText
    ? `${options.promptText}${result.text}`
    : result.text;
  return {
    id: completionId,
    object: "text_completion",
    created,
    model: modelName,
    system_fingerprint: openAiSystemFingerprint(cliConfig),
    choices: [
      {
        text: echoedText,
        index: 0,
        logprobs: null,
        finish_reason: "stop"
      }
    ],
    usage: buildOpenAiUsage(result, cliConfig)
  };
}

export function buildOpenAiResponseObject(responseId, created, modelName, cliConfig, result, options = {}) {
  const outputText = result.text || "";
  return {
    id: responseId,
    object: "response",
    created_at: created,
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: options.instructions || null,
    max_output_tokens: cliConfig.meta.maxNewTokens,
    model: modelName,
    output: [
      {
        id: `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: outputText,
            annotations: []
          }
        ]
      }
    ],
    output_text: outputText,
    temperature: cliConfig.meta.temperature,
    usage: buildOpenAiResponsesUsage(result, cliConfig)
  };
}
