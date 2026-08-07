// Reference chat descriptors -- the DATA that ships with each model.
// Kept side-effect free so tooling can import them to author model descriptors.
// chat_spec_parity.test.mjs proves these render byte-identically to the legacy
// hand-written formatters.

export const CHAT_SPECS = {
  "tinyllama-chatml": {
    join: "\n", addBos: true,
    system: { mode: "block", prefix: "<|system|>\n", suffix: "</s>", emitWhenEmpty: false },
    user: { prefix: "<|user|>\n", suffix: "</s>" },
    assistant: { prefix: "<|assistant|>\n", suffix: "</s>" },
    generationPrompt: "<|assistant|>"
  },
  qwen2: {
    join: "\n", addBos: false,
    system: { mode: "block", prefix: "<|im_start|>system\n", suffix: "<|im_end|>", emitWhenEmpty: true },
    user: { prefix: "<|im_start|>user\n", suffix: "<|im_end|>" },
    assistant: { prefix: "<|im_start|>assistant\n", suffix: "<|im_end|>" },
    generationPrompt: "<|im_start|>assistant\n"
  },
  phi3: {
    join: "\n", addBos: false,
    system: { mode: "block", prefix: "<|system|>\n", suffix: "<|end|>", emitWhenEmpty: true },
    user: { prefix: "<|user|>\n", suffix: "<|end|>" },
    assistant: { prefix: "<|assistant|>\n", suffix: "<|end|>" },
    generationPrompt: "<|assistant|>\n"
  },
  gemma: {
    join: "", addBos: true,
    system: { mode: "fold", foldSeparator: "\n\n" },
    user: { prefix: "<|turn>user\n", suffix: "<turn|>\n" },
    assistant: { prefix: "<|turn>model\n", suffix: "<turn|>\n" },
    generationPrompt: "<|turn>model\n"
  },
  mistral: {
    join: " ", trim: true, addBos: false,
    system: { mode: "fold", foldSeparator: "\n\n" },
    user: { prefix: "[INST] ", suffix: " [/INST]" },
    assistant: { prefix: "", suffix: "</s>" },
    generationPrompt: ""
  },
  llama3: {
    join: "", addBos: false, bosLiteral: "<|begin_of_text|>",
    system: { mode: "block", prefix: "<|start_header_id|>system<|end_header_id|>\n\n", suffix: "<|eot_id|>", emitWhenEmpty: true },
    user: { prefix: "<|start_header_id|>user<|end_header_id|>\n\n", suffix: "<|eot_id|>" },
    assistant: { prefix: "<|start_header_id|>assistant<|end_header_id|>\n\n", suffix: "<|eot_id|>" },
    generationPrompt: "<|start_header_id|>assistant<|end_header_id|>\n\n"
  },
  "deepseek-r1": {
    join: "", addBos: false, bosLiteral: "<｜begin▁of▁sentence｜>",
    system: { mode: "prepend", source: "explicit" },
    user: { prefix: "<｜User｜>", suffix: "" },
    assistant: { prefix: "<｜Assistant｜>", suffix: "<｜end▁of▁sentence｜>" },
    generationPrompt: "<｜Assistant｜>"
  },
  "deepseek-v2": {
    join: "", addBos: false, bosLiteral: "<｜begin▁of▁sentence｜>",
    system: { mode: "prepend", suffix: "\n\n" },
    user: { prefix: "User: ", suffix: "\n\n" },
    assistant: { prefix: "Assistant: ", suffix: "<｜end▁of▁sentence｜>" },
    generationPrompt: "Assistant:"
  }
};
CHAT_SPECS.llama4 = { ...CHAT_SPECS.llama3 };
CHAT_SPECS.qwen3_5 = { ...CHAT_SPECS.qwen2 };
CHAT_SPECS.qwen3 = { ...CHAT_SPECS.qwen2 };

// The reasoning prime moves OUT of the template and INTO the reasoning descriptor.
export const CHAT_PRIMES = {
  qwen3_5: { primeOn: "<think>\n", primeOff: "<think>\n\n</think>\n\n" },
  qwen3: { primeOn: "<think>\n", primeOff: "<think>\n\n</think>\n\n" },
  "deepseek-r1": { primeOn: "<think>\n", primeOff: "<think>\n" }  // R1 always reasons
};
