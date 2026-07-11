// Parity gate for "chat template as data".
//
// Renders every migratable template two ways — through its legacy hand-written
// formatter, and through the generic renderChat() driven by a declarative chat
// descriptor — and requires the prompts to be BYTE-IDENTICAL. This is what makes
// retiring the per-model formatters safe: a mismatch here is a prompt-format
// regression, which silently degrades every reply.
//
// llama2 / tinyllama are intentionally absent: they are not chat templates, they
// are bespoke prompt strategies (history compaction, injected instructions), so
// they stay as code.
//
//   node web/server/chat_spec_parity.test.mjs

import { buildPromptPackage } from "./prompting.mjs";
import { CHAT_SPECS, CHAT_PRIMES } from "./chat_specs.reference.mjs";

// The descriptors. In production these ship WITH each model (manifest / cpi.json);
// here they are the reference set the parity gate proves correct.
const MESSAGES = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Hi there" },
  { role: "assistant", content: "Hello! How can I help?" },
  { role: "user", content: "What is 2+2?" }
];
const SYSTEM = "You are a helpful assistant.";

let failures = 0;
for (const [template, chat] of Object.entries(CHAT_SPECS)) {
  for (const thinking of [false, true]) {
    const base = { template, systemPrompt: SYSTEM, thinking };
    const legacy = buildPromptPackage(MESSAGES, base);
    const spec = buildPromptPackage(MESSAGES, {
      ...base, chat, reasoning: CHAT_PRIMES[template] ?? {}
    });
    const same = legacy.prompt === spec.prompt;
    const bosSame = legacy.addBos === spec.addBos;
    if (!same || !bosSame) {
      failures++;
      console.log(`\nFAIL ${template} (thinking=${thinking})`);
      if (!bosSame) console.log(`  addBos: legacy=${legacy.addBos} spec=${spec.addBos}`);
      if (!same) {
        console.log(`  legacy: ${JSON.stringify(legacy.prompt)}`);
        console.log(`  spec  : ${JSON.stringify(spec.prompt)}`);
      }
    } else {
      console.log(`ok   ${template.padEnd(18)} thinking=${String(thinking).padEnd(5)} (${legacy.prompt.length} chars)`);
    }
  }
}
console.log(failures === 0
  ? `\nPARITY OK — all ${Object.keys(CHAT_SPECS).length} templates render byte-identically from data`
  : `\nPARITY FAILED — ${failures} mismatches`);
process.exit(failures === 0 ? 0 : 1);
