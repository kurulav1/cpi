// Text normalization for generated chat output, extracted from index.mjs so it
// can be unit-tested without starting the server. See text_normalize.test.mjs.
//
// The key invariant (regression-guarded): healthy output is returned intact.
// The lossy "drop trailing incomplete tail" salvage runs ONLY on degenerate
// (repetition-loop) output — unconditionally cutting at the last '.'/'?'/'!'
// truncated all code/DSL/structured text (ternaries, quoted ".", decimals,
// URLs). See docs/MORPH_DECODER_BUG.md. Mirrors the C++ gating in
// src/app/main_helpers.cpp (trim_incomplete_trailing_tail behind
// looks_degenerate_repetition).

// Detects degenerate repetition loops (the failure mode the aggressive cleanup
// pipeline exists for). Healthy output must NOT match.
// Mirrors looks_degenerate_repetition() in src/app/main_helpers.cpp.
export function looksDegenerateRepetition(text) {
  const s = String(text || "");
  // A word repeated 3+ times in a row ("the the the").
  if (/\b([A-Za-z][A-Za-z'-]*)\b(?:\s+\1\b){2,}/i.test(s)) return true;
  // The same letter 6+ times in a row ("aaaaaa").
  if (/([A-Za-z])\1{5,}/.test(s)) return true;
  // The same punctuation mark 3+ times in a row ("!!!!").
  if (/([!?,])\1{2,}/.test(s)) return true;
  // The same sentence emitted twice (or more) consecutively.
  const sentences = s.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
  let prev = "";
  for (const sentence of sentences) {
    const key = sentence.replace(/\s+/g, " ").trim().toLowerCase();
    if (key.length >= 16 && key === prev) return true;
    prev = key;
  }
  return false;
}

export function normalizeGeneratedChatText(text, template) {
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

  // Salvage tiny-model drift ONLY for degenerate output — see module header and
  // docs/MORPH_DECODER_BUG.md. Healthy output keeps its full content.
  if (looksDegenerateRepetition(cleaned)) {
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
  }

  return cleaned.trim();
}
