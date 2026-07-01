// Regression tests for generated-text normalization — guards the decoder-halt
// bug where an ungated "drop trailing incomplete tail" heuristic truncated
// healthy output at the last '.'/'?'/'!' (breaking ternaries, quoted ".",
// em-dashes — common in code/DSL output).
//
// Run: node web/server/text_normalize.test.mjs

import { normalizeGeneratedChatText, looksDegenerateRepetition } from "./text_normalize.mjs";

let failures = 0;
let checks = 0;
function check(cond, msg) {
  checks += 1;
  if (!cond) {
    failures += 1;
    console.error("FAIL: " + msg);
  }
}

// Healthy output containing '.'/'?'/'!'/em-dash must be returned INTACT (no
// truncation at the last sentence terminator). These are the engine outputs
// for the R1/R2 + em-dash repros.
check(normalizeGeneratedChatText('a=="0"?b then DONE', "qwen2") === 'a=="0"?b then DONE',
      "R2: ternary '?' not truncated");
check(normalizeGeneratedChatText('X then "." then DONE', "qwen2") === 'X then "." then DONE',
      "R1: quoted period not truncated");
check(normalizeGeneratedChatText('display=display=="0"?$item:display~$item', "qwen2")
        === 'display=display=="0"?$item:display~$item',
      "DSL ternary handler preserved");
check(normalizeGeneratedChatText("foo — bar then DONE", "qwen2") === "foo — bar then DONE",
      "em-dash content not truncated");
check(normalizeGeneratedChatText("result = x > 0.5 ? hi : lo", "llama3")
        === "result = x > 0.5 ? hi : lo",
      "decimal + ternary on llama3 preserved");

// Controls (no terminator) — unchanged.
check(normalizeGeneratedChatText("alpha bravo charlie then DONE", "qwen2")
        === "alpha bravo charlie then DONE",
      "R4 control: plain text preserved");

// Degenerate output STILL gets the salvage trim (the heuristic's real purpose):
// a repeated-word loop with an incomplete trailing tail is cut at the last
// terminal.
check(looksDegenerateRepetition("go go go go go. trailing junk") === true,
      "degenerate repetition detected");
check(normalizeGeneratedChatText("go go go go go. trailing junk", "qwen2")
        === "go go go go go.",
      "degenerate output: trailing tail still trimmed");

// Healthy output is not flagged degenerate.
check(looksDegenerateRepetition('a=="0"?b then DONE') === false,
      "healthy ternary not flagged degenerate");

console.log("text_normalize.test: " + (checks - failures) + "/" + checks + " checks passed");
process.exit(failures === 0 ? 0 : 1);
