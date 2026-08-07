// Does the markdown pipeline render LaTeX -- without eating money?
// Runs the SAME plugin chain and the SAME normalisation the app uses.
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";

// Mirrors formatModelMarkdown's math handling in src/App.jsx. Keep the two in step:
// this file is the gate that says the pipeline renders LaTeX WITHOUT eating money.
function normalize(t) {
  let c = t
    .replace(/\\\[([\s\S]+?)\\\]/g, (_m, b) => "$$" + b + "$$")
    .replace(/\\\(([\s\S]+?)\\\)/g, (_m, b) => "$" + b + "$");
  c = c.replace(
    /(^|[\s(])\$(\d[\d,]*(?:\.\d+)?)(?=$|[\s.,;:!?)])/gm,
    (_m, pre, num) => pre + "\\$" + num
  );
  return c;
}

const proc = unified()
  .use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkBreaks)
  .use(remarkRehype).use(rehypeKatex, { throwOnError: false, strict: false })
  .use(rehypeStringify);

const BS = "\\";
const cases = [
  ["the reported one",          "Average Case: $O(n " + BS + "log n)$ (Very fast)", true],
  ["exponent",                  "Worst Case: $O(n^2)$ occurs when...", true],
  ["display math",              "$$" + BS + "frac{1}{2}" + BS + "sum_{i=1}^{n} x_i^2$$", true],
  ["paren delimiters",          "Complexity is " + BS + "(O(n " + BS + "log n)" + BS + ") on average.", true],
  ["bracket delimiters",        BS + "[ E = mc^2 " + BS + "]", true],
  ["single symbol",             "Let $x$ be the pivot.", true],
  ["CURRENCY must NOT be math", "It costs $5 and the other is $7 each.", false],
  ["currency AND math",         "It costs $5 but runs in $O(n " + BS + "log n)$ time.", true],
  ["half-streamed, no throw",   "Average Case: $O(n " + BS + "lo", false],
  ["math starting with a digit", "The root is $2x + 1 = 0$ here.", true],
  ["big money",                  "Revenue was $1,200.50 last year.", false],
];

let bad = 0;
for (const [name, src, wantMath] of cases) {
  try {
    const html = String(await proc.process(normalize(src)));
    const isMath = html.includes("katex");
    const plain = html.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
    const ok = isMath === wantMath;
    if (!ok) bad += 1;
    console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(28)} ${isMath ? "rendered" : "plain"}`);
    if (name.startsWith("CURRENCY")) console.log(`       -> ${plain}`);
  } catch (e) {
    bad += 1;
    console.log(`THREW ${name}: ${e.message}`);
  }
}
console.log(bad === 0 ? "\nMATH OK -- renders LaTeX, leaves money alone, survives half-streamed formulas"
                      : `\n${bad} case(s) FAILED`);
