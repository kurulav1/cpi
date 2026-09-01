# Determinism: what is verified, and where it stops

Reproduce any row with `--verify-determinism`, which decodes greedily and prints
digests of both sides: what the model was asked, and what it answered.

```
cpi <model> --tokenizer <tok> --prompt "The capital of France is" --verify-determinism 64 --gpu-cache-all
[verify] input_hash=90b40b4551408b2e73ffeb9ad0ca98ff
[verify]   model     Llama-3.2-1B-Instruct-F16.gguf file=7e2a46fe01895b16776ed927aac8f56c bytes=2479595168
[verify]   tokenizer tokenizer.json                 file=79e3e522635f3171300913bb421464a8
[verify]   prompt    6 token ids                    sha256=7ee923c120792ad1240a31d85ef1ad65
[verify]   sampling  sha256=f0e95884c86bf3a453671186b850992d
[verify]   settings  temp=0 greedy=1 max_new=64 ctx=2048 quant=none kv_bits=16 paged=0 paged_blocks=0 gpu_cache_all=1
[verify] output_hash=9003b7d09a9eae93 tokens=64
[verify] backend=llama-cuda model=Llama-3.2-1B-Instruct-F16.gguf quant=none kv_bits=16 paged=0 gpu_cache_all=1 ctx=2048 temp=0 prompt_tokens=6
[verify] ids=12366,13,578,469,3168,...
```

Both halves are there on purpose. `output_hash` says two runs produced the same
tokens; `input_hash` says they were asked the same thing. A hash pasted without the
second is half a claim, because the reader cannot tell which weights file, which
tokenizer, which prompt or which settings produced it, and those are exactly what
differs between their machine and yours. When two people compare and disagree, the
components say which input moved before anyone goes looking for a determinism bug.

The digests are SHA-256, checked against the published vectors in `sha256_test` and
against `hashlib` on real multi-gigabyte files, so the value printed here is what
every other tool calls SHA-256. Two details are deliberate. The prompt digest covers
the token IDS rather than the text, because ids are what the model actually saw and
the same string tokenizes differently under a different tokenizer. And a model given
as a DIRECTORY is hashed as a manifest of names and sizes, labelled `manifest=`
rather than `file=`: that catches a different or truncated checkpoint and would not
catch an edited tensor, which is a weaker claim and is marked as one.

It is not a separate code path: it calls `generate()` exactly as a request does,
because a verifier that decoded differently from the engine would attest to
nothing. `tools/determinism_matrix.sh` runs the single-sequence table below,
`tools/determinism_batch.sh` covers batch size (which a single-sequence verifier
cannot see), `tools/determinism_backend.sh` compares CPU against CUDA, and
`tools/determinism_version.sh` compares today's build against an older one.
`tools/determinism_selftest.sh` runs each of them against a deliberately corrupted
build and fails if any reports no difference (see "Showing the checks can fail").

## Holds

Measured on one machine, CUDA, RTX 5090, greedy (temperature 0), 64 tokens, two
model families. Same hash across all of these:

| varied | models |
| --- | --- |
| repeated runs of the same command | Llama-3.2-1B, Gemma-4-E2B |
| `--gpu-cache-all` on and off | Llama-3.2-1B, Gemma-4-E2B |
| `--paged-kv-cache`, `--paged-blocks` | Llama-3.2-1B |
| `--max-context` 2048, 4096, 8192 | Llama-3.2-1B |
| container: `.ll2c` against the GGUF of the same checkpoint | Llama-3.2-1B only, see below |
| position within a batch: first of five against last of five | Llama-3.2-1B |
| batch size: 1, 2, 3, 5, 8 sequences in flight | Llama-3.2-1B |

Three of those are worth more than the rest. Reading the same weights out of two
different container formats lands on the same tokens; a request's position in a
batch does not move its output; and neither does how many other requests are
being served beside it.

**Container equivalence is verified per model and does not generalise.** That row
says `.ll2c` and GGUF agree for Llama-3.2-1B. It reads as saying container format
does not change the answer, which is a larger claim and a false one: the same
Qwen2.5-0.5B checkpoint is correct from `.ll2c` and generates fluent nonsense from
GGUF, at F16 and Q8_0 alike. Neither quantisation nor the tokenizer is at fault;
the qwen2 architecture mapping in the GGUF reader is, and GGUF is now refused for
qwen2 rather than trusted.

Two things follow. Every container row needs its model named beside it, because a
container format is a second implementation of what the weights mean and is only as
good as the architecture mapping it uses. And the axis hides a second variable:
which container a file is in also decides which ENGINE runs it (a safetensors
directory reaches Llama4CudaEngine, a `.ll2c` or GGUF reaches LlamaEngine), so
"same weights, different container" can quietly mean "different engine".

## Does not hold

| boundary | effect |
| --- | --- |
| `--weight-quant int8` / `int4` | different hash |
| `--kv-int4` **beyond ~64 tokens** | same hash at 64, different at 512 |

The first is expected rather than a defect: int4 weights are different numbers,
so they produce different logits. It is listed because a determinism claim has to
say which knobs are allowed to move the answer.

The second is length-dependent, which is easy to miss. At 64 tokens the KV cache
is small enough that the recent-window tier holds everything in fp16, so nothing
quantised is ever read and the hash matches. At 512 it diverges. Testing only
short outputs would have produced a false claim of KV-quant invariance.

## Batch size, and why it took a fix

Batch size was on the failing list until it was fixed, and it was the sharpest
limit while it lasted. Position within a batch was already invariant, but size
was not: the same prompt decoded alone and decoded alongside four others produced
the same opening and then split.

```
alone       ... The Eiffel Tower is located in Paris. The Louvre Museum is also ...
in a batch  ... The Eiffel Tower is located in Paris. The Eiffel Tower is one of ...
```

That is the one boundary a server cannot live with. An agent loop varies its
batch by definition, so "deterministic unless something else is in flight" means
a request's answer depends on unrelated traffic, and no amount of pinning the
seed recovers it.

The cause was not two code paths in CPI. The scheduler runs the same code for one
sequence and for five, and `linear_rowmajor_weight` always calls cuBLAS with the
batch as N. cuBLAS picks a different kernel for N=1, and the two round
differently on near-ties, so the divergence appears wherever the model was nearly
indifferent between two tokens.

The fix is to stop asking for N=1: projections and the LM head are padded to a
minimum of two columns, with the padding row zeroed so the extra work is
well-defined, and the second column is discarded. One kernel then serves every
batch size by construction rather than by agreement.

It costs about 6% of decode throughput at batch 1, measured on the tool-calling
harness at 200 calls per side:

```
padding on    60.1s   3.93 ms/token
padding off   56.9s   3.70 ms/token
```

It is on by default and `CPI_DET_BATCH=0` turns it off, which is also the control
that makes the test meaningful: with it off, batch 1 must disagree with the rest,
and if it does not the test is measuring nothing.

This applies to fp16 only, because the batched decode path refuses INT8 and INT4
weights outright. There is no quantised batched path for the single path to
disagree with, so the axis does not exist there rather than being unverified.
It does mean `--serve` cannot currently run quantised weights at all.

## Therefore

Deterministic, on one machine, for a given model and settings: repeated runs,
container format, paging, GPU cache policy, context size, position within a
batch, batch size, build version back to the start of the project, and CPU
against CUDA.

Not deterministic across: weight quantisation, and KV quantisation once
generation is long enough to read quantised KV.

Worth knowing why it holds, because it says when to doubt it. Greedy decoding
consumes only the *ordering* of the logits, not their values, so a numeric
difference has to be large enough to reorder the top of the distribution before
it can change a token. Most differences are not. That makes the property
strongest where the model is confident and weakest at high-entropy positions and
in long generations, where near-ties accumulate: a run that agrees for 512 tokens
is weaker evidence than the same run agreeing for 2048. Sampling at a non-zero
temperature is a different question and is not claimed here.

## Showing the checks can fail

Every row above is a script reporting "identical". That is evidence only if the
script would have said otherwise, and the divergence branch is the branch that
never runs. When it did finally run here it crashed, on a path-format bug it had
been carrying unnoticed the whole time.

So the failure case has a switch. `CPI_DET_PERTURB=<step>` replaces the token
generated at index `<step>` with a different valid one, deterministically, in
every engine that emits tokens. `tools/determinism_selftest.sh` runs each check
twice, once against an honest build and once against a corrupted one, and fails
if a check cannot tell them apart:

```
1. the perturbation switch itself
  ok    perturbation changes the stream        56f4e7d7733d0ffa -> c72e427cb8f685fe
  ok    corrupted runs announce themselves on stderr
2. each check passes on an honest build        matrix, batch, backend: exit 0
3. each check FAILS on a build corrupted at token 10
  ok    matrix / batch / backend all detect it
4. the batch fix's own control
  ok    batch check fails without the fix      exit 1
```

Writing that turned up three things the table had been resting on.

**Two of the four scripts could not fail.** `determinism_matrix.sh` and
`determinism_batch.sh` printed a table and exited 0 whatever they found, so in CI
they would have been green while reporting divergence on screen. Both now assert:
rows are grouped by what must not change the answer, and a group that disagrees
exits non-zero.

**The perturbation did not reach every engine.** Hooking the CPU engine, the CUDA
single-sequence loop, speculative decode and the batched scheduler covers the
Llama paths, but Gemma runs through the shared `decode_driver.cpp`, which had no
hook. The `gemma-selftest` row would not have been corrupted, that group would
never have fired, and the run would still have passed on the llama group. The
self-test now requires *every* group whose row was corrupted to have caught it,
rather than accepting one failure somewhere.

**The perturbation is at the token, not in a kernel.** A single ULP in the LM head
is more faithful to a real numeric fault, but whether it flips a token inside 64
steps depends on hitting a near-tie, so a control built on it can quietly test
nothing. Corrupting a known index is what lets the check assert *where* the
divergence was reported. It exercises the comparison plumbing rather than the
arithmetic, and the arithmetic has its own control: `CPI_DET_BATCH=0` restores the
cuBLAS N=1 kernel choice, a real divergence that the batch check must fail on.

## Across versions

Today's build reproduces the token stream of the oldest build that can run the
model at all: commit `7c883b3` (2026-05-24), which is 817 commits back and the
seventh commit in the repository. Identical output on six configurations,
`tools/determinism_version.sh`:

| config | result |
| --- | --- |
| 64 tokens, `--gpu-cache-all` | identical |
| 64 tokens, no GPU cache | identical |
| 256 tokens, `--gpu-cache-all` | identical |
| 256 tokens, no GPU cache | identical |
| 256 tokens, prose prompt | identical |
| 64 tokens, code prompt | identical |

So version stability already exists here rather than needing to be established;
what it needs is gating, so that it keeps holding.

Two limits on that claim. It is one model family, because the comparison has to
use a container both builds read: GGUF support landed mid-history, so the test
runs on `.ll2c`, and only one `.ll2c` checkpoint is on hand. And `--paged-kv-cache`
could not be compared at all, because the old build fails it with a missing
tensor at layer 16 of a 16-layer model. That is an old bug fixed since, not a
divergence, and it is listed because a config the old build cannot run says
nothing about whether the two agree on it. A row that produced no tokens on both
sides would have compared equal and looked like a pass.

## Across backends: CPU against CUDA

The CPU engine produces the same tokens as CUDA. Nine configurations, three
prompts, lengths to 2048 tokens, all identical (`tools/determinism_backend.sh`):

| config | result |
| --- | --- |
| 64, 256, 512, 1024 tokens | identical |
| 2048 tokens, `--max-context 4096` | identical |
| prose prompt, code prompt, 512 tokens | identical |
| `--paged-kv-cache`, `--max-context 8192` | identical |
| generating into the context limit | identical (2043 tokens) |

This was the axis most likely to break, and the reason it was worth running
before renting anything: different reduction orders, different accumulate
precision, no cuBLAS at all. It holds anyway. Greedy decoding only needs the
ordering of the logits to survive, not their exact values, so numeric
differences have to reach a near-tie before they can change a token.

That last row was a defect, and it was worth chasing rather than excusing.
Generating into the context limit, CUDA emitted 2048 tokens where the CPU emitted
2043, and the CPU stream was an exact prefix of the CUDA one. The obvious reading
is a harmless disagreement about when to stop. It was not.

The KV cache and position tables are allocated for exactly `max_context`
positions, and the engine already refuses a prompt that would exceed them,
because doing so "corrupts memory; observed as garbage output on some builds and
a 0xC0000409 stack-buffer-overrun on others". The decode loop carried no such
bound: it ran to `max_new_tokens` whatever the window said. With a 6-token prompt
and `--max-context 2048`, positions 0..2047 hold 2043 generated tokens, and CUDA
wrote past the end for the rest. Compared against the same generation with room
to run (`--max-context 4096`), it agrees to generated index 2042 and then leaves
the rails, ending `... 0 12 220 220 220`.

So the CPU was right and CUDA was overrunning its cache. The same loop's own
speculative variant bounds this correctly, and so does the op-plan decode driver;
the single-sequence CUDA path was the one that did not. Bounded now, and both
engines emit 2043 tokens with the same hash, matching the ctx-4096 reference
token for token. Two rows in `determinism_backend.sh` generate into the limit to
keep it that way, and a `PREFIX ONLY` result is treated as a failure rather than
a footnote.

The first attempt at this measurement returned matching hashes from two runs
that had both used CUDA, because the `[verify] backend=` field was a compile-time
`#if`: a CUDA build reported `cuda` whichever engine ran. The field whose only
job is to attribute a mismatch was reporting what the binary could do rather than
what it did. It now reports the engine resolved at runtime, and the script fails
outright if the two runs do not name different engines.

## The same property in the units of an agent loop

Everything above compares token ids. Nobody running an agent thinks in tokens;
what they need to know is whether the second run calls the same tools, with the
same arguments, in the same order, over the same number of turns. Token identity
implies that, so this is not a new property, only the same one stated in the
units of the thing people are actually afraid of.

`tools/traces/reference_trace.json` pins a fixed loop as data: a system prompt,
four tool schemas, a deterministic stub for every tool, and a user prompt.
`tools/verify_trace.py` drives it against a running server and hashes the
canonical record of what happened; `tools/determinism_trace.sh` runs it across
configurations. One hash everywhere:

| varied | trace hash |
| --- | --- |
| baseline | `c6c5b0b1f2acded0` |
| the same server, again | identical |
| five traces in flight at once | identical |
| `--gpu-cache-all` off | identical |
| `--max-context 4096` | identical |
| the CPU engine instead of CUDA | identical |

The trace itself:

```
turn 0.0  get_weather        {"city":"Paris","unit":"celsius"}
turn 1.0  convert_currency   {"amount":500,"from_currency":"GBP","to_currency":"EUR"}
turn 2    final              "I'd be happy to help you with your travel plans..."
```

This was the first check to exercise grammar state, prefix reuse, growing context
and the stop rules at once, and it immediately found two things the token-level
checks could not have.

**`--serve --cpu` started no server.** It printed the chat-template line and
exited 0: no server, no error, nothing to attribute. Serving was wired only
inside the CUDA branches of the engine switch. This also means the "CPU agrees
with CUDA" row above was measured through the CLI and said nothing about the CPU
serving path, which was not working at all. A row can be true and still be
narrower than its title.

**The serial transport ignored `tools` and `tool_choice`.** It parsed
`json_schema` but never compiled a tool schema, so a request demanding a tool
call came back 200 with unconstrained prose and `finish_reason: "length"`, which
is indistinguishable from a model that chose not to call anything. The batched
path had been fixed for this; the serial path, which is what CPU serving uses,
had not.

Both are fixed, which is why the CPU row above exists at all. Note what the
guards were worth: the first trace run returned a perfectly reproducible hash of
a trace containing zero tool calls, and a stable hash of nothing would have read
as a clean row.

That zero-call run was the first sighting of a real defect, though it took a real
framework to recognise it as one. The tools were never rendered into the prompt at
all, only compiled into a grammar, so the model was being asked to call functions
it had not been told existed; tool results were dropped from the history for the
same reason; and nothing looked for a call in unconstrained output. A standard
agent loop could not run. All three are fixed, and the loop now completes and
repeats under framework defaults. It is worth recording that this harness did not
catch it: the reference trace forces its opening turns, which no real framework
does, so it only ever exercised the configuration where the bug was absent.

Like the others, this check has been shown to fail. `CPI_DET_SELFTEST=100`
corrupts one token of a final answer and the comparison reports `988900bd75c47cb8`
against the baseline's `c6c5b0b1f2acded0`. The index matters: at 10 the corruption
lands inside a tool call, destroys its JSON, and no trace comes back at all, which
the script now reports as `INCONCLUSIVE` rather than counting as detection.

### A tool call arrives one of two ways, and only one is guaranteed

This is the sharpest boundary in the agent path, and it is not about determinism.

When `tool_choice` is `required`, or names a function, the reply is produced under
a grammar compiled from that tool's JSON schema. The call **cannot** come back
malformed: the shape is enforced during decoding, not checked afterwards.

When `tool_choice` is `auto` or absent, which is what every agent framework sends,
nothing is constrained. That is the correct reading of the parameter, since `auto`
means the model chooses whether to call anything, and a grammar would remove the
choice. The call is therefore **recognised in ordinary output** afterwards:
`find_tool_call_json` scans the reply for a balanced JSON object carrying `name`
and `arguments`.

So on the `auto` path a tool call is parsed, not guaranteed. What follows from
that, stated plainly because the difference is easy to miss when both paths return
the same `tool_calls` shape to the client:

- A model that emits no recognisable JSON produces no tool call, and the turn
  comes back as ordinary content. That is a model outcome, not an error.
- Arguments are whatever the model wrote. They are valid JSON by construction, since
  they would not have parsed otherwise, but nothing checks them against the tool's
  schema. A call with a misspelled field reaches the client intact.
- The constrained path's guarantee ("a requested tool call cannot come back
  malformed") applies only where a grammar ran. It does not extend to `auto`.

Determinism is unaffected either way: the same input still produces the same
tokens, so the same call is parsed out of them. What differs is whether the shape
was *enforced* or merely *observed*.

Two further limits. `--serve` refuses int8/int4 weights, so the quantised trace
rows do not exist rather than being untested. And a `BROKEN` row here flipped to
passing between runs before the harness learned to wait for the GPU to release its
memory: killing the process is not the same as getting the VRAM back, and a
server started too early quietly declines to hold weights resident. That was a
harness artifact in the costume of a finding.

## Not yet measured

- **Across machines.** Every row above is one RTX 5090. Cross-machine
  determinism is the claim worth making and it is untested. Renting a second
  NVIDIA SKU for an hour would settle it.
- **Across backends, Metal.** Unverified: no Apple Silicon was available. CPU
  against CUDA is measured above; Metal is the remaining backend.

Until those are run the honest statement is the narrow one: same machine, same
build, same settings, and the exclusions above.
