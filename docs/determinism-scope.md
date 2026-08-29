# Determinism: what is verified, and where it stops

Reproduce any row with `--verify-determinism`, which decodes greedily and prints a
hash of the token ids plus every setting that could change them:

```
cpi <model> --prompt "The capital of France is" --verify-determinism 64 --gpu-cache-all
[verify] hash=9003b7d09a9eae93 tokens=64
[verify] backend=cuda model=Llama-3.2-1B-Instruct-F16.gguf quant=none kv_bits=16 paged=0 \
         gpu_cache_all=1 ctx=2048 temp=0 prompt_tokens=6
[verify] ids=51354,13,578,469,3168,...
```

It is not a separate code path: it calls `generate()` exactly as a request does,
because a verifier that decoded differently from the engine would attest to
nothing. `tools/determinism_matrix.sh` runs the single-sequence table below;
`tools/determinism_batch.sh` covers batch size, which a single-sequence verifier
cannot see.

## Holds

Measured on one machine, CUDA, RTX 5090, greedy (temperature 0), 64 tokens, two
model families. Same hash across all of these:

| varied | models |
| --- | --- |
| repeated runs of the same command | Llama-3.2-1B, Gemma-4-E2B |
| `--gpu-cache-all` on and off | Llama-3.2-1B, Gemma-4-E2B |
| `--paged-kv-cache`, `--paged-blocks` | Llama-3.2-1B |
| `--max-context` 2048, 4096, 8192 | Llama-3.2-1B |
| container: `.ll2c` against the GGUF of the same checkpoint | Llama-3.2-1B |
| position within a batch: first of five against last of five | Llama-3.2-1B |
| batch size: 1, 2, 3, 5, 8 sequences in flight | Llama-3.2-1B |

Three of those are worth more than the rest. Reading the same weights out of two
different container formats lands on the same tokens; a request's position in a
batch does not move its output; and neither does how many other requests are
being served beside it.

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

Deterministic, on one machine and backend, for a given model and settings:
repeated runs, container format, paging, GPU cache policy, context size, position
within a batch, and batch size.

Not deterministic across: weight quantisation, and KV quantisation once
generation is long enough to read quantised KV.

## Not yet measured

- **Across machines.** Every row above is one RTX 5090. Cross-machine
  determinism is the claim worth making and it is untested. Renting a second
  NVIDIA SKU for an hour would settle it.
- **Across backends.** Metal is unverified: no Apple Silicon was available. CPU
  against CUDA is untested here.
- **Across versions.** Whether today's build agrees with one from six months ago
  is unknown, and it decides whether version stability already exists or has to
  start being gated now. This one needs no hardware.

Until those are run the honest statement is the narrow one: same machine, same
build, same settings, and the exclusions above.
