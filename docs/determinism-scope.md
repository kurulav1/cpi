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
nothing. `tools/determinism_matrix.sh` runs the table below.

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

Two of those are worth more than the rest. Reading the same weights out of two
different container formats lands on the same tokens, and a request's position in
a batch does not move its output.

## Does not hold

| boundary | effect |
| --- | --- |
| `--weight-quant int8` / `int4` | different hash |
| `--kv-int4` **beyond ~64 tokens** | same hash at 64, different at 512 |
| batch of 1 against a batch of several | different continuation |

The first is expected rather than a defect: int4 weights are different numbers,
so they produce different logits. It is listed because a determinism claim has to
say which knobs are allowed to move the answer.

The second is length-dependent, which is easy to miss. At 64 tokens the KV cache
is small enough that the recent-window tier holds everything in fp16, so nothing
quantized is ever read and the hash matches. At 512 it diverges. Testing only
short outputs would have produced a false claim of KV-quant invariance.

The third is the sharpest limit. Position within a batch is invariant, but batch
size is not: the same prompt decoded alone and decoded alongside four others
produces the same opening and then splits.

```
alone       ... The Eiffel Tower is located in Paris. The Louvre Museum is also ...
in a batch  ... The Eiffel Tower is located in Paris. The Eiffel Tower is one of ...
```

Both are stable on repetition, so this is not noise: it is two code paths. The
batched decode reaches the logits through batched GEMMs where the single path
uses a GEMV, and the two round differently on near-ties. The divergence appears
where the model was nearly indifferent.

## Therefore

Deterministic, on one machine and backend, for a given model and settings:
repeated runs, container format, paging, GPU cache policy, context size, and
position within a batch.

Not deterministic across: weight quantisation, KV quantisation once generation is
long enough to read quantised KV, and single against batched decode.

## Not yet measured

- **Across machines.** Every row above is one RTX 5090. Cross-machine
  determinism is the claim worth making and it is untested.
- **Across backends.** Metal is unverified: no Apple Silicon was available. CPU
  against CUDA is untested here.
- **Across versions.** Whether today's build agrees with one from six months ago
  is unknown, and it decides whether version stability already exists or has to
  start being gated now.

Until those are run the honest statement is the narrow one: same machine, same
build, same settings, and the exclusions above.
