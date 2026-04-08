# Phi-MoE Validation Report (2026-04-07)

## Scope
- Target 1: microsoft/Phi-tiny-MoE-instruct
- Target 2: microsoft/Phi-mini-MoE-instruct
- Modes: fp16 / int8 / int4

## Phi-tiny Artifact Status
- Model path: D:\codex-temp\phi_tiny_validation\Phi-tiny-MoE-instruct.ll2c
- Validation: PASS (tools/validate_ll2c.py)
- Tensor count: 8037
- Note: int8/int4 artifacts are hardlinks to the same validated .ll2c file containing appended quant tensors.

## Phi-tiny Perf + Parity (moe_cuda_bench)
- fp16 decode tok/s: 3.04
- int8 decode tok/s: 4.5
- int4 decode tok/s: 3.45
- int8 parity: overlap@40=0.0, mean_abs=Infinity, max_abs=Infinity
- int4 parity: overlap@40=0.475, mean_abs=5.765546551724136, max_abs=9.6417
- Bench gate: FAILED (int8 and int4 drift out of bounds)

## Phi-tiny Quality (eval_prompts)
- fp16 mean_score: 0.8658 (5/5)
- int8 mean_score: 0.0 (5/5)
- int4 mean_score: 0.79 (5/5)

## Phi-mini Flow Status
Attempted full flow download -> convert -> pack -> validate -> warmup -> quality.

Current result:
- Download started and progressed into shard 2.
- Failed with: No space left on device.
- Conversion/pack/validate/warmup/quality were not reachable in this run.

Files currently present under D:\codex-temp\phi_mini_validation\hf:


Name                                 Length SizeGB
----                                 ------ ------
model-00001-of-00004.safetensors 4993819264   4.65
README.md                             13164      0
configuration_slimmoe.py              12443      0
data_summary_card.md                   4348      0
SECURITY.md                            2656      0
NOTICE.md                              1772      0
LICENSE                                1105      0
config.json                            1069      0
CODE_OF_CONDUCT.md                      444      0
added_tokens.json                       315      0
generation_config.json                  177      0




## Blocking Constraint
- Current free space on D: ~0.95 GB after cleanup.
- Full Phi-mini flow requires substantially more free space (multi-shard download + conversion/packing intermediates).

## Conclusion
- Phi-tiny: completed fp16/int8/int4 validation runs; int8 quality/parity currently broken; int4 quality passes but parity drift exceeds configured bounds.
- Phi-mini: blocked by disk capacity before conversion stage.
