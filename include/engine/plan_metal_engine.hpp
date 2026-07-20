#pragma once

// The Metal executor for the op-plan IR -- the Apple Silicon sibling of
// PlanCudaEngine.
//
// Same plan, different backend: build_llama_plan() produces the op list, and this
// walks it. The IR carries weights as opaque handles, which here are MTLBuffer
// objects rather than raw device pointers -- that is exactly why op_plan.hpp types
// them as void* and not __half*.
//
// Unified memory removes most of what the CUDA engine spends code on: a weight is
// "uploaded" by memcpy into a shared buffer, and logits are read back by simply
// looking at the buffer. No streams, no staging, no async copies.
//
// Scope: fp16, dense, uniform-geometry decode (Llama/Qwen2). No quantization -- Metal
// has no __dp4a equivalent, so the int4/int8 paths need a different kernel and are
// deliberately out of scope here.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <map>
#include <utility>
#include <vector>

#include "engine/batch_scheduler.hpp"
#include "engine/engine_types.hpp"
#include "engine/op_plan.hpp"
#include "engine/op_plan_builder.hpp"
#include "model/llama_config.hpp"
#include "model/weight_loader.hpp"
#include "runtime/metal_context.hpp"

namespace engine {

struct GenerationConstraints;  // fwd (generation_constraints.hpp) — grammar-constrained decode

class PlanMetalEngine {
public:
  PlanMetalEngine();
  ~PlanMetalEngine();

  // False when there is no Metal GPU (any VM without a paravirtual device). The
  // caller must check rather than assume a Mac has one.
  bool available() const;
  std::string device_name() const;
  const std::string& last_error() const {
    return last_error_;
  }

  // Loads a .ll2c model, uploads its weights, and builds the plan.
  // quant_bits: 0 = fp16, or 4 / 8 for weight-only quantization of every projection
  // and the (untied) LM head. quant_group 0 defaults to 64.
  void open(const std::string& weights_path, int max_context = 2048, int quant_bits = 0,
            int quant_group = 0);

  // Runs the vision tower over ONE pre-patchified image and returns its soft tokens:
  // (grid_h*grid_w)/(merge^2) rows of vision_out_hidden_size, which equals the text model's
  // hidden size so they can be spliced straight into the token stream.
  //
  // `patches` is (grid_h*grid_w) x (in_channels * temporal_patch_size * patch_size^2), the
  // layout the tower's own patch embed expects. Throws if the model has no tower or the grid
  // is not a whole number of merge units.
  std::vector<float> encode_image(const std::vector<float>& patches, int grid_h, int grid_w);

  // Prefills a prompt containing image tokens. `embeds` is indexed by absolute position; an
  // empty row leaves that token's own embedding in place. Rows are spliced AS-IS.
  void prefill_multimodal(const std::vector<int>& tokens,
                          const std::vector<std::vector<float>>& embeds);
  int quant_bits() const {
    return quant_bits_;
  }

  // Bytes of GPU-resident WEIGHT buffers. This -- not RSS -- is the number that says
  // whether a model fits: the mmap'd fp16 file stays resident because the quantizer
  // reads it, but those pages are clean and file-backed, so they are evictable.
  std::size_t weight_bytes() const;

  const model::LlamaConfig& config() const {
    return cfg_;
  }

  // Runs one decode step and returns the logits (a view into the shared buffer,
  // valid until the next step).
  const std::vector<float>& forward_token(int token, int position);

  // Capture a .gputrace of whatever runs between these, for Xcode's Metal Debugger.
  //
  // metal_gemm_bench can already capture ITS dispatches -- but a bench is not a prefill, and
  // reading a limiter off one and calling it the kernel's was a real mistake here: the bench's
  // own per-shape times predict 137 ms of GEMM for a 541-token prefill that actually spends
  // ~207 ms. Same kernels, same shapes, 1.5x apart, and cache/shape-mix/chunking are all
  // eliminated. Only a trace of the PASS can say where that goes.
  //
  // Requires MTL_CAPTURE_ENABLED=1 before the process starts. Returns false (with
  // last_error()) rather than throwing: a profiling aid must not be able to break a run.
  bool begin_gputrace(const std::string& path) {
    return ctx_.begin_gputrace(path);
  }
  void end_gputrace() {
    ctx_.end_gputrace();
  }

  // ---- Batched paged decode (continuous batching) --------------------------
  //
  // Sizes the KV as one paged POOL per layer -- num_blocks blocks of block_size tokens,
  // shared by every sequence -- rather than a contiguous max_context run. A contiguous cache
  // cannot serve concurrent sequences without reserving max_context per slot; paging lets
  // them grow independently, and lets two sequences share an identical prompt prefix by
  // refcounting its blocks. Call before open().
  //
  // The pool layout and block-table arithmetic match the CUDA paged path exactly, so a table
  // built by the shared (and entirely backend-free) engine::SequenceBlockTable means the same
  // thing on either backend. Note the two layouts coincide when a block table is the identity
  // map -- which is what the parity gate exploits.
  void set_paged_kv(int num_blocks, int block_size);
  bool paged_kv() const {
    return paged_blocks_ > 0;
  }

  // One decode step for N independent sequences at once. tokens[b] is sequence b's newest
  // token, positions[b] its position, and block_tables_flat is a [N][max_blocks] row-major
  // table of physical block ids. out_logits is resized to N rows of vocab_size.
  //
  // This walks the SAME op plan the single-sequence path walks. Only rope (per-row
  // positions), the KV scatter and attention (paged gathers) differ, because every other op
  // is row-independent and already runs T rows at once for prefill.
  void decode_step_batched_logits(const std::vector<int>& tokens, const std::vector<int>& positions,
                                  const std::vector<int>& block_tables_flat, int max_blocks,
                                  std::vector<std::vector<float>>& out_logits);

  // Prefills `tokens` at start_position into ONE sequence's paged blocks. block_table maps
  // that sequence's logical blocks to physical ones; it must already cover the whole range.
  //
  // This is the batched paged path with the rows being consecutive tokens of one sequence
  // rather than different sequences: row t sits at start_position+t with a length of
  // start_position+t+1, which is exactly the causal mask prefill needs, and every row shares
  // the one block table. So it needs no kernels of its own.
  //
  // The catch, and it is deliberate for now: attention here is the DECODE kernel, so each
  // query walks the whole history and the pass is O(T^2) in device traffic -- the very thing
  // the single-sequence query-block prefill kernels exist to avoid. Correct first. The
  // contiguous path is untouched and keeps its fast kernels; only the paged (batched-server)
  // prefill pays this, and giving it the query-block treatment is a known follow-up.
  void prefill_paged(const std::vector<int>& tokens, int start_position,
                     const std::vector<int>& block_table);

  // The continuous-batching scheduler for this engine, built on first use.
  //
  // It is engine::BatchScheduler -- the SAME class LlamaEngine drives, not a Metal
  // reimplementation. Admission, newest-first preemption, block growth and the shared-prefix
  // LRU are identical to CUDA's by construction rather than by agreement, because this engine
  // supplies only the two operations that need a GPU (BatchBackend::prefill_suffix and
  // ::decode_batched_logits) and inherits every policy decision.
  //
  // Requires set_paged_kv() before open(): the scheduler hands out blocks from that pool.
  BatchScheduler& batch_scheduler(const BatchSchedulerOptions& opts);

  // Greedy decode. Argmax runs on the GPU so the vocab never crosses to the host.
  std::vector<int> generate_greedy(const std::vector<int>& prompt, int max_new);

  // Sampled decode, through CPI's shared sampler -- the same code path LlamaEngine
  // uses, not a second implementation. Greedy (temperature <= 0) still takes the
  // on-GPU argmax; anything else needs the logits on the host anyway, because
  // repetition penalty and n-gram blocking rescore tokens outside any top-k set.
  struct Sampling {
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    int no_repeat_ngram_size = 0;
    int eos_id = -1;
    unsigned seed = 0;
  };
  std::vector<int> generate(const std::vector<int>& prompt, int max_new, const Sampling& s);

  // Streaming sibling of generate(): calls on_token(id) for each new token and stops early if
  // it returns false. This is the interface the serving modes (REST / web bridge) drive.
  std::vector<int> generate_stream(const std::vector<int>& prompt, int max_new, const Sampling& s,
                                   const std::function<bool(int)>& on_token,
                                   const GenerationConstraints* constraints = nullptr);

  // Top-k (id, logit) for the token that would follow `prompt`. Used by the inspect endpoint.
  std::vector<std::pair<int, float>> inspect_next_logits(const std::vector<int>& prompt, int top_k);

  const BenchmarkStats& last_benchmark_stats() const {
    return bench_stats_;
  }

  // Wall time of the last prompt prefill, and how many tokens it covered.
  double last_prefill_ms() const {
    return prefill_ms_;
  }
  int last_prefill_tokens() const {
    return prefill_tokens_;
  }

  // GPU-busy time and dispatch/submission counts since the last reset -- for the
  // overhead-vs-kernel question. See MetalContext.
  double gpu_busy_ms() const {
    return ctx_.gpu_busy_ms();
  }
  std::uint64_t dispatch_count() const {
    return ctx_.dispatch_count();
  }
  std::uint64_t cmdbuf_count() const {
    return ctx_.cmdbuf_count();
  }
  void reset_gpu_counters() {
    ctx_.reset_counters();
  }

private:
  void execute_ops(const std::vector<opplan::Op>& ops, int layer, int position, int tokens);

 public:
  // CPI_METAL_PROFILE=1 accumulates GPU time by op kind; dump_profile() prints the split.
  void dump_profile() const;

  // CPI_METAL_GPUPROFILE=1: true per-kernel GPU nanoseconds, from the device timestamp counters,
  // rather than the host times dump_profile() reports. See MetalContext::enable_gpu_profile --
  // it serialises dispatches to bracket them, so the individual numbers are honest and their
  // sum is larger than a real pass.
  void dump_gpu_profile();

 private:
  void profile_tick(const char* name);
  std::map<std::string, double> profile_ms_;
  std::chrono::steady_clock::time_point profile_last_{};
  void encode_forward(int token, int position);
  // Encodes a whole prefill chunk: T tokens through the tower in one pass.
  void encode_prefill(const std::vector<int>& tokens, int start_position);

  // MetalWeights is defined inside plan_metal_engine.cpp, so it is an incomplete type in every
  // other translation unit -- including the vision tower's. This upcast is defined there, where
  // it is complete, rather than moving the class into the header just to satisfy one caller.
  opplan::WeightSource& weight_source();

  // How to cut a prefill of `n` tokens into chunks no larger than a slot. Splitting evenly
  // beats filling greedily -- see the definition.
  std::vector<int> prefill_chunks(int n) const;
  // How many key chunks a decode's attention should split into at this position. 1 means the
  // context is too short to be worth the merge pass.
  int attn_split_chunks(const opplan::Op& op, int position) const;
  void* slot(opplan::Slot s) const;

  runtime::MetalContext ctx_;
  model::WeightLoader weights_;
  model::LlamaConfig cfg_{};
  opplan::ModelPlan plan_;
  int max_context_ = 0;
  int max_prefill_ = 0;  // slots are sized for this many tokens at once
  int quant_bits_ = 0;
  int paged_blocks_ = 0;      // 0 = contiguous per-sequence KV (the single-request path)
  int paged_block_size_ = 0;  // tokens per block

  // Batched-decode scratch, grown on demand rather than sized for a worst-case batch.
  runtime::MetalBuffer batch_pos_buf_;     // int32[B]   -- each row's own position
  runtime::MetalBuffer batch_seqlen_buf_;  // int32[B]   -- each row's own length
  runtime::MetalBuffer batch_bt_buf_;      // int32[B * max_blocks]
  runtime::MetalBuffer batch_logits_buf_;  // float[B * vocab]
  int batch_cap_ = 0;        // rows the scratch is sized for
  int batch_bt_elems_ = 0;   // elements the block-table buffer is sized for

  // Non-null only while decode_step_batched_logits is running: it tells execute_ops to take
  // the per-row / paged variants of rope, KV store and attention. Everything else in the
  // plan is row-independent and needs no knowledge of batching at all.
  struct BatchCtx {
    int batch;
    int max_blocks;
    int block_size;
    // Prefill wants logits for the LAST row only -- its earlier tokens exist to fill the KV
    // cache, and a vocab GEMV per row would cost T of them instead of one. Batched DECODE
    // wants every row, because there each row is a different sequence's next token.
    bool logits_last_only;
  };
  const BatchCtx* batch_ = nullptr;

  // name -> device buffer. Owns every weight for the model's lifetime.
  std::unordered_map<std::string, runtime::MetalBuffer> wbuf_;

  std::vector<runtime::MetalBuffer> slots_;  // indexed by opplan::Slot
  std::vector<runtime::MetalBuffer> k_cache_;
  std::vector<runtime::MetalBuffer> v_cache_;

  // MoE: the router's selection, carried between ops on the device so a token never
  // round-trips to the host mid-layer. Only allocated for MoE models.
  // float[GEMM_SPLITK * tokens * out_dim] -- per-split partial sums for the split-K GEMM.
  // Allocated lazily, and only ever for the narrow-output projections that take that path.
  runtime::MetalBuffer gemm_partial_buf_;

  // Delta-net state, one contiguous buffer each, indexed by layer. Both are fp32 and both are
  // read-modify-write within a single token's forward, so a layer's slice must not be shared and
  // the ops that touch them must run exactly once per token. Zeroed at open(): the recurrence
  // starts empty, and starting it from stale memory produces fluent nonsense rather than a crash.
  runtime::MetalBuffer lin_conv_state_;       // [layers][conv_dim * (kernel-1)]
  runtime::MetalBuffer lin_recurrent_state_;  // [layers][num_v_heads * key_hd * value_hd]
  std::size_t lin_conv_stride_ = 0;
  std::size_t lin_recurrent_stride_ = 0;
  // True when the built plan contains a recurrent op, which forces prefill to advance one token
  // at a time. Derived from the plan in open(), not from the config.
  bool has_recurrent_ops_ = false;

  // Which buffer the current forward's token ids were staged into. Set by whichever entry point
  // staged them, because the token COUNT does not answer this -- a one-token prefill chunk still
  // arrives in the sequence buffer.
  bool tokens_in_seq_buf_ = false;

  // Borrowed for the duration of prefill_multimodal only, never owned. Cleared on the way out
  // (including on a throw) so a later plain prefill cannot read a dangling pointer.
  const std::vector<std::vector<float>>* embeds_ = nullptr;

  // Writes slot X after one layer, in the CPU engine's dump format. No-op unless CPI_Q35_DUMP is
  // set. See the definition for why both prefill and decode call it.
  void dump_layer_state(int layer_index, int position);

  // Writes one named intermediate slot, in the same format and under the same name the CPU engine
  // uses, so a divergence can be bisected to a buffer. No-op unless CPI_Q35_DUMP is set.
  void dump_named_slot(const char* name, int layer, int position, opplan::Slot sl, int n);

  runtime::MetalBuffer moe_idx_buf_;  // int32[top_k]  -- selected expert ids
  runtime::MetalBuffer moe_w_buf_;    // float[top_k]  -- their renormalised routing weights

  runtime::MetalBuffer tok_buf_;      // int32[1]
  runtime::MetalBuffer seq_tok_buf_;  // int32[max_prefill] -- a whole prompt chunk
  runtime::MetalBuffer pos_buf_;      // int32[1]
  // Split-KV decode attention scratch: per (head, chunk) softmax stats and the unnormalized
  // partial accumulators the merge pass folds together.
  runtime::MetalBuffer attn_part_m_;  // float[heads * maxchunks]
  runtime::MetalBuffer attn_part_l_;  // float[heads * maxchunks]
  runtime::MetalBuffer attn_part_o_;  // float[heads * maxchunks * head_dim]
  runtime::MetalBuffer logits_buf_;   // float[vocab]
  runtime::MetalBuffer argmax_val_;   // float[parts]
  runtime::MetalBuffer argmax_idx_;   // int32[parts]
  runtime::MetalBuffer argmax_out_;   // int32[1]

  std::vector<float> logits_;
  double prefill_ms_ = 0.0;
  int prefill_tokens_ = 0;
  BenchmarkStats bench_stats_{};
  std::vector<int> prev_seq_;  // last (prompt + generated) sequence, for shared-prefix KV reuse
  std::string last_error_;

  class MetalWeights;
  std::unique_ptr<MetalWeights> wsrc_;

  // Continuous batching. The allocator hands out block ids from the paged pool sized by
  // set_paged_kv(); the adapter is this engine's half of BatchBackend. Held as the base
  // pointer so the destructor does not need the adapter's definition.
  class BatchAdapter;
  std::unique_ptr<BlockAllocator> block_alloc_;
  std::unique_ptr<BatchBackend> batch_adapter_;
  std::unique_ptr<BatchScheduler> scheduler_;
};

}  // namespace engine
