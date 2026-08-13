#pragma once

// The continuous-batching scheduler, with no backend in it.
//
// Admission, preemption, block-table growth, shared-prefix reuse with an LRU cache,
// per-request sampling and retirement are all pure host bookkeeping over
// engine::BlockAllocator / SequenceBlockTable (which are themselves backend-free). This
// logic lived inside LlamaEngine, which made continuous batching a CUDA-only feature for
// no reason other than where the code sat: of its ~200 lines, exactly two touch a GPU.
//
// Those two are the BatchBackend interface below. A backend supplies a suffix prefill and a
// batched decode step; everything else is shared. That keeps the second backend honest
// Metal gets the same admission policy, the same preemption order and the same prefix cache
// as CUDA, because it is literally the same code, not a reimplementation that agrees today.
//
// Threading: not thread-safe, single owner. The engine serialises access, as before.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/paged_kv.hpp"
#include "engine/sampling.hpp"

namespace grammar {
class GrammarSampler;
}

namespace engine {

// Per-request generation parameters. Lives here rather than inside an engine because both
// the scheduler and the app-layer batch worker speak it, and neither should need a backend
// header to do so.
struct StreamParams {
  int max_new_tokens = 16;
  // Suppress EOS until this many tokens are generated. Ignored when `grammar` is set: a grammar
  // drives its own termination, so the scheduler does not also floor the length (see step()).
  int min_new_tokens = 0;
  float temperature = 0.0f;
  // Sampling shape, per request. These used to live on BatchSchedulerOptions (scheduler-global),
  // which silently ignored a client's top_p/top_k in the batched path and made every request in a
  // server share one setting; and, because repetition_penalty is global, a server started with
  // penalty > 1 could never take the device top-k fast path for anyone. The caller fills these
  // from the request (falling back to its own defaults); the scheduler reads only these.
  int top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  int no_repeat_ngram_size = 0;
  std::vector<int> stop_ids;
  // Non-owning; the caller keeps it alive until the request finishes.
  grammar::GrammarSampler* grammar = nullptr;
};

// One token for one request.
//
// finish_reason is "eos" | "stop" | "length" | "preempted". "Preempted" is part of the
// contract, not an error: it arrives with token == -1 and means the KV pool ran out and this
// request's blocks were reclaimed so older ones could finish. The caller is expected to
// re-admit it with (prompt + generated-so-far) and a decremented budget; the scheduler
// deliberately does not do that itself, because only the caller knows the request's remaining
// budget and whether it still wants the answer.
struct StreamEvent {
  std::string id;
  int token = 0;
  bool finished = false;
  const char* finish_reason = "";
};

// Per-row inputs to the device top-k fast path. penalty[b] <= 1 means "no repetition penalty for
// row b". penalty_ids / penalty_rows are the flattened unique seen token ids for the penalty rows
// (penalty_ids[i] belongs to row penalty_rows[i]); the backend applies the penalty on-device
// before the top-k, so the candidate set still matches the host sampler's slow path.
struct BatchTopkParams {
  std::vector<int> k;             // [batch] per-row top_k
  std::vector<float> penalty;     // [batch] per-row repetition_penalty (<= 1 => none)
  std::vector<int> penalty_ids;   // flattened unique seen ids across penalty rows
  std::vector<int> penalty_rows;  // parallel to penalty_ids: which row each id belongs to
};

// Per-row inputs to the device greedy (argmax) fast path. Same penalty layout as BatchTopkParams
// (the argmax is taken over the penalized, sanitized logits, matching the host greedy path);
// blocked[b] is a token id to exclude for row b (the EOS id when the row is still below
// min_new_tokens), or -1 for none.
struct BatchArgmaxParams {
  std::vector<float> penalty;     // [batch] per-row repetition_penalty (<= 1 => none)
  std::vector<int> penalty_ids;   // flattened unique seen ids across penalty rows
  std::vector<int> penalty_rows;  // parallel to penalty_ids: which row each id belongs to
  std::vector<int> blocked;       // [batch] id to exclude (EOS below min_new), -1 = none
};

// What the scheduler needs from a GPU. Two calls.
class BatchBackend {
public:
  virtual ~BatchBackend() = default;

  // Prefill prompt[shared .. end) into the sequence described by `block_table`. The KV for
  // [0, shared) is already in the pool via adopted (refcounted) blocks, so it must not be
  // recomputed; and must not be assumed contiguous with the suffix.
  //
  // kv_slot (here and per-row as kv_slots in the decode calls) is the sequence's stable
  // quality-slot id: assigned by the scheduler at admission, constant for the sequence's
  // lifetime, and reused only after the sequence retires. Backends with per-sequence side
  // state keyed off the sequence (CUDA's quant sink/window fp16 buffers) key it here;
  // backends without simply ignore it.
  //
  // Must be complete before returning: the scheduler may reuse shared engine state (a block
  // table upload, a scratch buffer) on the very next call.
  virtual void prefill_suffix(const std::vector<int>& prompt_tokens, int shared_tokens,
                              const std::vector<int>& block_table, int kv_slot) = 0;

  // One decode step for N sequences. tokens[b]/positions[b] are sequence b's newest token and
  // its position; block_tables_flat is [N][max_blocks] row-major; kv_slots[b] is sequence b's
  // stable quality-slot id (see prefill_suffix). out_logits is resized to N rows of vocab.
  virtual void decode_batched_logits(const std::vector<int>& tokens,
                                     const std::vector<int>& positions,
                                     const std::vector<int>& block_tables_flat, int max_blocks,
                                     const std::vector<int>& kv_slots,
                                     std::vector<std::vector<float>>& out_logits) = 0;

  // Optional fast path: one decode step returning each row's top-k candidate set via a device
  // top-k reduction, so only ~k candidates per row cross to the host instead of the full
  // [batch][vocab] logits. Returns false when the backend has no device top-k or k is unusable
  // (the caller then falls back to decode_batched_logits). The caller guarantees the batch needs
  // no full-vocabulary features (repetition penalty, n-gram blocking, grammar) and that every
  // row samples (temperature > 0), so the candidate set is exactly what the host sampler builds.
  virtual bool decode_batched_topk(const std::vector<int>& tokens,
                                   const std::vector<int>& positions,
                                   const std::vector<int>& block_tables_flat, int max_blocks,
                                   const std::vector<int>& kv_slots, const BatchTopkParams& sp,
                                   std::vector<std::vector<detail::SampleCandidate>>& out_cand) {
    (void)tokens;
    (void)positions;
    (void)block_tables_flat;
    (void)max_blocks;
    (void)kv_slots;
    (void)sp;
    (void)out_cand;
    return false;
  }

  // Optional fast path for greedy rows (temperature <= 0): one decode step returning each row's
  // argmax token via a device reduction, so only B ints cross to the host instead of the full
  // [batch][vocab] logits. Returns false when the backend has no device argmax. The caller
  // guarantees the batch needs no full-vocabulary host feature (n-gram blocking, grammar) and that
  // every row is greedy; repetition penalty is applied on-device before the argmax.
  virtual bool decode_batched_argmax(const std::vector<int>& tokens,
                                     const std::vector<int>& positions,
                                     const std::vector<int>& block_tables_flat, int max_blocks,
                                     const std::vector<int>& kv_slots, const BatchArgmaxParams& ap,
                                     std::vector<int>& out_ids) {
    (void)tokens;
    (void)positions;
    (void)block_tables_flat;
    (void)max_blocks;
    (void)kv_slots;
    (void)ap;
    (void)out_ids;
    return false;
  }
};

// Non-sampling limits the scheduler applies. Sampling shape (top_k / top_p / repetition_penalty /
// no_repeat_ngram_size) is per request on StreamParams, not here; it varies per client.
struct BatchSchedulerOptions {
  int paged_block_size = 32;
  int max_context = 2048;
  int eos_token_id = -1;
  bool verbose = false;
  // Skip shared-prefix adoption (and cache insertion). Set by backends whose per-sequence
  // side state cannot survive adoption: an adopted prefix was prefilled by another sequence,
  // so this sequence's quality-slot buffers (quant sink/window) would have holes for it.
  bool disable_prefix_reuse = false;
};

class BatchScheduler {
public:
  // Neither pointer is owned; both must outlive the scheduler.
  BatchScheduler(BatchBackend* backend, BlockAllocator* alloc, const BatchSchedulerOptions& opts);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  // THROWS if the KV pool cannot seat the prompt. The caller rejects the request (or, when
  // re-admitting a preempted one, retries later); it is not a scheduler decision.
  void admit(const std::string& id, const std::vector<int>& prompt_tokens,
             const StreamParams& params);

  // Advance every running request by one token. Returns false only when nothing ran; note it
  // returns true with events when all survivors were preempted, so the caller still sees them.
  bool step(std::vector<StreamEvent>& events);

  bool cancel(const std::string& id);
  int active() const;

  // The context window requests must fit. Exposed so the transports (batch
  // worker) can clamp an over-long prompt with a warning before admission
  // instead of learning the limit from an admit failure.
  int max_context() const { return opts_.max_context; }

  // Drop the shared-prefix cache. The caller must do this whenever the KV cache is wiped
  // underneath the scheduler (reset_kv_cache): the cached entries hold refcounted blocks whose
  // contents the wipe invalidates, and nothing about the block ids themselves would reveal
  // that; a later admit would happily adopt them and prefill garbage. Running sequences are
  // deliberately left alone, matching the behaviour this replaced.
  void clear_prefix_cache();

private:
  struct StreamSeq {
    std::string id;
    std::unique_ptr<SequenceBlockTable> blocks;
    std::vector<int> table;    // logical chunk -> physical block, grown on demand
    std::vector<int> history;  // full token history (prompt + generated)
    int pos = 0;
    int last_token = 0;
    int generated = 0;
    int kv_slot = 0;  // stable quality-slot id (see BatchBackend::prefill_suffix)
    StreamParams params;
  };

  // A small LRU of block-aligned prompt prefixes, each holding refcounted KV blocks. On admit
  // the longest whole-block common prefix is adopted instead of re-prefilled: identical tokens
  // at identical positions have bit-exact KV (causal attention + positional rope), so the
  // output is unchanged.
  struct CachedPrefix {
    std::unique_ptr<SequenceBlockTable> table;
    std::vector<int> tokens;
    std::uint64_t last_use = 0;
  };

  void grow_table(StreamSeq& s, int upto_pos);
  // Stable quality-slot allocation: smallest-free-first so backend side buffers stay dense.
  int acquire_kv_slot();
  void release_kv_slot(int slot);

  BatchBackend* backend_;
  BlockAllocator* alloc_;
  BatchSchedulerOptions opts_;
  std::vector<StreamSeq> seqs_;
  // Reused across steps so per-row logit buffers keep their vocab-sized capacity instead of
  // reallocating every decode step (the backend fills it via decode_batched_logits).
  std::vector<std::vector<float>> batch_logits_scratch_;
  // Reused per-row candidate scratch for the device top-k fast path.
  std::vector<std::vector<detail::SampleCandidate>> batch_cand_scratch_;
  BatchTopkParams batch_topk_scratch_;      // reused per-row k / penalty / seen-id inputs
  BatchArgmaxParams batch_argmax_scratch_;  // reused per-row penalty / blocked inputs (greedy path)
  std::vector<int> batch_argmax_out_;       // reused per-row winner ids (greedy path)
  std::vector<CachedPrefix> prefix_cache_;
  std::uint64_t prefix_cache_tick_ = 0;
  static constexpr std::size_t kPrefixCacheEntries = 32;
  std::vector<int> free_kv_slots_;  // released slot ids, reused smallest-first
  int next_kv_slot_ = 0;            // high-watermark of slots ever handed out
};

}  // namespace engine
