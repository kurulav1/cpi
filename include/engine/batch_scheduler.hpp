#pragma once

// The continuous-batching scheduler, with no backend in it.
//
// Admission, preemption, block-table growth, shared-prefix reuse with an LRU cache,
// per-request sampling and retirement are all pure host bookkeeping over
// engine::BlockAllocator / SequenceBlockTable (which are themselves backend-free). This
// logic lived inside LlamaEngine, which made continuous batching a CUDA-only feature for
// no reason other than where the code sat: of its ~200 lines, exactly TWO touch a GPU.
//
// Those two are the BatchBackend interface below. A backend supplies a suffix prefill and a
// batched decode step; everything else is shared. That keeps the second backend honest --
// Metal gets the same admission policy, the same preemption order and the same prefix cache
// as CUDA, because it is literally the same code, not a reimplementation that agrees today.
//
// Threading: not thread-safe, single owner. The engine serialises access, as before.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/paged_kv.hpp"

namespace grammar {
class GrammarSampler;
}

namespace engine {

// Per-request generation parameters. Lives here rather than inside an engine because both
// the scheduler and the app-layer batch worker speak it, and neither should need a backend
// header to do so.
struct StreamParams {
  int max_new_tokens = 16;
  int min_new_tokens = 0;
  float temperature = 0.0f;
  std::vector<int> stop_ids;
  // Non-owning; the caller keeps it alive until the request finishes.
  grammar::GrammarSampler* grammar = nullptr;
};

// One token for one request.
//
// finish_reason is "eos" | "stop" | "length" | "preempted". PREEMPTED IS PART OF THE
// CONTRACT, not an error: it arrives with token == -1 and means the KV pool ran out and this
// request's blocks were reclaimed so older ones could finish. The caller is expected to
// re-admit it with (prompt + generated-so-far) and a decremented budget -- the scheduler
// deliberately does not do that itself, because only the caller knows the request's remaining
// budget and whether it still wants the answer.
struct StreamEvent {
  std::string id;
  int token = 0;
  bool finished = false;
  const char* finish_reason = "";
};

// What the scheduler needs from a GPU. Two calls.
class BatchBackend {
public:
  virtual ~BatchBackend() = default;

  // Prefill prompt[shared .. end) into the sequence described by `block_table`. The KV for
  // [0, shared) is already in the pool via adopted (refcounted) blocks, so it must not be
  // recomputed -- and must not be assumed contiguous with the suffix.
  //
  // Must be complete before returning: the scheduler may reuse shared engine state (a block
  // table upload, a scratch buffer) on the very next call.
  virtual void prefill_suffix(const std::vector<int>& prompt_tokens, int shared_tokens,
                              const std::vector<int>& block_table) = 0;

  // One decode step for N sequences. tokens[b]/positions[b] are sequence b's newest token and
  // its position; block_tables_flat is [N][max_blocks] row-major. out_logits is resized to N
  // rows of vocab.
  virtual void decode_batched_logits(const std::vector<int>& tokens,
                                     const std::vector<int>& positions,
                                     const std::vector<int>& block_tables_flat, int max_blocks,
                                     std::vector<std::vector<float>>& out_logits) = 0;
};

// Sampling and limits the scheduler applies. These mirror the engine options the CUDA path
// read directly off LlamaEngine::options_; passing them in is what removes the dependency.
struct BatchSchedulerOptions {
  int paged_block_size = 32;
  int max_context = 2048;
  int eos_token_id = -1;
  int top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  int no_repeat_ngram_size = 0;
  bool verbose = false;
};

class BatchScheduler {
public:
  // Neither pointer is owned; both must outlive the scheduler.
  BatchScheduler(BatchBackend* backend, BlockAllocator* alloc, const BatchSchedulerOptions& opts);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  // THROWS if the KV pool cannot seat the prompt. The caller rejects the request (or, when
  // re-admitting a preempted one, retries later) -- it is not a scheduler decision.
  void admit(const std::string& id, const std::vector<int>& prompt_tokens,
             const StreamParams& params);

  // Advance every running request by one token. Returns false only when nothing ran; note it
  // returns TRUE with events when all survivors were preempted, so the caller still sees them.
  bool step(std::vector<StreamEvent>& events);

  bool cancel(const std::string& id);
  int active() const;

  // Drop the shared-prefix cache. The caller must do this whenever the KV cache is wiped
  // underneath the scheduler (reset_kv_cache): the cached entries hold refcounted blocks whose
  // CONTENTS the wipe invalidates, and nothing about the block ids themselves would reveal
  // that -- a later admit would happily adopt them and prefill garbage. Running sequences are
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

  BatchBackend* backend_;
  BlockAllocator* alloc_;
  BatchSchedulerOptions opts_;
  std::vector<StreamSeq> seqs_;
  std::vector<CachedPrefix> prefix_cache_;
  std::uint64_t prefix_cache_tick_ = 0;
  static constexpr std::size_t kPrefixCacheEntries = 32;
};

}  // namespace engine
