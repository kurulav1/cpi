#include "engine/batch_scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "engine/sampling.hpp"
#include "grammar/grammar_sampler.hpp"

namespace engine {

BatchScheduler::BatchScheduler(BatchBackend* backend, BlockAllocator* alloc,
                               const BatchSchedulerOptions& opts)
    : backend_(backend), alloc_(alloc), opts_(opts) {
  if (backend_ == nullptr || alloc_ == nullptr) {
    throw std::runtime_error("BatchScheduler: backend and allocator are required");
  }
}

BatchScheduler::~BatchScheduler() = default;

int BatchScheduler::active() const {
  return static_cast<int>(seqs_.size());
}

int BatchScheduler::acquire_kv_slot() {
  if (!free_kv_slots_.empty()) {
    // Smallest-first keeps the backend's per-slot side buffers dense.
    const auto it = std::min_element(free_kv_slots_.begin(), free_kv_slots_.end());
    const int slot = *it;
    free_kv_slots_.erase(it);
    return slot;
  }
  return next_kv_slot_++;
}

void BatchScheduler::release_kv_slot(int slot) {
  free_kv_slots_.push_back(slot);
}

void BatchScheduler::grow_table(StreamSeq& s, int upto_pos) {
  const int bs = opts_.paged_block_size > 0 ? opts_.paged_block_size : 32;
  if (!s.blocks->ensure_position(upto_pos)) {
    // The prefix cache pins KV blocks that are pure optimization, not live state. Under pool
    // pressure, drop it to reclaim those blocks and retry once before declaring exhaustion
    // (callers then reject the admit / preempt in decode).
    if (!prefix_cache_.empty()) {
      prefix_cache_.clear();
      if (opts_.verbose || std::getenv("CPI_PREFIX_LOG")) {
        std::fprintf(stderr, "[stream] prefix_cache evicted under KV pressure\n");
      }
    }
    if (!s.blocks->ensure_position(upto_pos)) {
      throw std::runtime_error(
          "stream scheduler: paged KV pool exhausted (concurrent length > max_context budget)");
    }
  }
  const int nblk = upto_pos / bs + 1;
  if (static_cast<int>(s.table.size()) != nblk) {
    s.table.resize(static_cast<std::size_t>(nblk));
    for (int c = 0; c < nblk; ++c) {
      s.table[static_cast<std::size_t>(c)] = s.blocks->block_for(c * bs);
    }
  }
}

void BatchScheduler::admit(const std::string& id, const std::vector<int>& prompt_tokens,
                           const StreamParams& params) {
  if (prompt_tokens.empty()) throw std::runtime_error("stream_admit: empty prompt");
  const int bs = opts_.paged_block_size > 0 ? opts_.paged_block_size : 32;

  StreamSeq s;
  s.id = id;
  s.blocks = std::make_unique<SequenceBlockTable>(alloc_, bs);
  s.history = prompt_tokens;
  s.last_token = prompt_tokens.back();
  s.pos = static_cast<int>(prompt_tokens.size()) - 1;
  s.params = params;

  // Shared-prefix reuse: adopt the cached prefix's KV blocks for the longest whole-BLOCK
  // common prefix, so only the suffix is prefilled. Capped below the prompt length so the
  // final block is always (re)prefilled to seed decode. Must run before grow_table allocates.
  // Skipped entirely when the backend's per-sequence side state cannot survive adoption
  // (opts_.disable_prefix_reuse; see BatchSchedulerOptions).
  int shared_tokens = 0;
  if (!opts_.disable_prefix_reuse) {
    const int cap = static_cast<int>(prompt_tokens.size()) - 1;
    CachedPrefix* best = nullptr;
    int best_whole = 0;
    for (auto& e : prefix_cache_) {
      const int mmax = std::min(static_cast<int>(e.tokens.size()), cap);
      int match = 0;
      while (match < mmax && prompt_tokens[static_cast<std::size_t>(match)] ==
                                 e.tokens[static_cast<std::size_t>(match)]) {
        ++match;
      }
      const int whole = (match / bs) * bs;  // share_prefix_from adopts whole blocks only
      if (whole > best_whole) {
        best_whole = whole;
        best = &e;
      }
    }
    if (best && best_whole >= bs && s.blocks->share_prefix_from(*best->table, best_whole)) {
      shared_tokens = best_whole;
      best->last_use = ++prefix_cache_tick_;
    }
  }

  grow_table(s, s.pos);

  // Slot assigned only after grow_table can no longer throw (pool exhaustion rejects the
  // admit before any slot is consumed); released again if the prefill itself throws.
  s.kv_slot = acquire_kv_slot();

  // The backend contract says this is complete on return, so shared engine state (a block
  // table upload, scratch buffers) is free for the next call.
  try {
    backend_->prefill_suffix(prompt_tokens, shared_tokens, s.table, s.kv_slot);
  } catch (...) {
    release_kv_slot(s.kv_slot);
    throw;
  }

  // Refresh the cache with this sequence's block-aligned prefix (a fresh table that refcounts
  // the blocks so they outlive this sequence). If this request exactly extends the entry it
  // reused, update that entry in place so a growing chat stays one slot; otherwise insert and
  // LRU-evict to the cap.
  const int cache_tokens = (static_cast<int>(prompt_tokens.size()) / bs) * bs;
  if (!opts_.disable_prefix_reuse && cache_tokens >= bs) {
    auto next_table = std::make_unique<SequenceBlockTable>(alloc_, bs);
    if (next_table->share_prefix_from(*s.blocks, cache_tokens)) {
      CachedPrefix* slot = nullptr;
      for (auto& e : prefix_cache_) {
        if (static_cast<int>(e.tokens.size()) == shared_tokens && shared_tokens > 0 &&
            std::equal(e.tokens.begin(), e.tokens.end(), prompt_tokens.begin())) {
          slot = &e;  // this request extends exactly this entry
          break;
        }
      }
      if (!slot) {
        if (prefix_cache_.size() >= kPrefixCacheEntries) {
          auto lru = std::min_element(
              prefix_cache_.begin(), prefix_cache_.end(),
              [](const CachedPrefix& a, const CachedPrefix& b) { return a.last_use < b.last_use; });
          *lru = CachedPrefix{};  // release its block refs before reuse
          slot = &*lru;
        } else {
          prefix_cache_.emplace_back();
          slot = &prefix_cache_.back();
        }
      }
      slot->table = std::move(next_table);
      slot->tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + cache_tokens);
      slot->last_use = ++prefix_cache_tick_;
    }
  }

  if ((opts_.verbose || std::getenv("CPI_PREFIX_LOG")) && shared_tokens > 0) {
    std::fprintf(stderr, "[stream] prefix_reuse tokens=%d/%zu\n", shared_tokens,
                 prompt_tokens.size());
  }
  seqs_.push_back(std::move(s));
}

bool BatchScheduler::step(std::vector<StreamEvent>& events) {
  events.clear();
  if (seqs_.empty()) return false;

  // Grow each running sequence's table to cover the position we will write. Under KV-pool
  // pressure grow_table first reclaims prefix-cache blocks; if a sequence still cannot grow,
  // preempt it (free its KV, report "preempted") rather than throwing and taking down the
  // whole batch. Sequences are held in admission order and grown oldest-first, so the newest
  // hit the wall first; preemption is newest-first, preserving older (closer-to-finish)
  // requests.
  for (std::size_t b = 0; b < seqs_.size();) {
    try {
      grow_table(seqs_[b], seqs_[b].pos);
      ++b;
    } catch (const std::exception&) {
      std::fprintf(stderr, "[stream] preempted %s: paged KV pool exhausted (%d running)\n",
                   seqs_[b].id.c_str(), static_cast<int>(seqs_.size()));
      events.push_back(StreamEvent{seqs_[b].id, -1, true, "preempted"});
      release_kv_slot(seqs_[b].kv_slot);
      seqs_.erase(seqs_.begin() + static_cast<std::ptrdiff_t>(b));
    }
  }
  const int B = static_cast<int>(seqs_.size());
  if (B == 0) return !events.empty();  // deliver any preempt events, then idle

  // Assemble the running batch over the survivors.
  int max_blocks = 0;
  for (auto& s : seqs_) {
    max_blocks = std::max(max_blocks, static_cast<int>(s.table.size()));
  }
  std::vector<int> toks(static_cast<std::size_t>(B)), poss(static_cast<std::size_t>(B));
  std::vector<int> slots(static_cast<std::size_t>(B));
  std::vector<int> flat(static_cast<std::size_t>(B) * static_cast<std::size_t>(max_blocks), 0);
  for (int b = 0; b < B; ++b) {
    toks[static_cast<std::size_t>(b)] = seqs_[static_cast<std::size_t>(b)].last_token;
    poss[static_cast<std::size_t>(b)] = seqs_[static_cast<std::size_t>(b)].pos;
    slots[static_cast<std::size_t>(b)] = seqs_[static_cast<std::size_t>(b)].kv_slot;
    const auto& t = seqs_[static_cast<std::size_t>(b)].table;
    for (std::size_t c = 0; c < t.size(); ++c) {
      flat[static_cast<std::size_t>(b) * static_cast<std::size_t>(max_blocks) + c] = t[c];
    }
  }

  // Try the device top-k fast path: only ~k candidates/row cross to the host instead of the full
  // [batch][vocab] logits. Valid only without full-vocabulary features (repetition penalty,
  // n-gram blocking) and without any grammar, and only when every row actually samples
  // (temperature > 0). Anything else uses the full-logits path.
  // CPI_BATCH_TOPK=0 forces the full-logits path (A/B lever and a safety switch).
  static const bool topk_enabled = [] {
    const char* e = std::getenv("CPI_BATCH_TOPK");
    return e == nullptr || e[0] != '0';
  }();
  // Eligibility is per row (each request's own params): the whole batch takes the fast path only
  // if every row samples (temperature > 0), asks for top-k, and uses no n-gram blocking or
  // grammar (which need the full vocab on the host). Repetition penalty is applied ON DEVICE
  // before the top-k, so it no longer vetoes the fast path; each penalty row's unique seen ids
  // are gathered and passed down.
  bool used_topk = false;
  if (topk_enabled) {
    bool eligible = true;
    BatchTopkParams& sp = batch_topk_scratch_;
    sp.k.clear();
    sp.penalty.clear();
    sp.penalty_ids.clear();
    sp.penalty_rows.clear();
    for (int b = 0; b < B; ++b) {
      const StreamParams& pr = seqs_[static_cast<std::size_t>(b)].params;
      if (pr.grammar || pr.temperature <= 0.0f || pr.top_k <= 0 || pr.no_repeat_ngram_size > 1) {
        eligible = false;
        break;
      }
      sp.k.push_back(pr.top_k);
      sp.penalty.push_back(pr.repetition_penalty);
      if (pr.repetition_penalty > 1.0f) {
        // Unique seen ids for this row; matches the host's unordered_set(history).
        const auto& hist = seqs_[static_cast<std::size_t>(b)].history;
        std::unordered_set<int> seen(hist.begin(), hist.end());
        for (int id : seen) {
          sp.penalty_ids.push_back(id);
          sp.penalty_rows.push_back(b);
        }
      }
    }
    if (eligible) {
      used_topk = backend_->decode_batched_topk(toks, poss, flat, max_blocks, slots, sp,
                                                batch_cand_scratch_);
    }
  }

  // Try the device greedy (argmax) fast path when the top-k path did not run: a batch of greedy
  // rows returns B winner ids via a device reduction instead of shipping [batch][vocab] logits for
  // a host argmax. Eligible only if every row is greedy (temperature <= 0) with no grammar and no
  // n-gram blocking (both need the full vocab on the host); repetition penalty is applied on-device
  // before the argmax, and EOS is excluded per row while below min_new_tokens.
  // CPI_BATCH_ARGMAX=0 forces the full-logits path (A/B lever and a safety switch).
  static const bool argmax_enabled = [] {
    const char* e = std::getenv("CPI_BATCH_ARGMAX");
    return e == nullptr || e[0] != '0';
  }();
  bool used_argmax = false;
  if (!used_topk && argmax_enabled) {
    bool eligible = true;
    BatchArgmaxParams& ap = batch_argmax_scratch_;
    ap.penalty.clear();
    ap.penalty_ids.clear();
    ap.penalty_rows.clear();
    ap.blocked.clear();
    for (int b = 0; b < B; ++b) {
      const StreamSeq& s = seqs_[static_cast<std::size_t>(b)];
      const StreamParams& pr = s.params;
      if (pr.grammar || pr.temperature > 0.0f || pr.no_repeat_ngram_size > 1) {
        eligible = false;
        break;
      }
      ap.penalty.push_back(pr.repetition_penalty);
      if (pr.repetition_penalty > 1.0f) {
        const auto& hist = s.history;
        std::unordered_set<int> seen(hist.begin(), hist.end());
        for (int id : seen) {
          ap.penalty_ids.push_back(id);
          ap.penalty_rows.push_back(b);
        }
      }
      // Suppress EOS below the min_new floor by excluding it from the argmax (the argmax-path
      // equivalent of the -inf logit the full path sets).
      const bool block_eos = s.generated < pr.min_new_tokens && opts_.eos_token_id >= 0;
      ap.blocked.push_back(block_eos ? opts_.eos_token_id : -1);
    }
    if (eligible) {
      used_argmax = backend_->decode_batched_argmax(toks, poss, flat, max_blocks, slots, ap,
                                                    batch_argmax_out_);
    }
  }

  std::vector<int> finished;  // indices into seqs_
  // Common post-sample bookkeeping, shared by both the candidate and full-logits paths.
  auto finish_row = [&](int b, int tok) {
    StreamSeq& s = seqs_[static_cast<std::size_t>(b)];
    if (s.params.grammar) s.params.grammar->accept(tok);
    s.history.push_back(tok);
    s.last_token = tok;
    ++s.pos;
    ++s.generated;
    const bool is_stop = std::find(s.params.stop_ids.begin(), s.params.stop_ids.end(), tok) !=
                         s.params.stop_ids.end();
    const char* reason = "";
    bool fin = false;
    if (is_stop) {
      fin = true;
      reason = (tok == opts_.eos_token_id) ? "eos" : "stop";
    } else if (s.generated >= s.params.max_new_tokens || s.pos >= opts_.max_context) {
      fin = true;
      reason = "length";
    }
    events.push_back(StreamEvent{s.id, tok, fin, reason});
    if (fin) finished.push_back(b);
  };

  if (used_topk) {
    for (int b = 0; b < B; ++b) {
      StreamSeq& s = seqs_[static_cast<std::size_t>(b)];
      auto& cand = batch_cand_scratch_[static_cast<std::size_t>(b)];
      // min_new_tokens: below the floor, drop EOS from the candidate set (the candidate-path
      // equivalent of the -inf logit the full path sets).
      if (s.generated < s.params.min_new_tokens && opts_.eos_token_id >= 0) {
        cand.erase(std::remove_if(cand.begin(), cand.end(),
                                  [&](const detail::SampleCandidate& c) {
                                    return c.id == opts_.eos_token_id;
                                  }),
                   cand.end());
      }
      finish_row(
          b, detail::dispatch_sample_from_candidates(cand, s.params.temperature, s.params.top_p));
    }
  } else if (used_argmax) {
    // Device already picked each row's winner (penalty + EOS suppression applied on-device).
    for (int b = 0; b < B; ++b) {
      finish_row(b, batch_argmax_out_[static_cast<std::size_t>(b)]);
    }
  } else {
    std::vector<std::vector<float>>& logits = batch_logits_scratch_;
    backend_->decode_batched_logits(toks, poss, flat, max_blocks, slots, logits);
    for (int b = 0; b < B; ++b) {
      StreamSeq& s = seqs_[static_cast<std::size_t>(b)];
      std::vector<float>& lg = logits[static_cast<std::size_t>(b)];
      if (s.params.grammar) s.params.grammar->apply_mask(lg);
      // Suppress EOS until min_new_tokens (skip when a grammar drives termination).
      if (s.generated < s.params.min_new_tokens && !s.params.grammar && opts_.eos_token_id >= 0 &&
          static_cast<std::size_t>(opts_.eos_token_id) < lg.size()) {
        lg[static_cast<std::size_t>(opts_.eos_token_id)] = -std::numeric_limits<float>::infinity();
      }
      finish_row(b, detail::dispatch_sample_from_logits(lg, s.params.temperature, s.params.top_k,
                                                        s.params.top_p, s.params.repetition_penalty,
                                                        s.params.no_repeat_ngram_size, s.history));
    }
  }

  // Free finished requests' blocks and remove them (iterate high->low to keep indices valid).
  // RAII on the unique_ptr releases blocks to the pool.
  for (auto it = finished.rbegin(); it != finished.rend(); ++it) {
    release_kv_slot(seqs_[static_cast<std::size_t>(*it)].kv_slot);
    seqs_.erase(seqs_.begin() + *it);
  }
  return true;
}

void BatchScheduler::clear_prefix_cache() {
  prefix_cache_.clear();
}

bool BatchScheduler::cancel(const std::string& id) {
  for (std::size_t i = 0; i < seqs_.size(); ++i) {
    if (seqs_[i].id == id) {
      release_kv_slot(seqs_[i].kv_slot);
      seqs_.erase(seqs_.begin() + static_cast<std::ptrdiff_t>(i));
      return true;
    }
  }
  return false;
}

}  // namespace engine
