#include "engine/batch_scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

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

  // Shared-prefix reuse: adopt the cached prefix's KV blocks for the longest WHOLE-BLOCK
  // common prefix, so only the suffix is prefilled. Capped below the prompt length so the
  // final block is always (re)prefilled to seed decode. Must run before grow_table allocates.
  int shared_tokens = 0;
  {
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

  // The backend contract says this is complete on return, so shared engine state (a block
  // table upload, scratch buffers) is free for the next call.
  backend_->prefill_suffix(prompt_tokens, shared_tokens, s.table);

  // Refresh the cache with this sequence's block-aligned prefix (a fresh table that refcounts
  // the blocks so they outlive this sequence). If this request exactly extends the entry it
  // reused, update that entry in place so a growing chat stays one slot; otherwise insert and
  // LRU-evict to the cap.
  const int cache_tokens = (static_cast<int>(prompt_tokens.size()) / bs) * bs;
  if (cache_tokens >= bs) {
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
  // hit the wall first -- preemption is newest-first, preserving older (closer-to-finish)
  // requests.
  for (std::size_t b = 0; b < seqs_.size();) {
    try {
      grow_table(seqs_[b], seqs_[b].pos);
      ++b;
    } catch (const std::exception&) {
      std::fprintf(stderr, "[stream] preempted %s: paged KV pool exhausted (%d running)\n",
                   seqs_[b].id.c_str(), static_cast<int>(seqs_.size()));
      events.push_back(StreamEvent{seqs_[b].id, -1, true, "preempted"});
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
  std::vector<int> flat(static_cast<std::size_t>(B) * static_cast<std::size_t>(max_blocks), 0);
  for (int b = 0; b < B; ++b) {
    toks[static_cast<std::size_t>(b)] = seqs_[static_cast<std::size_t>(b)].last_token;
    poss[static_cast<std::size_t>(b)] = seqs_[static_cast<std::size_t>(b)].pos;
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
  bool used_topk = false;
  if (topk_enabled && opts_.top_k > 0 && opts_.repetition_penalty <= 1.0f &&
      opts_.no_repeat_ngram_size <= 1) {
    bool eligible = true;
    for (const auto& s : seqs_) {
      if (s.params.grammar || s.params.temperature <= 0.0f) {
        eligible = false;
        break;
      }
    }
    if (eligible) {
      used_topk = backend_->decode_batched_topk(toks, poss, flat, max_blocks, opts_.top_k,
                                                 batch_cand_scratch_);
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
    const bool is_stop =
        std::find(s.params.stop_ids.begin(), s.params.stop_ids.end(), tok) != s.params.stop_ids.end();
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
      finish_row(b, detail::dispatch_sample_from_candidates(cand, s.params.temperature, opts_.top_p));
    }
  } else {
    std::vector<std::vector<float>>& logits = batch_logits_scratch_;
    backend_->decode_batched_logits(toks, poss, flat, max_blocks, logits);
    for (int b = 0; b < B; ++b) {
      StreamSeq& s = seqs_[static_cast<std::size_t>(b)];
      std::vector<float>& lg = logits[static_cast<std::size_t>(b)];
      if (s.params.grammar) s.params.grammar->apply_mask(lg);
      // Suppress EOS until min_new_tokens (skip when a grammar drives termination).
      if (s.generated < s.params.min_new_tokens && !s.params.grammar && opts_.eos_token_id >= 0 &&
          static_cast<std::size_t>(opts_.eos_token_id) < lg.size()) {
        lg[static_cast<std::size_t>(opts_.eos_token_id)] = -std::numeric_limits<float>::infinity();
      }
      finish_row(b, detail::dispatch_sample_from_logits(
                        lg, s.params.temperature, opts_.top_k, opts_.top_p,
                        opts_.repetition_penalty, opts_.no_repeat_ngram_size, s.history));
    }
  }

  // Free finished requests' blocks and remove them (iterate high->low to keep indices valid).
  // RAII on the unique_ptr releases blocks to the pool.
  for (auto it = finished.rbegin(); it != finished.rend(); ++it) {
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
      seqs_.erase(seqs_.begin() + static_cast<std::ptrdiff_t>(i));
      return true;
    }
  }
  return false;
}

}  // namespace engine
