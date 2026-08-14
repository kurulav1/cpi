// The continuous-batching scheduler loop, lifted out of the stdin/stdout worker
// so the in-binary HTTP server can drive the same code. See batch_worker.hpp.
#include "app/batch_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "app/main_helpers.hpp"
#include "grammar/grammar.hpp"
#include "grammar/grammar_sampler.hpp"
#include "grammar/json_schema_to_grammar.hpp"

namespace app {

using app::main_helpers::sanitize_stream_text;

BatchWorker::BatchWorker(engine::BatchScheduler& sched, model::Tokenizer& tokenizer,
                         BatchDefaults defaults, Sink sink)
    : sched_(sched),
      tokenizer_(tokenizer),
      defaults_(std::move(defaults)),
      sink_(std::move(sink)) {}

BatchWorker::~BatchWorker() = default;

void BatchWorker::emit(BatchEvent::Type type, const std::string& id, const std::string& text,
                       const std::string& finish_reason, const std::string& error, int generated,
                       double elapsed_ms, double tok_per_s, int prompt_tokens) {
  if (!sink_) return;
  BatchEvent e;
  e.type = type;
  e.id = id;
  e.text = text;
  e.finish_reason = finish_reason;
  e.error = error;
  e.generated = generated;
  e.prompt_tokens = prompt_tokens;
  e.elapsed_ms = elapsed_ms;
  e.tok_per_s = tok_per_s;
  sink_(e);
}

bool BatchWorker::submit(const std::string& id, const std::string& prompt,
                         const BatchOverrides& ov, std::string* error) {
  const auto fail = [&](const char* msg) {
    if (error) *error = msg;
    return false;
  };
  if (prompt.empty()) return fail("request missing 'prompt'");

  Incoming in;
  in.id = id;
  const bool add_bos = ov.has_add_bos ? ov.add_bos : defaults_.add_bos;
  try {
    in.tokens = tokenizer_.encode(prompt, add_bos);
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
  if (in.tokens.empty()) return fail("prompt encoded to zero tokens");
  // Same over-long-prompt clamp as the one-shot CLI and the interactive
  // worker: fit the prompt (keeping BOS plus the newest tokens) and warn on
  // stderr, rather than surfacing a scheduler admit failure to the client.
  main_helpers::clamp_prompt_to_context(in.tokens, sched_.max_context(), tokenizer_.bos_id());

  in.params.max_new_tokens = std::max(1, ov.max_new >= 0 ? ov.max_new : defaults_.max_new);
  in.params.temperature = std::max(0.0f, ov.temp >= 0.0f ? ov.temp : defaults_.temp);
  in.params.min_new_tokens = std::max(0, ov.min_new);
  in.params.top_k = ov.top_k >= 0 ? ov.top_k : defaults_.top_k;
  in.params.top_p = ov.top_p >= 0.0f ? ov.top_p : defaults_.top_p;
  in.params.repetition_penalty =
      ov.repeat_penalty >= 0.0f ? ov.repeat_penalty : defaults_.repeat_penalty;
  in.params.no_repeat_ngram_size =
      ov.no_repeat_ngram >= 0 ? ov.no_repeat_ngram : defaults_.no_repeat_ngram;
  in.json_schema = ov.json_schema;

  // Stop handling: the model's EOS plus any single-token stop strings. Multi-token
  // stop strings stay the transport's problem (it sees the decoded text).
  std::vector<int> stop_ids;
  if (tokenizer_.eos_id() >= 0) stop_ids.push_back(tokenizer_.eos_id());
  const std::vector<std::string>& stops =
      ov.stop_texts.empty() ? defaults_.stop_texts : ov.stop_texts;
  for (const auto& s : stops) {
    const auto t = tokenizer_.encode(s, /*add_bos=*/false);
    if (t.size() == 1 && std::find(stop_ids.begin(), stop_ids.end(), t[0]) == stop_ids.end()) {
      stop_ids.push_back(t[0]);
    }
  }
  in.params.stop_ids = std::move(stop_ids);

  {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back(std::move(in));
  }
  cv_.notify_all();
  return true;
}

void BatchWorker::cancel(const std::string& id) {
  if (id.empty()) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(id);
  }
  cv_.notify_all();
}

void BatchWorker::run_exclusive(const std::function<void()>& task) {
  ExclusiveTask slot;
  slot.fn = &task;
  {
    std::lock_guard<std::mutex> lk(mu_);
    exclusive_.push_back(&slot);
  }
  cv_.notify_all();
  std::unique_lock<std::mutex> lk(mu_);
  exclusive_cv_.wait(lk, [&]() { return slot.done; });
}

void BatchWorker::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
}

void BatchWorker::run() {
  // Per-request incremental-detokenization state (decode full history, diff text).
  struct DetokState {
    std::vector<int> ids;
    std::string prev_text;
    std::chrono::steady_clock::time_point t0;
    int prompt_tokens = 0;  // reported back as usage.prompt_tokens
  };
  std::unordered_map<std::string, DetokState> detok;
  // StreamParams.grammar is a non-owning pointer that must outlive the request,
  // so the samplers are owned here until done/cancel.
  std::unordered_map<std::string, std::unique_ptr<grammar::GrammarSampler>> grammars;

  // Requeue-and-resume for preempted requests: when the engine preempts under KV
  // pressure the request is paused (detok and grammar state kept) and re-admitted
  // once a slot frees. The pool is always >= max_context, so any one request fits
  // alone and a preempted request always resumes eventually.
  struct ResumeInfo {
    std::vector<int> prompt;
    engine::StreamParams params;
    int orig_max_new = 0;
    int orig_min_new = 0;
    int retries = 0;
  };
  std::unordered_map<std::string, ResumeInfo> resume;
  std::deque<std::string> waiting;
  bool slot_freed = false;
  constexpr int kMaxResumeRetries = 16;

  while (true) {
    // Exclusive tasks first: they need the engine to themselves, and running
    // them at the top of the iteration means no decode step is in flight.
    while (true) {
      ExclusiveTask* task = nullptr;
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (exclusive_.empty()) break;
        task = exclusive_.front();
        exclusive_.pop_front();
      }
      try {
        (*task->fn)();
      } catch (const std::exception& e) {
        std::cerr << "[batch] exclusive task threw: " << e.what() << "\n";
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        task->done = true;
      }
      exclusive_cv_.notify_all();
    }

    std::vector<Incoming> admits;
    std::vector<std::string> cancel_ids;
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (queue_.empty() && cancels_.empty() && exclusive_.empty() && waiting.empty() &&
          sched_.active() == 0) {
        if (stopping_) break;
        cv_.wait(lk, [&]() {
          return !queue_.empty() || !cancels_.empty() || !exclusive_.empty() || stopping_;
        });
      }
      while (!cancels_.empty()) {
        cancel_ids.push_back(std::move(cancels_.front()));
        cancels_.pop_front();
      }
      // Drop cancelled requests still queued (cancel beat admit).
      if (!cancel_ids.empty() && !queue_.empty()) {
        std::deque<Incoming> kept;
        while (!queue_.empty()) {
          if (std::find(cancel_ids.begin(), cancel_ids.end(), queue_.front().id) ==
              cancel_ids.end()) {
            kept.push_back(std::move(queue_.front()));
          }
          queue_.pop_front();
        }
        queue_.swap(kept);
      }
      while (!queue_.empty()) {
        admits.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }

    for (const auto& cid : cancel_ids) {
      sched_.cancel(cid);
      detok.erase(cid);
      grammars.erase(cid);
      resume.erase(cid);
      waiting.erase(std::remove(waiting.begin(), waiting.end(), cid), waiting.end());
    }

    for (auto& in : admits) {
      try {
        if (!in.json_schema.empty()) {
          try {
            grammar::Grammar g =
                grammar::Grammar::parse(grammar::json_schema_to_grammar(in.json_schema));
            auto sampler = std::make_unique<grammar::GrammarSampler>(
                std::move(g), tokenizer_.token_pieces(), tokenizer_.eos_id());
            in.params.grammar = sampler.get();
            grammars[in.id] = std::move(sampler);
          } catch (const std::exception& ge) {
            std::cerr << "[batch] grammar compile failed for " << in.id << " (" << ge.what()
                      << "); unconstrained\n";
          }
        }
        sched_.admit(in.id, in.tokens, in.params);
        DetokState st;
        st.t0 = std::chrono::steady_clock::now();
        st.prompt_tokens = static_cast<int>(in.tokens.size());
        detok[in.id] = std::move(st);
        resume[in.id] =
            ResumeInfo{in.tokens, in.params, in.params.max_new_tokens, in.params.min_new_tokens, 0};
        emit(BatchEvent::Type::Start, in.id, "", "", "", 0, 0.0, 0.0);
      } catch (const std::exception& e) {
        emit(BatchEvent::Type::Error, in.id, "", "", e.what(), 0, 0.0, 0.0);
      }
    }

    if (!waiting.empty() && (slot_freed || sched_.active() == 0)) {
      slot_freed = false;
      std::deque<std::string> still_waiting;
      while (!waiting.empty()) {
        const std::string rid = waiting.front();
        waiting.pop_front();
        auto rit = resume.find(rid);
        auto dit = detok.find(rid);
        if (rit == resume.end() || dit == detok.end()) continue;  // cancelled meanwhile
        const int done_so_far = static_cast<int>(dit->second.ids.size());
        if (++rit->second.retries > kMaxResumeRetries) {
          emit(BatchEvent::Type::Done, rid, dit->second.prev_text, "preempted", "", done_so_far,
               0.0, 0.0, dit->second.prompt_tokens);
          grammars.erase(rid);
          detok.erase(dit);
          resume.erase(rit);
          continue;
        }
        std::vector<int> full = rit->second.prompt;
        full.insert(full.end(), dit->second.ids.begin(), dit->second.ids.end());
        engine::StreamParams p = rit->second.params;
        p.max_new_tokens = std::max(1, rit->second.orig_max_new - done_so_far);
        p.min_new_tokens = std::max(0, rit->second.orig_min_new - done_so_far);
        try {
          sched_.admit(rid, full, p);  // no Start/detok reset: the stream continues
        } catch (const std::exception&) {
          still_waiting.push_back(rid);  // no room yet; retry after the next finish
        }
      }
      waiting = std::move(still_waiting);
    }

    if (sched_.active() == 0) {
      std::lock_guard<std::mutex> lk(mu_);
      if (stopping_ && queue_.empty() && cancels_.empty() && exclusive_.empty()) break;
      continue;
    }

    std::vector<engine::StreamEvent> events;
    sched_.step(events);
    for (const auto& e : events) {
      auto it = detok.find(e.id);
      if (it == detok.end()) continue;
      const std::string reason = e.finish_reason ? e.finish_reason : "";
      // Preemption is not a finish: pause and requeue.
      if (e.finished && reason == "preempted") {
        waiting.push_back(e.id);
        continue;
      }
      // A request ending on its stop token must not emit that token's text: a chat
      // model's EOS is special (decode drops it), but a grammar terminates on
      // tokenizer.eos_id() which can decode to a visible "</s>". A "length" finish
      // keeps its token (real content).
      const bool terminator = e.finished && (reason == "eos" || reason == "stop");
      if (!terminator && e.token >= 0) {
        it->second.ids.push_back(e.token);
        const std::string decoded = sanitize_stream_text(tokenizer_.decode(it->second.ids));
        if (decoded.size() > it->second.prev_text.size()) {
          const std::string delta = decoded.substr(it->second.prev_text.size());
          if (!delta.empty()) {
            emit(BatchEvent::Type::Delta, e.id, delta, "", "", 0, 0.0, 0.0);
          }
          it->second.prev_text = decoded;
        }
      }
      if (e.finished) {
        const int gen = static_cast<int>(it->second.ids.size());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - it->second.t0)
                              .count();
        const double tps = ms > 0.0 ? (gen * 1000.0 / ms) : 0.0;
        emit(BatchEvent::Type::Done, e.id, it->second.prev_text, reason, "", gen, ms, tps,
             it->second.prompt_tokens);
        detok.erase(it);
        grammars.erase(e.id);
        resume.erase(e.id);
        slot_freed = true;
      }
    }
  }
}

}  // namespace app
