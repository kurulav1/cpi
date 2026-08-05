// Multiplexed (continuous-batching) interactive worker.
//
// Opt-in via --interactive-batch. Unlike the single-request interactive loop,
// this drives the engine's streaming batch scheduler (stream_admit/stream_step):
// requests may arrive at any time, are prefilled into their own paged blocks,
// and decode together one step per outer iteration; finished requests free their
// blocks and drop out of the batch. Emits the same JSON-line events as the
// single worker (start / delta / done / error), tagged by request id, so the
// Node layer can multiplex several SSE streams onto one worker.
//
// Requires a tokenizer + --paged-blocks. Takes the scheduler, not an engine: the worker only
// needs admit/step/cancel/active, and engine::BatchScheduler is backend-free, so this file is
// GPU-agnostic (no #if CPI_HAS_CUDA needed).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "app/main_helpers.hpp"
#include "app/main_modes.hpp"
#include "engine/batch_scheduler.hpp"
#include "grammar/grammar.hpp"
#include "grammar/grammar_sampler.hpp"
#include "grammar/json_schema_to_grammar.hpp"
#include "model/tokenizer.hpp"

namespace app::main_modes {

using app::main_helpers::json_escape;
using app::main_helpers::json_get_bool;
using app::main_helpers::json_get_float;
using app::main_helpers::json_get_int;
using app::main_helpers::json_get_raw_value;
using app::main_helpers::json_get_string;
using app::main_helpers::json_get_string_array;
using app::main_helpers::sanitize_stream_text;

void run_interactive_batch(engine::BatchScheduler& sched, model::Tokenizer& tokenizer,
                           const std::vector<std::string>& default_stop_texts, bool default_add_bos,
                           int default_max_new, float default_temp, int default_top_k,
                           float default_top_p, float default_repeat_penalty,
                           int default_no_repeat) {
  std::mutex out_mu;
  const auto write_event = [&](const std::string& type, const std::string& id,
                               const std::string& extra) {
    std::lock_guard<std::mutex> lk(out_mu);
    std::cout << "{\"type\":\"" << type << "\"";
    if (!id.empty()) std::cout << ",\"id\":\"" << json_escape(id) << "\"";
    if (!extra.empty()) std::cout << "," << extra;
    std::cout << "}\n" << std::flush;
  };

  struct Incoming {
    std::string id;
    std::vector<int> tokens;
    engine::StreamParams params;
    std::string json_schema;  // raw structural schema, compiled to a grammar at admit
  };
  std::mutex q_mu;
  std::condition_variable q_cv;
  std::deque<Incoming> queue;
  std::deque<std::string> cancels;  // ids to cancel (client disconnected)
  std::atomic<bool> shutdown{false};

  // Reader thread: parse stdin request lines into the queue. Blocks on getline;
  // a {"shutdown":true} line or stdin EOF ends it.
  std::thread reader([&]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::string id = json_get_string(line, "id");
      if (json_get_bool(line, "shutdown", false)) break;
      // Cancel command {"cancel":"<id>"}: reclaim a disconnected request.
      const std::string cancel_id = json_get_string(line, "cancel");
      if (!cancel_id.empty()) {
        {
          std::lock_guard<std::mutex> lk(q_mu);
          cancels.push_back(cancel_id);
        }
        q_cv.notify_all();
        continue;
      }
      try {
        Incoming in;
        in.id = id;
        const std::string prompt = json_get_string(line, "prompt");
        if (prompt.empty()) throw std::runtime_error("interactive request missing 'prompt'");
        const bool add_bos = json_get_bool(line, "add_bos", default_add_bos);
        in.tokens = tokenizer.encode(prompt, add_bos);
        if (in.tokens.empty()) throw std::runtime_error("prompt encoded to zero tokens");

        int mn = json_get_int(line, "max_new", default_max_new);
        if (mn < 1) mn = 1;
        float tp = json_get_float(line, "temp", default_temp);
        if (tp < 0.0f) tp = 0.0f;
        in.params.max_new_tokens = mn;
        in.params.temperature = tp;
        in.params.min_new_tokens = std::max(0, json_get_int(line, "min_new", 0));
        // Per-request sampling shape (falls back to the server defaults). Previously these were
        // scheduler-global, so a client's top_p/top_k was silently ignored in the batched path.
        in.params.top_k = json_get_int(line, "top_k", default_top_k);
        in.params.top_p = json_get_float(line, "top_p", default_top_p);
        in.params.repetition_penalty =
            json_get_float(line, "repeat_penalty", default_repeat_penalty);
        in.params.no_repeat_ngram_size = json_get_int(line, "no_repeat_ngram", default_no_repeat);
        in.json_schema = json_get_raw_value(line, "json_schema");

        std::vector<int> stop_ids;
        if (tokenizer.eos_id() >= 0) stop_ids.push_back(tokenizer.eos_id());
        std::vector<std::string> stops = json_get_string_array(line, "stop_texts");
        if (stops.empty()) stops = default_stop_texts;
        for (const auto& s : stops) {
          const auto t = tokenizer.encode(s, /*add_bos=*/false);
          if (t.size() == 1 &&
              std::find(stop_ids.begin(), stop_ids.end(), t[0]) == stop_ids.end()) {
            stop_ids.push_back(t[0]);
          }
        }
        in.params.stop_ids = std::move(stop_ids);

        {
          std::lock_guard<std::mutex> lk(q_mu);
          queue.push_back(std::move(in));
        }
        q_cv.notify_all();
      } catch (const std::exception& e) {
        write_event("error", id, "\"error\":\"" + json_escape(e.what()) + "\"");
      }
    }
    shutdown.store(true);
    q_cv.notify_all();
  });

  // Per-request incremental-detokenization state (decode full history, diff text).
  struct DetokState {
    std::vector<int> ids;
    std::string prev_text;
    std::chrono::steady_clock::time_point t0;  // decode start (for tok/s)
  };
  std::unordered_map<std::string, DetokState> detok;
  // Per-request grammar samplers: StreamParams.grammar is a non-owning pointer
  // that must outlive the request, so the worker owns them here until done/cancel.
  std::unordered_map<std::string, std::unique_ptr<grammar::GrammarSampler>> grammars;

  // Requeue-and-resume for preempted requests. When the engine preempts a
  // request under KV pressure it emits a "preempted" event; instead of closing
  // the client stream we pause it here (keeping its detok/grammar state) and
  // re-admit it (re-prefilling prompt+generated-so-far) once a running request
  // finishes and frees blocks. Since the pool is always >= max_context, any one
  // request fits alone, so a preempted request always resumes eventually.
  struct ResumeInfo {
    std::vector<int> prompt;      // original prompt tokens (pre-generation)
    engine::StreamParams params;  // original params (grammar ptr preserved)
    int orig_max_new = 0;
    int orig_min_new = 0;
    int retries = 0;
  };
  std::unordered_map<std::string, ResumeInfo> resume;
  std::deque<std::string> waiting;  // preempted ids awaiting a free slot
  bool slot_freed = false;          // a running request finished this cycle
  constexpr int kMaxResumeRetries = 16;

  while (true) {
    // Admit any queued requests (prefill happens inside stream_admit).
    std::vector<Incoming> admits;
    std::vector<std::string> cancel_ids;
    {
      std::unique_lock<std::mutex> lk(q_mu);
      if (queue.empty() && cancels.empty() && waiting.empty() && sched.active() == 0) {
        if (shutdown.load()) break;
        q_cv.wait(lk, [&]() { return !queue.empty() || !cancels.empty() || shutdown.load(); });
      }
      while (!cancels.empty()) {
        cancel_ids.push_back(std::move(cancels.front()));
        cancels.pop_front();
      }
      // Drop cancelled requests still waiting in the queue (cancel beat admit).
      if (!cancel_ids.empty() && !queue.empty()) {
        std::deque<Incoming> kept;
        while (!queue.empty()) {
          if (std::find(cancel_ids.begin(), cancel_ids.end(), queue.front().id) == cancel_ids.end())
            kept.push_back(std::move(queue.front()));
          queue.pop_front();
        }
        queue.swap(kept);
      }
      while (!queue.empty()) {
        admits.push_back(std::move(queue.front()));
        queue.pop_front();
      }
    }
    // Apply cancels for already-running requests (frees their KV blocks).
    for (const auto& cid : cancel_ids) {
      const bool evicted = sched.cancel(cid);
      detok.erase(cid);
      grammars.erase(cid);
      resume.erase(cid);
      waiting.erase(std::remove(waiting.begin(), waiting.end(), cid), waiting.end());
      std::cerr << "[batch] cancel " << cid << (evicted ? " (evicted)" : " (not running)") << "\n";
    }
    for (auto& in : admits) {
      try {
        // Grammar-constrained decoding: compile the request's json_schema to a
        // grammar sampler (owned in `grammars`, applied per-step in stream_step).
        // An unparseable schema falls back to unconstrained generation.
        if (!in.json_schema.empty()) {
          try {
            grammar::Grammar g =
                grammar::Grammar::parse(grammar::json_schema_to_grammar(in.json_schema));
            auto sampler = std::make_unique<grammar::GrammarSampler>(
                std::move(g), tokenizer.token_pieces(), tokenizer.eos_id());
            in.params.grammar = sampler.get();
            grammars[in.id] = std::move(sampler);
          } catch (const std::exception& ge) {
            std::cerr << "[batch] grammar compile failed for " << in.id << " (" << ge.what()
                      << "); unconstrained\n";
          }
        }
        sched.admit(in.id, in.tokens, in.params);
        DetokState st;
        st.t0 = std::chrono::steady_clock::now();
        detok[in.id] = std::move(st);
        resume[in.id] =
            ResumeInfo{in.tokens, in.params, in.params.max_new_tokens, in.params.min_new_tokens, 0};
        write_event("start", in.id, "");
      } catch (const std::exception& e) {
        write_event("error", in.id, "\"error\":\"" + json_escape(e.what()) + "\"");
      }
    }

    // Resume preempted requests once a slot has freed (a running request
    // finished) or the pool is now empty. Re-prefill prompt+generated so the
    // client stream continues; if the pool still can't fit it, keep it waiting.
    if (!waiting.empty() && (slot_freed || sched.active() == 0)) {
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
          // Give up rather than hang the client; report what we produced.
          write_event("done", rid,
                      "\"finish_reason\":\"preempted\",\"text\":\"" +
                          json_escape(dit->second.prev_text) + "\"");
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
          sched.admit(rid, full, p);  // no "start"/detok reset: stream continues
          std::fprintf(stderr, "[batch] resumed %s (tokens=%zu, budget=%d, try=%d)\n", rid.c_str(),
                       full.size(), p.max_new_tokens, rit->second.retries);
        } catch (const std::exception&) {
          still_waiting.push_back(rid);  // no room yet; retry after the next finish
        }
      }
      waiting = std::move(still_waiting);
    }

    if (sched.active() == 0) {
      if (shutdown.load()) break;
      continue;
    }

    // One decode step over every running request.
    std::vector<engine::StreamEvent> events;
    sched.step(events);
    for (const auto& e : events) {
      auto it = detok.find(e.id);
      if (it == detok.end()) continue;
      // A request ending on its stop token (eos/stop) must not emit that token's
      // text: a chat model's EOS is special (decode drops it), but a grammar
      // terminates on tokenizer.eos_id() which can decode to a visible "</s>".
      // Keep the token for a "length" finish (it is real content).
      const std::string reason = e.finish_reason ? e.finish_reason : "";
      // Preemption is not a finish: pause the request and requeue it (its detok
      // and grammar state are kept) so it resumes when a slot frees.
      if (e.finished && reason == "preempted") {
        waiting.push_back(e.id);
        continue;
      }
      const bool terminator = e.finished && (reason == "eos" || reason == "stop");
      // token < 0 is the preempt sentinel (no real token to emit).
      if (!terminator && e.token >= 0) {
        it->second.ids.push_back(e.token);
        const std::string decoded = sanitize_stream_text(tokenizer.decode(it->second.ids));
        if (decoded.size() > it->second.prev_text.size()) {
          const std::string delta = decoded.substr(it->second.prev_text.size());
          if (!delta.empty()) {
            write_event("delta", e.id, "\"delta\":\"" + json_escape(delta) + "\"");
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
        char nums[160];
        std::snprintf(nums, sizeof(nums),
                      ",\"generated\":%d,\"elapsed_ms\":%.1f,\"tok_per_s\":%.2f", gen, ms, tps);
        write_event("done", e.id,
                    "\"finish_reason\":\"" + std::string(e.finish_reason) + "\",\"text\":\"" +
                        json_escape(it->second.prev_text) + "\"" + nums);
        detok.erase(it);
        grammars.erase(e.id);
        resume.erase(e.id);
        slot_freed = true;  // a slot opened; a waiting request can resume next cycle
      }
    }
  }

  if (reader.joinable()) reader.join();
}

}  // namespace app::main_modes
