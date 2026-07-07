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
// Requires a tokenizer + --paged-blocks + --gpu-cache-all (LlamaEngine).
#include "app/main_modes.hpp"

#if LLAMA_ENGINE_HAS_CUDA

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
#include "engine/llama_engine.hpp"
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

void run_interactive_batch(engine::LlamaEngine& eng, model::Tokenizer& tokenizer,
                           const std::vector<std::string>& default_stop_texts, bool default_add_bos,
                           int default_max_new, float default_temp) {
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
    engine::LlamaEngine::StreamParams params;
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
      // Cancel command: {"cancel":"<id>"} — reclaim a disconnected request.
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

  while (true) {
    // Admit any queued requests (prefill happens inside stream_admit).
    std::vector<Incoming> admits;
    std::vector<std::string> cancel_ids;
    {
      std::unique_lock<std::mutex> lk(q_mu);
      if (queue.empty() && cancels.empty() && eng.stream_active() == 0) {
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
      const bool evicted = eng.stream_cancel(cid);
      detok.erase(cid);
      grammars.erase(cid);
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
        eng.stream_admit(in.id, in.tokens, in.params);
        DetokState st;
        st.t0 = std::chrono::steady_clock::now();
        detok[in.id] = std::move(st);
        write_event("start", in.id, "");
      } catch (const std::exception& e) {
        write_event("error", in.id, "\"error\":\"" + json_escape(e.what()) + "\"");
      }
    }

    if (eng.stream_active() == 0) {
      if (shutdown.load()) break;
      continue;
    }

    // One decode step over every running request.
    std::vector<engine::LlamaEngine::StreamEvent> events;
    eng.stream_step(events);
    for (const auto& e : events) {
      auto it = detok.find(e.id);
      if (it == detok.end()) continue;
      // A request ending on its stop token (eos/stop) must not emit that token's
      // text: a chat model's EOS is special (decode drops it), but a grammar
      // terminates on tokenizer.eos_id() which can decode to a visible "</s>".
      // Keep the token for a "length" finish (it is real content).
      const std::string reason = e.finish_reason ? e.finish_reason : "";
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
      }
    }
  }

  if (reader.joinable()) reader.join();
}

}  // namespace app::main_modes

#endif  // LLAMA_ENGINE_HAS_CUDA
