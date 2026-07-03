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
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "app/main_helpers.hpp"
#include "engine/llama_engine.hpp"
#include "model/tokenizer.hpp"

namespace app::main_modes {

using app::main_helpers::json_escape;
using app::main_helpers::json_get_bool;
using app::main_helpers::json_get_float;
using app::main_helpers::json_get_int;
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
  };
  std::mutex q_mu;
  std::condition_variable q_cv;
  std::deque<Incoming> queue;
  std::atomic<bool> shutdown{false};

  // Reader thread: parse stdin request lines into the queue. Blocks on getline;
  // a {"shutdown":true} line or stdin EOF ends it.
  std::thread reader([&]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::string id = json_get_string(line, "id");
      if (json_get_bool(line, "shutdown", false)) break;
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
  };
  std::unordered_map<std::string, DetokState> detok;

  while (true) {
    // Admit any queued requests (prefill happens inside stream_admit).
    std::vector<Incoming> admits;
    {
      std::unique_lock<std::mutex> lk(q_mu);
      if (queue.empty() && eng.stream_active() == 0) {
        if (shutdown.load()) break;
        q_cv.wait(lk, [&]() { return !queue.empty() || shutdown.load(); });
      }
      while (!queue.empty()) {
        admits.push_back(std::move(queue.front()));
        queue.pop_front();
      }
    }
    for (auto& in : admits) {
      try {
        eng.stream_admit(in.id, in.tokens, in.params);
        detok[in.id] = DetokState{};
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
      it->second.ids.push_back(e.token);
      const std::string decoded = sanitize_stream_text(tokenizer.decode(it->second.ids));
      if (decoded.size() > it->second.prev_text.size()) {
        const std::string delta = decoded.substr(it->second.prev_text.size());
        if (!delta.empty()) {
          write_event("delta", e.id, "\"delta\":\"" + json_escape(delta) + "\"");
        }
        it->second.prev_text = decoded;
      }
      if (e.finished) {
        write_event("done", e.id,
                    "\"finish_reason\":\"" + std::string(e.finish_reason) + "\",\"text\":\"" +
                        json_escape(it->second.prev_text) + "\"");
        detok.erase(it);
      }
    }
  }

  if (reader.joinable()) reader.join();
}

}  // namespace app::main_modes

#endif  // LLAMA_ENGINE_HAS_CUDA
