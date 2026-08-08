// Multiplexed (continuous-batching) interactive worker: the stdin/stdout
// transport.
//
// Opt-in via --interactive-batch. The scheduler loop itself lives in
// app::BatchWorker (shared with the in-binary HTTP server); this file is only
// the adapter: it parses JSON request lines from stdin and renders the worker's
// events as the same JSON-line events the Node layer already consumes
// (start / delta / done / error, tagged by request id), so several SSE streams
// can be multiplexed onto one worker.
//
// Requires a tokenizer + --paged-blocks. Takes the scheduler, not an engine, so
// this file stays GPU-agnostic.
#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "app/batch_worker.hpp"
#include "app/main_helpers.hpp"
#include "app/main_modes.hpp"
#include "engine/batch_scheduler.hpp"
#include "model/tokenizer.hpp"

namespace app::main_modes {

using app::main_helpers::json_escape;
using app::main_helpers::json_get_bool;
using app::main_helpers::json_get_float;
using app::main_helpers::json_get_int;
using app::main_helpers::json_get_raw_value;
using app::main_helpers::json_get_string;
using app::main_helpers::json_get_string_array;

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

  BatchDefaults defaults;
  defaults.stop_texts = default_stop_texts;
  defaults.add_bos = default_add_bos;
  defaults.max_new = default_max_new;
  defaults.temp = default_temp;
  defaults.top_k = default_top_k;
  defaults.top_p = default_top_p;
  defaults.repeat_penalty = default_repeat_penalty;
  defaults.no_repeat_ngram = default_no_repeat;

  BatchWorker worker(sched, tokenizer, defaults, [&](const BatchEvent& e) {
    switch (e.type) {
      case BatchEvent::Type::Start:
        write_event("start", e.id, "");
        break;
      case BatchEvent::Type::Delta:
        write_event("delta", e.id, "\"delta\":\"" + json_escape(e.text) + "\"");
        break;
      case BatchEvent::Type::Done: {
        char nums[160];
        std::snprintf(nums, sizeof(nums),
                      ",\"generated\":%d,\"elapsed_ms\":%.1f,\"tok_per_s\":%.2f", e.generated,
                      e.elapsed_ms, e.tok_per_s);
        write_event("done", e.id,
                    "\"finish_reason\":\"" + e.finish_reason + "\",\"text\":\"" +
                        json_escape(e.text) + "\"" + nums);
        break;
      }
      case BatchEvent::Type::Error:
        write_event("error", e.id, "\"error\":\"" + json_escape(e.error) + "\"");
        break;
    }
  });

  // Reader thread: parse stdin request lines. A {"shutdown":true} line or EOF ends it.
  std::thread reader([&]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::string id = json_get_string(line, "id");
      if (json_get_bool(line, "shutdown", false)) break;
      const std::string cancel_id = json_get_string(line, "cancel");
      if (!cancel_id.empty()) {
        worker.cancel(cancel_id);
        std::cerr << "[batch] cancel " << cancel_id << "\n";
        continue;
      }
      BatchOverrides ov;
      ov.max_new = json_get_int(line, "max_new", -1);
      ov.temp = json_get_float(line, "temp", -1.0f);
      ov.min_new = json_get_int(line, "min_new", 0);
      ov.top_k = json_get_int(line, "top_k", -1);
      ov.top_p = json_get_float(line, "top_p", -1.0f);
      ov.repeat_penalty = json_get_float(line, "repeat_penalty", -1.0f);
      ov.no_repeat_ngram = json_get_int(line, "no_repeat_ngram", -1);
      ov.has_add_bos = line.find("\"add_bos\"") != std::string::npos;
      ov.add_bos = json_get_bool(line, "add_bos", default_add_bos);
      ov.stop_texts = json_get_string_array(line, "stop_texts");
      ov.json_schema = json_get_raw_value(line, "json_schema");

      std::string err;
      if (!worker.submit(id, json_get_string(line, "prompt"), ov, &err)) {
        write_event("error", id, "\"error\":\"" + json_escape(err) + "\"");
      }
    }
    worker.stop();
  });

  worker.run();
  if (reader.joinable()) reader.join();
}

}  // namespace app::main_modes
