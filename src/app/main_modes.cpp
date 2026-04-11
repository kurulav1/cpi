#include "app/main_modes.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "app/main_helpers.hpp"
#include "model/tokenizer.hpp"

namespace app::main_modes {
namespace {

using app::main_helpers::find_first_stop_pos;
using app::main_helpers::has_complete_sentence;
using app::main_helpers::json_escape;
using app::main_helpers::json_get_bool;
using app::main_helpers::json_get_float;
using app::main_helpers::json_get_int;
using app::main_helpers::json_get_string;
using app::main_helpers::json_get_string_array;
using app::main_helpers::normalize_final_response_text;
using app::main_helpers::sanitize_stream_text;

}  // namespace

void execute_engine_modes(const RunExecutionOptions& options,
                          const std::vector<int>& prompt_tokens,
                          const std::vector<int>& stop_token_ids,
                          const std::vector<std::string>& stop_texts,
                          model::Tokenizer* tokenizer,
                          const GenerateFn& generate,
                          const GenerateStreamFn& generate_stream,
                          const InspectNextLogitsFn& inspect_next_logits,
                          const LastBenchmarkStatsFn& last_benchmark_stats) {
  if (options.use_tokenizer && tokenizer == nullptr) {
    throw std::runtime_error("tokenizer is required for tokenizer-backed execution");
  }

  // --- Interactive NDJSON mode ---
  if (options.interactive_mode) {
    if (!options.use_tokenizer || tokenizer == nullptr) {
      throw std::runtime_error("--interactive requires --tokenizer");
    }

    const auto write_event = [&](const std::string& type,
                                 const std::string& id,
                                 const std::string& extra_json) {
      static std::mutex write_mu;
      std::lock_guard<std::mutex> lk(write_mu);
      std::cout << "{\"type\":\"" << type << "\"";
      if (!id.empty()) {
        std::cout << ",\"id\":\"" << json_escape(id) << "\"";
      }
      if (!extra_json.empty()) {
        std::cout << "," << extra_json;
      }
      std::cout << "}\n" << std::flush;
    };
    const auto append_moe_selected_json = [&](std::ostringstream& os,
                                              const engine::BenchmarkStats& stats) {
      if (stats.moe_topk_layers <= 0 || stats.moe_topk_k <= 0 ||
          stats.moe_topk_indices.empty() || stats.moe_topk_probs.empty()) {
        return;
      }
      os << ",\"moe_selected\":[";
      for (int layer = 0; layer < stats.moe_topk_layers; ++layer) {
        if (layer > 0) {
          os << ",";
        }
        os << "{\"layer\":" << layer << ",\"experts\":[";
        for (int k = 0; k < stats.moe_topk_k; ++k) {
          if (k > 0) {
            os << ",";
          }
          const std::size_t flat =
              static_cast<std::size_t>(layer) * static_cast<std::size_t>(stats.moe_topk_k) +
              static_cast<std::size_t>(k);
          const int idx =
              (flat < stats.moe_topk_indices.size()) ? stats.moe_topk_indices[flat] : -1;
          os << idx;
        }
        os << "],\"probs\":[";
        for (int k = 0; k < stats.moe_topk_k; ++k) {
          if (k > 0) {
            os << ",";
          }
          const std::size_t flat =
              static_cast<std::size_t>(layer) * static_cast<std::size_t>(stats.moe_topk_k) +
              static_cast<std::size_t>(k);
          const float prob =
              (flat < stats.moe_topk_probs.size()) ? stats.moe_topk_probs[flat] : 0.0f;
          os << std::fixed << std::setprecision(6) << prob;
        }
        os << "]}";
      }
      os << "]";
    };

    std::string request_json;
    while (std::getline(std::cin, request_json)) {
      if (request_json.empty()) {
        continue;
      }

      const std::string req_id = json_get_string(request_json, "id");
      if (json_get_bool(request_json, "shutdown", false)) {
        write_event("done", req_id, "\"text\":\"\",\"elapsed_ms\":0");
        break;
      }

      try {
        const std::string req_prompt = json_get_string(request_json, "prompt");
        if (req_prompt.empty()) {
          throw std::runtime_error("interactive request missing 'prompt'");
        }
        const int req_max_new_raw = json_get_int(request_json, "max_new", options.max_new);
        const int req_max_new = (req_max_new_raw < 1) ? 1 : req_max_new_raw;
        const float req_temp_raw = json_get_float(request_json, "temp", options.temp);
        const float req_temp = (req_temp_raw < 0.0f) ? 0.0f : req_temp_raw;
        const bool req_add_bos = json_get_bool(request_json, "add_bos", !options.force_no_bos);
        const bool req_sentence_stop = json_get_bool(request_json, "sentence_stop", options.sentence_stop);

        std::vector<std::string> req_stop_texts = json_get_string_array(request_json, "stop_texts");
        if (req_stop_texts.empty()) {
          req_stop_texts = stop_texts;
        }

        std::vector<int> req_stop_ids;
        if (tokenizer->eos_id() >= 0) {
          req_stop_ids.push_back(tokenizer->eos_id());
        }
        for (const auto& st : req_stop_texts) {
          const auto toks = tokenizer->encode(st, /*add_bos=*/false);
          if (toks.size() == 1) {
            const int tid = toks[0];
            if (std::find(req_stop_ids.begin(), req_stop_ids.end(), tid) == req_stop_ids.end()) {
              req_stop_ids.push_back(tid);
            }
          }
        }

        const std::vector<int> req_prompt_tokens = tokenizer->encode(req_prompt, req_add_bos);
        std::vector<int> generated_ids;
        std::mutex stream_mu;
        std::condition_variable stream_cv;
        std::atomic<bool> stream_done{false};
        std::atomic<bool> sentence_stop_hit{false};
        double prev_moe_router_ms = 0.0;
        double prev_moe_expert_ms = 0.0;
        double prev_moe_merge_ms = 0.0;
        const auto req_start = std::chrono::steady_clock::now();

        write_event("start", req_id, "");
        std::thread stream_thread([&]() {
          std::string prev_decoded_local;
          std::size_t seen_tokens = 0;
          while (true) {
            std::vector<int> snapshot;
            bool done_now = false;
            {
              std::unique_lock<std::mutex> lk(stream_mu);
              stream_cv.wait(lk, [&] {
                return stream_done.load() || generated_ids.size() != seen_tokens;
              });
              snapshot = generated_ids;
              seen_tokens = snapshot.size();
              done_now = stream_done.load();
            }

            const std::string decoded = sanitize_stream_text(tokenizer->decode(snapshot));
            if (decoded.size() > prev_decoded_local.size()) {
              const std::string delta = decoded.substr(prev_decoded_local.size());
              if (!delta.empty()) {
                write_event("delta", req_id, "\"delta\":\"" + json_escape(delta) + "\"");
              }
            }
            prev_decoded_local = decoded;
            if (req_sentence_stop && has_complete_sentence(decoded)) {
              sentence_stop_hit.store(true);
            }

            if (done_now) {
              break;
            }
          }
        });

        (void)generate_stream(req_prompt_tokens, req_max_new, req_temp, [&](int tok) {
          if (std::find(req_stop_ids.begin(), req_stop_ids.end(), tok) != req_stop_ids.end()) {
            return false;
          }
          if (sentence_stop_hit.load()) {
            return false;
          }
          int token_index = 0;
          {
            std::lock_guard<std::mutex> lk(stream_mu);
            generated_ids.push_back(tok);
            token_index = static_cast<int>(generated_ids.size());
          }
          stream_cv.notify_one();
          const auto& stats = last_benchmark_stats();
          const double token_router_ms = std::max(0.0, stats.decode_moe_router_ms - prev_moe_router_ms);
          const double token_expert_ms = std::max(0.0, stats.decode_moe_expert_ms - prev_moe_expert_ms);
          const double token_merge_ms = std::max(0.0, stats.decode_moe_merge_ms - prev_moe_merge_ms);
          prev_moe_router_ms = stats.decode_moe_router_ms;
          prev_moe_expert_ms = stats.decode_moe_expert_ms;
          prev_moe_merge_ms = stats.decode_moe_merge_ms;
          std::ostringstream metrics_extra;
          metrics_extra << "\"metrics\":{"
                        << "\"token_index\":" << token_index
                        << ",\"moe_quant_mode\":\"" << json_escape(stats.moe_quant_mode) << "\""
                        << ",\"moe_router_ms\":" << std::fixed << std::setprecision(4) << token_router_ms
                        << ",\"moe_expert_ms\":" << std::fixed << std::setprecision(4) << token_expert_ms
                        << ",\"moe_merge_ms\":" << std::fixed << std::setprecision(4) << token_merge_ms;
          append_moe_selected_json(metrics_extra, stats);
          metrics_extra << "}";
          write_event("metrics", req_id, metrics_extra.str());
          if (sentence_stop_hit.load()) {
            return false;
          }
          return true;
        });
        stream_done.store(true);
        stream_cv.notify_one();
        if (stream_thread.joinable()) {
          stream_thread.join();
        }

        std::string final_text = sanitize_stream_text(tokenizer->decode(generated_ids));
        const std::size_t stop_pos = find_first_stop_pos(final_text, req_stop_texts);
        if (stop_pos != std::string::npos) {
          final_text = final_text.substr(0, stop_pos);
        }
        final_text = normalize_final_response_text(final_text);
        const auto req_end = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(req_end - req_start).count();
        const int generated_tokens = static_cast<int>(generated_ids.size());
        double tok_per_s = 0.0;
        if (generated_tokens > 0 && elapsed_ms > 0) {
          tok_per_s = (1000.0 * static_cast<double>(generated_tokens)) /
                      static_cast<double>(elapsed_ms);
        }
        const auto& stats = last_benchmark_stats();
        std::ostringstream done_extra;
        done_extra << "\"text\":\"" << json_escape(final_text) << "\""
                   << ",\"elapsed_ms\":" << elapsed_ms
                   << ",\"generated_tokens\":" << generated_tokens
                   << ",\"tok_per_s\":" << std::fixed << std::setprecision(2) << tok_per_s
                   << ",\"metrics\":{"
                   << "\"moe_quant_mode\":\"" << json_escape(stats.moe_quant_mode) << "\""
                   << ",\"moe_router_ms\":" << std::fixed << std::setprecision(4) << stats.decode_moe_router_ms
                   << ",\"moe_expert_ms\":" << std::fixed << std::setprecision(4) << stats.decode_moe_expert_ms
                   << ",\"moe_merge_ms\":" << std::fixed << std::setprecision(4) << stats.decode_moe_merge_ms;
        append_moe_selected_json(done_extra, stats);
        done_extra << "}";
        write_event("done",
                    req_id,
                    done_extra.str());
      } catch (const std::exception& e) {
        write_event("error", req_id, "\"error\":\"" + json_escape(e.what()) + "\"");
      }
    }
    return;
  }

  // --- Diagnostic modes (inspect/trace) ---
  if (options.inspect_next_topk > 0) {
    const auto top = inspect_next_logits(prompt_tokens, options.inspect_next_topk);
    std::cout << "Next-token top-" << options.inspect_next_topk << ":\n";
    for (const auto& [id, logit] : top) {
      std::cout << "  id=" << id << " logit=" << std::fixed << std::setprecision(4) << logit;
      if (options.use_tokenizer) {
        std::cout << " text=\"" << sanitize_stream_text(tokenizer->decode({id})) << "\"";
      }
      std::cout << "\n";
    }
  }
  if (options.trace_steps > 0) {
    const int trace_topk = (options.inspect_next_topk > 0) ? options.inspect_next_topk : 5;
    std::vector<int> trace_tokens = prompt_tokens;
    std::cout << "Greedy trace:\n";
    for (int step = 0; step < options.trace_steps; ++step) {
      const auto top = inspect_next_logits(trace_tokens, trace_topk);
      if (top.empty()) {
        break;
      }
      const int chosen = top.front().first;
      std::cout << "  step=" << step << " choose id=" << chosen;
      if (options.use_tokenizer) {
        std::cout << " text=\"" << sanitize_stream_text(tokenizer->decode({chosen})) << "\"";
      }
      std::cout << " candidates=";
      for (std::size_t i = 0; i < top.size(); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << top[i].first;
        if (options.use_tokenizer) {
          std::cout << ":" << sanitize_stream_text(tokenizer->decode({top[i].first}));
        }
      }
      std::cout << "\n";
      trace_tokens.push_back(chosen);
    }
  }

  // --- Generation ---
  std::vector<int> out;
  long long gen_ms = 0;
  double avg_prefill_ms = 0.0;
  double avg_decode_ms = 0.0;
  double avg_transfer_ms = 0.0;
  double avg_total_ms = 0.0;
  double avg_decode_rmsnorm_ms = 0.0;
  double avg_decode_qkv_ms = 0.0;
  double avg_decode_kv_store_ms = 0.0;
  double avg_decode_attention_ms = 0.0;
  double avg_decode_wo_ms = 0.0;
  double avg_decode_mlp_ms = 0.0;
  double avg_decode_moe_router_ms = 0.0;
  double avg_decode_moe_expert_ms = 0.0;
  double avg_decode_moe_merge_ms = 0.0;
  double avg_decode_lm_head_ms = 0.0;
  double avg_prefill_tok_per_s = 0.0;
  double avg_decode_tok_per_s = 0.0;
  double avg_total_tok_per_s = 0.0;
  int avg_generated_tokens = 0;
  int avg_streamed_layer_copies = 0;
  int avg_tq3_cached_active = 0;

  const bool repeated_benchmark = options.benchmark_mode && (options.benchmark_reps > 1 || options.benchmark_warmup > 0);

  auto run_quiet_generation = [&]() {
    if (!options.use_tokenizer) {
      return generate(prompt_tokens, options.max_new, options.temp);
    }
    return generate_stream(prompt_tokens, options.max_new, options.temp, [&](int tok) {
      return std::find(stop_token_ids.begin(), stop_token_ids.end(), tok) ==
             stop_token_ids.end();
    });
  };

  if (repeated_benchmark) {
    // --- Repeated benchmark path ---
    if (options.benchmark_warmup > 0 && !options.quiet_output) {
      std::cout << "[bench] warmup_runs=" << options.benchmark_warmup << "\n";
    }
    for (int i = 0; i < options.benchmark_warmup; ++i) {
      (void)run_quiet_generation();
    }

    long long total_ms_sum = 0;
    double prefill_ms_sum = 0.0;
    double decode_ms_sum = 0.0;
    double transfer_ms_sum = 0.0;
    double decode_rmsnorm_ms_sum = 0.0;
    double decode_qkv_ms_sum = 0.0;
    double decode_kv_store_ms_sum = 0.0;
    double decode_attention_ms_sum = 0.0;
    double decode_wo_ms_sum = 0.0;
    double decode_mlp_ms_sum = 0.0;
    double decode_moe_router_ms_sum = 0.0;
    double decode_moe_expert_ms_sum = 0.0;
    double decode_moe_merge_ms_sum = 0.0;
    double decode_lm_head_ms_sum = 0.0;
    int generated_sum = 0;
    int streamed_layer_copy_sum = 0;
    int tq3_cached_active_sum = 0;
    for (int rep = 0; rep < options.benchmark_reps; ++rep) {
      const auto rep_start = std::chrono::steady_clock::now();
      out = run_quiet_generation();
      const auto rep_end = std::chrono::steady_clock::now();
      const auto rep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rep_end - rep_start).count();
      const auto& stats = last_benchmark_stats();
      total_ms_sum += rep_ms;
      prefill_ms_sum += stats.prefill_ms;
      decode_ms_sum += stats.decode_ms;
      transfer_ms_sum += stats.transfer_ms;
      decode_rmsnorm_ms_sum += stats.decode_rmsnorm_ms;
      decode_qkv_ms_sum += stats.decode_qkv_ms;
      decode_kv_store_ms_sum += stats.decode_kv_store_ms;
      decode_attention_ms_sum += stats.decode_attention_ms;
      decode_wo_ms_sum += stats.decode_wo_ms;
      decode_mlp_ms_sum += stats.decode_mlp_ms;
      decode_moe_router_ms_sum += stats.decode_moe_router_ms;
      decode_moe_expert_ms_sum += stats.decode_moe_expert_ms;
      decode_moe_merge_ms_sum += stats.decode_moe_merge_ms;
      decode_lm_head_ms_sum += stats.decode_lm_head_ms;
      generated_sum += stats.generated_tokens;
      streamed_layer_copy_sum += stats.streamed_layer_copies;
      tq3_cached_active_sum += stats.tq3_cached_active;
    }
    gen_ms = total_ms_sum / options.benchmark_reps;
    avg_total_ms = static_cast<double>(total_ms_sum) / static_cast<double>(options.benchmark_reps);
    avg_prefill_ms = prefill_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_ms = decode_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_transfer_ms = transfer_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_rmsnorm_ms = decode_rmsnorm_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_qkv_ms = decode_qkv_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_kv_store_ms = decode_kv_store_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_attention_ms = decode_attention_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_wo_ms = decode_wo_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_mlp_ms = decode_mlp_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_moe_router_ms = decode_moe_router_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_moe_expert_ms = decode_moe_expert_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_moe_merge_ms = decode_moe_merge_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_decode_lm_head_ms = decode_lm_head_ms_sum / static_cast<double>(options.benchmark_reps);
    avg_generated_tokens = generated_sum / options.benchmark_reps;
    avg_streamed_layer_copies = streamed_layer_copy_sum / options.benchmark_reps;
    avg_tq3_cached_active = tq3_cached_active_sum / options.benchmark_reps;
    const int prefill_tokens = static_cast<int>(prompt_tokens.size() > 0 ? prompt_tokens.size() - 1 : 0);
    if (prefill_tokens > 0 && avg_prefill_ms > 0.0) {
      avg_prefill_tok_per_s = 1000.0 * static_cast<double>(prefill_tokens) / avg_prefill_ms;
    }
    if (avg_generated_tokens > 0 && avg_decode_ms > 0.0) {
      avg_decode_tok_per_s = 1000.0 * static_cast<double>(avg_generated_tokens) / avg_decode_ms;
    }
    if (avg_generated_tokens > 0 && avg_total_ms > 0.0) {
      avg_total_tok_per_s = 1000.0 * static_cast<double>(avg_generated_tokens) / avg_total_ms;
    }
  } else {
    // --- Interactive / single-run path ---
    const auto gen_start = std::chrono::steady_clock::now();
    if (options.use_tokenizer) {
      // Async streaming display: GPU decode loop pushes tokens into a queue;
      // a background thread batches tokenizer.decode calls so the engine
      // never waits on the tokenizer.
      std::vector<int> token_queue;
      std::mutex queue_mutex;
      std::condition_variable queue_cv;
      std::atomic<bool> gen_done{false};

      std::thread display_thread([&]() {
        std::vector<int> display_ids;
        std::string prev_text;
        while (true) {
          std::vector<int> batch;
          {
            std::unique_lock<std::mutex> lk(queue_mutex);
            queue_cv.wait(lk, [&] { return !token_queue.empty() || gen_done.load(); });
            batch.swap(token_queue);
          }
          if (!batch.empty()) {
            display_ids.insert(display_ids.end(), batch.begin(), batch.end());
            const std::string decoded = sanitize_stream_text(tokenizer->decode(display_ids));
            if (decoded.size() > prev_text.size()) {
              std::cout << decoded.substr(prev_text.size()) << std::flush;
            }
            prev_text = decoded;
          }
          if (gen_done.load()) {
            std::vector<int> tail;
            { std::lock_guard<std::mutex> lk(queue_mutex); tail.swap(token_queue); }
            if (!tail.empty()) {
              display_ids.insert(display_ids.end(), tail.begin(), tail.end());
              const std::string decoded = sanitize_stream_text(tokenizer->decode(display_ids));
              if (decoded.size() > prev_text.size()) {
                std::cout << decoded.substr(prev_text.size()) << std::flush;
              }
            }
            break;
          }
        }
        if (!options.quiet_output) {
          std::cout << "\n";
        }
      });

      std::exception_ptr generation_error;
      try {
        out = generate_stream(prompt_tokens, options.max_new, options.temp, [&](int tok) {
          if (std::find(stop_token_ids.begin(), stop_token_ids.end(), tok) != stop_token_ids.end()) {
            return false;
          }
          { std::lock_guard<std::mutex> lk(queue_mutex); token_queue.push_back(tok); }
          queue_cv.notify_one();
          return true;
        });
      } catch (...) {
        generation_error = std::current_exception();
      }

      gen_done.store(true);
      queue_cv.notify_one();
      if (display_thread.joinable()) {
        display_thread.join();
      }
      if (generation_error) {
        std::rethrow_exception(generation_error);
      }
    } else {
      out = generate(prompt_tokens, options.max_new, options.temp);
    }
    const auto gen_end = std::chrono::steady_clock::now();
    gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
  }

  // --- Output ---
  if (!options.quiet_output) {
    std::cout << "Output tokens:";
    for (int t : out) {
      std::cout << ' ' << t;
    }
    std::cout << "\n";
  }

  if (options.use_tokenizer) {
    std::vector<int> generated_only;
    if (out.size() > prompt_tokens.size()) {
      const auto prompt_n = static_cast<std::vector<int>::difference_type>(prompt_tokens.size());
      generated_only.assign(out.begin() + prompt_n, out.end());
    }
      std::string final_text = sanitize_stream_text(tokenizer->decode(generated_only));
      const std::size_t stop_pos = find_first_stop_pos(final_text, stop_texts);
      if (stop_pos != std::string::npos) {
        final_text = final_text.substr(0, stop_pos);
      }
      final_text = normalize_final_response_text(final_text);
      if (!generated_only.empty()) {
      std::size_t special_count = 0;
      const auto& specials = tokenizer->special_ids();
      for (int id : generated_only) {
        if (std::find(specials.begin(), specials.end(), id) != specials.end()) {
          ++special_count;
        }
      }
      const double special_ratio = static_cast<double>(special_count) / static_cast<double>(generated_only.size());
      if (!options.quiet_output && special_ratio > 0.25) {
        std::cout << "[warn] generated special-token ratio is high (" << std::fixed << std::setprecision(2)
                  << (special_ratio * 100.0)
                  << "%). This often indicates a model/tokenizer mismatch or an incorrect chat template.\n";
      }
    }
    if (options.simple_mode) {
      std::cout << final_text << "\n";
    } else if (!options.quiet_output) {
      std::cout << "Decoded text:\n" << final_text << "\n";
    }
  }

  // --- Performance reporting ---
  const std::size_t generated_count = (out.size() > prompt_tokens.size()) ? (out.size() - prompt_tokens.size()) : 0;
  if (!options.quiet_output && generated_count > 0 && gen_ms > 0) {
    const std::size_t perf_tokens = repeated_benchmark ? static_cast<std::size_t>(avg_generated_tokens) : generated_count;
    const double perf_ms = repeated_benchmark ? avg_total_ms : static_cast<double>(gen_ms);
    const double toks_per_sec = (perf_tokens > 0 && perf_ms > 0.0) ? (1000.0 * static_cast<double>(perf_tokens)) / perf_ms : 0.0;
    std::cout << std::fixed << std::setprecision(2)
              << (repeated_benchmark ? "[perf-avg] " : "[perf] ")
              << "generated_tokens=" << perf_tokens
              << " elapsed_ms=" << perf_ms
              << " tok_per_s=" << toks_per_sec << "\n";
  }
  if (!options.quiet_output && options.benchmark_mode) {
    if (repeated_benchmark) {
      const int prefill_tokens = prompt_tokens.empty() ? 0 : static_cast<int>(prompt_tokens.size()) - 1;
      std::cout << std::fixed << std::setprecision(2)
                << "[bench-avg] reps=" << options.benchmark_reps
                << " prefill_tokens=" << prefill_tokens
                << " prefill_ms=" << avg_prefill_ms
                << " prefill_tok_per_s=" << avg_prefill_tok_per_s
                << " decode_tokens=" << avg_generated_tokens
                << " decode_ms=" << avg_decode_ms
                << " decode_tok_per_s=" << avg_decode_tok_per_s
                << " transfer_ms=" << avg_transfer_ms
                << " streamed_layer_copies=" << avg_streamed_layer_copies
                << " tq3_cached_active=" << avg_tq3_cached_active
                << " total_ms=" << avg_total_ms
                << " total_tok_per_s=" << avg_total_tok_per_s << "\n";
      if (options.benchmark_phases) {
        std::cout << std::fixed << std::setprecision(2)
                  << "[bench-phase-avg]"
                  << " rmsnorm_ms=" << avg_decode_rmsnorm_ms
                  << " qkv_ms=" << avg_decode_qkv_ms
                  << " kv_store_ms=" << avg_decode_kv_store_ms
                  << " attention_ms=" << avg_decode_attention_ms
                  << " wo_ms=" << avg_decode_wo_ms
                  << " mlp_ms=" << avg_decode_mlp_ms
                  << " moe_router_ms=" << avg_decode_moe_router_ms
                  << " moe_expert_ms=" << avg_decode_moe_expert_ms
                  << " moe_merge_ms=" << avg_decode_moe_merge_ms
                  << " lm_head_ms=" << avg_decode_lm_head_ms << "\n";
      }
    } else {
      const auto& stats = last_benchmark_stats();
      const int prefill_tokens = (stats.prompt_tokens > 0) ? (stats.prompt_tokens - 1) : 0;
      const double prefill_tok_per_s =
          (prefill_tokens > 0 && stats.prefill_ms > 0.0) ? (1000.0 * static_cast<double>(prefill_tokens) / stats.prefill_ms) : 0.0;
      const double decode_tok_per_s =
          (stats.generated_tokens > 0 && stats.decode_ms > 0.0) ? (1000.0 * static_cast<double>(stats.generated_tokens) / stats.decode_ms) : 0.0;
      std::cout << std::fixed << std::setprecision(2)
                << "[bench] prefill_tokens=" << prefill_tokens
                << " prefill_ms=" << stats.prefill_ms
                << " prefill_tok_per_s=" << prefill_tok_per_s
                << " decode_tokens=" << stats.generated_tokens
                << " decode_ms=" << stats.decode_ms
                << " decode_tok_per_s=" << decode_tok_per_s
                << " transfer_ms=" << stats.transfer_ms
                << " streamed_layer_copies=" << stats.streamed_layer_copies
                << " tq3_cached_active=" << stats.tq3_cached_active << "\n";
      if (options.benchmark_phases) {
        std::cout << std::fixed << std::setprecision(2)
                  << "[bench-phase]"
                  << " rmsnorm_ms=" << stats.decode_rmsnorm_ms
                  << " qkv_ms=" << stats.decode_qkv_ms
                  << " kv_store_ms=" << stats.decode_kv_store_ms
                  << " attention_ms=" << stats.decode_attention_ms
                  << " wo_ms=" << stats.decode_wo_ms
                  << " mlp_ms=" << stats.decode_mlp_ms
                  << " moe_router_ms=" << stats.decode_moe_router_ms
                  << " moe_expert_ms=" << stats.decode_moe_expert_ms
                  << " moe_merge_ms=" << stats.decode_moe_merge_ms
                  << " lm_head_ms=" << stats.decode_lm_head_ms << "\n";
      }
    }
  }
}

}  // namespace app::main_modes
