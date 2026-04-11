#include "app/main_cli.hpp"

#include <cstdlib>
#include <ostream>
#include <stdexcept>

#include "app/main_helpers.hpp"

namespace app::main_cli {
namespace {

constexpr int kDefaultMaxNewTokens = 16;
constexpr float kDefaultTemperature = 0.8f;
constexpr int kSimpleModeDefaultMaxNewTokens = 64;
constexpr float kSimpleModeDefaultTemperature = 0.2f;

std::string read_env(const char* name) {
#ifdef _WIN32
  char* raw = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&raw, &len, name) != 0 || raw == nullptr) {
    return {};
  }
  std::string out(raw);
  std::free(raw);
  return out;
#else
  const char* raw = std::getenv(name);
  return raw ? std::string(raw) : std::string();
#endif
}

template <typename T, typename ParseFn>
T parse_env_or_default(const char* name, const T& default_value, ParseFn&& parse_fn) {
  const std::string raw = read_env(name);
  if (raw.empty()) {
    return default_value;
  }
  try {
    return parse_fn(raw.c_str());
  } catch (...) {
    throw std::runtime_error(std::string("invalid value for env ") + name + ": " + raw);
  }
}

void init_options_from_env(engine::EngineOptions* opts) {
  opts->max_context = parse_env_or_default(
      "LLAMA_INFER_MAX_CONTEXT", opts->max_context, [](const char* v) { return std::stoi(v); });
  opts->top_k = parse_env_or_default(
      "LLAMA_INFER_TOP_K", opts->top_k, [](const char* v) { return std::stoi(v); });
  opts->top_p = parse_env_or_default(
      "LLAMA_INFER_TOP_P", opts->top_p, [](const char* v) { return std::stof(v); });
  opts->repetition_penalty = parse_env_or_default(
      "LLAMA_INFER_REPEAT_PENALTY", opts->repetition_penalty, [](const char* v) { return std::stof(v); });
  opts->no_repeat_ngram_size = parse_env_or_default(
      "LLAMA_INFER_NO_REPEAT_NGRAM", opts->no_repeat_ngram_size, [](const char* v) { return std::stoi(v); });
  opts->eos_token_id = parse_env_or_default(
      "LLAMA_INFER_EOS_TOKEN_ID", opts->eos_token_id, [](const char* v) { return std::stoi(v); });
  opts->gpu_cache_layers = parse_env_or_default(
      "LLAMA_INFER_GPU_CACHE_LAYERS", opts->gpu_cache_layers, [](const char* v) { return std::stoi(v); });
  opts->gpu_cache_limit_mb = static_cast<std::size_t>(parse_env_or_default(
      "LLAMA_INFER_GPU_CACHE_LIMIT_MB", static_cast<unsigned long long>(opts->gpu_cache_limit_mb),
      [](const char* v) { return std::stoull(v); }));
  opts->vram_safety_margin_mb = static_cast<std::size_t>(parse_env_or_default(
      "LLAMA_INFER_VRAM_MARGIN_MB", static_cast<unsigned long long>(opts->vram_safety_margin_mb),
      [](const char* v) { return std::stoull(v); }));
  opts->max_cpu_percent = parse_env_or_default(
      "LLAMA_INFER_MAX_CPU_PERCENT", opts->max_cpu_percent, [](const char* v) { return std::stod(v); });
  opts->max_memory_percent = parse_env_or_default(
      "LLAMA_INFER_MAX_MEMORY_PERCENT", opts->max_memory_percent, [](const char* v) { return std::stod(v); });
  opts->resource_sample_interval_ms = parse_env_or_default(
      "LLAMA_INFER_RESOURCE_SAMPLE_MS", opts->resource_sample_interval_ms, [](const char* v) { return std::stoi(v); });
  opts->resource_sustain_ms = parse_env_or_default(
      "LLAMA_INFER_RESOURCE_SUSTAIN_MS", opts->resource_sustain_ms, [](const char* v) { return std::stoi(v); });
  opts->resource_throttle_sleep_ms = parse_env_or_default(
      "LLAMA_INFER_RESOURCE_THROTTLE_MS", opts->resource_throttle_sleep_ms, [](const char* v) { return std::stoi(v); });
  opts->tq_cached_init_timeout_ms = parse_env_or_default(
      "LLAMA_INFER_TQ_CACHED_INIT_TIMEOUT_MS", opts->tq_cached_init_timeout_ms, [](const char* v) { return std::stoi(v); });
  opts->tq_first_token_timeout_ms = parse_env_or_default(
      "LLAMA_INFER_TQ_FIRST_TOKEN_TIMEOUT_MS", opts->tq_first_token_timeout_ms, [](const char* v) { return std::stoi(v); });
}

}  // namespace

void print_usage(std::ostream& os) {
  os << "Usage: llama_infer <model.ll2c|model_dir> [--prompt text --tokenizer tokenizer.model] "
        "[--tokens csv] [--max-new n] [--temp t] [--max-context n] "
        "[--gpu-cache-all] [--gpu-cache-layers n] [--gpu-cache-limit-mb n] [--vram-safety-margin-mb n] "
        "[--top-k n] [--top-p p] [--repeat-penalty r] [--no-repeat-ngram n] [--rope-theta f] "
        "[--stop-text text] [--chat-template tinyllama|tinyllama-chatml|llama2|llama3|mistral|phi3|qwen2|qwen3_5|llama4] "
          "[--dump-tokenizer-meta] [--dump-prompt-tokens] [--inspect-next-topk n] "
        "[--trace-steps n] [--sentence-stop] [--benchmark] [--benchmark-reps n] [--benchmark-warmup n] "
        "[--benchmark-phases] [--runtime-metrics] [--no-split-attention] "
        "[--enable-tq-cached] [--tq-mode auto|mse|prod] "
        "[--max-cpu-percent n] [--max-memory-percent n] "
        "[--resource-sample-ms n] [--resource-sustain-ms n] [--resource-throttle-ms n] "
        "[--no-resource-limits] "
        "[--tq-cached-init-timeout-ms n] [--tq-first-token-timeout-ms n] "
          "[--allow-legacy-chat-tokenizer] "
        "[--no-bos] [--eos-token n] [--no-loop-guard] [--int8-streaming|--int4-streaming|--weight-quant none|int8|int4] "
        "[--paged-kv-cache] [--web] [--interactive] [--simple]\n";
}

void apply_simple_mode_defaults(ParsedArgs* args) {
  if (!args->simple_mode) {
    return;
  }
  const int simple_default_max_new = parse_env_or_default(
      "LLAMA_INFER_SIMPLE_MAX_NEW", kSimpleModeDefaultMaxNewTokens, [](const char* v) { return std::stoi(v); });
  const float simple_default_temp = parse_env_or_default(
      "LLAMA_INFER_SIMPLE_TEMP", kSimpleModeDefaultTemperature, [](const char* v) { return std::stof(v); });

  if (!args->max_new_set) {
    args->max_new = simple_default_max_new;
  }
  if (!args->temp_set) {
    args->temp = simple_default_temp;
  }
  if (!args->cache_mode_set) {
    args->opts.gpu_cache_all = true;
  }
  if (!args->weight_quant_set) {
    args->opts.int8_streaming = true;
    args->opts.streaming_quant_bits = 8;
  }
  args->opts.enable_host_resource_limits = false;
  if (!args->chat_template_set && args->chat_template.empty()) {
    args->chat_template = main_helpers::guess_chat_template_from_model_path(args->opts.model_path);
  }
  args->sentence_stop = true;
  args->web_mode = true;
}

void validate_args(const ParsedArgs& args) {
  if (args.max_new < 1) {
    throw std::runtime_error("--max-new must be >= 1");
  }
  if (args.opts.max_context <= 0) {
    throw std::runtime_error("--max-context must be > 0");
  }
  if (args.benchmark_reps < 1) {
    throw std::runtime_error("--benchmark-reps must be >= 1");
  }
  if (args.benchmark_warmup < 0) {
    throw std::runtime_error("--benchmark-warmup must be >= 0");
  }
  if (args.opts.gpu_cache_layers < -1) {
    throw std::runtime_error("--gpu-cache-layers must be >= 0");
  }
  if (args.opts.max_cpu_percent <= 0.0 || args.opts.max_cpu_percent > 100.0) {
    throw std::runtime_error("--max-cpu-percent must be in (0, 100]");
  }
  if (args.opts.max_memory_percent <= 0.0 || args.opts.max_memory_percent > 100.0) {
    throw std::runtime_error("--max-memory-percent must be in (0, 100]");
  }
  if (args.opts.resource_sample_interval_ms < 0) {
    throw std::runtime_error("--resource-sample-ms must be >= 0");
  }
  if (args.opts.resource_sustain_ms <= 0) {
    throw std::runtime_error("--resource-sustain-ms must be > 0");
  }
  if (args.opts.resource_throttle_sleep_ms < 0) {
    throw std::runtime_error("--resource-throttle-ms must be >= 0");
  }
  if (args.opts.tq_cached_init_timeout_ms < 0) {
    throw std::runtime_error("--tq-cached-init-timeout-ms must be >= 0");
  }
  if (args.opts.tq_first_token_timeout_ms < 0) {
    throw std::runtime_error("--tq-first-token-timeout-ms must be >= 0");
  }
  if (args.opts.eos_token_id < -1) {
    throw std::runtime_error("--eos-token must be >= -1");
  }
  int gpu_cache_mode_count = 0;
  gpu_cache_mode_count += args.opts.gpu_cache_all ? 1 : 0;
  gpu_cache_mode_count += (args.opts.gpu_cache_layers >= 0) ? 1 : 0;
  gpu_cache_mode_count += (args.opts.gpu_cache_limit_mb > 0) ? 1 : 0;
  if (gpu_cache_mode_count > 1) {
    throw std::runtime_error("use only one of --gpu-cache-all, --gpu-cache-layers, or --gpu-cache-limit-mb");
  }
}

ParsedArgs parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error("missing model path");
  }

  ParsedArgs args;
  args.opts.model_path = argv[1];
  init_options_from_env(&args.opts);
  args.max_new = parse_env_or_default(
      "LLAMA_INFER_DEFAULT_MAX_NEW", kDefaultMaxNewTokens, [](const char* v) { return std::stoi(v); });
  args.temp = parse_env_or_default(
      "LLAMA_INFER_DEFAULT_TEMP", kDefaultTemperature, [](const char* v) { return std::stof(v); });

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_val = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--prompt") {
      args.prompt_text = need_val("--prompt");
    } else if (arg == "--tokenizer") {
      args.tokenizer_path = need_val("--tokenizer");
      args.tokenizer_set = true;
    } else if (arg == "--chat-template") {
      args.chat_template = need_val("--chat-template");
      args.chat_template_set = true;
    } else if (arg == "--tokens") {
      args.token_csv = need_val("--tokens");
    } else if (arg == "--max-new") {
      args.max_new = std::stoi(need_val("--max-new"));
      args.max_new_set = true;
    } else if (arg == "--temp") {
      args.temp = std::stof(need_val("--temp"));
      args.temp_set = true;
    } else if (arg == "--max-context") {
      args.opts.max_context = std::stoi(need_val("--max-context"));
    } else if (arg == "--gpu-cache-all") {
      args.opts.gpu_cache_all = true;
      args.cache_mode_set = true;
    } else if (arg == "--gpu-cache-layers") {
      args.opts.gpu_cache_layers = std::stoi(need_val("--gpu-cache-layers"));
      args.cache_mode_set = true;
    } else if (arg == "--gpu-cache-limit-mb") {
      args.opts.gpu_cache_limit_mb = static_cast<std::size_t>(
          std::stoull(need_val("--gpu-cache-limit-mb")));
      args.cache_mode_set = true;
    } else if (arg == "--vram-safety-margin-mb") {
      args.opts.vram_safety_margin_mb = static_cast<std::size_t>(
          std::stoull(need_val("--vram-safety-margin-mb")));
    } else if (arg == "--top-k") {
      args.opts.top_k = std::stoi(need_val("--top-k"));
    } else if (arg == "--top-p") {
      args.opts.top_p = std::stof(need_val("--top-p"));
    } else if (arg == "--repeat-penalty") {
      args.opts.repetition_penalty = std::stof(need_val("--repeat-penalty"));
    } else if (arg == "--no-repeat-ngram") {
      args.opts.no_repeat_ngram_size = std::stoi(need_val("--no-repeat-ngram"));
    } else if (arg == "--parity-check") {
      args.parity_check = true;
    } else if (arg == "--dump-tokenizer-meta") {
      args.dump_tokenizer_meta = true;
    } else if (arg == "--dump-prompt-tokens") {
      args.dump_prompt_tokens = true;
    } else if (arg == "--inspect-next-topk") {
      args.inspect_next_topk = std::stoi(need_val("--inspect-next-topk"));
    } else if (arg == "--trace-steps") {
      args.trace_steps = std::stoi(need_val("--trace-steps"));
    } else if (arg == "--sentence-stop") {
      args.sentence_stop = true;
    } else if (arg == "--benchmark") {
      args.benchmark_mode = true;
    } else if (arg == "--benchmark-reps") {
      args.benchmark_mode = true;
      args.benchmark_reps = std::stoi(need_val("--benchmark-reps"));
    } else if (arg == "--benchmark-warmup") {
      args.benchmark_mode = true;
      args.benchmark_warmup = std::stoi(need_val("--benchmark-warmup"));
    } else if (arg == "--benchmark-phases") {
      args.benchmark_mode = true;
      args.benchmark_phases = true;
    } else if (arg == "--runtime-metrics") {
      args.runtime_metrics = true;
    } else if (arg == "--no-split-attention") {
      args.opts.disable_split_attention = true;
    } else if (arg == "--allow-legacy-chat-tokenizer") {
      args.allow_legacy_chat_tokenizer = true;
    } else if (arg == "--no-bos") {
      args.force_no_bos = true;
    } else if (arg == "--eos-token") {
      args.opts.eos_token_id = std::stoi(need_val("--eos-token"));
    } else if (arg == "--no-loop-guard") {
      args.opts.loop_guard = false;
    } else if (arg == "--stop-text") {
      args.stop_texts.push_back(need_val("--stop-text"));
    } else if (arg == "--rope-theta") {
      args.opts.rope_theta = std::stof(need_val("--rope-theta"));
    } else if (arg == "--int8-streaming") {
      args.opts.int8_streaming = true;
      args.opts.streaming_quant_bits = 8;
      args.opts.prefer_lowbit_cache = true;
      args.weight_quant_set = true;
    } else if (arg == "--int4-streaming") {
      args.opts.int8_streaming = true;
      args.opts.streaming_quant_bits = 4;
      args.opts.prefer_lowbit_cache = true;
      args.weight_quant_set = true;
    } else if (arg == "--weight-quant") {
      const std::string mode = need_val("--weight-quant");
      args.weight_quant_set = true;
      if (mode == "none") {
        args.opts.int8_streaming = false;
        args.opts.prefer_lowbit_cache = false;
      } else if (mode == "int8") {
        args.opts.int8_streaming = true;
        args.opts.streaming_quant_bits = 8;
        args.opts.prefer_lowbit_cache = true;
      } else if (mode == "int4") {
        args.opts.int8_streaming = true;
        args.opts.streaming_quant_bits = 4;
        args.opts.prefer_lowbit_cache = true;
      } else {
        throw std::runtime_error("--weight-quant must be one of: none, int8, int4");
      }
    } else if (arg == "--paged-kv-cache") {
      args.opts.paged_kv_cache = true;
    } else if (arg == "--kv-int4") {
      args.opts.kv_cache_int4 = true;
    } else if (arg == "--enable-tq-cached") {
      args.opts.enable_tq_cached = true;
    } else if (arg == "--tq-mode") {
      args.opts.tq_mode = need_val("--tq-mode");
      if (args.opts.tq_mode != "auto" && args.opts.tq_mode != "mse" && args.opts.tq_mode != "prod") {
        throw std::runtime_error("--tq-mode must be one of: auto, mse, prod");
      }
    } else if (arg == "--max-cpu-percent") {
      args.opts.max_cpu_percent = std::stod(need_val("--max-cpu-percent"));
    } else if (arg == "--max-memory-percent") {
      args.opts.max_memory_percent = std::stod(need_val("--max-memory-percent"));
    } else if (arg == "--resource-sample-ms") {
      args.opts.resource_sample_interval_ms = std::stoi(need_val("--resource-sample-ms"));
    } else if (arg == "--resource-sustain-ms") {
      args.opts.resource_sustain_ms = std::stoi(need_val("--resource-sustain-ms"));
    } else if (arg == "--resource-throttle-ms") {
      args.opts.resource_throttle_sleep_ms = std::stoi(need_val("--resource-throttle-ms"));
    } else if (arg == "--no-resource-limits") {
      args.opts.enable_host_resource_limits = false;
    } else if (arg == "--tq-cached-init-timeout-ms") {
      args.opts.tq_cached_init_timeout_ms = std::stoi(need_val("--tq-cached-init-timeout-ms"));
    } else if (arg == "--tq-first-token-timeout-ms") {
      args.opts.tq_first_token_timeout_ms = std::stoi(need_val("--tq-first-token-timeout-ms"));
    } else if (arg == "--web") {
      args.web_mode = true;
    } else if (arg == "--interactive") {
      args.interactive_mode = true;
      args.web_mode = true;
    } else if (arg == "--simple") {
      args.simple_mode = true;
    } else if (arg == "--cpu") {
      args.force_cpu = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  apply_simple_mode_defaults(&args);
  validate_args(args);
  return args;
}

}  // namespace app::main_cli
