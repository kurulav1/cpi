// main.cpp - Command-line entry point for llama_infer.
//
// Parses CLI arguments, configures tokenizer/prompt state, auto-selects the
// execution engine, and dispatches runtime modes.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "app/main_cli.hpp"
#include "app/main_helpers.hpp"
#include "app/main_modes.hpp"
#include "engine/cpu_engine.hpp"
#include "engine/llama4_cpu_engine.hpp"
#if LLAMA_ENGINE_HAS_CUDA
#include "engine/llama4_cuda_engine.hpp"
#include "engine/llama_engine.hpp"
#endif
#include "model/tokenizer.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

using app::main_helpers::SingleInstanceGuard;
using app::main_helpers::auto_detect_tokenizer_path;
using app::main_helpers::build_chat_prompt;
using app::main_helpers::default_stop_texts_for_template;
using app::main_helpers::is_safetensors_model_dir;
using app::main_helpers::join_ints;
using app::main_helpers::parse_tokens;

}  // namespace

int main(int argc, char** argv) {
  // Disable stdout/stderr buffering so progress messages appear immediately
  // even when output is piped or redirected to a file.
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  SingleInstanceGuard instance_guard;
  if (!instance_guard.acquire()) {
    std::cerr << "Another llama_infer instance is already running.\n";
    return 3;
  }

  if (argc < 2) {
    app::main_cli::print_usage(std::cerr);
    return 1;
  }

  app::main_cli::ParsedArgs cli;
  try {
    cli = app::main_cli::parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    return 1;
  }

  try {
    const bool quiet_output = cli.web_mode || cli.simple_mode;
    cli.opts.verbose = !quiet_output;
    cli.opts.profile_decode_phases = cli.benchmark_phases || cli.runtime_metrics;

    std::vector<int> prompt_tokens;
    std::vector<int> stop_token_ids;
    model::Tokenizer tokenizer;
    const bool use_tokenizer = !cli.prompt_text.empty() || cli.interactive_mode;
    std::ostream& info_out = quiet_output ? std::cerr : std::cout;
    const bool is_llama4_safetensors = is_safetensors_model_dir(cli.opts.model_path);

    // --- Tokenizer setup ---
    if (use_tokenizer) {
      if (cli.tokenizer_path.empty()) {
        if (cli.simple_mode) {
          cli.tokenizer_path = auto_detect_tokenizer_path(cli.opts.model_path);
          if (cli.tokenizer_path.empty()) {
            throw std::runtime_error("could not auto-detect tokenizer; pass --tokenizer explicitly");
          }
        } else {
          throw std::runtime_error("--tokenizer is required when using --prompt or --interactive");
        }
      }
      if (!cli.interactive_mode && cli.chat_template.empty() && is_llama4_safetensors &&
          std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
        cli.chat_template = "llama4";
        info_out << "[info] defaulting to --chat-template llama4 for the configured safetensors model directory.\n";
      }
      if (!cli.interactive_mode && cli.stop_texts.empty()) {
        cli.stop_texts = default_stop_texts_for_template(cli.chat_template);
      }
      tokenizer.load(cli.tokenizer_path);
      if (tokenizer.eos_id() >= 0) {
        cli.opts.eos_token_id = tokenizer.eos_id();
      }

      if (!cli.interactive_mode) {
        bool tinyllama_plain_fallback =
            cli.chat_template == "tinyllama-chatml" && std::filesystem::path(cli.tokenizer_path).extension() != ".json" &&
            !cli.allow_legacy_chat_tokenizer;
        if (tinyllama_plain_fallback) {
          info_out << "[warn] TinyLlama tokenizer.model path is less reliable because this checkpoint ships a "
                      "tokenizer.json BPE tokenizer. Falling back to a plain instruction prompt.\n";
          info_out << "[hint] For best TinyLlama chat quality, use tokenizer.json from the same HF model.\n";
        } else if ((cli.chat_template == "tinyllama" || cli.chat_template == "tinyllama-chatml") &&
                   std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
          info_out << "[tokenizer] using native tokenizer.json BPE path\n";
        } else if (cli.chat_template == "llama4" && std::filesystem::path(cli.tokenizer_path).extension() != ".json") {
          info_out << "[warn] Llama4 is expected to use a HuggingFace tokenizer.json tokenizer.\n";
        } else if ((cli.chat_template == "llama3" || cli.chat_template == "phi3" || cli.chat_template == "qwen2") &&
                   std::filesystem::path(cli.tokenizer_path).extension() != ".json") {
          info_out << "[warn] " << cli.chat_template << " is expected to use a tokenizer.json (HF BPE). "
                      "Pass --tokenizer path/to/tokenizer.json for best results.\n";
        } else if (cli.chat_template == "mistral" && std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
          info_out << "[info] Using tokenizer.json for Mistral (HF BPE path).\n";
        } else if (cli.chat_template == "tinyllama-chatml" && cli.allow_legacy_chat_tokenizer) {
          info_out << "[warn] forcing legacy TinyLlama chat template with tokenizer.model; output quality may be poor.\n";
        }

        const std::string formatted_prompt = build_chat_prompt(cli.chat_template, cli.prompt_text, tinyllama_plain_fallback);
        bool add_bos = (cli.chat_template != "tinyllama") || tinyllama_plain_fallback;
        if (cli.chat_template == "tinyllama" || cli.chat_template == "llama4") {
          add_bos = false;
        }
        if (cli.force_no_bos) {
          add_bos = false;
        }

        prompt_tokens = tokenizer.encode(formatted_prompt, add_bos);
        if (cli.dump_tokenizer_meta) {
          info_out << "[tokenizer] bos_id=" << tokenizer.bos_id() << " eos_id=" << tokenizer.eos_id()
                   << " unk_id=" << tokenizer.unk_id() << "\n";
          info_out << "[tokenizer] special_ids(" << tokenizer.special_ids().size()
                   << "): " << join_ints(tokenizer.special_ids(), 64) << "\n";
        }
        if (cli.dump_prompt_tokens) {
          info_out << "[tokenizer] prompt_tokens(" << prompt_tokens.size()
                   << "): " << join_ints(prompt_tokens, 256) << "\n";
        }

        if (tokenizer.eos_id() >= 0) {
          stop_token_ids = {tokenizer.eos_id()};
        }
        for (const auto& st : cli.stop_texts) {
          const auto toks = tokenizer.encode(st, /*add_bos=*/false);
          if (toks.size() == 1) {
            const int tid = toks[0];
            if (std::find(stop_token_ids.begin(), stop_token_ids.end(), tid) == stop_token_ids.end()) {
              stop_token_ids.push_back(tid);
            }
          }
        }
      }
    } else {
      prompt_tokens = parse_tokens(cli.token_csv);
    }

    // --- Engine initialization: auto-detect GPU, fall back to CPU ---
#if LLAMA_ENGINE_HAS_CUDA
    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
#else
    const int cuda_device_count = 0;
#endif
    const bool is_llama4_model = is_safetensors_model_dir(cli.opts.model_path);
    const bool use_llama4_cpu_engine = is_llama4_model && (cli.force_cpu || cuda_device_count == 0);
#if LLAMA_ENGINE_HAS_CUDA
    const bool use_llama4_cuda_engine = is_llama4_model && !cli.force_cpu && cuda_device_count > 0;
#else
    const bool use_llama4_cuda_engine = false;
#endif
    const bool use_cpu_engine = use_llama4_cpu_engine || (!is_llama4_model && (cli.force_cpu || cuda_device_count == 0));
    if (!quiet_output) {
      if (use_llama4_cuda_engine) {
        std::cout << "[info] Detected a safetensors model. Using the Llama4 CUDA engine.\n";
      } else if (use_llama4_cpu_engine) {
        std::cout << "[info] Detected a safetensors model. Using the Llama4 CPU engine.\n";
      } else if (use_cpu_engine) {
#if LLAMA_ENGINE_HAS_CUDA
        std::cout << "[info] " << (cli.force_cpu ? "CPU engine forced via --cpu flag." : "No CUDA device found.")
                  << " Using CPU inference engine.\n";
#else
        std::cout << "[info] "
                  << (cli.force_cpu ? "CPU engine forced via --cpu flag." : "This binary was built without CUDA support.")
                  << " Using CPU inference engine.\n";
#endif
      }
    }

    app::main_modes::RunExecutionOptions run_opts;
    run_opts.interactive_mode = cli.interactive_mode;
    run_opts.quiet_output = quiet_output;
    run_opts.use_tokenizer = use_tokenizer;
    run_opts.sentence_stop = cli.sentence_stop;
    run_opts.benchmark_mode = cli.benchmark_mode;
    run_opts.benchmark_phases = cli.benchmark_phases;
    run_opts.simple_mode = cli.simple_mode;
    run_opts.force_no_bos = cli.force_no_bos;
    run_opts.max_new = cli.max_new;
    run_opts.temp = cli.temp;
    run_opts.inspect_next_topk = cli.inspect_next_topk;
    run_opts.trace_steps = cli.trace_steps;
    run_opts.benchmark_reps = cli.benchmark_reps;
    run_opts.benchmark_warmup = cli.benchmark_warmup;

    auto run_with_engine = [&](auto& eng) {
      eng.initialize(cli.opts);
#if LLAMA_ENGINE_HAS_CUDA
      if constexpr (std::is_same_v<std::decay_t<decltype(eng)>, engine::LlamaEngine>) {
        if (cli.parity_check) {
          eng.run_parity_check(prompt_tokens);
        }
      }
#endif

      app::main_modes::execute_engine_modes(
          run_opts,
          prompt_tokens,
          stop_token_ids,
          cli.stop_texts,
          use_tokenizer ? &tokenizer : nullptr,
          [&](const std::vector<int>& p, int max_new, float temperature) {
            return eng.generate(p, max_new, temperature);
          },
          [&](const std::vector<int>& p,
              int max_new,
              float temperature,
              const std::function<bool(int)>& on_token) {
            return eng.generate_stream(p, max_new, temperature, on_token);
          },
          [&](const std::vector<int>& p, int top_k) {
            return eng.inspect_next_logits(p, top_k);
          },
          [&]() -> const engine::BenchmarkStats& {
            return eng.last_benchmark_stats();
          });
    };

#if LLAMA_ENGINE_HAS_CUDA
    if (use_llama4_cuda_engine) {
      engine::Llama4CudaEngine llama4_cuda_eng;
      run_with_engine(llama4_cuda_eng);
    } else if (use_llama4_cpu_engine) {
#else
    if (use_llama4_cpu_engine) {
#endif
      engine::Llama4CpuEngine llama4_cpu_eng;
      run_with_engine(llama4_cpu_eng);
    } else if (use_cpu_engine) {
      engine::CpuLlamaEngine cpu_eng;
      run_with_engine(cpu_eng);
    } else {
#if LLAMA_ENGINE_HAS_CUDA
      engine::LlamaEngine gpu_eng;
      run_with_engine(gpu_eng);
#else
      throw std::runtime_error("CUDA inference was requested, but this binary was built without CUDA support");
#endif
    }
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
