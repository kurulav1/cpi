// main.cpp - Command-line entry point for llama_infer.
//
// Parses CLI arguments, configures tokenizer/prompt state, auto-selects the
// execution engine, and dispatches runtime modes.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
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
#include "engine/qwen35_cpu_engine.hpp"
#if LLAMA_ENGINE_HAS_CUDA
#include "engine/plan_cuda_engine.hpp"
#include "engine/llama4_cuda_engine.hpp"
#include "engine/llama_engine.hpp"
#include "engine/speculative_decoder.hpp"
#endif
#include "model/image_preprocess.hpp"
#include "model/png.hpp"
#include "model/tokenizer.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

using app::main_helpers::auto_detect_tokenizer_path;
using app::main_helpers::build_chat_prompt;
using app::main_helpers::default_stop_texts_for_template;
using app::main_helpers::infer_safetensors_model_family;
using app::main_helpers::is_safetensors_model_dir;
using app::main_helpers::join_ints;
using app::main_helpers::parse_tokens;
using app::main_helpers::SingleInstanceGuard;

}  // namespace

namespace {

// One image + a text question, end to end.
//
// The token stream is:  <bos> <turn>user  <boi> <image>x280 <eoi>  <text>  <turn|> ...
// The 280 image tokens are placeholders: their EMBEDDINGS are replaced by the vision
// tower's soft tokens (used as-is -- unlike text embeddings they get no sqrt(hidden)
// scale). Their attention limit is set to the end of the span, which makes the image
// bidirectional while the surrounding text stays causal.
void run_with_image(engine::PlanCudaEngine& eng, const app::main_cli::ParsedArgs& cli,
                    model::Tokenizer& tokenizer, std::ostream& info_out) {
  eng.open(cli.opts.model_path, cli.opts.max_context);
  if (!eng.has_vision()) throw std::runtime_error("this model has no vision tower");
  if (!eng.can_sequence_prefill())
    throw std::runtime_error("this plan cannot sequence-prefill, which images require");

  const auto& v = eng.vision_config();
  const model::image::Image img = model::image::load_png(cli.image_path);
  info_out << "[image] " << cli.image_path << "  " << img.width << "x" << img.height << "\n";

  const int soft_budget = 280;  // vision_soft_tokens_per_image
  // NOT padded. HF pads the patch grid (for batching) and then masks the padding out of
  // attention; we encode one image, so simply not padding is equivalent -- and it avoids
  // the trap that without such a mask the padding patches take part in the encoder's
  // bidirectional attention as zero-valued keys and dilute every softmax. The oracle gate
  // missed this: its synthetic grid happened to need no padding.
  const model::image::PatchGrid grid = model::image::to_patches(
      img, v.patch_size, v.pooling_kernel, soft_budget, /*pad_to_budget=*/false);
  const std::vector<float> soft =
      eng.encode_image(grid.pixels, grid.pos_x, grid.pos_y, grid.soft_tokens);
  const int H = static_cast<int>(soft.size()) / grid.soft_tokens;
  // The text stream reserves one token per LIVE pooled cell, not per padded cell: the
  // processor inserts (grid_w/k)*(grid_h/k) image tokens. Using the padded budget
  // instead misaligns the span against the real soft tokens, and the model then reports
  // seeing no image at all.
  const int n_image = grid.live_soft_tokens;
  if (const char* dump = std::getenv("CPI_DUMP_SOFT")) {  // debug tap for the parity check
    std::ofstream df(dump, std::ios::binary);
    const int dims[2] = {n_image, H};
    df.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    df.write(reinterpret_cast<const char*>(soft.data()),
             static_cast<std::streamsize>(static_cast<std::size_t>(n_image) * H * sizeof(float)));
  }
  info_out << "[image] " << grid.grid_w << "x" << grid.grid_h << " patches -> " << n_image
           << " image tokens (the encoder emits " << grid.soft_tokens
           << "; the rest are padding cells)\n";

  // Gemma 4's image markers.
  constexpr int kBoi = 255999, kImage = 258880, kEoi = 258882;

  std::vector<int> tokens;
  std::vector<std::vector<float>> embeds;
  auto push_text = [&](const std::string& text, bool add_bos) {
    for (int t : tokenizer.encode(text, add_bos)) {
      tokens.push_back(t);
      embeds.emplace_back();
    }
  };

  push_text("<|turn>user\n", /*add_bos=*/true);
  const int span_start = static_cast<int>(tokens.size());
  tokens.push_back(kBoi);
  embeds.emplace_back();
  for (int i = 0; i < n_image; ++i) {
    tokens.push_back(kImage);
    embeds.emplace_back(soft.begin() + static_cast<std::size_t>(i) * H,
                        soft.begin() + static_cast<std::size_t>(i + 1) * H);
  }
  tokens.push_back(kEoi);
  embeds.emplace_back();
  const int span_end = static_cast<int>(tokens.size());
  push_text("\n" + cli.prompt_text + "<turn|>\n<|turn>model\n", /*add_bos=*/false);

  // Bidirectional image span: every token inside it sees the whole span; text is causal.
  // CPI_IMAGE_CAUSAL=1 falls back to a fully causal prompt (bisection aid).
  std::vector<int> limits(tokens.size());
  for (int i = 0; i < static_cast<int>(tokens.size()); ++i)
    limits[static_cast<std::size_t>(i)] = (i >= span_start && i < span_end) ? span_end : i + 1;
  if (std::getenv("CPI_IMAGE_CAUSAL")) limits.clear();

  info_out << "[image] prompt = " << tokens.size() << " tokens (" << (span_end - span_start)
           << " of them the image)\n\n";

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> out =
      eng.generate_multimodal(tokens, embeds, limits, cli.max_new, cli.temp, nullptr);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  // Decode the whole sequence at once -- decoding token by token loses the word-boundary
  // markers and runs every word together.
  std::cout << tokenizer.decode(out) << "\n";
  std::cout << "\n[perf] generated_tokens=" << out.size() << " elapsed_ms=" << ms
            << " tok_per_s=" << (out.empty() ? 0.0 : out.size() / (ms / 1000.0)) << "\n";
}

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
    // Single model-identity classification (the capability seam). Everything
    // below keys off this instead of scattering extension/dir/family checks.
    const app::main_helpers::ModelProbe model_probe =
        app::main_helpers::probe_model(cli.opts.model_path);
    const bool is_llama4_safetensors = model_probe.is_safetensors_dir;
    const std::string safetensors_family = model_probe.safetensors_family;

    // --- Tokenizer setup ---
    if (use_tokenizer) {
      if (cli.tokenizer_path.empty()) {
        if (cli.simple_mode) {
          cli.tokenizer_path = auto_detect_tokenizer_path(cli.opts.model_path);
          if (cli.tokenizer_path.empty()) {
            throw std::runtime_error(
                "could not auto-detect tokenizer; pass --tokenizer explicitly");
          }
        } else {
          throw std::runtime_error("--tokenizer is required when using --prompt or --interactive");
        }
      }
      if (!cli.interactive_mode && cli.chat_template.empty() && is_llama4_safetensors &&
          std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
        if (safetensors_family == "qwen3_5") {
          cli.chat_template = "qwen3_5";
          info_out << "[info] defaulting to --chat-template qwen3_5 for the configured Qwen3.5 "
                      "safetensors model directory.\n";
        } else if (safetensors_family == "gemma4") {
          cli.chat_template = "gemma";
          info_out << "[info] defaulting to --chat-template gemma for the configured Gemma 4 "
                      "safetensors model directory.\n";
        } else {
          cli.chat_template = "llama4";
          info_out << "[info] defaulting to --chat-template llama4 for the configured safetensors "
                      "model directory.\n";
        }
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
            cli.chat_template == "tinyllama-chatml" &&
            std::filesystem::path(cli.tokenizer_path).extension() != ".json" &&
            !cli.allow_legacy_chat_tokenizer;
        if (tinyllama_plain_fallback) {
          info_out << "[warn] TinyLlama tokenizer.model path is less reliable because this "
                      "checkpoint ships a "
                      "tokenizer.json BPE tokenizer. Falling back to a plain instruction prompt.\n";
          info_out << "[hint] For best TinyLlama chat quality, use tokenizer.json from the same HF "
                      "model.\n";
        } else if ((cli.chat_template == "tinyllama" || cli.chat_template == "tinyllama-chatml") &&
                   std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
          info_out << "[tokenizer] using native tokenizer.json BPE path\n";
        } else if (cli.chat_template == "llama4" &&
                   std::filesystem::path(cli.tokenizer_path).extension() != ".json") {
          info_out << "[warn] Llama4 is expected to use a HuggingFace tokenizer.json tokenizer.\n";
        } else if ((cli.chat_template == "llama3" || cli.chat_template == "phi3" ||
                    cli.chat_template == "qwen2" || cli.chat_template == "qwen3_5") &&
                   std::filesystem::path(cli.tokenizer_path).extension() != ".json") {
          info_out << "[warn] " << cli.chat_template
                   << " is expected to use a tokenizer.json (HF BPE). "
                      "Pass --tokenizer path/to/tokenizer.json for best results.\n";
        } else if (cli.chat_template == "mistral" &&
                   std::filesystem::path(cli.tokenizer_path).extension() == ".json") {
          info_out << "[info] Using tokenizer.json for Mistral (HF BPE path).\n";
        } else if (cli.chat_template == "tinyllama-chatml" && cli.allow_legacy_chat_tokenizer) {
          info_out << "[warn] forcing legacy TinyLlama chat template with tokenizer.model; output "
                      "quality may be poor.\n";
        }

        const std::string formatted_prompt =
            build_chat_prompt(cli.chat_template, cli.prompt_text, tinyllama_plain_fallback);
        bool add_bos = (cli.chat_template != "tinyllama") || tinyllama_plain_fallback;
        if (cli.chat_template == "tinyllama" || cli.chat_template == "llama4") {
          add_bos = false;
        }
        if (cli.force_no_bos) {
          add_bos = false;
        }

        prompt_tokens = tokenizer.encode(formatted_prompt, add_bos);
        if (cli.dump_tokenizer_meta) {
          info_out << "[tokenizer] bos_id=" << tokenizer.bos_id()
                   << " eos_id=" << tokenizer.eos_id() << " unk_id=" << tokenizer.unk_id() << "\n";
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
            if (std::find(stop_token_ids.begin(), stop_token_ids.end(), tid) ==
                stop_token_ids.end()) {
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
    // Resolve the whole dispatch decision — model family (probe) + device
    // situation — into a single engine choice, replacing the former is_X/use_X
    // boolean tangle. (Gemma 4 without CUDA throws here, as before.)
#if LLAMA_ENGINE_HAS_CUDA
    const bool cuda_available = cuda_device_count > 0;
#else
    const bool cuda_available = false;
#endif
    const app::main_helpers::EngineChoice engine_choice =
        app::main_helpers::resolve_engine(model_probe, cuda_available, cli.force_cpu);

    if (!quiet_output) {
      using app::main_helpers::EngineChoice;
      switch (engine_choice) {
        case EngineChoice::PlanCuda:
          std::cout << "[info] Detected a Gemma 4 (.cpi) model. Using the generic op-plan CUDA "
                       "engine.\n";
          break;
        case EngineChoice::Qwen35Cuda:
          std::cout
              << "[info] Detected a Qwen3.5 safetensors model. Using the Qwen3.5 CUDA engine.\n";
          break;
        case EngineChoice::Qwen35Cpu:
          std::cout << "[info] Detected a Qwen3.5 safetensors model. Using the Qwen3.5 CPU engine.\n";
          break;
        case EngineChoice::Llama4Cuda:
          std::cout << "[info] Detected a safetensors model. Using the Llama4 CUDA engine.\n";
          break;
        case EngineChoice::Llama4Cpu:
          std::cout << "[info] Detected a safetensors model. Using the Llama4 CPU engine.\n";
          break;
        case EngineChoice::LlamaCpu:
#if LLAMA_ENGINE_HAS_CUDA
          std::cout << "[info] "
                    << (cli.force_cpu ? "CPU engine forced via --cpu flag."
                                      : "No CUDA device found.")
                    << " Using CPU inference engine.\n";
#else
          std::cout << "[info] "
                    << (cli.force_cpu ? "CPU engine forced via --cpu flag."
                                      : "This binary was built without CUDA support.")
                    << " Using CPU inference engine.\n";
#endif
          break;
        case EngineChoice::LlamaCuda:
          break;  // default fast path — no banner
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
          // Pure gate: run the check and exit 0 (PASS) / 1 (FAIL) so a script/CI
          // can verify a forward-path change preserved correctness.
          const bool ok = eng.run_parity_check(prompt_tokens);
          std::cout.flush();
          std::exit(ok ? 0 : 1);
        }
        if (cli.batched_check > 0) {
          eng.run_batched_decode_check(prompt_tokens, cli.batched_check);
          return;
        }
        if (cli.scheduler_check > 0) {
          eng.run_scheduler_check(prompt_tokens, cli.scheduler_check, -1);
          return;
        }
        if (cli.batch_bench > 0) {
          eng.run_batch_bench(prompt_tokens, cli.batch_bench);
          return;
        }
        if (cli.interactive_batch) {
          if (!use_tokenizer) {
            throw std::runtime_error("--interactive-batch requires --tokenizer");
          }
          app::main_modes::run_interactive_batch(eng, tokenizer, cli.stop_texts, !cli.force_no_bos,
                                                 cli.max_new, cli.temp);
          return;
        }
      }
#endif

      app::main_modes::execute_engine_modes(
          run_opts, prompt_tokens, stop_token_ids, cli.stop_texts,
          use_tokenizer ? &tokenizer : nullptr,
          [&](const std::vector<int>& p, int max_new, float temperature) {
            return eng.generate(p, max_new, temperature);
          },
          [&](const std::vector<int>& p, int max_new, float temperature,
              const std::function<bool(int)>& on_token,
              const engine::GenerationConstraints* constraints) {
            return eng.generate_stream(p, max_new, temperature, on_token, constraints);
          },
          [&](const std::vector<int>& p, int top_k) { return eng.inspect_next_logits(p, top_k); },
          [&]() -> const engine::BenchmarkStats& { return eng.last_benchmark_stats(); });
    };

    using app::main_helpers::EngineChoice;
    switch (engine_choice) {
#if LLAMA_ENGINE_HAS_CUDA
      // Gemma 4 and Qwen3.5 are both just op plans; the executor is the same.
      case EngineChoice::PlanCuda:
      case EngineChoice::Qwen35Cuda: {
        engine::PlanCudaEngine plan_eng;
        if (!cli.image_path.empty()) {
          run_with_image(plan_eng, cli, tokenizer, info_out);
          break;
        }
        run_with_engine(plan_eng);
        break;
      }
      case EngineChoice::Llama4Cuda: {
        engine::Llama4CudaEngine llama4_cuda_eng;
        run_with_engine(llama4_cuda_eng);
        break;
      }
#endif
      case EngineChoice::Qwen35Cpu: {
        engine::Qwen35CpuEngine qwen35_cpu_eng;
        run_with_engine(qwen35_cpu_eng);
        break;
      }
      case EngineChoice::Llama4Cpu: {
        engine::Llama4CpuEngine llama4_cpu_eng;
        run_with_engine(llama4_cpu_eng);
        break;
      }
      case EngineChoice::LlamaCpu: {
        engine::CpuLlamaEngine cpu_eng;
        run_with_engine(cpu_eng);
        break;
      }
#if LLAMA_ENGINE_HAS_CUDA
      case EngineChoice::LlamaCuda: {
      if (!cli.draft_model_path.empty()) {
        // Speculative decoding: target (this model) + a small draft model that
        // shares the tokenizer. Initialize the target first so the draft's VRAM
        // budgeting sees the remaining free memory.
        engine::LlamaEngine target_eng;
        target_eng.initialize(cli.opts);

        engine::EngineOptions draft_opts = cli.opts;
        draft_opts.model_path = cli.draft_model_path;
        // Quantize the (fp16) draft to int8 at load so it fits fully in the VRAM
        // left after the int4 target — a partially-cached, layer-streamed draft
        // is far too slow to be a useful speculator. int8 is near-lossless for a
        // small model, so acceptance is essentially unchanged.
        draft_opts.int8_streaming = true;
        draft_opts.streaming_quant_bits = 8;
        draft_opts.prefer_lowbit_cache = true;
        draft_opts.gpu_cache_all = true;
        engine::LlamaEngine draft_eng;
        draft_eng.initialize(draft_opts);

        engine::SpeculativeDecoder spec(draft_eng, target_eng, cli.spec_tokens);
        const int eos = cli.opts.eos_token_id;
        app::main_modes::execute_engine_modes(
            run_opts, prompt_tokens, stop_token_ids, cli.stop_texts,
            use_tokenizer ? &tokenizer : nullptr,
            [&](const std::vector<int>& p, int max_new, float /*temperature*/) {
              return spec.generate(p, max_new, eos, nullptr);
            },
            [&](const std::vector<int>& p, int max_new, float temperature,
                const std::function<bool(int)>& on_token,
                const engine::GenerationConstraints* constraints) {
              // Grammar-constrained decoding can't run on the speculative verify
              // path (it argmaxes K drafts on-device, which a logit mask can't
              // reach). Fall back to non-speculative single-token decode on the
              // target engine, which honours the grammar. Unconstrained requests
              // still use the fast speculative path.
              if (constraints != nullptr && constraints->grammar != nullptr) {
                return target_eng.generate_stream(p, max_new, temperature, on_token, constraints);
              }
              return spec.generate(p, max_new, eos, on_token);
            },
            [&](const std::vector<int>& p, int top_k) {
              return target_eng.inspect_next_logits(p, top_k);
            },
            [&]() -> const engine::BenchmarkStats& { return target_eng.last_benchmark_stats(); });

        if (!quiet_output) {
          const auto& s = spec.stats();
          std::cerr << "[spec] rounds=" << s.rounds << " drafted=" << s.drafted
                    << " accepted=" << s.accepted << " emitted=" << s.emitted
                    << " accept_rate=" << s.accept_rate()
                    << " tokens_per_round=" << s.tokens_per_round()
                    << " tree2_recovery_rate=" << s.tree2_recovery_rate()
                    << " spec_tokens=" << cli.spec_tokens << "\n";
        }
      } else {
        engine::LlamaEngine gpu_eng;
        run_with_engine(gpu_eng);
      }
        break;
      }
#endif
      default:
        // Reachable only in a no-CUDA build if a *Cuda choice were resolved, which
        // resolve_engine never returns without a CUDA device.
        throw std::runtime_error(
            "CUDA inference was requested, but this binary was built without CUDA support");
    }
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
