// main.cpp - Command-line entry point for cpi.
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
// The image splicer is a template over the engine now (CUDA and Metal both use it), and its
// header pulls in no backend types -- so it sits outside both guards.
#include "app/image_prompt.hpp"
#if CPI_HAS_CUDA
#include "engine/llama4_cuda_engine.hpp"
#include "engine/llama_engine.hpp"
#include "engine/plan_cuda_engine.hpp"
#include "engine/speculative_decoder.hpp"
#endif
#if CPI_ENABLE_METAL
#include "engine/plan_metal_engine.hpp"
#include "engine/speculative_decoder.hpp"  // header-only template; harmless if CUDA already pulled it
#include "model/image_preprocess.hpp"
#include "runtime/metal_context.hpp"
#endif
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

#if CPI_HAS_CUDA

// One image + a text question, end to end. The chat text carries a single `<|image|>`
// placeholder; app::image_prompt::expand turns it into the real image span.
void run_with_image(engine::PlanCudaEngine& eng, const app::main_cli::ParsedArgs& cli,
                    model::Tokenizer& tokenizer, std::ostream& info_out) {
  eng.open(cli.opts.model_path, cli.opts.max_context);

  const std::string text = "<|turn>user\n<|image|>\n" + cli.prompt_text + "<turn|>\n<|turn>model\n";
  const std::vector<int> base = tokenizer.encode(text, /*add_bos=*/true);
  const auto ip = app::image_prompt::expand(eng, base, cli.image_path,
                                            std::getenv("CPI_IMAGE_CAUSAL") != nullptr);
  info_out << "[image] " << cli.image_path << "  " << ip.grid_w << "x" << ip.grid_h
           << " patches -> " << ip.image_tokens << " image tokens; prompt = " << ip.tokens.size()
           << " tokens\n\n";

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> out =
      eng.generate_multimodal(ip.tokens, ip.embeds, ip.limits, cli.max_new, cli.temp, nullptr);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  // Decode the whole sequence at once -- decoding token by token loses the word-boundary
  // markers and runs every word together.
  std::cout << tokenizer.decode(out) << "\n";
  std::cout << "\n[perf] generated_tokens=" << out.size() << " elapsed_ms=" << ms
            << " tok_per_s=" << (out.empty() ? 0.0 : out.size() / (ms / 1000.0)) << "\n";
}

// Binds the multimodal generator when the engine actually has a vision tower. The
// overload for every other engine type returns null, so this is resolved by TYPE rather
// than by a runtime model-name check.
app::main_modes::GenerateMultimodalFn make_multimodal_fn(engine::PlanCudaEngine& eng) {
  return [&eng](const std::vector<int>& base, const std::string& image_path, int max_new,
                float temp, const std::function<bool(int)>& on_token) {
    const auto ip = app::image_prompt::expand(eng, base, image_path,
                                              std::getenv("CPI_IMAGE_CAUSAL") != nullptr);
    return eng.generate_multimodal(ip.tokens, ip.embeds, ip.limits, max_new, temp, on_token);
  };
}

#endif  // CPI_HAS_CUDA

// The fallback for every engine without a vision tower -- including all of them in a
// CUDA-free build, where the overload above does not exist.
template <typename E>
app::main_modes::GenerateMultimodalFn make_multimodal_fn(E&) {
  return nullptr;
}

#if CPI_ENABLE_METAL
// One image + a text question through the Metal Qwen3.5 tower, end to end. This is the CLI face
// of the pipeline metal_vision_test gates: the tower, the splice, M-RoPE and the two-position
// decode are all verified there against HuggingFace (MULTIMODAL_stream 8/8). What this adds is
// only the two ends the test stubs -- real image preprocessing (qwen2vl_preprocess, byte-exact vs
// the HF processor) and detokenised output.
//
// The special-token ids are Qwen3.5's and are NOT in the .ll2c container, so they are stated here
// with provenance. A different vision model would need its own ids and template.
// The reusable core: base prompt tokens + an image path -> generated tokens, streaming each one
// through on_token. Both the one-shot CLI (--image) and the REST/web bridge's multimodal route go
// through THIS, so there is one path and it is the one metal_vision_test gates -- rather than a
// second, unverified copy behind the server.
std::vector<int> generate_multimodal_metal(engine::PlanMetalEngine& meng,
                                           const std::vector<int>& base_tokens,
                                           const std::string& image_path, int max_new, int eos,
                                           const std::function<bool(int)>& on_token,
                                           std::ostream* info_out) {
  // GEMMA 4's tower first: the SAME expand + generate_multimodal pair the CUDA path runs --
  // the splice layout lives in one templated header, and the engines carry the same surface.
  // (meng.has_vision() is the Gemma tower; config().has_vision_tower() is Qwen3.5's, carried
  // by container v7. A model has one or the other, never both.)
  if (meng.has_vision()) {
    const auto ip = app::image_prompt::expand(meng, base_tokens, image_path,
                                              std::getenv("CPI_IMAGE_CAUSAL") != nullptr);
    if (info_out) {
      *info_out << "[image] " << image_path << "  " << ip.grid_w << "x" << ip.grid_h
                << " patches -> " << ip.image_tokens
                << " image tokens; prompt = " << ip.tokens.size() << " tokens\n\n";
    }
    (void)eos;  // the engine stops on its own config's eos, exactly as PlanCudaEngine does
    return meng.generate_multimodal(ip.tokens, ip.embeds, ip.limits, max_new, /*temp=*/0.0f,
                                    on_token);
  }

  constexpr int kImageToken = 248056;   // <|image_pad|>
  constexpr int kVisionStart = 248053;  // <|vision_start|>
  constexpr int kVisionEnd = 248054;    // <|vision_end|>
  // Qwen2VLImageProcessor's pixel-area budget (config size.shortest/longest_edge). Held here
  // because the container does not carry it; qwen2vl_preproc_test pins these same values.
  constexpr long kMinPixels = 65536;
  constexpr long kMaxPixels = 16777216;

  const model::LlamaConfig& c = meng.config();
  if (!c.has_vision_tower()) {
    throw std::runtime_error("--image: this model has no vision tower (text-only container)");
  }
  const int merge = c.vision_spatial_merge_size;

  // 1. Preprocess the image exactly as HuggingFace does (gated: qwen2vl_preproc_test).
  const model::image::Image img = model::image::load_png(image_path);
  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float stdv[3] = {0.5f, 0.5f, 0.5f};
  const model::image::Qwen2VLPatches pp =
      model::image::qwen2vl_preprocess(img, c.vision_patch_size, c.vision_temporal_patch_size,
                                       merge, mean, stdv, kMinPixels, kMaxPixels);

  // 2. Tower -> soft tokens.
  const std::vector<float> soft = meng.encode_image(pp.pixels, pp.grid_h, pp.grid_w);
  const int out_hidden = c.vision_out_hidden_size;
  const int n_soft = static_cast<int>(soft.size()) / out_hidden;
  const int mh = pp.grid_h / merge, mw = pp.grid_w / merge;

  // 3. Build the prompt: <vision_start> + n_soft image placeholders + <vision_end> + the
  //    question. embeds[i] holds a soft token for each placeholder position, empty elsewhere; the
  //    engine splices the soft tokens over the placeholders' embeddings.
  // The image span goes at the FRONT, before the caller's tokens. If the prompt already carries a
  // single <|image_pad|> placeholder the span replaces it in place instead, so a chat template
  // that positions the image inside the user turn keeps that position.
  std::vector<int> toks;
  std::vector<std::vector<float>> embeds;
  auto push_image_span = [&]() {
    toks.push_back(kVisionStart);
    embeds.emplace_back();
    for (int i = 0; i < n_soft; ++i) {
      toks.push_back(kImageToken);
      embeds.emplace_back(soft.begin() + static_cast<std::size_t>(i) * out_hidden,
                          soft.begin() + static_cast<std::size_t>(i + 1) * out_hidden);
    }
    toks.push_back(kVisionEnd);
    embeds.emplace_back();
  };
  const auto ph = std::find(base_tokens.begin(), base_tokens.end(), kImageToken);
  if (ph != base_tokens.end()) {
    for (auto it = base_tokens.begin(); it != ph; ++it) {
      toks.push_back(*it);
      embeds.emplace_back();
    }
    push_image_span();
    for (auto it = ph + 1; it != base_tokens.end(); ++it) {
      toks.push_back(*it);
      embeds.emplace_back();
    }
  } else {
    push_image_span();
    for (int t : base_tokens) {
      toks.push_back(t);
      embeds.emplace_back();
    }
  }

  if (info_out != nullptr) {
    *info_out << "[image] " << image_path << "  " << pp.grid_w << "x" << pp.grid_h << " patches -> "
              << n_soft << " soft tokens; prompt = " << toks.size() << " tokens\n\n";
  }

  // 4. M-RoPE positions, prefill all but the last token, then decode with the rotary and cache
  //    positions kept apart (an image span advances the rotary counter by its MERGED extent).
  const std::vector<std::int32_t> mpos = engine::build_mrope_positions(toks, kImageToken, mh, mw);
  const int n_tok = static_cast<int>(toks.size());
  const std::vector<int> head(toks.begin(), toks.end() - 1);
  const std::vector<std::vector<float>> head_embeds(embeds.begin(), embeds.end() - 1);
  std::vector<std::int32_t> head_mpos(static_cast<std::size_t>(3) * (n_tok - 1));
  for (int axis = 0; axis < 3; ++axis) {
    for (int t = 0; t < n_tok - 1; ++t) {
      head_mpos[static_cast<std::size_t>(axis) * (n_tok - 1) + t] =
          mpos[static_cast<std::size_t>(axis) * n_tok + t];
    }
  }
  meng.prefill_multimodal(head, head_embeds, head_mpos);

  const int rope_pos0 = engine::mrope_next_position(toks, kImageToken, mh, mw) - 1;
  const int cache_pos0 = static_cast<int>(head.size());

  std::vector<int> gen;
  int next = toks.back();
  for (int i = 0; i < max_new; ++i) {
    const std::vector<float>& lg = meng.forward_token(next, rope_pos0 + i, cache_pos0 + i);
    int best = 0;
    for (std::size_t j = 1; j < lg.size(); ++j) {
      if (lg[j] > lg[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
    }
    if (eos >= 0 && best == eos) break;
    gen.push_back(best);
    next = best;
    if (on_token && !on_token(best)) break;  // caller asked to stop (client disconnect, stop text)
  }
  return gen;
}

// One image + a text question on the command line. A thin wrapper over the shared core above.
void run_with_image_metal(engine::PlanMetalEngine& meng, const app::main_cli::ParsedArgs& cli,
                          model::Tokenizer& tokenizer, std::ostream& info_out) {
  // Gemma's chat text carries the placeholder inside the user turn -- the same string CUDA's
  // run_with_image builds, so the two CLIs ask the model the same question the same way.
  const std::string text =
      meng.has_vision() ? "<|turn>user\n<|image|>\n" + cli.prompt_text + "<turn|>\n<|turn>model\n"
                        : cli.prompt_text;
  const std::vector<int> base = tokenizer.encode(text, /*add_bos=*/meng.has_vision());
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> gen = generate_multimodal_metal(meng, base, cli.image_path, cli.max_new,
                                                         cli.opts.eos_token_id, nullptr, &info_out);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << tokenizer.decode(gen) << "\n";
  std::cout << "\n[perf] generated_tokens=" << gen.size() << " elapsed_ms=" << ms
            << " tok_per_s=" << (gen.empty() ? 0.0 : gen.size() / (ms / 1000.0)) << "\n";
}
#endif

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

  // Backward-compat: the LLAMA_INFER_* environment variables were renamed to CPI_* in v0.2.0.
  // Map any still-set old var onto its new name (unless the new one is already set) so existing
  // .env files and scripts keep working, with a one-line deprecation notice.
  app::main_cli::apply_legacy_env_aliases();

  // --version / --help are answered before anything else (no model, no instance lock), so they
  // work as plain informational commands.
#ifndef CPI_VERSION
#define CPI_VERSION "0.0.0"
#endif
#ifndef CPI_GIT_SHA
#define CPI_GIT_SHA "unknown"
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--version" || a == "-V") {
      const char* backends =
#if defined(CPI_HAS_CUDA) && CPI_HAS_CUDA
          "cpu, cuda"
#elif defined(CPI_HAS_METAL) && CPI_HAS_METAL
          "cpu, metal"
#else
          "cpu"
#endif
          ;
      std::cout << "cpi " CPI_VERSION " (" CPI_GIT_SHA ")\n" << "backends: " << backends << "\n";
      return 0;
    }
    if (a == "--help" || a == "-h") {
      app::main_cli::print_usage(std::cout);
      return 0;
    }
  }

  SingleInstanceGuard instance_guard;
  if (!instance_guard.acquire()) {
    std::cerr << "Another cpi instance is already running.\n";
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
#if CPI_HAS_CUDA
    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
#else
    const int cuda_device_count = 0;
#endif
    // Resolve the whole dispatch decision — model family (probe) + device
    // situation — into a single engine choice, replacing the former is_X/use_X
    // boolean tangle. (Gemma 4 without CUDA throws here, as before.)
#if CPI_HAS_CUDA
    const bool cuda_available = cuda_device_count > 0;
#else
    const bool cuda_available = false;
#endif
    // Apple Silicon: probe for a real Metal GPU (nil inside a GPU-less VM). Only matters when
    // there is no CUDA device -- Metal is the Mac's GPU fast path.
    bool metal_available = false;
#if CPI_ENABLE_METAL
    if (!cuda_available && !cli.force_cpu) {
      runtime::MetalContext mctx;
      metal_available = mctx.available();
    }
#endif
    const app::main_helpers::EngineChoice engine_choice = app::main_helpers::resolve_engine(
        model_probe, cuda_available, metal_available, cli.force_cpu);

    if (!quiet_output) {
      using app::main_helpers::EngineChoice;
      switch (engine_choice) {
        case EngineChoice::PlanCuda:
          std::cout << "[info] Detected a Gemma 4 (.cpi) model. Using the generic op-plan CUDA "
                       "engine.\n";
          break;
        case EngineChoice::Qwen35Cuda:
          std::cout << "[info] Detected a Qwen3.5 model. Using the Qwen3.5 CUDA engine.\n";
          break;
        case EngineChoice::Qwen35Cpu:
          std::cout << "[info] Detected a Qwen3.5 model. Using the Qwen3.5 CPU engine.\n";
          break;
        case EngineChoice::Llama4Cuda:
          std::cout << "[info] Detected a safetensors model. Using the Llama4 CUDA engine.\n";
          break;
        case EngineChoice::Llama4Cpu:
          std::cout << "[info] Detected a safetensors model. Using the Llama4 CPU engine.\n";
          // SAY that the GPU is being skipped. On a Mac this path is reached with a perfectly
          // good Metal device present -- Llama4 has its own engine (llama4_cuda_engine) with no
          // Metal counterpart, so resolve_engine falls back to CPU. That fallback used to be
          // silent, which reads as "CPI is slow on this model" rather than "CPI is not using
          // your GPU". A degradation nobody can attribute is the same failure mode as --image
          // and --draft-model silently doing nothing.
          if (metal_available && !cli.force_cpu) {
            std::cout << "[warn] A Metal GPU is present but Llama4 has no Metal engine yet, so "
                         "this will run on the CPU and be far slower. Only the Llama/Qwen2/Qwen3 "
                         "and Qwen3.5 families have a Metal path.\n";
          }
          break;
        case EngineChoice::LlamaCpu:
#if CPI_HAS_CUDA
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
        case EngineChoice::LlamaMetal:
        case EngineChoice::Qwen35Metal:
        case EngineChoice::Gemma4Metal:
          std::cout << "[info] No CUDA device found. Using the Metal GPU engine (Apple Silicon).\n";
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
#if CPI_HAS_CUDA
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
          app::main_modes::run_interactive_batch(
              eng.batch_scheduler(), tokenizer, cli.stop_texts, !cli.force_no_bos, cli.max_new,
              cli.temp, cli.opts.top_k, cli.opts.top_p, cli.opts.repetition_penalty,
              cli.opts.no_repeat_ngram_size);
          // Plain `return`: this block is inside a lambda, not main(). The Metal branch's
          // identical-looking `return 0` IS in main(). Getting these the same way round breaks
          // one compiler or the other, and each hides the other's error -- this one is behind
          // #if CPI_HAS_CUDA, so a Mac build never sees it.
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
          [&]() -> const engine::BenchmarkStats& { return eng.last_benchmark_stats(); },
          // Images: only the plan executor has a vision tower. Everything else passes
          // null, and an image request against it fails cleanly rather than quietly
          // dropping the picture.
          make_multimodal_fn(eng));
    };

    using app::main_helpers::EngineChoice;
    switch (engine_choice) {
#if CPI_HAS_CUDA
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
#if CPI_ENABLE_METAL
      // All three reach the same executor: Qwen3.5 and Gemma 4 are op plans like any other, and
      // the engine picks its builder from the container's family. The separate enum members exist
      // so the choice is recorded rather than inferred.
      case EngineChoice::Qwen35Metal:
      case EngineChoice::Gemma4Metal:
      case EngineChoice::LlamaMetal: {
        // Apple Silicon GPU path for the main binary -- the same PlanMetalEngine the metal_infer
        // tool uses, wired into the standard serving modes so the REST/web bridge runs on a Mac.
        // What actually works here, verified on an M4 rather than assumed -- the comment this
        // replaces claimed "fp16 for now; grammar-constrained decode and a Metal vision tower are
        // not yet wired", and two thirds of that was stale:
        //
        //   int4/int8 weight-quant  yes (--weight-quant, below)
        //   continuous batching     yes (--interactive-batch, two concurrent streams interleaved)
        //   grammar / JSON-schema   yes -- proven by forcing a field name the model would never
        //                           emit unprompted; without the schema it produces "name"/"id"
        //   vision tower            yes, both of them: Qwen3.5 (gated 8/8 vs HF) and Gemma 4
        //                           (expand + generate_multimodal, gated vs the CUDA tower's
        //                           soft tokens) -- run_with_image_metal routes by which tower
        //                           the model carries.
        //   speculative decoding    yes (--draft-model, the block just below).
        if (!cli.image_path.empty() && !use_tokenizer) {
          throw std::runtime_error("--image requires --tokenizer");
        }
        // Speculative decoding: a small DRAFT proposes K tokens, this model (the TARGET) checks
        // them in one parallel verify pass, and only its own argmax is ever emitted -- so the
        // output is identical to plain greedy decoding of the target, just faster when the draft
        // guesses well. Two PlanMetalEngine instances; unified memory means each just holds its
        // own weights. Not compatible with --interactive-batch (that path is its own server).
        if (!cli.draft_model_path.empty()) {
          if (cli.interactive_batch) {
            throw std::runtime_error(
                "--draft-model and --interactive-batch are mutually exclusive");
          }
          engine::PlanMetalEngine target_eng;
          const int tgt_quant = cli.opts.int8_streaming ? cli.opts.streaming_quant_bits : 0;
          target_eng.open(cli.opts.model_path, cli.opts.max_context, tgt_quant, 0,
                          cli.opts.rope_theta);

          // Quantize the draft to int8 at load. A small model loses almost nothing to int8, and
          // a draft that does not fit fully resident is far too slow to be worth speculating
          // with. This mirrors the CUDA path's draft handling.
          engine::PlanMetalEngine draft_eng;
          draft_eng.open(cli.draft_model_path, cli.opts.max_context, 8, 0, cli.opts.rope_theta);

          engine::SpeculativeDecoder<engine::PlanMetalEngine> spec(draft_eng, target_eng,
                                                                   cli.spec_tokens);
          const int eos = cli.opts.eos_token_id;
          app::main_modes::execute_engine_modes(
              run_opts, prompt_tokens, stop_token_ids, cli.stop_texts,
              use_tokenizer ? &tokenizer : nullptr,
              [&](const std::vector<int>& p, int max_new, float /*temperature*/) {
                // spec.generate returns prompt+generated; PlanMetalEngine::generate returns
                // generated-only, so strip the prompt to keep the two Metal paths' output
                // identical rather than have speculation echo the prompt.
                std::vector<int> full = spec.generate(p, max_new, eos, nullptr);
                return std::vector<int>(full.begin() + std::min(p.size(), full.size()), full.end());
              },
              [&](const std::vector<int>& p, int max_new, float temperature,
                  const std::function<bool(int)>& on_token,
                  const engine::GenerationConstraints* constraints) {
                // Grammar can't ride the speculative verify path (it argmaxes drafts on-device,
                // which a logit mask can't reach), so a constrained request falls back to the
                // target's own single-token decode -- same choice the CUDA path makes.
                if (constraints != nullptr && constraints->grammar != nullptr) {
                  engine::PlanMetalEngine::Sampling s;
                  s.temperature = temperature;
                  s.top_k = cli.opts.top_k;
                  s.top_p = cli.opts.top_p;
                  s.repetition_penalty = cli.opts.repetition_penalty;
                  s.no_repeat_ngram_size = cli.opts.no_repeat_ngram_size;
                  s.eos_id = eos;
                  return target_eng.generate_stream(p, max_new, s, on_token, constraints);
                }
                // Strip the prompt, as generate_stream does: on_token already streamed the
                // generated tokens live, and the return value feeds the same display path.
                std::vector<int> full = spec.generate(p, max_new, eos, on_token);
                return std::vector<int>(full.begin() + std::min(p.size(), full.size()), full.end());
              },
              [&](const std::vector<int>& p, int top_k) {
                return target_eng.inspect_next_logits(p, top_k);
              },
              [&]() -> const engine::BenchmarkStats& { return target_eng.last_benchmark_stats(); },
              nullptr);

          if (!quiet_output) {
            const auto& s = spec.stats();
            std::cerr << "[spec] rounds=" << s.rounds << " drafted=" << s.drafted
                      << " accepted=" << s.accepted << " emitted=" << s.emitted
                      << " accept_rate=" << s.accept_rate()
                      << " tokens_per_round=" << s.tokens_per_round()
                      << " spec_tokens=" << cli.spec_tokens << "\n";
          }
          break;
        }
        engine::PlanMetalEngine meng;
        // --weight-quant int4/int8 maps to on-load host quantization;
        // otherwise fp16. This is what lets a large model (e.g. an 8B) serve on a 16 GB Mac.
        const int metal_quant_bits = cli.opts.int8_streaming ? cli.opts.streaming_quant_bits : 0;
        // Continuous batching needs the KV as a paged pool, and the pool is sized by open(),
        // so this has to be decided BEFORE the model loads -- not when the first request
        // arrives. Sized to the same budget the CUDA path uses: enough blocks for
        // --paged-blocks-tokens, defaulting to a few max_context-worth of concurrent sequences.
        if (cli.interactive_batch) {
          const int bs = cli.opts.paged_block_size > 0 ? cli.opts.paged_block_size : 32;
          // Room for ~4 concurrent max_context sequences. CUDA instead sizes its pool from a
          // VRAM budget, which does not transfer: on unified memory the KV competes with the
          // weights and the OS for the same pool, so "how many bytes are free" is a different
          // question here and deserves its own answer rather than a copied heuristic. Until
          // then this is a fixed, stated default -- the scheduler preempts newest-first when
          // it runs out, which is a real behaviour rather than a crash.
          const int pool_tokens = cli.opts.max_context * 4;
          meng.set_paged_kv((pool_tokens + bs - 1) / bs, bs);
        }
        meng.open(cli.opts.model_path, cli.opts.max_context, metal_quant_bits, /*quant_group=*/0,
                  cli.opts.rope_theta);

        // One-shot `--image`. After open() so it shares the engine with the serving modes below
        // rather than loading the weights a second time.
        if (!cli.image_path.empty() && !cli.web_mode && !cli.interactive_batch) {
          run_with_image_metal(meng, cli, tokenizer, std::cerr);
          break;
        }

        if (cli.interactive_batch) {
          if (!use_tokenizer) {
            throw std::runtime_error("--interactive-batch requires --tokenizer");
          }
          // The same worker and the same scheduler CUDA drives -- this branch exists only to
          // build the engine, not to reimplement serving.
          engine::BatchSchedulerOptions bo;
          bo.max_context = cli.opts.max_context;
          bo.eos_token_id = cli.opts.eos_token_id;
          bo.verbose = cli.opts.verbose;
          app::main_modes::run_interactive_batch(
              meng.batch_scheduler(bo), tokenizer, cli.stop_texts, !cli.force_no_bos, cli.max_new,
              cli.temp, cli.opts.top_k, cli.opts.top_p, cli.opts.repetition_penalty,
              cli.opts.no_repeat_ngram_size);
          return 0;
        }
        auto make_samp = [&](float temperature) {
          engine::PlanMetalEngine::Sampling s;
          s.temperature = temperature;
          s.top_k = cli.opts.top_k;
          s.top_p = cli.opts.top_p;
          // --repeat-penalty and --no-repeat-ngram were parsed, stored, and then dropped here.
          // The batch path below forwards them per-request via StreamParams, so the flags worked
          // under --interactive-batch and silently did nothing everywhere else -- which is the
          // shape of bug that reads as "the model is repetitive on Mac" rather than as a missing
          // feature. Sampling has carried both fields all along.
          s.repetition_penalty = cli.opts.repetition_penalty;
          s.no_repeat_ngram_size = cli.opts.no_repeat_ngram_size;
          s.eos_id = cli.opts.eos_token_id;
          return s;
        };
        app::main_modes::execute_engine_modes(
            run_opts, prompt_tokens, stop_token_ids, cli.stop_texts,
            use_tokenizer ? &tokenizer : nullptr,
            [&](const std::vector<int>& p, int max_new, float temperature) {
              return meng.generate(p, max_new, make_samp(temperature));
            },
            [&](const std::vector<int>& p, int max_new, float temperature,
                const std::function<bool(int)>& on_token,
                const engine::GenerationConstraints* constraints) {
              return meng.generate_stream(p, max_new, make_samp(temperature), on_token,
                                          constraints);
            },
            [&](const std::vector<int>& p, int top_k) {
              return meng.inspect_next_logits(p, top_k);
            },
            [&]() -> const engine::BenchmarkStats& { return meng.last_benchmark_stats(); },
            // Multimodal over the bridge. Null when the container has no vision tower, so a
            // text-only model still fails the request cleanly ("this model cannot take images")
            // instead of pretending. This used to be an unconditional nullptr, which meant the
            // REST/web image route was dead on Metal even after the CLI path worked.
            // has_vision_tower() is Qwen3.5's tower (container v7); has_vision() is Gemma 4's.
            (meng.config().has_vision_tower() || meng.has_vision())
                ? app::main_modes::GenerateMultimodalFn(
                      [&](const std::vector<int>& base, const std::string& image_path, int max_new,
                          float /*temperature*/, const std::function<bool(int)>& on_token) {
                        return generate_multimodal_metal(meng, base, image_path, max_new,
                                                         cli.opts.eos_token_id, on_token, nullptr);
                      })
                : nullptr);
        break;
      }
#endif
      case EngineChoice::LlamaCpu: {
        engine::CpuLlamaEngine cpu_eng;
        run_with_engine(cpu_eng);
        break;
      }
#if CPI_HAS_CUDA
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
