// shard_planner -- pre-deployment "will it fit, on how many GPUs" report. Reads a model's config.json
// (or takes dims on the command line), then prints the EXACT per-rank VRAM footprint for a chosen
// tensor/pipeline/expert split and scans for the minimum world size that fits a target GPU. Pure host
// arithmetic (no GPU, no cluster); validate it by checking world=1 reproduces the known single-GPU
// footprints. See engine/shard_plan.hpp for the byte model.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "engine/shard_plan.hpp"
#include "model/json_mini.hpp"

namespace {

using engine::Quant;

Quant parse_quant(const std::string& s) {
  if (s == "int4") return Quant::INT4;
  if (s == "int8") return Quant::INT8;
  return Quant::FP16;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "";
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string humize(std::size_t bytes) {
  char buf[64];
  const double gb = bytes / (1024.0 * 1024.0 * 1024.0);
  if (gb >= 1.0) {
    std::snprintf(buf, sizeof(buf), "%.2f GB", gb);
  } else {
    std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
  }
  return buf;
}

int arg_int(int argc, char** argv, const char* flag, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
  return def;
}
std::string arg_str(int argc, char** argv, const char* flag, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return def;
}

void usage() {
  std::printf(
      "usage: shard_planner [--config <config.json|model_dir>] [dims...] [options]\n"
      "  --config PATH      HuggingFace config.json (or a dir containing one)\n"
      "  dims (override/without a config): --hidden N --layers N --heads N --kv-heads N\n"
      "       --head-dim N --inter N --vocab N --experts N --expert-inter N --tie 0|1\n"
      "  --quant fp16|int8|int4     weight quant (default fp16)\n"
      "  --kv-quant fp16|int8|int4  KV-cache quant (default fp16)\n"
      "  --seq N            context length for the KV cache (default 4096)\n"
      "  --batch N          concurrent sequences (default 1)\n"
      "  --tp N --pp N --ep N       parallelism degrees to report (default 1/1/1)\n"
      "  --gpu-mem-gb G     per-GPU memory for the fit scan (default 80)\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    usage();
    return 0;
  }
  using engine::mini::json_get_bool;
  using engine::mini::json_get_int;

  engine::ModelDims d;
  std::string cfg_path = arg_str(argc, argv, "--config", "");
  if (!cfg_path.empty()) {
    // Accept either a config.json or a directory holding one.
    if (cfg_path.size() < 5 || cfg_path.substr(cfg_path.size() - 5) != ".json")
      cfg_path += "/config.json";
    const std::string j = read_file(cfg_path);
    if (j.empty()) {
      std::fprintf(stderr, "error: could not read %s\n", cfg_path.c_str());
      return 1;
    }
    d.name = cfg_path;
    // Field names try HF first, then CPI's container schema (see config_json.hpp).
    d.hidden = json_get_int(j, "hidden_size", 0);
    d.num_layers = json_get_int(j, "num_hidden_layers", json_get_int(j, "num_layers", 0));
    d.num_heads = json_get_int(j, "num_attention_heads", json_get_int(j, "num_heads", 0));
    d.num_kv_heads = json_get_int(j, "num_key_value_heads", json_get_int(j, "num_kv_heads", d.num_heads));
    d.head_dim = json_get_int(j, "head_dim", d.num_heads > 0 ? d.hidden / d.num_heads : 0);
    d.intermediate = json_get_int(j, "intermediate_size", 0);
    d.vocab = json_get_int(j, "vocab_size", 0);
    d.num_experts = json_get_int(j, "num_local_experts", json_get_int(j, "num_experts", 0));
    d.expert_intermediate =
        json_get_int(j, "moe_intermediate_size", json_get_int(j, "expert_intermediate_size", 0));
    d.tie_embeddings = json_get_bool(j, "tie_word_embeddings", false);
    // MoE with a single intermediate field: it describes the experts, and there's no dense MLP.
    if (d.num_experts > 0 && d.expert_intermediate == 0) {
      d.expert_intermediate = d.intermediate;
      d.intermediate = 0;
    }
  }
  // Command-line overrides (also the way to plan a model with no config file).
  d.hidden = arg_int(argc, argv, "--hidden", d.hidden);
  d.num_layers = arg_int(argc, argv, "--layers", d.num_layers);
  d.num_heads = arg_int(argc, argv, "--heads", d.num_heads);
  d.num_kv_heads = arg_int(argc, argv, "--kv-heads", d.num_kv_heads ? d.num_kv_heads : d.num_heads);
  d.head_dim = arg_int(argc, argv, "--head-dim",
                       d.head_dim ? d.head_dim : (d.num_heads > 0 ? d.hidden / d.num_heads : 0));
  d.intermediate = arg_int(argc, argv, "--inter", d.intermediate);
  d.vocab = arg_int(argc, argv, "--vocab", d.vocab);
  d.num_experts = arg_int(argc, argv, "--experts", d.num_experts);
  d.expert_intermediate = arg_int(argc, argv, "--expert-inter", d.expert_intermediate);
  d.tie_embeddings = arg_int(argc, argv, "--tie", d.tie_embeddings ? 1 : 0) != 0;
  if (d.name.empty()) d.name = "(from flags)";

  if (d.hidden <= 0 || d.num_layers <= 0 || d.num_heads <= 0) {
    std::fprintf(stderr, "error: need at least --hidden, --layers, --heads (or a --config)\n");
    usage();
    return 1;
  }
  if (d.num_kv_heads <= 0) d.num_kv_heads = d.num_heads;
  if (d.head_dim <= 0) d.head_dim = d.hidden / d.num_heads;

  engine::PlanConfig pc;
  pc.weight_quant = parse_quant(arg_str(argc, argv, "--quant", "fp16"));
  pc.kv_quant = parse_quant(arg_str(argc, argv, "--kv-quant", "fp16"));
  pc.seq_len = arg_int(argc, argv, "--seq", 4096);
  pc.batch = arg_int(argc, argv, "--batch", 1);
  pc.tp = arg_int(argc, argv, "--tp", 1);
  pc.pp = arg_int(argc, argv, "--pp", 1);
  pc.ep = arg_int(argc, argv, "--ep", 1);
  const double gpu_gb = static_cast<double>(arg_int(argc, argv, "--gpu-mem-gb", 80));
  const std::size_t budget = static_cast<std::size_t>(gpu_gb * 1024.0 * 1024.0 * 1024.0);

  std::printf("Model: %s\n", d.name.c_str());
  std::printf("  hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d inter=%d vocab=%d\n", d.hidden,
              d.num_layers, d.num_heads, d.num_kv_heads, d.head_dim, d.intermediate, d.vocab);
  if (d.is_moe())
    std::printf("  MoE: experts=%d expert_inter=%d\n", d.num_experts, d.expert_intermediate);
  std::printf("  weight_quant=%s kv_quant=%s seq=%d batch=%d\n", engine::quant_name(pc.weight_quant),
              engine::quant_name(pc.kv_quant), pc.seq_len, pc.batch);

  const engine::Footprint f = estimate_footprint(d, pc);
  std::printf("\nPer-rank footprint at tp=%d pp=%d ep=%d (world=%d), %d layers on the heaviest rank:\n",
              pc.tp, pc.pp, pc.ep, pc.world(), f.layers_on_rank);
  std::printf("  embedding    %s\n", humize(f.embedding).c_str());
  std::printf("  lm head      %s%s\n", humize(f.head).c_str(), d.tie_embeddings ? "  (tied -> 0)" : "");
  std::printf("  attention    %s\n", humize(f.attn).c_str());
  if (f.dense_mlp) std::printf("  dense MLP    %s\n", humize(f.dense_mlp).c_str());
  if (f.experts) std::printf("  experts      %s\n", humize(f.experts).c_str());
  std::printf("  weights      %s  (subtotal)\n", humize(f.weights()).c_str());
  std::printf("  KV cache     %s  (seq=%d batch=%d)\n", humize(f.kv_cache).c_str(), pc.seq_len,
              pc.batch);
  std::printf("  activations  %s  (approx working-set)\n", humize(f.activations).c_str());
  std::printf("  -------------------------------\n");
  std::printf("  PER RANK     %s\n", humize(f.total()).c_str());
  std::printf("  fits %.0f GB GPU? %s\n", gpu_gb, f.total() <= budget ? "YES" : "NO");

  // Minimum-world scan.
  const auto opts =
      engine::scan_fit(d, pc.weight_quant, pc.kv_quant, pc.seq_len, pc.batch, budget);
  std::printf("\nMinimum world size to fit %.0f GB/GPU (weight_quant=%s, kv=%s, seq=%d, batch=%d):\n",
              gpu_gb, engine::quant_name(pc.weight_quant), engine::quant_name(pc.kv_quant),
              pc.seq_len, pc.batch);
  if (opts.empty()) {
    std::printf("  no tp/pp/ep split up to 64x fits -- lower seq/batch, quantise further, or raise GPU mem.\n");
  } else {
    const int best_world = opts.front().world();
    int shown = 0;
    for (const auto& o : opts) {
      if (o.world() > best_world && shown >= 1) break;  // show the smallest world + its variants only
      std::printf("  world=%2d  tp=%d pp=%d ep=%d  ->  %s / rank\n", o.world(), o.tp, o.pp, o.ep,
                  humize(o.per_rank_bytes).c_str());
      ++shown;
      if (shown >= 6) break;
    }
  }
  return 0;
}
