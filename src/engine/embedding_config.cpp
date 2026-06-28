#include "engine/embedding_config.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine {
namespace {

std::optional<std::string> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::nullopt;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Finds `"key"` at top level and returns the index just after its colon.
std::size_t find_value(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t p = json.find(needle);
  if (p == std::string::npos) {
    return std::string::npos;
  }
  p = json.find(':', p + needle.size());
  if (p == std::string::npos) {
    return std::string::npos;
  }
  ++p;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) {
    ++p;
  }
  return p;
}

std::optional<double> json_num(const std::string& json, const std::string& key) {
  const std::size_t p = find_value(json, key);
  if (p == std::string::npos) {
    return std::nullopt;
  }
  std::size_t e = p;
  while (e < json.size() && (std::isdigit(static_cast<unsigned char>(json[e])) || json[e] == '-' ||
                             json[e] == '+' || json[e] == '.' || json[e] == 'e' || json[e] == 'E')) {
    ++e;
  }
  if (e == p) {
    return std::nullopt;
  }
  try {
    return std::stod(json.substr(p, e - p));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> json_bool(const std::string& json, const std::string& key) {
  const std::size_t p = find_value(json, key);
  if (p == std::string::npos) {
    return std::nullopt;
  }
  if (json.compare(p, 4, "true") == 0) return true;
  if (json.compare(p, 5, "false") == 0) return false;
  return std::nullopt;
}

std::optional<std::string> json_str(const std::string& json, const std::string& key) {
  const std::size_t p = find_value(json, key);
  if (p == std::string::npos || p >= json.size() || json[p] != '"') {
    return std::nullopt;
  }
  std::string out;
  std::size_t i = p + 1;
  while (i < json.size() && json[i] != '"') {
    if (json[i] == '\\' && i + 1 < json.size()) {
      const char e = json[i + 1];
      switch (e) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        default: out.push_back(e); break;
      }
      i += 2;
    } else {
      out.push_back(json[i]);
      ++i;
    }
  }
  return out;
}

}  // namespace

EmbeddingConfig EmbeddingConfig::load(const std::string& model_dir) {
  const auto cfg_text = read_file(model_dir + "/config.json");
  if (!cfg_text) {
    throw std::runtime_error("embedding: missing config.json in " + model_dir);
  }
  const std::string& c = *cfg_text;
  EmbeddingConfig cfg;
  const auto need_int = [&](const char* k) {
    const auto v = json_num(c, k);
    if (!v) {
      throw std::runtime_error(std::string("embedding: config.json missing ") + k);
    }
    return static_cast<int>(*v);
  };
  cfg.hidden_size = need_int("hidden_size");
  cfg.num_layers = need_int("num_hidden_layers");
  cfg.num_heads = need_int("num_attention_heads");
  cfg.intermediate_size = need_int("intermediate_size");
  cfg.max_position_embeddings = need_int("max_position_embeddings");
  cfg.vocab_size = need_int("vocab_size");
  if (const auto v = json_num(c, "type_vocab_size")) cfg.type_vocab_size = static_cast<int>(*v);
  if (const auto v = json_num(c, "layer_norm_eps")) cfg.layer_norm_eps = static_cast<float>(*v);
  if (const auto v = json_str(c, "hidden_act")) cfg.hidden_act = *v;
  if (const auto v = json_str(c, "position_embedding_type")) cfg.position_embedding_type = *v;
  if (const auto v = json_str(c, "model_type")) cfg.model_type = *v;
  cfg.dimension = cfg.hidden_size;
  cfg.max_tokens = cfg.max_position_embeddings;

  // Pooling default from sentence-transformers 1_Pooling/config.json.
  if (const auto pool_text = read_file(model_dir + "/1_Pooling/config.json")) {
    if (json_bool(*pool_text, "pooling_mode_cls_token").value_or(false)) {
      cfg.pooling = PoolingMode::Cls;
    } else if (json_bool(*pool_text, "pooling_mode_mean_tokens").value_or(false)) {
      cfg.pooling = PoolingMode::Mean;
    }
  }

  // Embedding-specific knobs (CPI-authored, model-agnostic schema).
  if (const auto e_text = read_file(model_dir + "/cpi_embed.json")) {
    const std::string& e = *e_text;
    if (const auto v = json_str(e, "pooling")) cfg.pooling = (*v == "mean") ? PoolingMode::Mean : PoolingMode::Cls;
    if (const auto v = json_bool(e, "normalize")) cfg.normalize = *v;
    if (const auto v = json_num(e, "dimension")) cfg.dimension = static_cast<int>(*v);
    if (const auto v = json_num(e, "max_tokens")) cfg.max_tokens = static_cast<int>(*v);
    if (const auto v = json_str(e, "query_prefix")) cfg.query_prefix = *v;
    if (const auto v = json_str(e, "doc_prefix")) cfg.doc_prefix = *v;
    if (const auto v = json_bool(e, "lowercase")) cfg.lowercase = *v;
    if (const auto v = json_bool(e, "strip_accents")) cfg.strip_accents = *v;
  }

  if (cfg.max_tokens > cfg.max_position_embeddings) {
    cfg.max_tokens = cfg.max_position_embeddings;
  }
  return cfg;
}

}  // namespace engine
