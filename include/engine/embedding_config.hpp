#pragma once

// Config-driven description of a BERT-family embedding model. Architecture is
// read from the model's own `config.json`; the embedding-specific knobs (pooling
// mode, normalization, query/document prefixes, output dimension) come from an
// optional `cpi_embed.json` in the same directory, falling back to the model's
// `1_Pooling/config.json` and sensible defaults. This is the "core" abstraction:
// a new embedding model is served by dropping its directory in, no code change.

#include <stdexcept>
#include <string>

namespace engine {

enum class PoolingMode { Cls, Mean };

struct EmbeddingConfig {
  // Architecture (from config.json).
  int hidden_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int intermediate_size = 0;
  int max_position_embeddings = 0;
  int vocab_size = 0;
  int type_vocab_size = 2;
  float layer_norm_eps = 1e-12f;
  std::string hidden_act = "gelu";                   // gelu (only supported act)
  std::string position_embedding_type = "absolute";  // absolute (only supported)
  std::string model_type = "bert";

  // Embedding behaviour (from cpi_embed.json / 1_Pooling/config.json).
  PoolingMode pooling = PoolingMode::Cls;
  bool normalize = true;     // L2-normalize the output
  int dimension = 0;         // output dim (defaults to hidden_size)
  int max_tokens = 512;      // truncate inputs to this many tokens
  std::string query_prefix;  // prepended to input_type=="query"
  std::string doc_prefix;    // prepended to input_type=="document"

  bool lowercase = true;  // tokenizer normalizer (from config/defaults)
  bool strip_accents = true;

  // Loads and merges config.json + cpi_embed.json + 1_Pooling/config.json from
  // `model_dir`. Throws std::runtime_error on a missing/invalid config.json.
  static EmbeddingConfig load(const std::string& model_dir);
};

}  // namespace engine
