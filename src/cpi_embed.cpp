// cpi_embed: CPI's embedding worker. Loads a BERT-family embedding model
// (config-driven) and serves embeddings over a line-delimited JSON protocol on
// stdin/stdout, mirroring how cpi serves chat. Driven by the Node
// server's /v1/embeddings route.
//
// Usage: cpi_embed <model_dir>
// Request line:  {"id":"x","input":"text" | ["t1","t2"],"input_type":"query"|"document"}
// Response line: {"id":"x","dim":384,"embeddings":[[...],...],"tokens":[n1,...]}
//                {"id":"x","error":"..."}
// Shutdown:      {"shutdown":true}

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "app/main_helpers.hpp"
#include "model/wordpiece_tokenizer.hpp"

// One embedder, chosen at compile time. The two implementations deliberately expose the same
// initialize/embed/dim/max_tokens/config surface, so everything below this point -- the protocol,
// the tokenizer, the batching -- is backend-agnostic and this is the only place that knows.
//
// CUDA wins when both are available, matching resolve_engine's preference for the discrete GPU.
#if CPI_HAS_CUDA
#include "engine/bert_embedder.hpp"
using CpiEmbedder = engine::BertEmbedder;
#elif defined(CPI_ENABLE_METAL)
#include "engine/metal_bert_embedder.hpp"
using CpiEmbedder = engine::MetalBertEmbedder;
#else
#error "cpi_embed needs a GPU backend: enable CUDA or Metal"
#endif

int main(int argc, char** argv) {
  using app::main_helpers::json_escape;
  using app::main_helpers::json_get_bool;
  using app::main_helpers::json_get_raw_value;
  using app::main_helpers::json_get_string;
  using app::main_helpers::json_get_string_array;

  if (argc < 2) {
    std::cerr << "usage: cpi_embed <model_dir>\n";
    return 2;
  }
  const std::string model_dir = argv[1];

  CpiEmbedder embedder;
  model::WordPieceTokenizer tokenizer;
  try {
    embedder.initialize(model_dir);
    const auto& cfg = embedder.config();
    tokenizer.load(model_dir, cfg.lowercase, cfg.strip_accents);
  } catch (const std::exception& e) {
    std::cerr << "[embed] init failed: " << e.what() << "\n";
    return 1;
  }
  const engine::EmbeddingConfig& cfg = embedder.config();
  std::cerr << "[embed] ready dim=" << embedder.dim()
            << " pool=" << (cfg.pooling == engine::PoolingMode::Mean ? "mean" : "cls")
            << " max_tokens=" << embedder.max_tokens() << " model=" << model_dir << "\n"
            << std::flush;

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    if (json_get_bool(line, "shutdown", false)) {
      break;
    }
    const std::string id = json_get_string(line, "id");
    const std::string input_type = json_get_string(line, "input_type");
    const std::string prefix = (input_type == "query") ? cfg.query_prefix : cfg.doc_prefix;

    // input may be a single string or an array of strings (order preserved).
    std::vector<std::string> inputs;
    const std::string raw = json_get_raw_value(line, "input");
    if (!raw.empty() && raw[0] == '[') {
      inputs = json_get_string_array(line, "input");
    } else if (!raw.empty()) {
      inputs.push_back(json_get_string(line, "input"));
    }

    try {
      std::ostringstream embs;
      std::ostringstream toks;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<int> ids = tokenizer.encode(prefix + inputs[i], embedder.max_tokens());
        const std::vector<float> v = embedder.embed(ids);
        if (i) {
          embs << ",";
          toks << ",";
        }
        embs << "[";
        for (std::size_t d = 0; d < v.size(); ++d) {
          if (d) embs << ",";
          char buf[24];
          std::snprintf(buf, sizeof(buf), "%.7g", v[d]);
          embs << buf;
        }
        embs << "]";
        toks << ids.size();
      }
      std::cout << "{\"id\":\"" << json_escape(id) << "\",\"dim\":" << embedder.dim()
                << ",\"embeddings\":[" << embs.str() << "],\"tokens\":[" << toks.str() << "]}\n"
                << std::flush;
    } catch (const std::exception& e) {
      std::cout << "{\"id\":\"" << json_escape(id) << "\",\"error\":\"" << json_escape(e.what())
                << "\"}\n"
                << std::flush;
    }
  }
  return 0;
}
