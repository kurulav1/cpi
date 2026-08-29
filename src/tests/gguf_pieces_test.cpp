// The invariant grammar-constrained decoding rests on: the per-token byte table
// the grammar masks with must be exactly what the detokenizer emits for that
// token.
//
// If they disagree for even one token, the sampler can allow a token believing it
// writes some bytes while generation writes different ones. The grammar then
// reports success and the output violates the schema, which is the failure mode
// worth caring about most: silently malformed structured output.
//
// This exists because tokenizer_pieces_test covers only the tokenizer.json path,
// with a synthetic fixture. The GGUF vocabulary path (load_from_vocab) had no
// coverage, and a real Llama-3 GGUF produced a tool name of "a\xC1\xA4d" against
// a grammar whose enum allowed only "add" and "get_weather".
//
// Needs a real GGUF, so it SKIPs when none is given. Pass one as argv[1] or in
// CPI_TEST_GGUF.

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "model/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string path;
  if (argc > 1) {
    path = argv[1];
  } else if (const char* env = std::getenv("CPI_TEST_GGUF")) {
    path = env;
  }
  if (path.empty()) {
    std::printf("[gguf_pieces] SKIP: no GGUF given (argv[1] or CPI_TEST_GGUF)\n");
    return 0;
  }

  model::Tokenizer tok;
  if (!tok.load_from_gguf(path)) {
    std::printf("[gguf_pieces] SKIP: %s is not a readable GGUF with a vocabulary\n", path.c_str());
    return 0;
  }

  const std::vector<std::string>& pieces = tok.token_pieces();
  if (pieces.empty()) {
    std::printf("[gguf_pieces] FAIL: token_pieces() is empty, so the grammar can mask nothing\n");
    return 1;
  }

  int mismatches = 0;
  int checked = 0;
  for (std::size_t id = 0; id < pieces.size(); ++id) {
    if (pieces[id].empty()) {
      continue;  // special/added tokens carry no bytes by contract
    }
    const std::string decoded = tok.decode({static_cast<int>(id)});
    ++checked;
    if (decoded != pieces[id]) {
      if (mismatches < 10) {
        std::printf("[gguf_pieces] id=%zu piece=%s decode=%s\n", id, pieces[id].c_str(),
                    decoded.c_str());
      }
      ++mismatches;
    }
  }
  std::printf("[gguf_pieces] checked %d tokens, %d mismatch(es)\n", checked, mismatches);
  if (mismatches > 0) {
    std::printf("[gguf_pieces] FAIL: the grammar masks in a different byte space than decode\n");
    return 1;
  }
  std::printf("[gguf_pieces] PASS\n");
  return 0;
}
