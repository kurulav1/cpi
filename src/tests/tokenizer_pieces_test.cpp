// Unit test for HfBpeTokenizer::token_pieces(): the per-token byte table used
// by grammar-constrained decoding. Writes a minimal SentencePiece-style
// tokenizer.json fixture at runtime, loads it, and asserts the byte mapping for
// a leading-space token, a digit token, a special token (empty), and a
// byte-fallback token. Pure C++ / GPU-free.
//
// Exit code 0 = all passed, 1 = a failure.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "model/hf_bpe_tokenizer.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::cerr << "FAIL: " << msg << "\n";
  }
}

// A minimal BPE tokenizer.json. The word-boundary marker is the default U+2581
// (the SentencePiece lower-block char), written as the 6-char JSON escape
// backslash-u-2581 so this source stays pure ASCII. byte_fallback is on
// so "<0x0A>" decodes to a newline byte. ids 0-2 are special.
const std::string kFixture =
    "{\n"
    "  \"added_tokens\": [\n"
    "    {\"id\": 0, \"content\": \"<unk>\", \"special\": true},\n"
    "    {\"id\": 1, \"content\": \"<s>\",   \"special\": true},\n"
    "    {\"id\": 2, \"content\": \"</s>\",  \"special\": true}\n"
    "  ],\n"
    "  \"model\": {\n"
    "    \"type\": \"BPE\",\n"
    "    \"byte_fallback\": true,\n"
    "    \"vocab\": {\n"
    "      \"<unk>\": 0,\n"
    "      \"<s>\": 1,\n"
    "      \"</s>\": 2,\n"
    "      \"\\u2581the\": 3,\n"
    "      \"3\": 4,\n"
    "      \"<0x0A>\": 5,\n"
    "      \"code\": 6\n"
    "    },\n"
    "    \"merges\": []\n"
    "  }\n"
    "}\n";

}  // namespace

int main() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cpi_tokenizer_pieces_test.json";
  {
    std::ofstream out(path, std::ios::binary);
    out << kFixture;
  }

  model::HfBpeTokenizer tok;
  try {
    tok.load(path.string());
  } catch (const std::exception& e) {
    std::cerr << "FAIL: tokenizer load threw: " << e.what() << "\n";
    return 1;
  }

  const std::vector<std::string>& pieces = tok.token_pieces();
  check(pieces.size() >= 7, "table covers all vocab ids");

  // Leading word-boundary marker becomes a leading space (not stripped).
  check(pieces.size() > 3 && pieces[3] == " the", "leading-space token -> ' the'");
  // A plain digit token maps to its literal byte.
  check(pieces.size() > 4 && pieces[4] == "3", "digit token '3' -> '3'");
  // A plain word token is unchanged.
  check(pieces.size() > 6 && pieces[6] == "code", "word token 'code' -> 'code'");
  // Byte-fallback token decodes to its raw byte.
  check(pieces.size() > 5 && pieces[5] == "\n", "byte-fallback '<0x0A>' -> newline");
  // Special tokens carry no surface bytes.
  check(pieces.size() > 2 && pieces[0].empty() && pieces[1].empty() && pieces[2].empty(),
        "special tokens map to empty bytes");

  std::error_code ec;
  std::filesystem::remove(path, ec);

  std::cout << "tokenizer_pieces_test: " << (g_checks - g_failures) << "/" << g_checks
            << " checks passed\n";
  if (g_failures != 0) {
    std::cerr << "tokenizer_pieces_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  return 0;
}
