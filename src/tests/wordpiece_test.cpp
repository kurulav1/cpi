// Unit test for the hand-rolled WordPiece tokenizer against known
// bert-base-uncased token ids (bge-small-en-v1.5 uses that 30522 vocab).
// Requires the bge model dir (vocab.txt). Pure C++, no CUDA.

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "model/wordpiece_tokenizer.hpp"

namespace {
int g_fail = 0, g_checks = 0;
void check(bool c, const std::string& m) {
  ++g_checks;
  if (!c) {
    ++g_fail;
    std::cerr << "FAIL: " << m << "\n";
  }
}
std::string vec(const std::vector<int>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}
}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "artifacts/hub/BAAI__bge-small-en-v1.5";
  // The bge model dir is gitignored, so it is absent in CI and other clean
  // checkouts. Report a CTest SKIP (return code 77) rather than a failure when
  // it is missing; the test still runs fully wherever the model is present.
  if (!std::filesystem::exists(dir)) {
    std::cerr << "SKIP: bge model dir not found at " << dir
              << " (gitignored; provide it to run this test)\n";
    return 77;
  }
  model::WordPieceTokenizer tok;
  try {
    tok.load(dir, /*lowercase=*/true, /*strip_accents=*/true);
  } catch (const std::exception& e) {
    std::cerr << "load failed: " << e.what() << "\n";
    return 2;
  }

  check(tok.cls_id() == 101 && tok.sep_id() == 102 && tok.unk_id() == 100,
        "special ids: CLS=101 SEP=102 UNK=100");

  // Known bert-base-uncased: hello=7592 world=2088 ,=1010 !=999
  const std::vector<int> hw = tok.encode("Hello, world!", 512);
  check(hw == std::vector<int>({101, 7592, 1010, 2088, 999, 102}),
        "Hello, world! -> [101,7592,1010,2088,999,102], got " + vec(hw));

  // Lowercasing: "HELLO" == "hello"
  check(tok.encode("HELLO", 512) == tok.encode("hello", 512), "uppercase lowercased");

  // Determinism: identical text -> identical ids
  check(tok.encode("the capital of France", 512) == tok.encode("the capital of France", 512),
        "deterministic");

  // Subwording: a rare word splits into multiple ## pieces, all real ids.
  const std::vector<int> sub = tok.encode("tokenization", 512);
  check(sub.size() > 3 && sub.front() == 101 && sub.back() == 102,
        "subword splits + CLS/SEP wrap, got " + vec(sub));

  // Graceful truncation: long input capped at max_tokens incl CLS/SEP.
  std::string longtext;
  for (int i = 0; i < 1000; ++i) longtext += "word ";
  const std::vector<int> trunc = tok.encode(longtext, 64);
  check(trunc.size() == 64 && trunc.front() == 101 && trunc.back() == 102,
        "truncated to 64 with CLS/SEP, got size " + std::to_string(trunc.size()));

  std::cout << "wordpiece_test: " << (g_checks - g_fail) << "/" << g_checks << " passed\n";
  if (g_fail) {
    std::cerr << g_fail << " failure(s)\n";
    return 1;
  }
  return 0;
}
