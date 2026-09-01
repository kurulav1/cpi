#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "util/sha256.hpp"

// Pinning the INPUT side of a reproducibility claim.
//
// The token-stream hash says two runs produced the same answer. On its own that is
// half a claim: it does not say what they were asked. Someone reading a pasted hash
// cannot tell which weights file, which tokenizer, which prompt or which sampling
// settings produced it, and those are exactly the things that differ between their
// machine and yours. So the verifier emits digests for all four, plus one aggregate
// over them, and a reader can check the inputs match before wondering why the
// outputs do not.
namespace cpi::util {

struct InputDigest {
  std::string model;       // sha256 of the weights file, or a manifest for a directory
  std::string model_kind;  // "file" or "manifest" -- these are not the same claim
  std::uintmax_t model_bytes = 0;
  std::string tokenizer;
  std::string tokenizer_kind;
  std::string prompt;    // sha256 over the token ids actually fed to the model
  std::string sampling;  // sha256 over the canonical settings string
  std::string settings;  // the settings themselves, readable
  std::string aggregate;  // sha256 over the four above, in a fixed order
};

// Streams the file rather than mapping it: this runs on weights that can be tens of
// gigabytes, and a verification command has no business doubling the process's
// footprint to hash one.
[[nodiscard]] inline std::string sha256_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return "unreadable";
  }
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = f.gcount();
    if (got > 0) {
      h.update(buf.data(), static_cast<std::size_t>(got));
    }
  }
  return h.hex();
}

// A model given as a directory (a HuggingFace checkpoint) is hashed as a manifest of
// relative paths and sizes, sorted, NOT as its contents. That is a weaker statement
// and is labelled as one: it catches a different checkpoint or a truncated download,
// and would not catch an edited tensor. Hashing every shard of a 30 GB directory to
// print one line is a cost nobody asked for; the label is what keeps it honest.
[[nodiscard]] inline std::string sha256_dir_manifest(const std::filesystem::path& dir,
                                                     std::uintmax_t* total_bytes) {
  std::vector<std::string> entries;
  std::error_code ec;
  for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
       it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec) || ec) continue;
    const auto rel = std::filesystem::relative(it->path(), dir, ec);
    if (ec) continue;
    const auto sz = it->file_size(ec);
    if (ec) continue;
    if (total_bytes) *total_bytes += sz;
    // Forward slashes so the digest does not depend on which OS produced it.
    std::string name = rel.generic_string();
    entries.push_back(name + ":" + std::to_string(sz));
  }
  std::sort(entries.begin(), entries.end());
  Sha256 h;
  for (const auto& e : entries) {
    h.update(e);
    h.update("\n", 1);
  }
  return h.hex();
}

// model_path may be a file or a directory; tokenizer_path may be empty, which means
// the tokenizer travelled inside the weights file and is already covered by it.
[[nodiscard]] inline InputDigest compute_input_digest(const std::string& model_path,
                                                      const std::string& tokenizer_path,
                                                      const std::vector<int>& prompt_tokens,
                                                      const std::string& settings) {
  InputDigest d;
  std::error_code ec;
  const std::filesystem::path mp(model_path);
  if (std::filesystem::is_directory(mp, ec)) {
    d.model_kind = "manifest";
    d.model = sha256_dir_manifest(mp, &d.model_bytes);
  } else {
    d.model_kind = "file";
    d.model = sha256_file(mp);
    d.model_bytes = std::filesystem::file_size(mp, ec);
    if (ec) d.model_bytes = 0;
  }

  if (tokenizer_path.empty()) {
    d.tokenizer_kind = "embedded";
    d.tokenizer = "in-model";
  } else {
    d.tokenizer_kind = "file";
    d.tokenizer = sha256_file(std::filesystem::path(tokenizer_path));
  }

  // The token IDS, not the prompt text. Two different strings can tokenize to the
  // same ids and the same string can tokenize differently under another tokenizer;
  // what the model actually saw is the ids, so that is what gets pinned.
  {
    Sha256 h;
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
      if (i) h.update(",", 1);
      const std::string s = std::to_string(prompt_tokens[i]);
      h.update(s);
    }
    d.prompt = h.hex();
  }

  d.settings = settings;
  d.sampling = sha256_hex(settings);

  // One value that changes if any input changed, so a reader has a single thing to
  // compare before reading the four components to find out which one moved.
  Sha256 agg;
  agg.update(d.model);
  agg.update("|", 1);
  agg.update(d.tokenizer);
  agg.update("|", 1);
  agg.update(d.prompt);
  agg.update("|", 1);
  agg.update(d.sampling);
  d.aggregate = agg.hex();
  return d;
}

}  // namespace cpi::util
