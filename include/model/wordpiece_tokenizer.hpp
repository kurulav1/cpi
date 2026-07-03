#pragma once

// Minimal, dependency-free WordPiece tokenizer for BERT-family embedding models
// (bge, gte, e5, ...). Mirrors HuggingFace's BertTokenizer pipeline:
//   1. BertNormalizer  — clean control chars, normalize whitespace, (optionally)
//      lowercase + strip accents, isolate CJK characters.
//   2. BertPreTokenizer — split on whitespace and isolate punctuation.
//   3. WordPiece        — greedy longest-match subwording with a "##"
//      continuation prefix; unmatched words map to [UNK].
//
// Loaded from a HuggingFace model directory's `vocab.txt` (one token per line,
// id = line index). English-first: accent stripping covers the common Latin
// precomposed letters.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace model {

class WordPieceTokenizer {
public:
  // Loads vocab.txt from `model_dir`. `lowercase` and `strip_accents` follow the
  // model's normalizer (bge-small-en is uncased: both true). Throws on error.
  void load(const std::string& model_dir, bool lowercase = true, bool strip_accents = true);

  // Encodes `text` into token ids, wrapped as [CLS] tokens... [SEP] and
  // truncated so the total length is at most max_tokens (graceful truncation).
  std::vector<int> encode(const std::string& text, int max_tokens) const;

  int cls_id() const {
    return cls_id_;
  }
  int sep_id() const {
    return sep_id_;
  }
  int pad_id() const {
    return pad_id_;
  }
  int unk_id() const {
    return unk_id_;
  }
  std::size_t vocab_size() const {
    return id_to_token_.size();
  }

private:
  // Greedy WordPiece over one normalized, whitespace-free word.
  void wordpiece(const std::string& word, std::vector<int>& out) const;

  std::unordered_map<std::string, int> vocab_;  // token -> id
  std::vector<std::string> id_to_token_;        // id -> token
  bool lowercase_ = true;
  bool strip_accents_ = true;
  int cls_id_ = 101;
  int sep_id_ = 102;
  int pad_id_ = 0;
  int unk_id_ = 100;
  int max_input_chars_per_word_ = 100;
};

}  // namespace model
