#pragma once

// Minimal Hugging Face tokenizer.json BPE runtime for LLaMA-family models.
//
// Supports the TinyLlama tokenizer layout used in this project:
//   - Sequence normalizer with prepend + space replacement
//   - BPE vocab/merges
//   - Byte fallback for out-of-vocabulary characters
//   - Special added tokens such as <s> and </s>
//
// Use HfBpeTokenizer directly only when you need low-level BPE access; in
// most cases prefer the model::Tokenizer facade which selects this backend
// automatically.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace model {

// Self-contained BPE tokenizer loaded from a Hugging Face tokenizer.json file.
// Instances are not thread-safe during load() but are safe for concurrent
// encode()/decode() calls after initialisation.
class HfBpeTokenizer {
public:
  // Loads vocabulary, merge rules, and special tokens from the tokenizer.json
  // file at path.  Throws on missing file or JSON parse errors.
  void load(const std::string& path);

  // Encodes text into a sequence of BPE token IDs.
  // If add_bos is true, the BOS token ID is prepended to the result.
  std::vector<int> encode(const std::string& text, bool add_bos) const;

  // Decodes a sequence of token IDs back into a UTF-8 string.
  // Byte-fallback tokens are collapsed back into their original bytes. Non-
  // special added tokens (e.g. "</think>") are emitted as their literal text;
  // special tokens (BOS/EOS/…) are dropped, except any listed in keep_special,
  // which are emitted as their literal text (reasoning delimiters like Gemma's
  // "<|channel>", which are `special` but must survive for the stream splitter).
  std::string decode(const std::vector<int>& ids,
                     const std::unordered_set<int>* keep_special = nullptr) const;
  // Byte-level/SentencePiece decode of a run of ordinary vocab ids (no added/
  // special token handling); used by decode() between added-token boundaries.
  std::string decode_run(const std::vector<int>& ids) const;

  // Returns the beginning-of-sequence token ID, or -1 if not present in vocab.
  int bos_id() const {
    return bos_id_;
  }

  // Returns the end-of-sequence token ID, or -1 if not present in vocab.
  int eos_id() const {
    return eos_id_;
  }

  // Returns the unknown token ID, or -1 if not present in vocab.
  int unk_id() const {
    return unk_id_;
  }

  // Returns the full set of special token IDs loaded from the added_tokens
  // section of tokenizer.json.
  const std::vector<int>& special_ids() const {
    return special_ids_;
  }

  // Returns a per-token byte table: token_pieces()[id] is the raw bytes token
  // `id` emits (byte-level / byte-fallback resolved, word-boundary marker -> a
  // space, no sequence-level leading-space strip). Special/added tokens map to
  // an empty string. Built once at load(); used by grammar-constrained decoding.
  const std::vector<std::string>& token_pieces() const {
    return token_pieces_;
  }

private:
  // Resolves a single token's vocabulary piece to the raw bytes it emits,
  // mirroring decode() for one token but without the leading-space strip.
  std::string piece_to_bytes(const std::string& piece) const;
  // Encodes a single pre-tokenised segment using BPE merge rules.
  // prepend_boundary controls whether the word-boundary marker is prepended.
  std::vector<int> encode_segment(const std::string& text, bool prepend_boundary) const;

  // Splits text into UTF-8 codepoint-aligned whitespace-separated pieces
  // following the normaliser's split behaviour.
  std::vector<std::string> split_utf8_pieces(const std::string& text) const;

  // Encodes a single text piece, falling back to individual byte tokens when
  // a piece is not found in the vocabulary (requires byte_fallback_ == true).
  std::vector<int> encode_piece_with_fallback(const std::string& piece) const;

  // Returns the canonical merge-table key for a BPE pair (left, right).
  std::string merge_key(const std::string& left, const std::string& right) const;

  std::unordered_map<std::string, int> vocab_;  // Piece-to-ID mapping from the vocab section.
  std::vector<std::string> id_to_piece_;        // Reverse ID-to-piece mapping for decode.
  std::vector<std::string>
      token_pieces_;  // Per-token emitted bytes for grammar masking (see token_pieces()).
  std::unordered_map<std::string, int>
      merge_ranks_;  // BPE merge pair -> priority rank (lower = earlier).
  std::vector<std::pair<std::string, int>> added_tokens_;  // Special/added tokens with their IDs.
  std::vector<int> special_ids_;                           // IDs of all added/special tokens.

  std::string word_boundary_ =
      "\xE2\x96\x81";           // UTF-8 encoding of the U+2581 word-boundary marker (▁).
  bool byte_fallback_ = false;  // When true, unknown pieces are encoded as individual byte tokens.
  bool strip_leading_space_ =
      true;                  // When true, a leading word-boundary marker is stripped during decode.
  bool byte_level_ = false;  // When true, use GPT-2 byte-to-unicode encoding (ByteLevel BPE).
  int bos_id_ = -1;          // Cached BOS token ID.
  int eos_id_ = -1;          // Cached EOS token ID.
  int unk_id_ = -1;          // Cached UNK token ID.

  // GPT-2 byte-to-unicode table: byte_to_unicode_[b] is the UTF-8 string for the
  // Unicode code point that byte b maps to.  Only populated when byte_level_ == true.
  std::string byte_to_unicode_[256];
  // Reverse mapping: Unicode code point -> byte.  Used during decode.
  std::unordered_map<std::uint32_t, unsigned char> unicode_to_byte_;
};

}  // namespace model
