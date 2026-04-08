#pragma once

// Unified tokenizer facade that selects the appropriate backend at runtime.
//
// The Tokenizer class supports two backends:
//   - HfBpeTokenizer: a built-in BPE implementation loaded from a
//     Hugging Face tokenizer.json file (always available).
//   - SentencePiece: loaded via the sentencepiece library and invoked through
//     helper executables, available only when LLAMA_ENGINE_HAS_SENTENCEPIECE
//     is defined at compile time.
//
// Callers interact exclusively through this class; backend selection is
// performed transparently inside load().

#include <string>
#include <vector>

namespace model {

// Tokenizer wraps either a Hugging Face BPE tokenizer or a SentencePiece model
// behind a common encode/decode interface.  Use load() to initialise the
// tokenizer from a model directory before calling any other method.
class Tokenizer {
 public:
  ~Tokenizer();

  // Loads the tokenizer from the given model directory path.
  // If a tokenizer.json file is present the HfBpeTokenizer backend is used.
  // Otherwise, the SentencePiece backend is used (requires the
  // LLAMA_ENGINE_HAS_SENTENCEPIECE compile-time flag).
  // Throws on missing or malformed tokenizer files.
  void load(const std::string& path);

  // Encodes text into a sequence of token IDs.
  // If add_bos is true, the BOS token ID is prepended to the result.
  std::vector<int> encode(const std::string& text, bool add_bos) const;

  // Decodes a sequence of token IDs back into a UTF-8 string.
  std::string decode(const std::vector<int>& ids) const;

  // Returns the beginning-of-sequence token ID, or -1 if not defined.
  int bos_id() const { return bos_id_; }

  // Returns the end-of-sequence token ID, or -1 if not defined.
  int eos_id() const { return eos_id_; }

  // Returns the unknown token ID, or -1 if not defined.
  int unk_id() const { return unk_id_; }

  // Returns the full set of special token IDs (e.g. BOS, EOS, PAD).
  const std::vector<int>& special_ids() const { return special_ids_; }

  // Returns the token IDs that should terminate generation (typically EOS
  // and any other end-of-turn tokens defined by the model).
  std::vector<int> generation_stop_ids() const;

  // Returns ids with all special token IDs removed.
  std::vector<int> strip_special_ids(const std::vector<int>& ids) const;

 private:
  std::string model_path_;           // Directory that contains the tokenizer files.
  std::string tokenizer_json_path_;  // Resolved path to tokenizer.json, if present.
  std::string spm_encode_exe_;       // Path to the spm_encode helper binary (SPM backend).
  std::string spm_decode_exe_;       // Path to the spm_decode helper binary (SPM backend).
  int bos_id_ = -1;                  // Cached BOS token ID from the active backend.
  int eos_id_ = -1;                  // Cached EOS token ID from the active backend.
  int unk_id_ = -1;                  // Cached UNK token ID from the active backend.
  std::vector<int> special_ids_;     // All special token IDs collected from the backend.
  void* hf_bpe_ = nullptr;           // Owning pointer to HfBpeTokenizer (heap-allocated).
#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
  void* sp_ = nullptr;               // Owning pointer to the SentencePiece processor.
#endif
};

}  // namespace model
