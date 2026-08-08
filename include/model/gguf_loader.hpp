#pragma once

// GGUF reader: run the files people already have.
//
// GGUF is llama.cpp's container, and it is what almost every published local
// checkpoint is distributed as. Reading it directly means a user's existing
// collection runs without a conversion step, and it means CPI and llama.cpp can
// be benchmarked on byte-identical weights instead of two conversions of the
// same checkpoint.
//
// The loader presents the same surface as WeightLoader / SafetensorsLoader
// (has_tensor / tensor_data / tensor_bytes / tensor_dtype / config), so the
// engines consume it without knowing which container they were handed.
//
// Two things are not free:
//   - Quantized tensors cannot be handed out as raw pointers, because CPI's
//     fp16 paths expect fp16. They are dequantized on first access into an
//     owned cache, so a Q4_0 file costs its fp16 size in RAM once warm. Keeping
//     block-quantized weights packed end-to-end is a follow-up, not a
//     correctness question.
//   - LLaMA-family GGUFs store Q and K permuted for ggml's interleaved RoPE.
//     CPI (like HuggingFace) rotates split halves, so those two tensors are
//     un-permuted at load. Skipping this produces fluent-looking garbage rather
//     than an error, which is exactly the class of bug worth naming here.
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/llama_config.hpp"
#include "platform/mmap_file.hpp"

namespace model {

// True when `path` names a file whose first four bytes are the GGUF magic.
// Used for engine/container selection before anything is mapped.
[[nodiscard]] bool is_gguf_file(const std::string& path);

class GgufLoader {
public:
  // Maps the file and parses the header, metadata and tensor table. Throws with
  // a specific message on a bad magic, an unsupported version, or a tensor type
  // this build cannot dequantize.
  void open(const std::string& path);

  [[nodiscard]] const std::byte* tensor_data(const std::string& name) const;
  [[nodiscard]] std::size_t tensor_bytes(const std::string& name) const;
  [[nodiscard]] bool has_tensor(const std::string& name) const;
  [[nodiscard]] std::string tensor_dtype(const std::string& name) const;
  [[nodiscard]] std::vector<std::string> tensor_names() const;

  [[nodiscard]] const LlamaConfig& config() const {
    return config_;
  }

  // Tokenizer data carried inside the GGUF (tokenizer.ggml.*). Empty when the
  // file has none, in which case the caller still needs --tokenizer.
  struct TokenizerData {
    std::string model;  // "llama" (SPM), "gpt2" (BPE), ...
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<std::int32_t> token_types;
    std::vector<std::string> merges;
    int bos_id = -1;
    int eos_id = -1;
    int unk_id = -1;
    bool add_bos = true;
    [[nodiscard]] bool empty() const {
      return tokens.empty();
    }
  };
  [[nodiscard]] const TokenizerData& tokenizer() const {
    return tokenizer_;
  }

  // The raw metadata, for diagnostics and for fields the config does not model.
  [[nodiscard]] std::string metadata_string(const std::string& key) const;

  // Every tensor as the file declares it (ggml name, ggml type id, dims), for
  // diagnostics. Reading a container you did not write means being able to see
  // what it actually says rather than what it was expected to say.
  struct RawTensor {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> dims;
    std::size_t elements = 0;
  };
  [[nodiscard]] std::vector<RawTensor> raw_tensors() const;

  // The tensor's bytes exactly as stored (still quantized), for code that
  // consumes the packed form rather than the fp16 expansion. Null if unknown.
  [[nodiscard]] const std::byte* raw_tensor_bytes(const std::string& gguf_name) const;

  // Non-empty when the file's architecture has no trustworthy CPI mapping. The
  // file still reads (inspection, dequant checks); it is running it that is
  // refused, at the engine boundary. See build_config for the reasoning.
  [[nodiscard]] const std::string& unsupported_reason() const {
    return unsupported_reason_;
  }

private:
  // ggml tensor types this reader understands. The numbering is ggml's.
  enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // "k-quants": 256-element super-blocks carrying quantized per-sub-block
    // scales. Not optional in practice -- even a file named Q4_0 ships k-quant
    // tensors for its embedding and output matrices.
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
  };

  struct TensorInfo {
    std::string gguf_name;
    std::vector<std::uint64_t> dims;  // ggml order: dims[0] is the fastest axis
    GgmlType type = GgmlType::F16;
    std::uint64_t offset = 0;  // from the start of the tensor data region
    std::size_t elements = 0;
    std::size_t fp16_bytes = 0;
  };

  void parse(const std::string& path);
  void build_config();
  void map_names();
  // Materializes `info` as fp16 into the cache and returns it.
  const std::byte* materialize(const std::string& cpi_name, const TensorInfo& info) const;

  platform::MMapFile mmap_;
  const std::byte* data_base_ = nullptr;  // start of the tensor data region
  std::size_t data_bytes_ = 0;
  LlamaConfig config_{};
  TokenizerData tokenizer_;
  std::string architecture_;
  std::string unsupported_reason_;

  // Metadata values, kept as strings (numbers formatted) for a uniform lookup.
  std::unordered_map<std::string, std::string> meta_;
  // GGUF-name -> info, and the CPI-name -> GGUF-name mapping built from it.
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::unordered_map<std::string, std::string> cpi_to_gguf_;
  // Tensors needing the RoPE un-permute (LLaMA-family Q and K).
  std::unordered_map<std::string, int> permute_heads_;

  // Dequantized (or un-permuted) tensors, owned. mutable because materializing
  // is a caching detail of a const read.
  mutable std::unordered_map<std::string, std::vector<std::byte>> cache_;
};

}  // namespace model
