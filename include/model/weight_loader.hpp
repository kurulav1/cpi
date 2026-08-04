#pragma once

// Memory-mapped weight loader for the project's compact custom binary format.
//
// The binary format layout is:
//   [Header] [TensorTable entries] [raw tensor bytes]
//
// WeightLoader opens the file via a platform memory-map so that tensor data
// can be accessed with zero-copy pointer arithmetic.  The model's LlamaConfig
// is embedded in the header and is exposed through config().

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/llama_config.hpp"
#include "platform/mmap_file.hpp"

namespace model {

// Byte range of a single named tensor within the mapped weight file.
// offset and bytes are relative to the start of the raw tensor data region.
struct TensorSlice {
  std::size_t offset = 0;  // Byte offset from the start of the tensor data region.
  std::size_t bytes = 0;   // Size of the tensor data in bytes.
};

// Reads only the model family from a .ll2c container, without mapping its weights.
//
// Engine selection has to know the family before it commits to an engine, and opening a
// multi-gigabyte container to ask one question is not an option. Returns ModelFamily::Llama for
// anything it cannot read; a missing or unreadable file is not this function's error to report,
// and the caller's normal open() path will produce a better message.
[[nodiscard]] ModelFamily peek_container_family(const std::string& path);

// Memory-mapped weight loader for a compact custom binary format.
// Format:
//   [Header][TensorTable entries][raw tensor bytes]
class WeightLoader {
public:
  // Opens and memory-maps the weight file at path, then parses its header and
  // tensor table.  Throws on I/O errors or format violations.
  void open(const std::string& path);

  // Returns a read-only pointer to the raw bytes of the tensor named name.
  // The pointer is valid for the lifetime of this WeightLoader object.
  // Throws std::out_of_range if name is not present in the weight file.
  [[nodiscard]] const std::byte* tensor_data(const std::string& name) const;

  // Returns the size in bytes of the tensor named name.
  // Throws std::out_of_range if name is not present in the weight file.
  [[nodiscard]] std::size_t tensor_bytes(const std::string& name) const;

  // Returns true if a tensor with the given name exists in the weight file.
  [[nodiscard]] bool has_tensor(const std::string& name) const;

  // Part of the loader surface SafetensorsLoader also presents, so a consumer can be templated on
  // the loader rather than written twice. A .ll2c is fp16 by construction; pack_ll2c.py converts
  // at packing time; so this is a constant here, and the answer is what lets an uploader convert
  // BF16 from a HuggingFace checkpoint without special-casing which container it came from.
  [[nodiscard]] std::string tensor_dtype(const std::string& name) const {
    return has_tensor(name) ? std::string("F16") : std::string();
  }

  // Every tensor name in the container, sorted. Needed to REPACK a .ll2c into another container
  // without a hardcoded name list; the list would be a second place the model's tensor set is
  // written down, and those drift (see the v7 vision-geometry whitelist).
  [[nodiscard]] std::vector<std::string> tensor_names() const;

  // Returns the LlamaConfig that was embedded in the weight file header.
  [[nodiscard]] const LlamaConfig& config() const {
    return config_;
  }

private:
  // Reads the header and populates config_ and tensors_ from the mapped file.
  void parse_manifest();

  platform::MMapFile mmap_;  // Memory-mapped handle to the weight file.
  LlamaConfig config_{};     // Model config decoded from the file header.
  std::unordered_map<std::string, TensorSlice> tensors_;  // Name-to-slice index for all tensors.
};

}  // namespace model
