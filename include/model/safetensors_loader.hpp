#pragma once

// Minimal multi-shard safetensors weight loader.
// Supports read-only access to tensors stored across multiple .safetensors
// shard files (e.g. Llama4's 50-shard layout).  All shard files are
// memory-mapped so that the OS page cache can evict pages automatically
// when RAM pressure is high.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "platform/mmap_file.hpp"

namespace model {

class SafetensorsLoader {
public:
  // Opens and memory-maps all .safetensors files in model_dir (sorted by
  // filename), parses their JSON headers, and builds a tensor name→location
  // index. Throws on I/O or parse errors.
  void open(const std::string& model_dir);

  // Returns a read-only pointer to the start of tensor `name`'s raw bytes.
  // Valid for the lifetime of this object. Throws if name is not present.
  [[nodiscard]] const std::byte* tensor_ptr(const std::string& name) const;

  // Alias for tensor_ptr, so this loader and model::WeightLoader present the SAME three-method
  // surface (has_tensor / tensor_bytes / tensor_data). That lets a consumer be templated on the
  // loader instead of duplicated per container format -- which is the point of the .cpi work.
  [[nodiscard]] const std::byte* tensor_data(const std::string& name) const {
    return tensor_ptr(name);
  }

  // Returns the byte size of tensor `name`. Throws if not present.
  [[nodiscard]] std::size_t tensor_bytes(const std::string& name) const;

  // Returns true if `name` is present in the index.
  [[nodiscard]] bool has_tensor(const std::string& name) const;

  // The tensor's dtype exactly as the container declares it ("F16", "BF16", "F32"). Empty if the
  // tensor is absent. Consumers that memcpy the bytes to a device MUST check this: CPI's own
  // containers are F16, but a HuggingFace checkpoint is typically BF16, and the two are the same
  // WIDTH with a different exponent split -- so reinterpreting one as the other produces
  // plausibly-scaled garbage instead of an error. model::WeightLoader answers "F16" to the same
  // question, which is what lets a consumer be templated on the loader.
  [[nodiscard]] std::string tensor_dtype(const std::string& name) const;

  // Every tensor in the container, so a consumer can walk the model without knowing its naming
  // scheme. model::WeightLoader answers the same question. Sorted, not hash order: callers that
  // rank tensors must resolve ties the same way on every load.
  [[nodiscard]] std::vector<std::string> tensor_names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
  }

  // The container's `__metadata__` block, verbatim JSON. A `.cpi` stores its whole LlamaConfig
  // here -- the self-describing replacement for the .ll2c binary header. Empty for a plain HF
  // checkpoint, which keeps its config in a separate config.json.
  [[nodiscard]] bool has_metadata() const { return !metadata_json_.empty(); }
  [[nodiscard]] const std::string& metadata_json() const { return metadata_json_; }

private:
  struct TensorMeta {
    int shard_idx;
    std::size_t data_start;  // byte offset within shard binary payload
    std::size_t data_end;    // exclusive end (data_end - data_start = size)
    std::string dtype;       // as written in the header: "F16", "BF16", "F32", ...
  };

  std::vector<platform::MMapFile> shards_;
  std::vector<std::size_t> shard_data_offsets_;  // 8 + header_size per shard
  std::unordered_map<std::string, TensorMeta> tensors_;
  std::string metadata_json_;
};

}  // namespace model
