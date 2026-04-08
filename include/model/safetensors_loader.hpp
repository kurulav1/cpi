#pragma once

// Minimal multi-shard safetensors weight loader.
// Supports read-only access to tensors stored across multiple .safetensors
// shard files (e.g. Llama4's 50-shard layout).  All shard files are
// memory-mapped so that the OS page cache can evict pages automatically
// when RAM pressure is high.

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

  // Returns the byte size of tensor `name`. Throws if not present.
  [[nodiscard]] std::size_t tensor_bytes(const std::string& name) const;

  // Returns true if `name` is present in the index.
  [[nodiscard]] bool has_tensor(const std::string& name) const;

 private:
  struct TensorMeta {
    int shard_idx;
    std::size_t data_start;  // byte offset within shard binary payload
    std::size_t data_end;    // exclusive end (data_end - data_start = size)
  };

  std::vector<platform::MMapFile> shards_;
  std::vector<std::size_t> shard_data_offsets_;  // 8 + header_size per shard
  std::unordered_map<std::string, TensorMeta> tensors_;
};

}  // namespace model
