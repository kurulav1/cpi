#include "model/weight_loader.hpp"

#include <cstring>

#include "common.hpp"

namespace model {
namespace {

#pragma pack(push, 1)
struct HeaderV1 {
  char magic[8];
  std::int32_t version;
  std::int32_t vocab_size;
  std::int32_t hidden_size;
  std::int32_t intermediate_size;
  std::int32_t num_layers;
  std::int32_t num_heads;
  std::int32_t max_seq_len;
  std::int32_t tensor_parallel;
  std::int32_t tensor_count;
  std::int64_t table_offset;
};

struct HeaderV2 {
  char magic[8];
  std::int32_t version;
  std::int32_t vocab_size;
  std::int32_t hidden_size;
  std::int32_t intermediate_size;
  std::int32_t num_layers;
  std::int32_t num_heads;
  std::int32_t num_kv_heads;
  std::int32_t max_seq_len;
  std::int32_t tensor_parallel;
  std::int32_t tensor_count;
  std::int64_t table_offset;
};

// HeaderV3 adds architecture metadata needed for multi-family model support:
// rope_theta, norm_eps, sliding_window, a flags bitfield, and a model_family_id.
// Flags bit layout: bit 0 = tie_word_embeddings, bit 1 = has_qkv_bias,
// bit 2 = use_layernorm.

struct HeaderV3 {
  char magic[8];
  std::int32_t version;
  std::int32_t vocab_size;
  std::int32_t hidden_size;
  std::int32_t intermediate_size;
  std::int32_t num_layers;
  std::int32_t num_heads;
  std::int32_t num_kv_heads;
  std::int32_t max_seq_len;
  std::int32_t tensor_parallel;
  std::int32_t tensor_count;
  std::int64_t table_offset;
  float        rope_theta;      // RoPE base frequency (0 = use family default).
  float        norm_eps;        // RMSNorm epsilon.
  std::int32_t sliding_window;  // Sliding window attention size (0 = disabled).
  std::int32_t flags;           // Bit 0: tie_word_embeddings. Bit 1: has_qkv_bias. Bit 2: use_layernorm.
  std::int32_t model_family_id; // ModelFamily enum value.
};

// HeaderV4 extends HeaderV3 with sparse-MoE metadata.
// num_local_experts == 0 indicates dense FFN (legacy behavior).
struct HeaderV4 {
  char magic[8];
  std::int32_t version;
  std::int32_t vocab_size;
  std::int32_t hidden_size;
  std::int32_t intermediate_size;
  std::int32_t num_layers;
  std::int32_t num_heads;
  std::int32_t num_kv_heads;
  std::int32_t max_seq_len;
  std::int32_t tensor_parallel;
  std::int32_t tensor_count;
  std::int64_t table_offset;
  float        rope_theta;
  float        norm_eps;
  std::int32_t sliding_window;
  std::int32_t flags;
  std::int32_t model_family_id;
  std::int32_t num_local_experts;
  std::int32_t num_experts_per_tok;
  std::int32_t expert_intermediate_size;
};

struct TensorEntry {
  char name[64];
  std::int64_t offset;
  std::int64_t bytes;
};
#pragma pack(pop)

constexpr const char kMagic[] = "LL2CUDA";

}  // namespace

void WeightLoader::open(const std::string& path) {
  mmap_.open(path);
  // Hint the OS to prefetch the file into the page cache immediately after mapping,
  // so subsequent tensor accesses don't stall on page faults during GPU loading.
  mmap_.prefetch();
  parse_manifest();
}

const std::byte* WeightLoader::tensor_data(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    LLAMA_ENGINE_THROW("tensor not found: " + name);
  }
  return mmap_.data() + it->second.offset;
}

std::size_t WeightLoader::tensor_bytes(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    LLAMA_ENGINE_THROW("tensor not found: " + name);
  }
  return it->second.bytes;
}

bool WeightLoader::has_tensor(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

void WeightLoader::parse_manifest() {
  if (!mmap_.valid() || mmap_.size() < sizeof(HeaderV1)) {
    LLAMA_ENGINE_THROW("invalid model file");
  }

  const auto* magic = reinterpret_cast<const char*>(mmap_.data());
  if (std::memcmp(magic, kMagic, sizeof(kMagic) - 1) != 0) {
    LLAMA_ENGINE_THROW("unsupported weights format. expected LL2CUDA manifest");
  }

  const auto version = *reinterpret_cast<const std::int32_t*>(mmap_.data() + 8);
  std::int32_t tensor_count = 0;
  std::int64_t table_offset = 0;

  if (version >= 4) {
    if (mmap_.size() < sizeof(HeaderV4)) {
      LLAMA_ENGINE_THROW("invalid v4 model header");
    }
    const auto* hdr = reinterpret_cast<const HeaderV4*>(mmap_.data());
    config_.vocab_size = hdr->vocab_size;
    config_.hidden_size = hdr->hidden_size;
    config_.intermediate_size = hdr->intermediate_size;
    config_.num_layers = hdr->num_layers;
    config_.num_heads = hdr->num_heads;
    config_.num_kv_heads = hdr->num_kv_heads;
    config_.max_seq_len = hdr->max_seq_len;
    config_.tensor_parallel = hdr->tensor_parallel;
    tensor_count = hdr->tensor_count;
    table_offset = hdr->table_offset;
    config_.rope_theta = hdr->rope_theta;
    config_.norm_eps = hdr->norm_eps > 0.0f ? hdr->norm_eps : 1e-5f;
    config_.sliding_window = hdr->sliding_window;
    config_.tie_word_embeddings = (hdr->flags & 1) != 0;
    config_.has_qkv_bias = (hdr->flags & 2) != 0;
    config_.use_layernorm = (hdr->flags & 4) != 0;
    config_.model_family = static_cast<ModelFamily>(hdr->model_family_id);
    config_.num_local_experts = hdr->num_local_experts;
    config_.num_experts_per_tok = hdr->num_experts_per_tok;
    config_.expert_intermediate_size = hdr->expert_intermediate_size;
  } else if (version >= 3) {
    if (mmap_.size() < sizeof(HeaderV3)) {
      LLAMA_ENGINE_THROW("invalid v3 model header");
    }
    const auto* hdr = reinterpret_cast<const HeaderV3*>(mmap_.data());
    config_.vocab_size = hdr->vocab_size;
    config_.hidden_size = hdr->hidden_size;
    config_.intermediate_size = hdr->intermediate_size;
    config_.num_layers = hdr->num_layers;
    config_.num_heads = hdr->num_heads;
    config_.num_kv_heads = hdr->num_kv_heads;
    config_.max_seq_len = hdr->max_seq_len;
    config_.tensor_parallel = hdr->tensor_parallel;
    tensor_count = hdr->tensor_count;
    table_offset = hdr->table_offset;
    config_.rope_theta = hdr->rope_theta;
    config_.norm_eps = hdr->norm_eps > 0.0f ? hdr->norm_eps : 1e-5f;
    config_.sliding_window = hdr->sliding_window;
    config_.tie_word_embeddings = (hdr->flags & 1) != 0;
    config_.has_qkv_bias = (hdr->flags & 2) != 0;
    config_.use_layernorm = (hdr->flags & 4) != 0;
    config_.model_family = static_cast<ModelFamily>(hdr->model_family_id);
    config_.num_local_experts = 0;
    config_.num_experts_per_tok = 0;
    config_.expert_intermediate_size = 0;
  } else if (version >= 2) {
    if (mmap_.size() < sizeof(HeaderV2)) {
      LLAMA_ENGINE_THROW("invalid v2 model header");
    }
    const auto* hdr = reinterpret_cast<const HeaderV2*>(mmap_.data());
    config_.vocab_size = hdr->vocab_size;
    config_.hidden_size = hdr->hidden_size;
    config_.intermediate_size = hdr->intermediate_size;
    config_.num_layers = hdr->num_layers;
    config_.num_heads = hdr->num_heads;
    config_.num_kv_heads = hdr->num_kv_heads;
    config_.max_seq_len = hdr->max_seq_len;
    config_.tensor_parallel = hdr->tensor_parallel;
    tensor_count = hdr->tensor_count;
    table_offset = hdr->table_offset;
    config_.num_local_experts = 0;
    config_.num_experts_per_tok = 0;
    config_.expert_intermediate_size = 0;
    config_.use_layernorm = false;
  } else {
    const auto* hdr = reinterpret_cast<const HeaderV1*>(mmap_.data());
    config_.vocab_size = hdr->vocab_size;
    config_.hidden_size = hdr->hidden_size;
    config_.intermediate_size = hdr->intermediate_size;
    config_.num_layers = hdr->num_layers;
    config_.num_heads = hdr->num_heads;
    config_.num_kv_heads = hdr->num_heads;
    config_.max_seq_len = hdr->max_seq_len;
    config_.tensor_parallel = hdr->tensor_parallel;
    tensor_count = hdr->tensor_count;
    table_offset = hdr->table_offset;
    config_.num_local_experts = 0;
    config_.num_experts_per_tok = 0;
    config_.expert_intermediate_size = 0;
    config_.use_layernorm = false;
  }

  if (config_.num_kv_heads <= 0 || config_.num_heads <= 0 || config_.num_heads % config_.num_kv_heads != 0) {
    LLAMA_ENGINE_THROW("invalid attention head config in model header");
  }

  const auto* table = reinterpret_cast<const TensorEntry*>(mmap_.data() + table_offset);
  tensors_.clear();
  for (int i = 0; i < tensor_count; ++i) {
    const auto& e = table[i];
    std::string name(e.name, strnlen(e.name, sizeof(e.name)));
    tensors_[name] = TensorSlice{static_cast<std::size_t>(e.offset),
                                 static_cast<std::size_t>(e.bytes)};
  }
}

}  // namespace model
