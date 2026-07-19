#include "model/weight_loader.hpp"

#include <cstring>
#include <fstream>

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
  float rope_theta;             // RoPE base frequency (0 = use family default).
  float norm_eps;               // RMSNorm epsilon.
  std::int32_t sliding_window;  // Sliding window attention size (0 = disabled).
  std::int32_t flags;  // Bit 0: tie_word_embeddings. Bit 1: has_qkv_bias. Bit 2: use_layernorm.
  std::int32_t model_family_id;  // ModelFamily enum value.
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
  float rope_theta;
  float norm_eps;
  std::int32_t sliding_window;
  std::int32_t flags;
  std::int32_t model_family_id;
  std::int32_t num_local_experts;
  std::int32_t num_experts_per_tok;
  std::int32_t expert_intermediate_size;
};

// HeaderV5 extends HeaderV4 with richer architecture metadata used by
// mixed-attention families such as Qwen3.5.
struct HeaderV5 {
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
  float rope_theta;
  float norm_eps;
  std::int32_t sliding_window;
  std::int32_t flags;
  std::int32_t model_family_id;
  std::int32_t num_local_experts;
  std::int32_t num_experts_per_tok;
  std::int32_t expert_intermediate_size;
  float partial_rotary_factor;
  std::int32_t linear_num_key_heads;
  std::int32_t linear_num_value_heads;
  std::int32_t attention_type_count;
  std::int64_t attention_type_offset;
};

// HeaderV6 appends the delta-net dimensions to V5. It is a STRICT APPEND: every field before
// them keeps its offset, so a v5 reader parses a v6 file correctly (it simply stops early) and a
// v6 reader parses a v5 file by leaving the new fields at 0. That is the only reason this can be
// a format bump rather than a migration.
struct HeaderV6 {
  HeaderV5 v5;
  std::int32_t linear_key_head_dim;
  std::int32_t linear_value_head_dim;
  std::int32_t linear_conv_kernel_dim;
};

struct TensorEntry {
  char name[64];
  std::int64_t offset;
  std::int64_t bytes;
};
#pragma pack(pop)

// The packer (tools/pack_ll2c.py) states these layouts a second time, as a struct format string,
// and nothing but arithmetic keeps the two in step. Pin the sizes: a field added on one side and
// not the other shifts every field after it, which does not fail to build or to load -- it reads
// plausible garbage out of the wrong offsets. pack_ll2c.py asserts the same two numbers.
static_assert(sizeof(HeaderV5) == 112, "HeaderV5 layout drifted from pack_ll2c.py's HEADER_FMT");
static_assert(sizeof(HeaderV6) == 124, "HeaderV6 layout drifted from pack_ll2c.py's HEADER_FMT");

constexpr const char kMagic[] = "LL2CUDA";

}  // namespace

// Deliberately reads through HeaderV5 rather than hand-computing model_family_id's byte offset:
// the offset lives in one place, the struct, and stays correct if the format grows again.
// model_family_id has been at a fixed offset since v4, so a v4 file reads correctly too.
ModelFamily peek_container_family(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return ModelFamily::Unknown;
  HeaderV5 h{};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || std::memcmp(h.magic, kMagic, sizeof(kMagic) - 1) != 0 || h.version < 4) {
    return ModelFamily::Unknown;
  }
  return static_cast<ModelFamily>(h.model_family_id);
}

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
  config_.layer_attention_kinds.clear();
  config_.partial_rotary_factor = 1.0f;
  config_.linear_num_key_heads = 0;
  config_.linear_num_value_heads = 0;
  config_.linear_key_head_dim = 0;
  config_.linear_value_head_dim = 0;
  config_.linear_conv_kernel_dim = 0;

  if (version >= 5) {
    if (mmap_.size() < sizeof(HeaderV5)) {
      LLAMA_ENGINE_THROW("invalid v5 model header");
    }
    const auto* hdr = reinterpret_cast<const HeaderV5*>(mmap_.data());
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
    config_.has_qk_norm = (hdr->flags & 8) != 0;
    config_.mlp_gelu = (hdr->flags & 16) != 0;
    config_.scale_embeddings = (hdr->flags & 32) != 0;
    config_.attn_output_gate = (hdr->flags & 64) != 0;
    config_.model_family = static_cast<ModelFamily>(hdr->model_family_id);
    config_.num_local_experts = hdr->num_local_experts;
    config_.num_experts_per_tok = hdr->num_experts_per_tok;
    config_.expert_intermediate_size = hdr->expert_intermediate_size;
    config_.partial_rotary_factor =
        hdr->partial_rotary_factor > 0.0f ? hdr->partial_rotary_factor : 1.0f;
    config_.linear_num_key_heads = hdr->linear_num_key_heads;
    config_.linear_num_value_heads = hdr->linear_num_value_heads;

    // v6 appended the delta-net dimensions. Guard on BOTH the version and the mapped size: a
    // truncated file that claims v6 would otherwise read past the mapping.
    if (version >= 6 && mmap_.size() >= sizeof(HeaderV6)) {
      const auto* h6 = reinterpret_cast<const HeaderV6*>(mmap_.data());
      config_.linear_key_head_dim = h6->linear_key_head_dim;
      config_.linear_value_head_dim = h6->linear_value_head_dim;
      config_.linear_conv_kernel_dim = h6->linear_conv_kernel_dim;
    }

    if (hdr->attention_type_count > 0) {
      const std::size_t bytes =
          static_cast<std::size_t>(hdr->attention_type_count) * sizeof(std::int32_t);
      if (hdr->attention_type_offset < 0 ||
          static_cast<std::size_t>(hdr->attention_type_offset) + bytes > mmap_.size()) {
        LLAMA_ENGINE_THROW("invalid v5 attention metadata table");
      }
      const auto* kinds =
          reinterpret_cast<const std::int32_t*>(mmap_.data() + hdr->attention_type_offset);
      config_.layer_attention_kinds.reserve(static_cast<std::size_t>(hdr->attention_type_count));
      for (int i = 0; i < hdr->attention_type_count; ++i) {
        config_.layer_attention_kinds.push_back(static_cast<AttentionKind>(kinds[i]));
      }
    }
  } else if (version >= 4) {
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
    config_.has_qk_norm = (hdr->flags & 8) != 0;
    config_.mlp_gelu = (hdr->flags & 16) != 0;
    config_.scale_embeddings = (hdr->flags & 32) != 0;
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

  if (!config_.layer_attention_kinds.empty() &&
      static_cast<int>(config_.layer_attention_kinds.size()) != config_.num_layers) {
    LLAMA_ENGINE_THROW("attention metadata count does not match num_layers");
  }

  if (config_.num_kv_heads <= 0 || config_.num_heads <= 0 ||
      config_.num_heads % config_.num_kv_heads != 0) {
    LLAMA_ENGINE_THROW("invalid attention head config in model header");
  }

  const auto* table = reinterpret_cast<const TensorEntry*>(mmap_.data() + table_offset);
  tensors_.clear();
  for (int i = 0; i < tensor_count; ++i) {
    const auto& e = table[i];
    std::string name(e.name, strnlen(e.name, sizeof(e.name)));
    tensors_[name] =
        TensorSlice{static_cast<std::size_t>(e.offset), static_cast<std::size_t>(e.bytes)};
  }
}

}  // namespace model
