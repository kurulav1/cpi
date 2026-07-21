#include "model/safetensors_loader.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

namespace fs = std::filesystem;

namespace model {
namespace {

struct JsonParser {
  const char* s;
  const char* end;

  JsonParser(const char* data, std::size_t len) : s(data), end(data + len) {}

  void skip_ws() {
    while (s < end && (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')) {
      ++s;
    }
  }

  void expect(char c) {
    skip_ws();
    if (s >= end || *s != c) {
      throw std::runtime_error(std::string("safetensors JSON: expected '") + c + "' got '" +
                               (s < end ? *s : '?') + "'");
    }
    ++s;
  }

  std::string parse_string() {
    skip_ws();
    if (s >= end || *s != '"') {
      throw std::runtime_error("safetensors JSON: expected '\"'");
    }
    ++s;

    std::string result;
    result.reserve(32);
    while (s < end && *s != '"') {
      if (*s == '\\') {
        ++s;
        if (s >= end) {
          throw std::runtime_error("safetensors JSON: unterminated escape");
        }
        switch (*s) {
          case '"':
            result += '"';
            break;
          case '\\':
            result += '\\';
            break;
          case '/':
            result += '/';
            break;
          case 'n':
            result += '\n';
            break;
          case 'r':
            result += '\r';
            break;
          case 't':
            result += '\t';
            break;
          case 'b':
            result += '\b';
            break;
          case 'f':
            result += '\f';
            break;
          case 'u':
            if (s + 4 >= end) {
              throw std::runtime_error("safetensors JSON: bad \\u escape");
            }
            result += '?';
            s += 4;
            break;
          default:
            result += *s;
            break;
        }
      } else {
        result += *s;
      }
      ++s;
    }

    if (s >= end || *s != '"') {
      throw std::runtime_error("safetensors JSON: unterminated string");
    }
    ++s;
    return result;
  }

  std::int64_t parse_integer() {
    skip_ws();
    if (s >= end) {
      throw std::runtime_error("safetensors JSON: expected integer, got EOF");
    }
    bool neg = false;
    if (*s == '-') {
      neg = true;
      ++s;
    }
    if (s >= end || *s < '0' || *s > '9') {
      throw std::runtime_error("safetensors JSON: expected digit");
    }
    std::int64_t value = 0;
    while (s < end && *s >= '0' && *s <= '9') {
      value = value * 10 + (*s - '0');
      ++s;
    }
    return neg ? -value : value;
  }

  void skip_value() {
    skip_ws();
    if (s >= end) {
      throw std::runtime_error("safetensors JSON: unexpected EOF in value");
    }

    if (*s == '"') {
      (void)parse_string();
      return;
    }

    if (*s == '{') {
      ++s;
      skip_ws();
      if (s < end && *s == '}') {
        ++s;
        return;
      }
      while (true) {
        (void)parse_string();
        expect(':');
        skip_value();
        skip_ws();
        if (s < end && *s == ',') {
          ++s;
          skip_ws();
          if (s < end && *s == '}') {
            break;
          }
          continue;
        }
        break;
      }
      expect('}');
      return;
    }

    if (*s == '[') {
      ++s;
      skip_ws();
      if (s < end && *s == ']') {
        ++s;
        return;
      }
      while (true) {
        skip_value();
        skip_ws();
        if (s < end && *s == ',') {
          ++s;
          skip_ws();
          if (s < end && *s == ']') {
            break;
          }
          continue;
        }
        break;
      }
      expect(']');
      return;
    }

    if (*s == 't') {
      if (s + 4 <= end && std::strncmp(s, "true", 4) == 0) {
        s += 4;
        return;
      }
      throw std::runtime_error("safetensors JSON: bad literal 'true'");
    }

    if (*s == 'f') {
      if (s + 5 <= end && std::strncmp(s, "false", 5) == 0) {
        s += 5;
        return;
      }
      throw std::runtime_error("safetensors JSON: bad literal 'false'");
    }

    if (*s == 'n') {
      if (s + 4 <= end && std::strncmp(s, "null", 4) == 0) {
        s += 4;
        return;
      }
      throw std::runtime_error("safetensors JSON: bad literal 'null'");
    }

    if (*s == '-' || (*s >= '0' && *s <= '9')) {
      if (*s == '-') {
        ++s;
      }
      while (s < end && ((*s >= '0' && *s <= '9') || *s == '.' || *s == 'e' || *s == 'E' ||
                         *s == '+' || *s == '-')) {
        ++s;
      }
      return;
    }

    throw std::runtime_error(std::string("safetensors JSON: unexpected char '") + *s + "'");
  }

  bool parse_tensor_object(std::int64_t& offset_start, std::int64_t& offset_end,
                           std::string& dtype) {
    expect('{');
    offset_start = -1;
    offset_end = -1;
    dtype.clear();
    skip_ws();
    if (s < end && *s == '}') {
      ++s;
      return false;
    }

    while (true) {
      const std::string key = parse_string();
      expect(':');
      if (key == "dtype") {
        // Captured because the value decides whether a consumer must CONVERT. Everything CPI
        // packs itself is F16, but a HuggingFace checkpoint is usually BF16 -- same width,
        // different exponent split, so reading one as the other is silent garbage rather than
        // an error.
        dtype = parse_string();
      } else if (key == "data_offsets") {
        expect('[');
        offset_start = parse_integer();
        expect(',');
        offset_end = parse_integer();
        skip_ws();
        if (s < end && *s == ',') {
          ++s;
        }
        expect(']');
      } else {
        skip_value();
      }
      skip_ws();
      if (s < end && *s == ',') {
        ++s;
        skip_ws();
        if (s < end && *s == '}') {
          break;
        }
        continue;
      }
      break;
    }
    expect('}');
    return offset_start >= 0 && offset_end >= 0;
  }

  std::string metadata_raw;

  template <typename Callback>
  void parse_header(Callback&& cb) {
    expect('{');
    skip_ws();
    if (s < end && *s == '}') {
      ++s;
      return;
    }

    while (true) {
      const std::string name = parse_string();
      expect(':');
      if (name == "__metadata__") {
        // CAPTURE it rather than discard. For a `.cpi` this block is the model's whole config --
        // the self-describing replacement for the .ll2c binary header -- so throwing it away
        // would leave the container unreadable by anything but a shape-guesser.
        skip_ws();
        const char* mstart = s;
        skip_value();
        metadata_raw.assign(mstart, static_cast<std::size_t>(s - mstart));
      } else {
        std::int64_t start = -1;
        std::int64_t end_offset = -1;
        std::string dtype;
        if (parse_tensor_object(start, end_offset, dtype)) {
          cb(name, static_cast<std::size_t>(start), static_cast<std::size_t>(end_offset), dtype);
        }
      }
      skip_ws();
      if (s < end && *s == ',') {
        ++s;
        skip_ws();
        if (s < end && *s == '}') {
          break;
        }
        continue;
      }
      break;
    }

    skip_ws();
    if (s < end && *s == '}') {
      ++s;
    }
  }
};

}  // namespace

void SafetensorsLoader::open(const std::string& model_dir) {
  shards_.clear();
  shard_data_offsets_.clear();
  tensors_.clear();

  fs::path root(model_dir);
  if (!fs::exists(root)) {
    LLAMA_ENGINE_THROW("safetensors path not found: " + model_dir);
  }

  std::vector<fs::path> shard_paths;
  bool single_file = false;
  if (fs::is_regular_file(root)) {
    if (root.extension() == ".safetensors") {
      // Long-standing behaviour: naming one shard of a sharded model means "the model beside
      // it". Preserved exactly -- PlanCudaEngine relies on it.
      root = root.parent_path();
    } else {
      // A SELF-CONTAINED container (.cpi): one file that IS the whole model, in safetensors
      // layout. Globbing its parent would be wrong twice over -- it would miss this file (the
      // extension does not match) and could pick up an unrelated model sitting next to it.
      shard_paths.push_back(root);
      single_file = true;
    }
  }

  if (!single_file) {
    if (!fs::is_directory(root)) {
      LLAMA_ENGINE_THROW("safetensors path is not a directory: " + model_dir);
    }
    for (const auto& entry : fs::directory_iterator(root)) {
      if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
        shard_paths.push_back(entry.path());
      }
    }
  }

  if (shard_paths.empty()) {
    throw std::runtime_error("safetensors: no .safetensors files found in: " + root.string());
  }

  std::sort(shard_paths.begin(), shard_paths.end(),
            [](const fs::path& left, const fs::path& right) {
              return left.filename().string() < right.filename().string();
            });

  shards_.reserve(shard_paths.size());
  shard_data_offsets_.reserve(shard_paths.size());

  for (std::size_t shard_idx = 0; shard_idx < shard_paths.size(); ++shard_idx) {
    const fs::path& path = shard_paths[shard_idx];

    platform::MMapFile mmap;
    mmap.open(path.string());

    const std::byte* base = mmap.data();
    const std::size_t file_size = mmap.size();
    if (file_size < 8) {
      throw std::runtime_error("safetensors: shard too small: " + path.string());
    }

    std::uint64_t header_size = 0;
    std::memcpy(&header_size, base, sizeof(header_size));
    if (8 + header_size > file_size) {
      throw std::runtime_error("safetensors: header_size exceeds file size in: " + path.string());
    }

    const std::size_t data_offset = 8 + static_cast<std::size_t>(header_size);
    const std::size_t payload_size = file_size - data_offset;
    shard_data_offsets_.push_back(data_offset);

    const char* json = reinterpret_cast<const char*>(base + 8);
    JsonParser parser(json, static_cast<std::size_t>(header_size));
    const int shard_index = static_cast<int>(shard_idx);
    parser.parse_header([&](const std::string& name, std::size_t start, std::size_t end_offset,
                            const std::string& dtype) {
      if (end_offset < start || end_offset > payload_size) {
        throw std::runtime_error("safetensors: invalid data_offsets for tensor " + name + " in " +
                                 path.string());
      }
      if (tensors_.find(name) != tensors_.end()) {
        return;
      }
      tensors_.emplace(name, TensorMeta{shard_index, start, end_offset, dtype});
    });
    // A `.cpi` carries its config here. Sharded HF models normally do not, so an empty block is
    // not an error -- only the .cpi path requires one, and it says so itself.
    if (!parser.metadata_raw.empty()) metadata_json_ = parser.metadata_raw;

    shards_.push_back(std::move(mmap));
  }
}

const std::byte* SafetensorsLoader::tensor_ptr(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("safetensors: tensor not found: " + name);
  }
  const TensorMeta& meta = it->second;
  const std::size_t file_offset =
      shard_data_offsets_[static_cast<std::size_t>(meta.shard_idx)] + meta.data_start;
  return shards_[static_cast<std::size_t>(meta.shard_idx)].data() + file_offset;
}

std::size_t SafetensorsLoader::tensor_bytes(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("safetensors: tensor not found: " + name);
  }
  const TensorMeta& meta = it->second;
  return meta.data_end - meta.data_start;
}

std::string SafetensorsLoader::tensor_dtype(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? std::string() : it->second.dtype;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

}  // namespace model
