#pragma once

// Minimal JSON reader + bf16 helpers, shared by the model loaders.
//
// Extracted from the Qwen3.5 engine so the plan executor can parse the same
// safetensors-style config.json and convert the same bf16 weights without
// duplicating ~200 lines. Hand-rolled (no third-party JSON dep), per CPI policy.

#include <cuda_fp16.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/safetensors_loader.hpp"

namespace engine {
namespace mini {

inline float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &word, sizeof(out));
  return out;
}

inline std::uint16_t float_to_half_bits(float value) {
  const __half h = __float2half(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

inline std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open config file: " + path.string());
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline void skip_ws(const std::string& json, std::size_t& pos) {
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
}

inline std::size_t json_find_key(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t pos = 0;
  while ((pos = json.find(needle, pos)) != std::string::npos) {
    std::size_t value_pos = pos + needle.size();
    skip_ws(json, value_pos);
    if (value_pos < json.size() && json[value_pos] == ':') {
      return value_pos + 1;
    }
    pos += needle.size();
  }
  return std::string::npos;
}

inline std::string json_read_string(const std::string& json, std::size_t& pos) {
  if (pos >= json.size() || json[pos] != '"') {
    return "";
  }
  ++pos;
  std::string result;
  while (pos < json.size()) {
    const char c = json[pos++];
    if (c == '"') break;
    if (c == '\\' && pos < json.size()) {
      const char escaped = json[pos++];
      switch (escaped) {
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '/':
          result.push_back('/');
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        default:
          result.push_back(escaped);
          break;
      }
      continue;
    }
    result.push_back(c);
  }
  return result;
}

inline std::string json_extract_object(const std::string& json, const std::string& key) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return "";
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '{') return "";
  const std::size_t start = pos;
  int depth = 0;
  bool in_string = false;
  for (; pos < json.size(); ++pos) {
    const char c = json[pos];
    if (in_string) {
      if (c == '\\') {
        ++pos;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') {
      ++depth;
      continue;
    }
    if (c == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(start, pos - start + 1);
      }
    }
  }
  return "";
}

inline std::string json_get_string(const std::string& json, const std::string& key,
                            const std::string& def = "") {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '"') return def;
  return json_read_string(json, pos);
}

inline int json_get_int(const std::string& json, const std::string& key, int def = 0) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) ||
                               json[end] == '-' || json[end] == '+')) {
    ++end;
  }
  if (end == pos) return def;
  try {
    return std::stoi(json.substr(pos, end - pos));
  } catch (...) {
    return def;
  }
}

inline float json_get_float(const std::string& json, const std::string& key, float def = 0.0f) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-' ||
          json[end] == '+' || json[end] == '.' || json[end] == 'e' || json[end] == 'E')) {
    ++end;
  }
  if (end == pos) return def;
  try {
    return std::stof(json.substr(pos, end - pos));
  } catch (...) {
    return def;
  }
}

inline std::vector<std::string> json_get_string_array(const std::string& json, const std::string& key) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return {};
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '[') return {};
  ++pos;
  std::vector<std::string> out;
  while (pos < json.size()) {
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] == ']') break;
    if (json[pos] == '"') {
      out.push_back(json_read_string(json, pos));
    } else {
      ++pos;
    }
    while (pos < json.size() && json[pos] != '"' && json[pos] != ']') ++pos;
  }
  return out;
}

inline const std::uint16_t* require_bf16_tensor(const model::SafetensorsLoader& loader,
                                         const std::string& name) {
  if (!loader.has_tensor(name)) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return reinterpret_cast<const std::uint16_t*>(loader.tensor_ptr(name));
}

inline const float* require_f32_tensor(const model::SafetensorsLoader& loader, const std::string& name) {
  if (!loader.has_tensor(name)) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return reinterpret_cast<const float*>(loader.tensor_ptr(name));
}

}  // namespace mini
}  // namespace engine
