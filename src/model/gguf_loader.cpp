// GGUF reader. See gguf_loader.hpp for scope and the two caveats.
#include "model/gguf_loader.hpp"

#include "model/gguf_kquant.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "model/json_mini.hpp"

#if CPI_HAS_CUDA
#  include <cuda_fp16.h>
#  include <cuda_runtime.h>

#  include "runtime/kernels.cuh"
#endif

namespace model {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;  // "GGUF" little-endian

// GGUF metadata value types (the container's own enum).
enum class MetaType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

// A bounds-checked cursor over the mapped file. Every read is checked: a
// truncated or hostile GGUF should produce a clear error, not a read past the
// mapping.
class Cursor {
public:
  Cursor(const std::byte* base, std::size_t size) : base_(base), size_(size) {}

  void require(std::size_t n) const {
    if (pos_ + n > size_) throw std::runtime_error("gguf: truncated file (read past end)");
  }

  template <typename T>
  T read() {
    require(sizeof(T));
    T v{};
    std::memcpy(&v, base_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }

  std::string read_string() {
    const std::uint64_t len = read<std::uint64_t>();
    if (len > (1u << 30)) throw std::runtime_error("gguf: implausible string length");
    require(static_cast<std::size_t>(len));
    std::string s(reinterpret_cast<const char*>(base_ + pos_), static_cast<std::size_t>(len));
    pos_ += static_cast<std::size_t>(len);
    return s;
  }

  [[nodiscard]] std::size_t pos() const {
    return pos_;
  }
  void align_to(std::size_t alignment) {
    if (alignment == 0) return;
    const std::size_t rem = pos_ % alignment;
    if (rem != 0) pos_ += alignment - rem;
  }

private:
  const std::byte* base_ = nullptr;
  std::size_t size_ = 0;
  std::size_t pos_ = 0;
};

std::string format_float(double v) {
  std::ostringstream os;
  os.precision(9);
  os << v;
  return os.str();
}

// Reads one metadata value, returning it as a string. Arrays of strings are
// returned joined by \x1f (a separator that cannot appear in a token), which is
// how the tokenizer vocabulary comes through without a second value model.
std::string read_meta_value(Cursor& cur, MetaType type, std::vector<std::string>* array_out,
                            std::vector<float>* float_array_out,
                            std::vector<std::int32_t>* int_array_out) {
  switch (type) {
    case MetaType::UInt8:
      return std::to_string(static_cast<unsigned>(cur.read<std::uint8_t>()));
    case MetaType::Int8:
      return std::to_string(static_cast<int>(cur.read<std::int8_t>()));
    case MetaType::UInt16:
      return std::to_string(cur.read<std::uint16_t>());
    case MetaType::Int16:
      return std::to_string(cur.read<std::int16_t>());
    case MetaType::UInt32:
      return std::to_string(cur.read<std::uint32_t>());
    case MetaType::Int32:
      return std::to_string(cur.read<std::int32_t>());
    case MetaType::Float32:
      return format_float(cur.read<float>());
    case MetaType::Bool:
      return cur.read<std::uint8_t>() ? "true" : "false";
    case MetaType::String:
      return cur.read_string();
    case MetaType::UInt64:
      return std::to_string(cur.read<std::uint64_t>());
    case MetaType::Int64:
      return std::to_string(cur.read<std::int64_t>());
    case MetaType::Float64:
      return format_float(cur.read<double>());
    case MetaType::Array: {
      const auto elem_type = static_cast<MetaType>(cur.read<std::uint32_t>());
      const std::uint64_t count = cur.read<std::uint64_t>();
      for (std::uint64_t i = 0; i < count; ++i) {
        std::string v = read_meta_value(cur, elem_type, nullptr, nullptr, nullptr);
        if (elem_type == MetaType::String && array_out) {
          array_out->push_back(std::move(v));
        } else if (elem_type == MetaType::Float32 && float_array_out) {
          float_array_out->push_back(std::stof(v));
        } else if (int_array_out &&
                   (elem_type == MetaType::Int32 || elem_type == MetaType::UInt32)) {
          int_array_out->push_back(static_cast<std::int32_t>(std::stol(v)));
        }
      }
      return "[array:" + std::to_string(count) + "]";
    }
  }
  throw std::runtime_error("gguf: unknown metadata value type");
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t man = h & 0x3FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (man != 0) {
      // Subnormal: normalize it.
      int e = -1;
      std::uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    } else {
      bits = sign;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::uint16_t float_to_half(float f) {
  return engine::mini::float_to_half_bits(f);
}

}  // namespace

namespace kquant {

// ---- k-quant super-blocks -------------------------------------------------
//
// The k-quants pack 256 weights per super-block and quantize the per-sub-block
// scales themselves, which is why they beat the flat Q4_0/Q8_0 formats at the
// same bit width. The layouts below follow ggml's block structs exactly; they
// are a wire format, so the field order and the bit packing are the spec, and
// deviating anywhere produces plausible-looking weights that are simply wrong.

// Q4_K/Q5_K scales: 8 sub-blocks x (6-bit scale, 6-bit min) packed into 12 bytes.
void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<std::uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

void dequant_q4_k(const std::uint8_t* base, std::size_t blocks, std::uint16_t* out) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
    const std::uint8_t* p = base + static_cast<std::size_t>(b) * 144;
    std::uint16_t dh = 0;
    std::uint16_t mh = 0;
    std::memcpy(&dh, p, 2);
    std::memcpy(&mh, p + 2, 2);
    const float d = half_to_float(dh);
    const float dmin = half_to_float(mh);
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* q = p + 4 + 12;
    std::uint16_t* y = out + static_cast<std::size_t>(b) * kSuperBlock;
    int is = 0;
    for (std::size_t j = 0; j < kSuperBlock; j += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (int l = 0; l < 32; ++l) *y++ = float_to_half(d1 * (q[l] & 0x0F) - m1);
      for (int l = 0; l < 32; ++l) *y++ = float_to_half(d2 * (q[l] >> 4) - m2);
      q += 32;
      is += 2;
    }
  }
}

void dequant_q5_k(const std::uint8_t* base, std::size_t blocks, std::uint16_t* out) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
    const std::uint8_t* p = base + static_cast<std::size_t>(b) * 176;
    std::uint16_t dh = 0;
    std::uint16_t mh = 0;
    std::memcpy(&dh, p, 2);
    std::memcpy(&mh, p + 2, 2);
    const float d = half_to_float(dh);
    const float dmin = half_to_float(mh);
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 4 + 12;
    const std::uint8_t* ql = p + 4 + 12 + 32;
    std::uint16_t* y = out + static_cast<std::size_t>(b) * kSuperBlock;
    int is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (std::size_t j = 0; j < kSuperBlock; j += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (int l = 0; l < 32; ++l) {
        const float v = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        *y++ = float_to_half(v);
      }
      for (int l = 0; l < 32; ++l) {
        const float v = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        *y++ = float_to_half(v);
      }
      ql += 32;
      is += 2;
      u1 = static_cast<std::uint8_t>(u1 << 2);
      u2 = static_cast<std::uint8_t>(u2 << 2);
    }
  }
}

void dequant_q6_k(const std::uint8_t* base, std::size_t blocks, std::uint16_t* out) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
    const std::uint8_t* p = base + static_cast<std::size_t>(b) * 210;
    const std::uint8_t* ql = p;
    const std::uint8_t* qh = p + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(p + 128 + 64);
    std::uint16_t dh = 0;
    std::memcpy(&dh, p + 128 + 64 + 16, 2);
    const float d = half_to_float(dh);
    std::uint16_t* y = out + static_cast<std::size_t>(b) * kSuperBlock;
    for (std::size_t n = 0; n < kSuperBlock; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = static_cast<int>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = float_to_half(d * sc[is + 0] * static_cast<float>(q1));
        y[l + 32] = float_to_half(d * sc[is + 2] * static_cast<float>(q2));
        y[l + 64] = float_to_half(d * sc[is + 4] * static_cast<float>(q3));
        y[l + 96] = float_to_half(d * sc[is + 6] * static_cast<float>(q4));
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void dequant_q2_k(const std::uint8_t* base, std::size_t blocks, std::uint16_t* out) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
    const std::uint8_t* p = base + static_cast<std::size_t>(b) * 84;
    const std::uint8_t* scales = p;
    const std::uint8_t* q = p + 16;
    std::uint16_t dh = 0;
    std::uint16_t mh = 0;
    std::memcpy(&dh, p + 16 + 64, 2);
    std::memcpy(&mh, p + 16 + 64 + 2, 2);
    const float d = half_to_float(dh);
    const float dmin = half_to_float(mh);
    std::uint16_t* y = out + static_cast<std::size_t>(b) * kSuperBlock;
    int is = 0;
    for (std::size_t n = 0; n < kSuperBlock; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        std::uint8_t sc = scales[is++];
        float dl = d * (sc & 0x0F);
        float ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = float_to_half(dl * static_cast<float>((q[l] >> shift) & 3) - ml);
        }
        sc = scales[is++];
        dl = d * (sc & 0x0F);
        ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = float_to_half(dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml);
        }
        shift += 2;
      }
      q += 32;
    }
  }
}

void dequant_q3_k(const std::uint8_t* base, std::size_t blocks, std::uint16_t* out) {
  constexpr std::uint32_t kmask1 = 0x03030303u;
  constexpr std::uint32_t kmask2 = 0x0f0f0f0fu;
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
    const std::uint8_t* p = base + static_cast<std::size_t>(b) * 110;
    const std::uint8_t* hm = p;
    const std::uint8_t* q = p + 32;
    std::uint16_t dh = 0;
    std::memcpy(&dh, p + 32 + 64 + 12, 2);
    const float d_all = half_to_float(dh);

    // The 6-bit scales are stored across 12 bytes in a packed layout; ggml
    // rebuilds them into 16 signed values through this shuffle.
    std::uint32_t aux[4];
    std::memcpy(aux, p + 32 + 64, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales = reinterpret_cast<const std::int8_t*>(aux);

    std::uint16_t* y = out + static_cast<std::size_t>(b) * kSuperBlock;
    std::uint8_t m = 1;
    int is = 0;
    for (std::size_t n = 0; n < kSuperBlock; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int base = static_cast<int>((q[l] >> shift) & 3);
          *y++ = float_to_half(dl * static_cast<float>(base - ((hm[l] & m) ? 0 : 4)));
        }
        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int base = static_cast<int>((q[l + 16] >> shift) & 3);
          *y++ = float_to_half(dl * static_cast<float>(base - ((hm[l + 16] & m) ? 0 : 4)));
        }
        shift += 2;
        m = static_cast<std::uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

}  // namespace kquant

bool is_gguf_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  return f.gcount() == static_cast<std::streamsize>(sizeof(magic)) && magic == kGgufMagic;
}

GgufLoader::~GgufLoader() {
#if CPI_HAS_CUDA
  if (device_packed_ != nullptr) cudaFree(device_packed_);
#endif
  device_packed_ = nullptr;
  device_packed_bytes_ = 0;
}

void GgufLoader::open(const std::string& path) {
  parse(path);
  build_config();
  map_names();

  // A quantized file that silently costs fp16 residency is a surprise worth
  // spending one line on: the download is small, the resident model is not,
  // and the flag that fixes it also makes decode faster (fewer weight bytes
  // per token). Only printed when the file actually carries quantized weights.
  bool quantized = false;
  for (const auto& [cpi_name, gguf_name] : cpi_to_gguf_) {
    (void)cpi_name;
    const GgmlType t = tensors_.at(gguf_name).type;
    if (t != GgmlType::F32 && t != GgmlType::F16 && t != GgmlType::BF16) {
      quantized = true;
      break;
    }
  }
  if (quantized) {
    std::fprintf(stderr,
                 "[gguf] quantized weights are dequantized to fp16 at load, so this model is "
                 "resident at its fp16 size. Pass --weight-quant int4 (or int8) to keep it packed "
                 "on the GPU: less VRAM, and faster decode.\n");
  }
}

void GgufLoader::parse(const std::string& path) {
  mmap_.open(path);
  // Same reasoning as the .ll2c loader: pull the file into the page cache up
  // front so the upload does not stall on faults tensor by tensor.
  mmap_.prefetch();
  const auto* base = static_cast<const std::byte*>(mmap_.data());
  const std::size_t size = mmap_.size();
  Cursor cur(base, size);

  if (cur.read<std::uint32_t>() != kGgufMagic) {
    throw std::runtime_error("gguf: bad magic (not a GGUF file): " + path);
  }
  const std::uint32_t version = cur.read<std::uint32_t>();
  if (version < 2 || version > 3) {
    throw std::runtime_error("gguf: unsupported version " + std::to_string(version) +
                             " (this reader handles v2 and v3)");
  }
  const std::uint64_t tensor_count = cur.read<std::uint64_t>();
  const std::uint64_t meta_count = cur.read<std::uint64_t>();
  if (tensor_count > (1u << 20) || meta_count > (1u << 20)) {
    throw std::runtime_error("gguf: implausible tensor/metadata count");
  }

  for (std::uint64_t i = 0; i < meta_count; ++i) {
    const std::string key = cur.read_string();
    const auto type = static_cast<MetaType>(cur.read<std::uint32_t>());
    if (key == "tokenizer.ggml.tokens") {
      read_meta_value(cur, type, &tokenizer_.tokens, nullptr, nullptr);
      meta_[key] = "[tokens]";
    } else if (key == "tokenizer.ggml.scores") {
      read_meta_value(cur, type, nullptr, &tokenizer_.scores, nullptr);
      meta_[key] = "[scores]";
    } else if (key == "tokenizer.ggml.token_type") {
      read_meta_value(cur, type, nullptr, nullptr, &tokenizer_.token_types);
      meta_[key] = "[types]";
    } else if (key == "tokenizer.ggml.merges") {
      read_meta_value(cur, type, &tokenizer_.merges, nullptr, nullptr);
      meta_[key] = "[merges]";
    } else {
      meta_[key] = read_meta_value(cur, type, nullptr, nullptr, nullptr);
    }
  }

  std::size_t alignment = 32;
  if (const auto it = meta_.find("general.alignment"); it != meta_.end()) {
    alignment = static_cast<std::size_t>(std::stoul(it->second));
  }

  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo info;
    info.gguf_name = cur.read_string();
    const std::uint32_t n_dims = cur.read<std::uint32_t>();
    if (n_dims == 0 || n_dims > 4) throw std::runtime_error("gguf: bad tensor rank");
    info.elements = 1;
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      const std::uint64_t dim = cur.read<std::uint64_t>();
      info.dims.push_back(dim);
      info.elements *= static_cast<std::size_t>(dim);
    }
    info.type = static_cast<GgmlType>(cur.read<std::uint32_t>());
    info.offset = cur.read<std::uint64_t>();
    info.fp16_bytes = info.elements * sizeof(std::uint16_t);
    // Deliberately no type check here. A container may carry tensors this
    // engine never reads (per-layer scales, an architecture's extra
    // projections), and refusing to open the whole file because one of them
    // uses a type we cannot dequantize would reject files that work fine. The
    // check happens where it matters: when a tensor is actually requested.
    tensors_[info.gguf_name] = std::move(info);
  }

  cur.align_to(alignment);
  if (cur.pos() > size) throw std::runtime_error("gguf: tensor data starts past end of file");
  data_base_ = base + cur.pos();
  data_bytes_ = size - cur.pos();
}

std::vector<GgufLoader::RawTensor> GgufLoader::raw_tensors() const {
  std::vector<RawTensor> out;
  out.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_) {
    RawTensor r;
    r.name = name;
    r.type = static_cast<std::uint32_t>(info.type);
    r.dims = info.dims;
    r.elements = info.elements;
    out.push_back(std::move(r));
  }
  std::sort(out.begin(), out.end(),
            [](const RawTensor& a, const RawTensor& b) { return a.name < b.name; });
  return out;
}

bool GgufLoader::fill_device_fp16(const std::string& cpi_name, void* dst, void* stream) const {
#if CPI_HAS_CUDA
  if (dst == nullptr) return false;
  // Kill switch, and the A/B lever the device path was verified with: the host
  // route must produce the same weights, so both must be runnable.
  static const bool disabled = []() {
    const char* env = std::getenv("CPI_GGUF_DEVICE_DEQUANT");
    return env != nullptr && env[0] == '0';
  }();
  if (disabled) return false;
  const auto it = cpi_to_gguf_.find(cpi_name);
  if (it == cpi_to_gguf_.end()) return false;
  const TensorInfo& info = tensors_.at(it->second);
  // Only the k-quants have a device kernel today. fp16 is already served
  // zero-copy from the mapping, and the flat quants are cheap enough on the
  // host that they are not worth a second path until measured.
  kernels::KQuantType kt;
  switch (info.type) {
    case GgmlType::Q4_K:
      kt = kernels::KQuantType::Q4_K;
      break;
    case GgmlType::Q5_K:
      kt = kernels::KQuantType::Q5_K;
      break;
    case GgmlType::Q6_K:
      kt = kernels::KQuantType::Q6_K;
      break;
    default:
      return false;
  }
  if (info.elements % kquant::kSuperBlock != 0) return false;
  if (permute_heads_.find(cpi_name) != permute_heads_.end()) {
    // Q/K need the RoPE un-permute, which lives on the host path for now.
    return false;
  }
  const std::size_t blocks = info.elements / kquant::kSuperBlock;
  std::size_t block_bytes = 0;
  switch (info.type) {
    case GgmlType::Q4_K:
      block_bytes = kquant::kQ4KBlockBytes;
      break;
    case GgmlType::Q5_K:
      block_bytes = kquant::kQ5KBlockBytes;
      break;
    default:
      block_bytes = kquant::kQ6KBlockBytes;
      break;
  }
  const std::size_t packed_bytes = blocks * block_bytes;
  if (info.offset + packed_bytes > data_bytes_) return false;

  auto* cuda_stream = static_cast<cudaStream_t>(stream);
  // A staging buffer sized to the largest tensor seen so far, reused across
  // layers: the packed bytes are a fraction of the fp16 they expand to, so this
  // is small next to what the old path allocated on the host.
  if (packed_bytes > device_packed_bytes_) {
    if (device_packed_ != nullptr) {
      cudaStreamSynchronize(cuda_stream);
      cudaFree(device_packed_);
      device_packed_ = nullptr;
      device_packed_bytes_ = 0;
    }
    if (cudaMalloc(&device_packed_, packed_bytes) != cudaSuccess) {
      device_packed_ = nullptr;
      return false;  // fall back to the host path rather than fail the load
    }
    device_packed_bytes_ = packed_bytes;
  }
  if (cudaMemcpyAsync(device_packed_, data_base_ + info.offset, packed_bytes,
                      cudaMemcpyHostToDevice, cuda_stream) != cudaSuccess) {
    return false;
  }
  kernels::launch_dequant_kquant(static_cast<const std::uint8_t*>(device_packed_), kt, blocks,
                                 static_cast<__half*>(dst), cuda_stream);
  return true;
#else
  (void)cpi_name;
  (void)dst;
  (void)stream;
  return false;
#endif
}

GgufLoader::PackedTensor GgufLoader::packed_kquant(const std::string& cpi_name) const {
  PackedTensor out;
  const auto it = cpi_to_gguf_.find(cpi_name);
  if (it == cpi_to_gguf_.end()) return out;
  const TensorInfo& info = tensors_.at(it->second);
  std::size_t block_bytes = 0;
  switch (info.type) {
    case GgmlType::Q4_K:
      out.kind = 0;
      block_bytes = kquant::kQ4KBlockBytes;
      break;
    case GgmlType::Q5_K:
      out.kind = 1;
      block_bytes = kquant::kQ5KBlockBytes;
      break;
    case GgmlType::Q6_K:
      out.kind = 2;
      block_bytes = kquant::kQ6KBlockBytes;
      break;
    default:
      return out;  // not a k-quant: no packed route
  }
  if (info.elements % kquant::kSuperBlock != 0) {
    out.kind = -1;
    return out;
  }
  // Q and K carry the converter's RoPE row interleave. It is reported rather
  // than applied: the un-permute reorders whole rows, so a packed consumer can
  // undo it by gathering row-sized byte runs.
  const auto perm = permute_heads_.find(cpi_name);
  if (perm != permute_heads_.end()) {
    out.permute_heads = perm->second;
  }
  out.bytes = (info.elements / kquant::kSuperBlock) * block_bytes;
  if (info.offset + out.bytes > data_bytes_) {
    out.kind = -1;
    return out;
  }
  out.data = data_base_ + info.offset;
  // ggml stores dims fastest-axis-first: dims[0] is the input width.
  out.cols = info.dims.size() > 0 ? static_cast<int>(info.dims[0]) : 0;
  out.rows = info.dims.size() > 1 ? static_cast<int>(info.dims[1]) : 1;
  return out;
}

const std::byte* GgufLoader::raw_tensor_bytes(const std::string& gguf_name) const {
  const auto it = tensors_.find(gguf_name);
  if (it == tensors_.end() || data_base_ == nullptr) return nullptr;
  if (it->second.offset > data_bytes_) return nullptr;
  return data_base_ + it->second.offset;
}

std::string GgufLoader::metadata_string(const std::string& key) const {
  const auto it = meta_.find(key);
  return it == meta_.end() ? std::string() : it->second;
}

void GgufLoader::build_config() {
  architecture_ = metadata_string("general.architecture");
  if (architecture_.empty()) throw std::runtime_error("gguf: missing general.architecture");
  const auto num = [&](const std::string& suffix, long fallback) -> long {
    const std::string v = metadata_string(architecture_ + "." + suffix);
    if (v.empty()) return fallback;
    try {
      return std::stol(v);
    } catch (const std::exception&) {
      return fallback;
    }
  };
  const auto fnum = [&](const std::string& suffix, float fallback) -> float {
    const std::string v = metadata_string(architecture_ + "." + suffix);
    if (v.empty()) return fallback;
    try {
      return std::stof(v);
    } catch (const std::exception&) {
      return fallback;
    }
  };

  config_.num_layers = static_cast<std::int32_t>(num("block_count", 32));
  config_.hidden_size = static_cast<std::int32_t>(num("embedding_length", 4096));
  config_.intermediate_size = static_cast<std::int32_t>(num("feed_forward_length", 11008));
  config_.num_heads = static_cast<std::int32_t>(num("attention.head_count", 32));
  config_.num_kv_heads =
      static_cast<std::int32_t>(num("attention.head_count_kv", config_.num_heads));
  config_.max_seq_len = static_cast<std::int32_t>(num("context_length", 4096));
  config_.rope_theta = fnum("rope.freq_base", 10000.0f);
  config_.norm_eps = fnum("attention.layer_norm_rms_epsilon", 1e-5f);
  config_.dtype = "fp16";

  // Vocabulary size comes from the embedding tensor when the metadata omits it
  // (some converters only write tokenizer.ggml.tokens).
  if (const auto it = tensors_.find("token_embd.weight"); it != tensors_.end()) {
    if (it->second.dims.size() >= 2) {
      config_.vocab_size = static_cast<std::int32_t>(it->second.dims[1]);
    }
  }
  if (!tokenizer_.tokens.empty()) {
    config_.vocab_size = static_cast<std::int32_t>(tokenizer_.tokens.size());
  }

  // Architecture mapping, and the refusal that matters more than the mapping.
  //
  // An architecture CPI has not been taught still has blk.N.attn_q-style tensors,
  // so a wrong mapping does not fail to load: it loads and answers nonsense.
  // Anything not known good is therefore refused by name. The cost of being wrong
  // in the other direction (a supported model rejected) is an error message; the
  // cost here is a user believing an answer.
  if (architecture_ == "llama") {
    // Verified: bit-exact against a .ll2c of the same checkpoint, and generation
    // is token-identical. Q/K are un-permuted (see map_names).
    config_.model_family = ModelFamily::LLaMA3;
  } else if (architecture_ == "qwen2") {
    // Same layout as llama minus the Q/K permutation (llama.cpp gives Qwen2 the
    // NeoX rope convention, which is already CPI's), plus QKV biases.
    // Implemented from the format, not yet verified against a Qwen2 GGUF.
    config_.model_family = ModelFamily::Qwen2;
    config_.has_qkv_bias = true;
  } else {
    // Recorded rather than thrown: reading an unmapped file is still useful
    // (inspecting it, checking a dequant), and the diagnostics depend on it.
    // The refusal happens at the engine boundary, in WeightLoader::open.
    config_.model_family = ModelFamily::Unknown;
    unsupported_reason_ =
        "gguf: architecture '" + architecture_ +
        "' is not mapped yet. Its tensors would load under the wrong conventions and produce "
        "confident nonsense rather than an error, so it is refused instead. Supported today: "
        "llama (verified), qwen2. Gemma/Gemma 4 and DeepSeek-V2 need their config and norm "
        "conventions mapped (Gemma's (1+w) norms, Gemma 4's per-layer embeddings, DeepSeek's MLA "
        "projections) before their GGUFs can be trusted; convert those to .ll2c/.cpi meanwhile.";
  }

  // Tied embeddings: no separate output tensor means the head shares the table.
  config_.tie_word_embeddings = tensors_.find("output.weight") == tensors_.end();

  tokenizer_.model = metadata_string("tokenizer.ggml.model");
  const auto id_of = [&](const char* key, int fallback) {
    const std::string v = metadata_string(key);
    if (v.empty()) return fallback;
    try {
      return static_cast<int>(std::stol(v));
    } catch (const std::exception&) {
      return fallback;
    }
  };
  tokenizer_.bos_id = id_of("tokenizer.ggml.bos_token_id", -1);
  tokenizer_.eos_id = id_of("tokenizer.ggml.eos_token_id", -1);
  tokenizer_.unk_id = id_of("tokenizer.ggml.unknown_token_id", -1);
  tokenizer_.add_bos = metadata_string("tokenizer.ggml.add_bos_token") != "false";
}

void GgufLoader::map_names() {
  // GGUF (ggml) names -> CPI names. Same tensors, different spelling.
  const auto add = [&](const std::string& cpi, const std::string& gguf) {
    if (tensors_.count(gguf)) cpi_to_gguf_[cpi] = gguf;
  };
  add("tok_embeddings.weight", "token_embd.weight");
  add("norm.weight", "output_norm.weight");
  add("output.weight", "output.weight");

  const int head_dim = config_.num_heads > 0 ? config_.hidden_size / config_.num_heads : 128;
  const bool permute_qk = architecture_ == "llama";
  for (int l = 0; l < config_.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string b = "blk." + std::to_string(l) + ".";
    add(p + "attention_norm.weight", b + "attn_norm.weight");
    add(p + "attention.wq", b + "attn_q.weight");
    add(p + "attention.wk", b + "attn_k.weight");
    add(p + "attention.wv", b + "attn_v.weight");
    add(p + "attention.wo", b + "attn_output.weight");
    add(p + "ffn_norm.weight", b + "ffn_norm.weight");
    add(p + "feed_forward.w1", b + "ffn_gate.weight");
    add(p + "feed_forward.w2", b + "ffn_down.weight");
    add(p + "feed_forward.w3", b + "ffn_up.weight");
    // Qwen2-style biases.
    add(p + "attention.bq", b + "attn_q.bias");
    add(p + "attention.bk", b + "attn_k.bias");
    add(p + "attention.bv", b + "attn_v.bias");

    // The RoPE convention difference. ggml rotates adjacent pairs (x[2i],
    // x[2i+1]); CPI and HuggingFace rotate split halves (x[i], x[i+d/2]).
    // convert_hf_to_gguf.py permutes Q and K for LLaMA so ggml's rope
    // reproduces HF's result, so reading those tensors back for a split-half
    // rope requires undoing it. Only the two rotated projections are affected.
    if (permute_qk) {
      permute_heads_[p + "attention.wq"] = config_.num_heads;
      permute_heads_[p + "attention.wk"] = config_.num_kv_heads;
    }
  }
  (void)head_dim;
}

bool GgufLoader::has_tensor(const std::string& name) const {
  return cpi_to_gguf_.count(name) != 0;
}

std::string GgufLoader::tensor_dtype(const std::string& name) const {
  return has_tensor(name) ? std::string("F16") : std::string();
}

std::vector<std::string> GgufLoader::tensor_names() const {
  std::vector<std::string> out;
  out.reserve(cpi_to_gguf_.size());
  for (const auto& [cpi, gguf] : cpi_to_gguf_) out.push_back(cpi);
  std::sort(out.begin(), out.end());
  return out;
}

std::size_t GgufLoader::tensor_bytes(const std::string& name) const {
  const auto it = cpi_to_gguf_.find(name);
  if (it == cpi_to_gguf_.end()) throw std::out_of_range("gguf: no tensor named " + name);
  return tensors_.at(it->second).fp16_bytes;
}

const std::byte* GgufLoader::tensor_data(const std::string& name) const {
  const auto it = cpi_to_gguf_.find(name);
  if (it == cpi_to_gguf_.end()) throw std::out_of_range("gguf: no tensor named " + name);
  const TensorInfo& info = tensors_.at(it->second);
  // An fp16 tensor that needs no un-permutation is already in the layout CPI
  // wants, so hand out the mapped bytes instead of copying them. That is nearly
  // every tensor of an F16 file: copying them all cost ~15 s of extra startup on
  // an 8B and a second full-size copy in RAM, for nothing.
  if (info.type == GgmlType::F16 && permute_heads_.find(name) == permute_heads_.end()) {
    if (info.offset + info.fp16_bytes > data_bytes_) {
      throw std::runtime_error("gguf: tensor '" + name + "' extends past the data region");
    }
    return data_base_ + info.offset;
  }
  const auto cached = cache_.find(name);
  if (cached != cache_.end()) return cached->second.data();
  return materialize(name, info);
}

const std::byte* GgufLoader::materialize(const std::string& cpi_name,
                                         const TensorInfo& info) const {
  if (info.offset > data_bytes_) throw std::runtime_error("gguf: tensor offset past end of data");
  const std::byte* src = data_base_ + info.offset;
  const std::size_t n = info.elements;

  // Access-time type gate (see parse): say precisely what is unsupported, and
  // for the k-quants also that the tensor is not a whole number of 256-element
  // super-blocks, which would otherwise read past the block table.
  switch (info.type) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::Q8_0:
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
      break;
    case GgmlType::Q2_K:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
      if (n % kquant::kSuperBlock != 0) {
        throw std::runtime_error("gguf: tensor '" + info.gguf_name +
                                 "' uses a k-quant type but its element count " +
                                 std::to_string(n) + " is not a multiple of 256");
      }
      break;
    default:
      throw std::runtime_error(
          "gguf: tensor '" + info.gguf_name + "' uses ggml type " +
          std::to_string(static_cast<std::uint32_t>(info.type)) +
          ", which this reader does not dequantize (supported: F32/F16/BF16, "
          "Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the k-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K)");
  }

  std::vector<std::uint16_t> half(n);
  switch (info.type) {
    case GgmlType::F16: {
      std::memcpy(half.data(), src, n * sizeof(std::uint16_t));
      break;
    }
    case GgmlType::BF16: {
      const auto* bf = reinterpret_cast<const std::uint16_t*>(src);
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        half[i] = float_to_half(engine::mini::bf16_to_float(bf[i]));
      }
      break;
    }
    case GgmlType::F32: {
      const auto* f = reinterpret_cast<const float*>(src);
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        half[i] = float_to_half(f[i]);
      }
      break;
    }
    case GgmlType::Q8_0: {
      // block: fp16 scale, then 32 int8 quants. value = d * q
      constexpr std::size_t kBlock = 32;
      const std::size_t blocks = n / kBlock;
      const auto* base = reinterpret_cast<const std::uint8_t*>(src);
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
        const std::uint8_t* p = base + static_cast<std::size_t>(b) * (2 + kBlock);
        std::uint16_t dh = 0;
        std::memcpy(&dh, p, sizeof(dh));
        const float d = half_to_float(dh);
        const auto* q = reinterpret_cast<const std::int8_t*>(p + 2);
        for (std::size_t i = 0; i < kBlock; ++i) {
          half[static_cast<std::size_t>(b) * kBlock + i] = float_to_half(d * static_cast<float>(q[i]));
        }
      }
      break;
    }
    case GgmlType::Q4_0: {
      // block: fp16 scale, then 16 bytes of packed nibbles. value = d * (q - 8)
      // Nibble order is ggml's: low nibbles hold the first half of the block.
      constexpr std::size_t kBlock = 32;
      const std::size_t blocks = n / kBlock;
      const auto* base = reinterpret_cast<const std::uint8_t*>(src);
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
        const std::uint8_t* p = base + static_cast<std::size_t>(b) * (2 + kBlock / 2);
        const std::size_t o = static_cast<std::size_t>(b) * kBlock;
        std::uint16_t dh = 0;
        std::memcpy(&dh, p, sizeof(dh));
        const float d = half_to_float(dh);
        const std::uint8_t* q = p + 2;
        for (std::size_t i = 0; i < kBlock / 2; ++i) {
          const int lo = (q[i] & 0x0F) - 8;
          const int hi = (q[i] >> 4) - 8;
          half[o + i] = float_to_half(d * static_cast<float>(lo));
          half[o + kBlock / 2 + i] = float_to_half(d * static_cast<float>(hi));
        }
      }
      break;
    }
    case GgmlType::Q4_1: {
      constexpr std::size_t kBlock = 32;
      const std::size_t blocks = n / kBlock;
      const auto* p = reinterpret_cast<const std::uint8_t*>(src);
      for (std::size_t b = 0; b < blocks; ++b) {
        std::uint16_t dh = 0;
        std::uint16_t mh = 0;
        std::memcpy(&dh, p, sizeof(dh));
        std::memcpy(&mh, p + 2, sizeof(mh));
        const float d = half_to_float(dh);
        const float m = half_to_float(mh);
        const std::uint8_t* q = p + 4;
        for (std::size_t i = 0; i < kBlock / 2; ++i) {
          half[b * kBlock + i] = float_to_half(d * static_cast<float>(q[i] & 0x0F) + m);
          half[b * kBlock + kBlock / 2 + i] = float_to_half(d * static_cast<float>(q[i] >> 4) + m);
        }
        p += 4 + kBlock / 2;
      }
      break;
    }
    case GgmlType::Q5_0:
    case GgmlType::Q5_1: {
      const bool q5_1 = info.type == GgmlType::Q5_1;
      constexpr std::size_t kBlock = 32;
      const std::size_t blocks = n / kBlock;
      const auto* p = reinterpret_cast<const std::uint8_t*>(src);
      for (std::size_t b = 0; b < blocks; ++b) {
        std::uint16_t dh = 0;
        std::memcpy(&dh, p, sizeof(dh));
        const float d = half_to_float(dh);
        float m = 0.0f;
        std::size_t head = 2;
        if (q5_1) {
          std::uint16_t mh = 0;
          std::memcpy(&mh, p + 2, sizeof(mh));
          m = half_to_float(mh);
          head = 4;
        }
        std::uint32_t qh = 0;
        std::memcpy(&qh, p + head, sizeof(qh));
        const std::uint8_t* q = p + head + 4;
        for (std::size_t i = 0; i < kBlock / 2; ++i) {
          const int xh0 = static_cast<int>((qh >> i) & 1u) << 4;
          const int xh1 = static_cast<int>((qh >> (i + 16)) & 1u) << 4;
          const int v0 = static_cast<int>(q[i] & 0x0F) | xh0;
          const int v1 = static_cast<int>(q[i] >> 4) | xh1;
          if (q5_1) {
            half[b * kBlock + i] = float_to_half(d * static_cast<float>(v0) + m);
            half[b * kBlock + kBlock / 2 + i] = float_to_half(d * static_cast<float>(v1) + m);
          } else {
            half[b * kBlock + i] = float_to_half(d * static_cast<float>(v0 - 16));
            half[b * kBlock + kBlock / 2 + i] = float_to_half(d * static_cast<float>(v1 - 16));
          }
        }
        p += head + 4 + kBlock / 2;
      }
      break;
    }
    case GgmlType::Q4_K:
      kquant::dequant_q4_k(reinterpret_cast<const std::uint8_t*>(src), n / kquant::kSuperBlock, half.data());
      break;
    case GgmlType::Q5_K:
      kquant::dequant_q5_k(reinterpret_cast<const std::uint8_t*>(src), n / kquant::kSuperBlock, half.data());
      break;
    case GgmlType::Q6_K:
      kquant::dequant_q6_k(reinterpret_cast<const std::uint8_t*>(src), n / kquant::kSuperBlock, half.data());
      break;
    case GgmlType::Q2_K:
      kquant::dequant_q2_k(reinterpret_cast<const std::uint8_t*>(src), n / kquant::kSuperBlock, half.data());
      break;
    case GgmlType::Q3_K:
      kquant::dequant_q3_k(reinterpret_cast<const std::uint8_t*>(src), n / kquant::kSuperBlock, half.data());
      break;
    default:
      throw std::runtime_error("gguf: unsupported tensor type at materialize time");
  }

  // Undo the converter's Q/K permutation for split-half RoPE.
  const auto perm = permute_heads_.find(cpi_name);
  if (perm != permute_heads_.end() && info.dims.size() >= 2) {
    const std::size_t in_features = static_cast<std::size_t>(info.dims[0]);
    const std::size_t out_features = static_cast<std::size_t>(info.dims[1]);
    const std::size_t heads = static_cast<std::size_t>(perm->second);
    if (heads > 0 && out_features % heads == 0) {
      const std::size_t hd = out_features / heads;  // rows per head
      const std::size_t hh = hd / 2;
      std::vector<std::uint16_t> out(half.size());
      for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t r = 0; r < hd; ++r) {
          // Converter did: row r of the head came from interleaving the halves.
          // Inverse: even rows are the first half, odd rows the second.
          const std::size_t src_row = (r < hh) ? (r * 2) : ((r - hh) * 2 + 1);
          const std::uint16_t* srow = half.data() + ((h * hd) + src_row) * in_features;
          std::uint16_t* drow = out.data() + ((h * hd) + r) * in_features;
          std::memcpy(drow, srow, in_features * sizeof(std::uint16_t));
        }
      }
      half.swap(out);
    }
  }

  std::vector<std::byte> bytes(half.size() * sizeof(std::uint16_t));
  std::memcpy(bytes.data(), half.data(), bytes.size());
  auto [it, _] = cache_.emplace(cpi_name, std::move(bytes));
  return it->second.data();
}

}  // namespace model
