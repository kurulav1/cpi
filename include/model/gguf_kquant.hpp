#pragma once

// ggml k-quant super-block layouts, shared between the host reader and the GPU
// kernels that will consume these blocks directly.
//
// Why this header exists: the GGUF loader dequantizes k-quants on the CPU into
// fp16, which costs the model's fp16 size in RAM and throws away the reason the
// file was quantized. The path to keeping them packed on the GPU starts with the
// same unpacking arithmetic running in a kernel, and the only way to trust that
// kernel is to check it against the host implementation that is already verified
// against an fp16 oracle. So the host functions are declared here rather than
// hidden in the loader's translation unit, and the device kernels are gated
// against them (see kquant_dequant_test).
//
// The layouts are ggml's block structs; they are a wire format, so the byte
// sizes and field order below are the specification, not a choice.
#include <cstddef>
#include <cstdint>

namespace model {
namespace kquant {

// ggml's QK_K: every k-quant type packs this many weights per super-block.
constexpr std::size_t kSuperBlock = 256;

// Bytes per super-block, per type.
constexpr std::size_t kQ2KBlockBytes = 84;   // scales[16] + qs[64] + d,dmin
constexpr std::size_t kQ3KBlockBytes = 110;  // hmask[32] + qs[64] + scales[12] + d
constexpr std::size_t kQ4KBlockBytes = 144;  // d,dmin + scales[12] + qs[128]
constexpr std::size_t kQ5KBlockBytes = 176;  // d,dmin + scales[12] + qh[32] + qs[128]
constexpr std::size_t kQ6KBlockBytes = 210;  // ql[128] + qh[64] + scales[16] + d

// Host dequantization: `blocks` super-blocks at `p` become blocks * kSuperBlock
// fp16 values at `out`. These are the reference implementations.
void dequant_q2_k(const std::uint8_t* p, std::size_t blocks, std::uint16_t* out);
void dequant_q3_k(const std::uint8_t* p, std::size_t blocks, std::uint16_t* out);
void dequant_q4_k(const std::uint8_t* p, std::size_t blocks, std::uint16_t* out);
void dequant_q5_k(const std::uint8_t* p, std::size_t blocks, std::uint16_t* out);
void dequant_q6_k(const std::uint8_t* p, std::size_t blocks, std::uint16_t* out);

}  // namespace kquant
}  // namespace model
