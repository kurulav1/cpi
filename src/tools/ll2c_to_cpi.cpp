// Repack a .ll2c container into a .cpi (safetensors layout + JSON __metadata__).
//
//   ll2c_to_cpi <in.ll2c> <out.cpi>
//
// Written in C++ rather than Python ON PURPOSE: model::WeightLoader already parses every header
// version (v1..v7) and yields a LlamaConfig. A Python repacker would be a second place the header
// layout is written down, and this codebase's recurring defect is exactly that; a constant or a
// field list living in two files and drifting (the v7 vision-geometry whitelist, the GEMM tile
// constants, the five copies of f32->f16).
//
// tensor NAMES are PRESERVED exactly. This is a container change, not a naming change: the same
// names go in and out, so no name map is involved and the op plan is untouched. That is what
// makes the gate meaningful; .cpi and .ll2c must generate identical tokens, and if they do, the
// only thing that changed is how the bytes were stored.
//
// Shapes are written flat ([num_fp16_elements]). A .ll2c stores no per-tensor shape at all
// shapes are derived from the config; and the consumer only needs a pointer and a byte count.
// Inventing 2-D shapes here would mean re-deriving them from the config, i.e. a third place the
// model's geometry is written down, for no gain.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "model/config_json.hpp"
#include "model/weight_loader.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: ll2c_to_cpi <in.ll2c> <out.cpi>\n");
    return 2;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  model::WeightLoader wl;
  try {
    wl.open(in_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "cannot open %s: %s\n", in_path.c_str(), e.what());
    return 1;
  }

  const model::LlamaConfig& cfg = wl.config();
  const std::vector<std::string> names = wl.tensor_names();
  std::printf("[repack] %s: %zu tensors, vocab=%d hidden=%d layers=%d\n", in_path.c_str(),
              names.size(), cfg.vocab_size, cfg.hidden_size, cfg.num_layers);

  // Build the safetensors JSON header: {"name": {"dtype","shape","data_offsets"}, ...,
  // "__metadata__": {...}}. Offsets are relative to the start of the data region.
  std::string hdr = "{";
  std::size_t offset = 0;
  for (const std::string& n : names) {
    const std::size_t bytes = wl.tensor_bytes(n);
    hdr += "\"";
    hdr += n;
    hdr += "\":{\"dtype\":\"F16\",\"shape\":[";
    hdr += std::to_string(bytes / 2);  // fp16 elements
    hdr += "],\"data_offsets\":[";
    hdr += std::to_string(offset);
    hdr += ",";
    hdr += std::to_string(offset + bytes);
    hdr += "]},";
    offset += bytes;
  }
  const std::string meta = model::config_to_json(cfg);
  hdr += "\"__metadata__\":";
  hdr += meta;
  hdr += "}";

  // safetensors requires the data region to start 8-byte aligned after the header; pad the
  // header with spaces (legal JSON whitespace) rather than shifting offsets.
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  const std::uint64_t hdr_len = hdr.size();
  out.write(reinterpret_cast<const char*>(&hdr_len), sizeof(hdr_len));
  out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));

  std::size_t written = 0;
  for (const std::string& n : names) {
    const std::byte* p = wl.tensor_data(n);
    const std::size_t bytes = wl.tensor_bytes(n);
    out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(bytes));
    written += bytes;
  }
  out.close();
  if (!out) {
    std::fprintf(stderr, "write failed (disk full?)\n");
    return 1;
  }

  std::printf("[repack] wrote %s: header %zu B + data %.2f GB\n", out_path.c_str(), hdr.size(),
              static_cast<double>(written) / 1e9);
  return 0;
}
