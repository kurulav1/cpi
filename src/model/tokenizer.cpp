#include "model/tokenizer.hpp"

#include "model/hf_bpe_tokenizer.hpp"

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

// Implements the top-level Tokenizer facade that supports two backends:
//   1. HfBpeTokenizer  — a built-in BPE implementation that reads HuggingFace
//      tokenizer.json files directly, requiring no external tools at runtime.
//   2. SentencePiece   — the reference libsentencepiece library (optional; gated
//      by LLAMA_ENGINE_HAS_SENTENCEPIECE), or, when the library is absent, a
//      subprocess-based fallback that shells out to the spm_encode/spm_decode
//      command-line programs found either on PATH or in the vcpkg tools tree.
//
// The subprocess fallback exists so that inference can still run on machines
// where sentencepiece cannot be compiled (e.g. MSVC with limited CMake config)
// at the cost of launching an external process per encode/decode call.

namespace model {
namespace {

// Returns the platform-appropriate path for a SentencePiece command-line tool.
// On Windows the tools are bundled inside the vcpkg installed tree; on other
// platforms the binary is expected to already be on PATH so only the bare name
// is returned.
std::string default_tool_path(const char* exe_name) {
#ifdef _WIN32
  const std::filesystem::path p = std::filesystem::path("third_party") / "vcpkg" / "installed" / "x64-windows" /
                                  "tools" / "sentencepiece" / exe_name;
#else
  const std::filesystem::path p = std::filesystem::path(exe_name);
#endif
  return p.string();
}

// Reads the entire contents of a file into a string using binary mode so that
// line-ending translation does not corrupt token ids written by spm_encode.
std::string read_text_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    // The file may have been deleted between the existence check and the open,
    // or permissions may have changed — surface a clear error either way.
    throw std::runtime_error("failed reading file: " + p.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Writes text to a file in binary mode so that the subprocess receives the
// exact bytes that were passed in, without platform line-ending translation.
void write_text_file(const std::filesystem::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed writing file: " + p.string());
  }
  out << text;
}

}  // namespace

// Destructor explicitly deletes the type-erased backend pointers.  The
// void* storage avoids including heavy backend headers (sentencepiece,
// hf_bpe_tokenizer) in the public tokenizer.hpp header, keeping compile
// times and transitive dependencies minimal.
Tokenizer::~Tokenizer() {
  delete reinterpret_cast<HfBpeTokenizer*>(hf_bpe_);
  hf_bpe_ = nullptr;
#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
  delete reinterpret_cast<sentencepiece::SentencePieceProcessor*>(sp_);
  sp_ = nullptr;
#endif
}

// Loads the tokenizer from the file at `path` and selects the appropriate
// backend based on the file extension:
//   - ".json" → HfBpeTokenizer (HuggingFace tokenizer.json format)
//   - anything else → SentencePiece model (.model binary format)
//
// This function resets all prior state so it is safe to call multiple times
// on the same Tokenizer instance (e.g. to hot-reload a model).
void Tokenizer::load(const std::string& path) {
  model_path_ = path;
  tokenizer_json_path_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;
  special_ids_.clear();
  // Release any previously loaded HF-BPE backend before constructing a new one.
  delete reinterpret_cast<HfBpeTokenizer*>(hf_bpe_);
  hf_bpe_ = nullptr;
  if (std::filesystem::path(path).extension() == ".json") {
    tokenizer_json_path_ = path;
    hf_bpe_ = new HfBpeTokenizer();
    auto* tok = reinterpret_cast<HfBpeTokenizer*>(hf_bpe_);
    tok->load(path);
    bos_id_ = tok->bos_id();
    eos_id_ = tok->eos_id();
    unk_id_ = tok->unk_id();
    special_ids_ = tok->special_ids();
    return;
  }

  // Locate the spm_encode/spm_decode helper executables.  The caller may
  // override the search directory via the LLAMA_SPM_TOOL_DIR environment
  // variable, which is useful in CI or custom install layouts.  The Windows
  // path uses _dupenv_s (the safe CRT variant) to avoid deprecation warnings.
  std::string tool_dir;
#ifdef _WIN32
  char* env_buf = nullptr;
  std::size_t env_len = 0;
  if (_dupenv_s(&env_buf, &env_len, "LLAMA_SPM_TOOL_DIR") == 0 && env_buf != nullptr) {
    tool_dir.assign(env_buf);
  }
  if (env_buf) {
    free(env_buf);
  }
#else
  if (const char* dir = std::getenv("LLAMA_SPM_TOOL_DIR")) {
    tool_dir.assign(dir);
  }
#endif

  if (!tool_dir.empty()) {
    const char* dir = tool_dir.c_str();
    spm_encode_exe_ = (std::filesystem::path(dir) / "spm_encode.exe").string();
    spm_decode_exe_ = (std::filesystem::path(dir) / "spm_decode.exe").string();
  } else {
    spm_encode_exe_ = default_tool_path("spm_encode.exe");
    spm_decode_exe_ = default_tool_path("spm_decode.exe");
  }

#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
  if (!sp_) {
    sp_ = new sentencepiece::SentencePieceProcessor();
  }
  auto* proc = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(sp_);
  const auto st = proc->Load(path);
  if (!st.ok()) {
    // SentencePiece returns a Status object; convert to string for the exception
    // message so callers see a human-readable description of the load failure.
    throw std::runtime_error("failed loading sentencepiece model: " + std::string(st.ToString()));
  }
  bos_id_ = proc->bos_id();
  eos_id_ = proc->eos_id();
  unk_id_ = proc->unk_id();
  if (bos_id_ >= 0) special_ids_.push_back(bos_id_);
  if (eos_id_ >= 0) special_ids_.push_back(eos_id_);
  if (unk_id_ >= 0) special_ids_.push_back(unk_id_);
#endif

  // If no backend populated the special token ids (e.g. sentencepiece was not
  // compiled in and the subprocess path is being used), fall back to the
  // conventional LLaMA token id assignments so generation stop logic works.
  if (special_ids_.empty()) {
    bos_id_ = 1;
    eos_id_ = 2;
    unk_id_ = 0;
    special_ids_ = {0, 1, 2};
  }
}

// Encodes `text` into a sequence of vocabulary ids.  Dispatches to the
// appropriate backend depending on which was loaded:
//   - HfBpeTokenizer for JSON-format tokenizer files
//   - In-process SentencePiece library when compiled in
//   - Subprocess fallback (spm_encode) as a last resort
//
// The `add_bos` flag prepends the beginning-of-sequence token, which most
// inference workflows require for the first (and only the first) segment.
std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
  if (!tokenizer_json_path_.empty()) {
    auto* tok = reinterpret_cast<const HfBpeTokenizer*>(hf_bpe_);
    if (!tok) {
      throw std::runtime_error("hf bpe tokenizer is not loaded");
    }
    return tok->encode(text, add_bos);
  }

#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
  if (sp_) {
    auto* proc = reinterpret_cast<const sentencepiece::SentencePieceProcessor*>(sp_);
    std::vector<int> ids;
    const auto st = proc->Encode(text, &ids);
    if (!st.ok()) {
      throw std::runtime_error("tokenizer encode failed: " + std::string(st.ToString()));
    }
    if (add_bos) {
      ids.insert(ids.begin(), proc->bos_id());
    }
    return ids;
  }
#endif

  // Subprocess fallback: write input text to a temp file, invoke spm_encode,
  // then read back the whitespace-separated integer ids.  Temp files are used
  // rather than stdin/stdout to avoid cross-platform pipe complexity.
  if (model_path_.empty()) {
    throw std::runtime_error("tokenizer is not loaded");
  }

  if (!std::filesystem::exists(spm_encode_exe_)) {
    throw std::runtime_error("spm_encode executable not found: " + spm_encode_exe_);
  }

  const auto tmp_dir = std::filesystem::temp_directory_path();
  const auto in_path = tmp_dir / "llama_spm_in.txt";
  const auto out_path = tmp_dir / "llama_spm_out.txt";

  write_text_file(in_path, text);
  const std::string cmd = spm_encode_exe_ + " --model=" + model_path_ +
                          " --output_format=id --input=" + in_path.string() +
                          " --output=" + out_path.string();
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    // A non-zero exit code means spm_encode could not tokenize the input;
    // include the full command string to aid diagnosis.
    throw std::runtime_error("spm_encode command failed: " + cmd);
  }

  std::vector<int> ids;
  std::stringstream ss(read_text_file(out_path));
  int id = 0;
  while (ss >> id) {
    ids.push_back(id);
  }

  if (add_bos) {
    // LLaMA-1 convention: BOS is always token id 1.  This is only reached when
    // no backend supplied the id at load time.
    ids.insert(ids.begin(), 1);
  }

  return ids;
}

// Decodes a sequence of vocabulary ids back to a UTF-8 string.  Special tokens
// (BOS, EOS, UNK, and any model-specific extras) are stripped before decoding
// because they carry no surface-form text and would otherwise appear as
// artefacts in the output.
std::string Tokenizer::decode(const std::vector<int>& ids) const {
  const std::vector<int> filtered = strip_special_ids(ids);

  if (!tokenizer_json_path_.empty()) {
    auto* tok = reinterpret_cast<const HfBpeTokenizer*>(hf_bpe_);
    if (!tok) {
      throw std::runtime_error("hf bpe tokenizer is not loaded");
    }
    return tok->decode(filtered);
  }

#ifdef LLAMA_ENGINE_HAS_SENTENCEPIECE
  if (sp_) {
    auto* proc = reinterpret_cast<const sentencepiece::SentencePieceProcessor*>(sp_);
    std::string out;
    const auto st = proc->Decode(filtered, &out);
    if (!st.ok()) {
      throw std::runtime_error("tokenizer decode failed: " + std::string(st.ToString()));
    }
    return out;
  }
#endif

  // Subprocess fallback: write space-separated ids to a temp file, invoke
  // spm_decode, and read the resulting text back.
  if (model_path_.empty()) {
    throw std::runtime_error("tokenizer is not loaded");
  }
  if (!std::filesystem::exists(spm_decode_exe_)) {
    throw std::runtime_error("spm_decode executable not found: " + spm_decode_exe_);
  }

  const auto tmp_dir = std::filesystem::temp_directory_path();
  const auto in_path = tmp_dir / "llama_spm_ids.txt";
  const auto out_path = tmp_dir / "llama_spm_text.txt";

  std::ostringstream id_stream;
  for (std::size_t i = 0; i < filtered.size(); ++i) {
    if (i > 0) {
      id_stream << ' ';
    }
    id_stream << filtered[i];
  }
  write_text_file(in_path, id_stream.str());

  const std::string cmd = spm_decode_exe_ + " --model=" + model_path_ +
                          " --input_format=id --input=" + in_path.string() +
                          " --output=" + out_path.string();
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("spm_decode command failed: " + cmd);
  }

  std::string result = read_text_file(out_path);
  // spm_decode appends a trailing newline to its output file; strip it so that
  // the streaming display diff (decoded.substr(prev_text.size())) stays aligned.
  if (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }
  return result;
}

// Returns the set of token ids that should halt autoregressive generation.
// EOS is always included when valid.  Other special tokens (e.g. <eot>,
// end-of-turn markers) are included because models that use them during
// training expect generation to stop on those tokens too.  BOS and UNK are
// excluded: they can legitimately appear inside a sequence and should not
// trigger early stopping.
//
// The result is sorted and deduplicated so callers can binary-search it.
std::vector<int> Tokenizer::generation_stop_ids() const {
  std::vector<int> out;
  if (eos_id_ >= 0) {
    out.push_back(eos_id_);
  }
  for (int id : special_ids_) {
    if (id < 0 || id == bos_id_ || id == unk_id_ || id == eos_id_) {
      continue;
    }
    out.push_back(id);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Removes special token ids from `ids` before decoding.  A std::set is
// constructed from special_ids_ to give O(log n) lookup per token, which is
// preferable to a linear scan given that the special set is typically small
// but the id sequence can be long.
std::vector<int> Tokenizer::strip_special_ids(const std::vector<int>& ids) const {
  if (special_ids_.empty()) {
    return ids;
  }
  std::set<int> special(special_ids_.begin(), special_ids_.end());
  std::vector<int> out;
  out.reserve(ids.size());
  for (int id : ids) {
    if (special.find(id) == special.end()) {
      out.push_back(id);
    }
  }
  return out;
}

}  // namespace model
