#include "model/hf_bpe_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

// Implements HfBpeTokenizer: a self-contained BPE tokenizer that loads
// HuggingFace tokenizer.json files without depending on the transformers
// Python library or any external C++ JSON library.
//
// Design rationale:
//   - A hand-rolled recursive-descent JSON parser (JsonCursor) is used instead
//     of a third-party library (nlohmann/json, RapidJSON, etc.) to keep the
//     dependency footprint small and avoid version conflicts in downstream
//     projects.
//   - BPE merge rules are stored in a flat hash map keyed by a composite
//     "left\x1Fright" string.  The U+001F unit-separator byte is chosen as the
//     delimiter because it cannot appear in a valid vocabulary piece, so the
//     key is unambiguous without escaping.
//   - The encode loop applies the greedy lowest-rank merge strategy (also known
//     as the standard BPE algorithm): at each step the adjacent pair with the
//     smallest rank in merge_ranks_ is merged.  This runs in O(n^2) in the
//     number of pieces but n is bounded by the token piece length, which is
//     typically small.

namespace model {
namespace {

// Reads an entire file into a std::string using binary mode to avoid
// platform-specific line-ending transformations corrupting the JSON content.
std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed reading tokenizer json: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Encodes a Unicode code point as a UTF-8 byte sequence and appends it to
// `out`.  This is needed to handle \uXXXX escape sequences inside JSON strings
// (e.g. special token content fields) without a full Unicode library.
//
// The encoding follows RFC 3629:
//   U+0000..U+007F   → 1 byte  (0xxxxxxx)
//   U+0080..U+07FF   → 2 bytes (110xxxxx 10xxxxxx)
//   U+0800..U+FFFF   → 3 bytes (1110xxxx 10xxxxxx 10xxxxxx)
//   U+10000..U+10FFFF → 4 bytes (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
void append_utf8(std::string* out, std::uint32_t cp) {
  if (cp <= 0x7F) {
    out->push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out->push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out->push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// A minimal, non-allocating recursive-descent JSON cursor used exclusively to
// parse the tokenizer.json format produced by HuggingFace tokenizers.
//
// Only the subset of JSON required by that format is implemented:
//   - Objects, arrays, strings, integers, booleans, and null.
//   - \uXXXX unicode escapes and the standard single-character escape
//     sequences (\n \r \t \b \f \\ \" \/).
//
// The cursor advances a single position index (pos_) through the underlying
// string.  All methods throw std::runtime_error on a parse error so callers
// do not need to check return values for validity.
class JsonCursor {
 public:
  explicit JsonCursor(const std::string& text) : text_(text) {}

  // Advances past any leading whitespace characters.
  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  // Consumes `ch` if it is the next non-whitespace character.
  // Returns true if consumed, false otherwise (cursor is not advanced on false).
  bool consume(char ch) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == ch) {
      ++pos_;
      return true;
    }
    return false;
  }

  // Asserts that the next non-whitespace character is `ch` and advances past it.
  // Throws if the expected character is not found.
  void expect(char ch) {
    skip_ws();
    if (pos_ >= text_.size() || text_[pos_] != ch) {
      throw std::runtime_error(std::string("json parse error: expected '") + ch + "'");
    }
    ++pos_;
  }

  // Parses a JSON string literal and returns the decoded content.
  // Handles all standard escape sequences including \uXXXX.
  std::string parse_string() {
    skip_ws();
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      char ch = text_[pos_++];
      if (ch == '"') {
        return out;
      }
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) {
        throw std::runtime_error("json parse error: truncated escape");
      }
      const char esc = text_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          // Parse exactly 4 hex digits following \u and convert the resulting
          // code point to its UTF-8 representation.
          if (pos_ + 4 > text_.size()) {
            throw std::runtime_error("json parse error: truncated unicode escape");
          }
          std::uint32_t cp = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') {
              cp |= static_cast<std::uint32_t>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              cp |= static_cast<std::uint32_t>(10 + h - 'a');
            } else if (h >= 'A' && h <= 'F') {
              cp |= static_cast<std::uint32_t>(10 + h - 'A');
            } else {
              throw std::runtime_error("json parse error: invalid unicode escape");
            }
          }
          append_utf8(&out, cp);
          break;
        }
        default:
          throw std::runtime_error("json parse error: unsupported escape");
      }
    }
    throw std::runtime_error("json parse error: unterminated string");
  }

  // Parses a JSON integer (with optional leading sign) and returns it as int.
  int parse_int() {
    skip_ws();
    const std::size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
      ++pos_;
    }
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (start == pos_) {
      throw std::runtime_error("json parse error: expected integer");
    }
    return std::stoi(text_.substr(start, pos_ - start));
  }

  // Parses a JSON boolean literal ("true" or "false").
  bool parse_bool() {
    skip_ws();
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      return true;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      return false;
    }
    throw std::runtime_error("json parse error: expected bool");
  }

  // Skips over a complete JSON value of any type.  Used when the parser
  // encounters a key that the tokenizer does not need, so the cursor is
  // advanced past the entire value without allocating or storing it.
  void skip_value() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("json parse error: unexpected eof");
    }
    const char ch = text_[pos_];
    if (ch == '"') {
      (void)parse_string();
      return;
    }
    if (ch == '{') {
      expect('{');
      if (consume('}')) {
        return;
      }
      while (true) {
        (void)parse_string();
        expect(':');
        skip_value();
        if (consume('}')) {
          return;
        }
        expect(',');
      }
    }
    if (ch == '[') {
      expect('[');
      if (consume(']')) {
        return;
      }
      while (true) {
        skip_value();
        if (consume(']')) {
          return;
        }
        expect(',');
      }
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+') {
      (void)parse_int();
      return;
    }
    if (text_.compare(pos_, 4, "true") == 0 || text_.compare(pos_, 5, "false") == 0) {
      (void)parse_bool();
      return;
    }
    if (text_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      return;
    }
    throw std::runtime_error("json parse error: unsupported value");
  }

 private:
  const std::string& text_;
  std::size_t pos_ = 0;
};

// Formats a raw byte value as the HuggingFace byte-fallback token string
// "<0xXX>" (e.g. byte 0x20 → "<0x20>").  This representation is used by
// models trained with byte-level fallback to encode bytes that have no
// regular vocabulary entry.
std::string to_hex_byte(unsigned char value) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string out = "<0x";
  out.push_back(kHex[(value >> 4) & 0x0F]);
  out.push_back(kHex[value & 0x0F]);
  out.push_back('>');
  return out;
}

// Returns true if `piece` is a valid byte-fallback token in the form "<0xXX>"
// and, if so, sets `*value` to the decoded byte.  The exact 6-character format
// check avoids false-positive matches on longer tokens that happen to start
// with "<0x".
bool is_byte_token(const std::string& piece, unsigned char* value) {
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') {
    return false;
  }
  auto hex = [](char ch) -> int {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
      return 10 + ch - 'A';
    }
    return -1;
  };
  const int hi = hex(piece[3]);
  const int lo = hex(piece[4]);
  if (hi < 0 || lo < 0) {
    return false;
  }
  *value = static_cast<unsigned char>((hi << 4) | lo);
  return true;
}

}  // namespace

// Loads a HuggingFace tokenizer.json file and populates all internal tables.
//
// The function parses four top-level sections of the JSON:
//   "added_tokens"  — special tokens (BOS, EOS, etc.) with their ids.
//   "normalizer"    — identifies the word-boundary marker string (default
//                     U+2581 LOWER ONE EIGHTH BLOCK, "▁", used by SentencePiece
//                     and many HF models) so spaces can be reconstructed on
//                     decode.
//   "decoder"       — determines whether a leading space should be stripped
//                     from the decoded output (strip_leading_space_).
//   "model"         — the core BPE vocabulary ("vocab") and ordered merge rules
//                     ("merges").
//
// After parsing, added_tokens_ is sorted by descending content length so that
// the longest match wins during encode (greedy maximal-munch matching).
void HfBpeTokenizer::load(const std::string& path) {
  vocab_.clear();
  id_to_piece_.clear();
  merge_ranks_.clear();
  added_tokens_.clear();
  special_ids_.clear();
  // Default word-boundary marker is U+2581 (▁), the SentencePiece convention.
  word_boundary_ = "\xE2\x96\x81";
  byte_fallback_ = false;
  strip_leading_space_ = true;
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;

  const std::string json = read_text_file(path);
  JsonCursor cur(json);
  cur.expect('{');

  while (!cur.consume('}')) {
    const std::string key = cur.parse_string();
    cur.expect(':');

    if (key == "added_tokens") {
      cur.expect('[');
      if (!cur.consume(']')) {
        while (true) {
          cur.expect('{');
          int id = -1;
          std::string content;
          bool special = false;
          while (!cur.consume('}')) {
            const std::string subkey = cur.parse_string();
            cur.expect(':');
            if (subkey == "id") {
              id = cur.parse_int();
            } else if (subkey == "content") {
              content = cur.parse_string();
            } else if (subkey == "special") {
              special = cur.parse_bool();
            } else {
              cur.skip_value();
            }
            if (cur.consume('}')) {
              break;
            }
            cur.expect(',');
          }
          if (!content.empty() && id >= 0) {
            added_tokens_.push_back({content, id});
            if (special) {
              special_ids_.push_back(id);
            }
            // Identify BOS/EOS/UNK by conventional token strings rather than
            // by a separate "role" field, which is absent in many tokenizer.json
            // files produced by earlier versions of the tokenizers library.
            if (content == "<s>" || content == "<|begin_of_text|>") {
              bos_id_ = id;
            } else if (content == "</s>" || content == "<|end_of_text|>" || content == "<|eot|>") {
              if (eos_id_ < 0) eos_id_ = id;
            } else if (content == "<unk>") {
              unk_id_ = id;
            }
          }
          if (cur.consume(']')) {
            break;
          }
          cur.expect(',');
        }
      }
    } else if (key == "normalizer") {
      // Llama4 and some other models set normalizer to null; handle gracefully.
      cur.skip_ws();
      if (!cur.consume('{')) {
        cur.skip_value();
      } else {
        while (!cur.consume('}')) {
          const std::string subkey = cur.parse_string();
          cur.expect(':');
          if (subkey == "normalizers") {
            cur.expect('[');
            if (!cur.consume(']')) {
              while (true) {
                cur.expect('{');
                std::string type;
                std::string prepend;
                std::string replace_content;
                while (!cur.consume('}')) {
                  const std::string nk = cur.parse_string();
                  cur.expect(':');
                  if (nk == "type") {
                    type = cur.parse_string();
                  } else if (nk == "prepend") {
                    prepend = cur.parse_string();
                  } else if (nk == "content") {
                    replace_content = cur.parse_string();
                  } else {
                    cur.skip_value();
                  }
                  if (cur.consume('}')) {
                    break;
                  }
                  cur.expect(',');
                }
                // Extract the word-boundary marker from the normalizer chain.
                // A "Prepend" normalizer with a non-empty prepend string is the
                // canonical way HF models specify the boundary; a "Replace"
                // normalizer that replaces spaces with a custom string is an
                // alternative used by some models.
                if (type == "Prepend" && !prepend.empty()) {
                  word_boundary_ = prepend;
                } else if (type == "Replace" && !replace_content.empty()) {
                  word_boundary_ = replace_content;
                }
                if (cur.consume(']')) {
                  break;
                }
                cur.expect(',');
              }
            }
          } else {
            cur.skip_value();
          }
          if (cur.consume('}')) {
            break;
          }
          cur.expect(',');
        }
      }
    } else if (key == "decoder") {
      cur.skip_ws();
      if (!cur.consume('{')) {
        cur.skip_value();
      } else {
      while (!cur.consume('}')) {
        const std::string subkey = cur.parse_string();
        cur.expect(':');
        if (subkey == "type") {
          const std::string dtype = cur.parse_string();
          if (dtype == "ByteLevel") {
            byte_level_ = true;
          }
        } else if (subkey == "decoders") {
          cur.expect('[');
          if (!cur.consume(']')) {
            while (true) {
              cur.expect('{');
              std::string type;
              std::string content;
              int start = 0;
              while (!cur.consume('}')) {
                const std::string dk = cur.parse_string();
                cur.expect(':');
                if (dk == "type") {
                  type = cur.parse_string();
                } else if (dk == "content") {
                  content = cur.parse_string();
                } else if (dk == "start") {
                  start = cur.parse_int();
                } else {
                  cur.skip_value();
                }
                if (cur.consume('}')) {
                  break;
                }
                cur.expect(',');
              }
              // A Strip decoder with content=" " and start=1 means "remove the
              // first space from the decoded output", which is how models that
              // prepend a space to the first word signal that the space is an
              // encoding artefact rather than genuine whitespace.
              if (type == "Strip" && content == " " && start == 1) {
                strip_leading_space_ = true;
              }
              if (cur.consume(']')) {
                break;
              }
              cur.expect(',');
            }
          }
        } else {
          cur.skip_value();
        }
        if (cur.consume('}')) {
          break;
        }
        cur.expect(',');
      }
      } // end else (decoder was an object)
    } else if (key == "model") {
      cur.expect('{');
      std::string model_type;
      while (!cur.consume('}')) {
        const std::string subkey = cur.parse_string();
        cur.expect(':');
        if (subkey == "type") {
          model_type = cur.parse_string();
        } else if (subkey == "byte_fallback") {
          byte_fallback_ = cur.parse_bool();
        } else if (subkey == "vocab") {
          cur.expect('{');
          int max_id = -1;
          while (!cur.consume('}')) {
            const std::string piece = cur.parse_string();
            cur.expect(':');
            const int id = cur.parse_int();
            vocab_[piece] = id;
            max_id = std::max(max_id, id);
            if (cur.consume('}')) {
              break;
            }
            cur.expect(',');
          }
          // Build the reverse id → piece mapping as a dense vector indexed by
          // id so that decode() can do O(1) lookups rather than a reverse hash
          // map scan.  The vector is sized to max_id+1 to cover all valid ids.
          if (max_id >= 0) {
            id_to_piece_.assign(static_cast<std::size_t>(max_id + 1), std::string());
            for (const auto& [piece, id] : vocab_) {
              id_to_piece_[static_cast<std::size_t>(id)] = piece;
            }
          }
        } else if (subkey == "merges") {
          // The merges array is ordered from highest priority (rank 0) to lowest.
          // Storing rank as the map value lets encode_segment() find the
          // globally best merge in a single pass over adjacent pairs.
          //
          // Two formats exist in HuggingFace tokenizer.json files:
          //   1. String format: "left right"  (classic BPE / SentencePiece style)
          //   2. Array format:  ["left", "right"]  (used by ByteLevel BPE models)
          cur.expect('[');
          int rank = 0;
          if (!cur.consume(']')) {
            while (true) {
              cur.skip_ws();
              std::string left, right;
              if (cur.consume('[')) {
                // Array format: ["left_piece", "right_piece"]
                left = cur.parse_string();
                cur.expect(',');
                right = cur.parse_string();
                cur.expect(']');
              } else {
                // String format: "left_piece right_piece"
                const std::string merge = cur.parse_string();
                const std::size_t sep = merge.find(' ');
                if (sep != std::string::npos) {
                  left = merge.substr(0, sep);
                  right = merge.substr(sep + 1);
                }
              }
              if (!left.empty() && !right.empty()) {
                merge_ranks_[merge_key(left, right)] = rank++;
              }
              if (cur.consume(']')) {
                break;
              }
              cur.expect(',');
            }
          }
        } else {
          cur.skip_value();
        }
        if (cur.consume('}')) {
          break;
        }
        cur.expect(',');
      }
      if (model_type != "BPE") {
        // Only BPE models are supported; other model types (e.g. WordPiece,
        // Unigram) have different tokenization algorithms that this class
        // does not implement.
        throw std::runtime_error("unsupported tokenizer.json model type: " + model_type);
      }
    } else {
      cur.skip_value();
    }

    if (cur.consume('}')) {
      break;
    }
    cur.expect(',');
  }

  // Deduplicate special_ids_ (a token may appear in both the "special" flag
  // and be explicitly named as BOS/EOS/UNK).
  std::sort(special_ids_.begin(), special_ids_.end());
  special_ids_.erase(std::unique(special_ids_.begin(), special_ids_.end()), special_ids_.end());

  // Sort added tokens by descending length so that the encode() loop performs
  // greedy longest-match when scanning for added/special token occurrences.
  std::sort(added_tokens_.begin(),
            added_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  // Fall back to vocabulary lookup for BOS/EOS/UNK if they were not found in
  // the added_tokens list.  Some older tokenizer.json files omit them there
  // but do include them in the main vocab object.
  if (bos_id_ < 0) {
    auto it = vocab_.find("<s>");
    if (it != vocab_.end()) {
      bos_id_ = it->second;
    }
  }
  if (eos_id_ < 0) {
    auto it = vocab_.find("</s>");
    if (it != vocab_.end()) {
      eos_id_ = it->second;
    }
  }
  if (unk_id_ < 0) {
    auto it = vocab_.find("<unk>");
    if (it != vocab_.end()) {
      unk_id_ = it->second;
    }
  }

  // Build GPT-2 byte-to-unicode tables when byte_level_ is active.
  // The mapping follows the standard GPT-2 bytes_to_unicode() function:
  //   bytes 33-126, 161-172, 174-255 → same code point (Latin-1 printable)
  //   bytes 0-32, 127, 128-160, 173  → U+0100-U+0143 (in that order)
  if (byte_level_) {
    std::uint32_t extra = 0x100;
    for (int b = 0; b < 256; ++b) {
      std::uint32_t cp;
      if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        cp = static_cast<std::uint32_t>(b);
      } else {
        cp = extra++;
      }
      std::string utf8;
      append_utf8(&utf8, cp);
      byte_to_unicode_[b] = utf8;
      unicode_to_byte_[cp] = static_cast<unsigned char>(b);
    }
  }
}

// Constructs the composite key used to look up a merge rule in merge_ranks_.
// U+001F (ASCII unit separator) is used as the internal delimiter because it
// is a control character that cannot appear in a valid BPE vocabulary piece,
// making the resulting key unambiguous.
std::string HfBpeTokenizer::merge_key(const std::string& left, const std::string& right) const {
  std::string out;
  out.reserve(left.size() + right.size() + 1);
  out += left;
  out.push_back('\x1F');
  out += right;
  return out;
}

// Splits `text` into individual UTF-8 code-point strings.  Each element of the
// returned vector holds exactly one Unicode code point (1–4 bytes).
//
// The function examines the high bits of each leading byte to determine the
// code point width according to the UTF-8 encoding rules.  If the leading byte
// indicates a multi-byte sequence that would extend beyond the end of the
// string (i.e. the input is truncated or malformed), the byte is treated as a
// single-byte unit rather than throwing, to match the lenient behaviour of
// Python's tokenizers library.
std::vector<std::string> HfBpeTokenizer::split_utf8_pieces(const std::string& text) const {
  std::vector<std::string> pieces;
  for (std::size_t i = 0; i < text.size();) {
    unsigned char ch = static_cast<unsigned char>(text[i]);
    std::size_t len = 1;
    if ((ch & 0x80U) == 0U) {
      len = 1;
    } else if ((ch & 0xE0U) == 0xC0U) {
      len = 2;
    } else if ((ch & 0xF0U) == 0xE0U) {
      len = 3;
    } else if ((ch & 0xF8U) == 0xF0U) {
      len = 4;
    }
    if (i + len > text.size()) {
      len = 1;
    }
    pieces.push_back(text.substr(i, len));
    i += len;
  }
  return pieces;
}

// Encodes a single vocabulary piece string to one or more token ids.
// If the piece exists verbatim in the vocabulary it is returned directly.
// Otherwise, when byte_fallback_ is enabled, each raw byte of the piece is
// converted to its "<0xXX>" token and looked up individually.  This ensures
// that no byte is silently dropped even for out-of-vocabulary pieces.
// When byte_fallback_ is disabled the piece is mapped to unk_id_, matching
// the standard SentencePiece "unk" treatment.
std::vector<int> HfBpeTokenizer::encode_piece_with_fallback(const std::string& piece) const {
  const auto it = vocab_.find(piece);
  if (it != vocab_.end()) {
    return {it->second};
  }
  if (!byte_fallback_) {
    return (unk_id_ >= 0) ? std::vector<int>{unk_id_} : std::vector<int>{};
  }

  // Byte fallback: encode each byte of the unrecognised piece separately.
  // This preserves all input bytes at the cost of a longer token sequence.
  std::vector<int> out;
  out.reserve(piece.size());
  for (unsigned char byte : piece) {
    const auto bit = vocab_.find(to_hex_byte(byte));
    if (bit != vocab_.end()) {
      out.push_back(bit->second);
    } else if (unk_id_ >= 0) {
      out.push_back(unk_id_);
    }
  }
  return out;
}

// Encodes a plain-text segment (no added/special tokens) into token ids using
// the standard greedy BPE algorithm.
//
// Steps:
//   1. Normalize: replace ASCII spaces with the word_boundary_ marker and
//      optionally prepend the marker at the start (to indicate a word boundary
//      at the segment boundary, e.g. after a special token).
//   2. Split the normalized string into individual UTF-8 code points.
//   3. Iteratively merge the adjacent pair with the smallest merge rank until
//      no more merges apply.  This is the standard BPE "greedy lowest rank"
//      algorithm.
//   4. Map each resulting piece to token id(s) via encode_piece_with_fallback.
//
// The algorithm is O(n^2) in the number of pieces per segment.  In practice
// segments are short (bounded by the longest word in the input), so this is
// acceptable.  A priority-queue optimisation could reduce this to O(n log n)
// if profiling identifies this as a bottleneck.
std::vector<int> HfBpeTokenizer::encode_segment(const std::string& text, bool prepend_boundary) const {
  if (text.empty()) {
    return {};
  }

  std::string normalized;
  if (byte_level_) {
    // ByteLevel BPE: map each byte to its GPT-2 unicode character.
    // No word_boundary_ prepend — the space byte (0x20) becomes Ġ (U+0120) which
    // the BPE merges treat as a word-boundary marker naturally.
    normalized.reserve(text.size() * 2);
    for (unsigned char b : text) {
      normalized += byte_to_unicode_[b];
    }
  } else {
    normalized.reserve((prepend_boundary ? word_boundary_.size() : 0) + text.size() * 2);
    if (prepend_boundary) {
      normalized = word_boundary_;
    }
    for (char ch : text) {
      if (ch == ' ') {
        normalized += word_boundary_;
      } else {
        normalized.push_back(ch);
      }
    }
  }

  std::vector<std::string> pieces = split_utf8_pieces(normalized);
  while (pieces.size() > 1) {
    // Find the adjacent pair with the globally lowest merge rank.
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_pos = pieces.size();
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
      const auto it = merge_ranks_.find(merge_key(pieces[i], pieces[i + 1]));
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = i;
      }
    }
    if (best_pos == pieces.size()) {
      // No applicable merge found; the sequence is fully tokenised.
      break;
    }
    // Perform the merge in-place: concatenate pieces[best_pos+1] into
    // pieces[best_pos] then erase the now-redundant element.
    pieces[best_pos] += pieces[best_pos + 1];
    pieces.erase(pieces.begin() + static_cast<std::vector<std::string>::difference_type>(best_pos + 1));
  }

  std::vector<int> ids;
  for (const auto& piece : pieces) {
    const auto piece_ids = encode_piece_with_fallback(piece);
    ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
  }
  return ids;
}

// Encodes `text` to a sequence of token ids, handling added/special tokens
// before applying BPE to the remaining plain-text segments.
//
// The encoding pipeline:
//   1. Optionally prepend BOS.
//   2. Scan the input left-to-right, checking for the longest matching
//      added token at each position.  Added tokens (which include special
//      tokens like <s>, </s>, <pad>, <unk>, and model-specific control tokens)
//      are emitted verbatim as their pre-assigned ids without going through the
//      BPE segmentation step.
//   3. Any plain-text between added tokens is encoded via encode_segment.
//
// Added tokens are sorted by descending length (done in load()) so the first
// match in the inner loop is always the longest possible match, implementing
// greedy maximal-munch at no extra cost.
std::vector<int> HfBpeTokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<int> ids;
  if (add_bos && bos_id_ >= 0) {
    ids.push_back(bos_id_);
  }

  std::size_t pos = 0;
  std::string current;
  bool first_segment = true;
  while (pos < text.size()) {
    bool matched_added = false;
    for (const auto& [content, id] : added_tokens_) {
      if (content.empty() || pos + content.size() > text.size()) {
        continue;
      }
      if (text.compare(pos, content.size(), content) == 0) {
        // Flush any accumulated plain text before emitting the special token.
        const auto seg_ids = encode_segment(current, first_segment);
        ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        current.clear();
        first_segment = false;
        ids.push_back(id);
        pos += content.size();
        matched_added = true;
        break;
      }
    }
    if (!matched_added) {
      current.push_back(text[pos++]);
    }
  }

  // Encode any remaining plain-text tail after the last special token.
  const auto tail_ids = encode_segment(current, first_segment);
  ids.insert(ids.end(), tail_ids.begin(), tail_ids.end());
  return ids;
}

// Decodes a sequence of token ids back to a UTF-8 string.
//
// Steps:
//   1. Convert each id to its vocabulary piece string.  Byte-fallback tokens
//      (of the form "<0xXX>") are converted back to their raw byte values.
//   2. Concatenate all pieces into a raw string that still contains the
//      word_boundary_ marker in place of spaces.
//   3. Replace every occurrence of the word_boundary_ marker with a space
//      character.
//   4. Optionally strip a leading space (controlled by strip_leading_space_)
//      which is an artefact of the prepend-space normalisation used during
//      encoding.
//
// Out-of-range ids are silently skipped to gracefully handle truncated or
// otherwise malformed id sequences.
std::string HfBpeTokenizer::decode(const std::vector<int>& ids) const {
  std::string raw;
  raw.reserve(ids.size() * 4);

  for (int id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_piece_.size()) {
      // Skip invalid ids rather than throwing, matching Python tokenizers
      // behaviour for out-of-range inputs.
      continue;
    }
    const std::string& piece = id_to_piece_[static_cast<std::size_t>(id)];
    unsigned char byte = 0;
    if (is_byte_token(piece, &byte)) {
      raw.push_back(static_cast<char>(byte));
    } else {
      raw += piece;
    }
  }

  if (byte_level_) {
    // ByteLevel BPE decode: each Unicode char in raw is a GPT-2-encoded byte.
    // Iterate over UTF-8 code points and reverse-map each back to its byte.
    std::string bytes_out;
    bytes_out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size();) {
      unsigned char ch = static_cast<unsigned char>(raw[i]);
      std::size_t clen = 1;
      if ((ch & 0x80U) == 0U) {
        clen = 1;
      } else if ((ch & 0xE0U) == 0xC0U) {
        clen = 2;
      } else if ((ch & 0xF0U) == 0xE0U) {
        clen = 3;
      } else if ((ch & 0xF8U) == 0xF0U) {
        clen = 4;
      }
      if (i + clen > raw.size()) clen = 1;
      // Decode the UTF-8 code point.
      std::uint32_t cp = 0;
      if (clen == 1) {
        cp = ch;
      } else if (clen == 2) {
        cp = (static_cast<std::uint32_t>(ch & 0x1FU) << 6) |
             static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+1]) & 0x3FU);
      } else if (clen == 3) {
        cp = (static_cast<std::uint32_t>(ch & 0x0FU) << 12) |
             (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+1]) & 0x3FU) << 6) |
             static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+2]) & 0x3FU);
      } else {
        cp = (static_cast<std::uint32_t>(ch & 0x07U) << 18) |
             (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+1]) & 0x3FU) << 12) |
             (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+2]) & 0x3FU) << 6) |
             static_cast<std::uint32_t>(static_cast<unsigned char>(raw[i+3]) & 0x3FU);
      }
      i += clen;
      const auto it = unicode_to_byte_.find(cp);
      if (it != unicode_to_byte_.end()) {
        bytes_out.push_back(static_cast<char>(it->second));
      }
      // Unknown code points (e.g. special token text like <|eot_id|>) are skipped.
    }
    return bytes_out;
  }

  // SentencePiece-style decode: replace word-boundary markers with spaces.
  std::string text;
  text.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size();) {
    if (raw.compare(i, word_boundary_.size(), word_boundary_) == 0) {
      text.push_back(' ');
      i += word_boundary_.size();
    } else {
      text.push_back(raw[i++]);
    }
  }

  if (strip_leading_space_ && !text.empty() && text.front() == ' ') {
    text.erase(text.begin());
  }
  return text;
}

}  // namespace model
