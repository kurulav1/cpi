#include "model/wordpiece_tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace model {
namespace {

// --- UTF-8 helpers -----------------------------------------------------------

// Decodes `s` into Unicode codepoints (lenient: malformed bytes pass through as
// their raw value).
std::vector<std::uint32_t> to_codepoints(const std::string& s) {
  std::vector<std::uint32_t> out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    std::uint32_t cp = c;
    int extra = 0;
    if ((c & 0x80) == 0x00) { cp = c; extra = 0; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { ++i; out.push_back(c); continue; }
    if (i + extra >= s.size()) { ++i; out.push_back(c); continue; }
    bool ok = true;
    for (int k = 1; k <= extra; ++k) {
      const unsigned char cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) { ++i; out.push_back(c); continue; }
    out.push_back(cp);
    i += extra + 1;
  }
  return out;
}

void append_utf8(std::string& s, std::uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// --- character classification (matching BERT's BasicTokenizer) ---------------

bool is_whitespace(std::uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0C || cp == 0x2028 ||
         cp == 0x2029 || cp == 0x00A0;
}

bool is_control(std::uint32_t cp) {
  if (cp == '\t' || cp == '\n' || cp == '\r') return false;
  return cp < 0x20 || cp == 0x7F;
}

bool is_punct(std::uint32_t cp) {
  // ASCII punctuation ranges, plus common Unicode punctuation blocks. BERT also
  // treats any Unicode P* category as punctuation; this covers the practical set.
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126)) {
    return true;
  }
  return (cp >= 0x2000 && cp <= 0x206F) ||  // general punctuation
         (cp >= 0x3000 && cp <= 0x303F);    // CJK symbols and punctuation
}

bool is_cjk(std::uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x3040 && cp <= 0x30FF) ||  // hiragana/katakana
         (cp >= 0xAC00 && cp <= 0xD7A3);    // hangul
}

// Maps a (lowercased) accented Latin codepoint to its ASCII base, approximating
// HF's NFD + drop-combining-marks. English-first: covers Latin-1 Supplement and
// the common Latin Extended-A letters; everything else is returned unchanged.
std::uint32_t strip_accent(std::uint32_t cp) {
  switch (cp) {
    case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: case 0x101:
    case 0x103: case 0x105: return 'a';
    case 0xE7: case 0x107: case 0x109: case 0x10D: return 'c';
    case 0xE8: case 0xE9: case 0xEA: case 0xEB: case 0x113: case 0x115: case 0x117:
    case 0x119: case 0x11B: return 'e';
    case 0xEC: case 0xED: case 0xEE: case 0xEF: case 0x129: case 0x12B: case 0x12F:
      return 'i';
    case 0xF1: case 0x144: case 0x148: return 'n';
    case 0xF2: case 0xF3: case 0xF4: case 0xF5: case 0xF6: case 0xF8: case 0x14D:
    case 0x14F: case 0x151: return 'o';
    case 0xF9: case 0xFA: case 0xFB: case 0xFC: case 0x169: case 0x16B: case 0x16D:
    case 0x16F: case 0x171: return 'u';
    case 0xFD: case 0xFF: return 'y';
    case 0xDF: return 's';  // sharp s -> s (approximation)
    default: return cp;
  }
}

std::uint32_t to_lower(std::uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;  // Latin-1 uppercase
  return cp;
}

}  // namespace

void WordPieceTokenizer::load(const std::string& model_dir, bool lowercase, bool strip_accents) {
  lowercase_ = lowercase;
  strip_accents_ = strip_accents;
  std::ifstream in(model_dir + "/vocab.txt", std::ios::binary);
  if (!in) {
    throw std::runtime_error("wordpiece: failed to open vocab.txt in " + model_dir);
  }
  std::string line;
  int id = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    vocab_.emplace(line, id);
    id_to_token_.push_back(line);
    ++id;
  }
  if (id_to_token_.empty()) {
    throw std::runtime_error("wordpiece: empty vocab.txt");
  }
  const auto lookup = [&](const char* tok, int def) {
    const auto it = vocab_.find(tok);
    return it != vocab_.end() ? it->second : def;
  };
  cls_id_ = lookup("[CLS]", 101);
  sep_id_ = lookup("[SEP]", 102);
  pad_id_ = lookup("[PAD]", 0);
  unk_id_ = lookup("[UNK]", 100);
}

void WordPieceTokenizer::wordpiece(const std::string& word, std::vector<int>& out) const {
  if (static_cast<int>(word.size()) > max_input_chars_per_word_) {
    out.push_back(unk_id_);
    return;
  }
  std::size_t start = 0;
  std::vector<int> sub;
  bool bad = false;
  while (start < word.size()) {
    std::size_t end = word.size();
    int cur = -1;
    while (start < end) {
      std::string piece = word.substr(start, end - start);
      if (start > 0) {
        piece = "##" + piece;
      }
      const auto it = vocab_.find(piece);
      if (it != vocab_.end()) {
        cur = it->second;
        break;
      }
      --end;
    }
    if (cur < 0) {
      bad = true;
      break;
    }
    sub.push_back(cur);
    start = end;
  }
  if (bad) {
    out.push_back(unk_id_);
  } else {
    out.insert(out.end(), sub.begin(), sub.end());
  }
}

std::vector<int> WordPieceTokenizer::encode(const std::string& text, int max_tokens) const {
  // 1. Normalize to a list of pre-tokens (whitespace-split, punctuation/CJK
  //    isolated), each as a normalized UTF-8 word string.
  const std::vector<std::uint32_t> cps = to_codepoints(text);
  std::vector<std::string> words;
  std::string cur;
  const auto flush = [&]() {
    if (!cur.empty()) {
      words.push_back(cur);
      cur.clear();
    }
  };
  for (std::uint32_t cp : cps) {
    if (cp == 0 || cp == 0xFFFD || is_control(cp)) {
      continue;
    }
    if (is_whitespace(cp)) {
      flush();
      continue;
    }
    if (lowercase_) {
      cp = to_lower(cp);
    }
    if (strip_accents_) {
      cp = strip_accent(cp);
    }
    if (is_punct(cp) || is_cjk(cp)) {
      flush();
      std::string s;
      append_utf8(s, cp);
      words.push_back(s);
      continue;
    }
    append_utf8(cur, cp);
  }
  flush();

  // 2. WordPiece each pre-token; 3. wrap with [CLS] ... [SEP], truncating the
  //    middle so the total stays within max_tokens (graceful truncation).
  std::vector<int> piece_ids;
  for (const std::string& w : words) {
    wordpiece(w, piece_ids);
  }
  const int budget = max_tokens > 2 ? max_tokens - 2 : 0;
  if (static_cast<int>(piece_ids.size()) > budget) {
    piece_ids.resize(static_cast<std::size_t>(budget));
  }
  std::vector<int> ids;
  ids.reserve(piece_ids.size() + 2);
  ids.push_back(cls_id_);
  ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
  ids.push_back(sep_id_);
  return ids;
}

}  // namespace model
