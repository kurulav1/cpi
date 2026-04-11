#include "app/main_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace app::main_helpers {
namespace {

constexpr const char* kDefaultSystemPrompt = "You are a concise helpful assistant.";
constexpr const char* kDefaultSingleInstanceMutex = "Local\\llama_infer_single_instance";
constexpr const char* kDefaultSingleInstanceLockPath = "/tmp/llama_infer_single_instance.lock";

std::string env_or_default_string(const char* env_key, const char* fallback) {
#ifdef _WIN32
  char* raw = nullptr;
  std::size_t raw_len = 0;
  if (_dupenv_s(&raw, &raw_len, env_key) == 0 && raw != nullptr) {
    const std::string value(raw);
    std::free(raw);
    if (!value.empty()) {
      return value;
    }
  } else if (raw != nullptr) {
    std::free(raw);
  }
  return fallback;
#else
  const char* raw = std::getenv(env_key);
  return (raw != nullptr && raw[0] != '\0') ? raw : fallback;
#endif
}

std::size_t json_find_key(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  std::size_t p = 0;
  int depth = 0;
  while (p < json.size()) {
    const char c = json[p];
    if (c == '"') {
      if (depth <= 1 && p + needle.size() <= json.size() &&
          json.compare(p, needle.size(), needle) == 0) {
        return p + needle.size();
      }
      ++p;
      while (p < json.size() && json[p] != '"') {
        if (json[p] == '\\') {
          ++p;
        }
        ++p;
      }
      if (p < json.size()) {
        ++p;
      }
    } else if (c == '{' || c == '[') {
      ++depth;
      ++p;
    } else if (c == '}' || c == ']') {
      --depth;
      ++p;
    } else {
      ++p;
    }
  }
  return std::string::npos;
}

std::string json_read_string(const std::string& json, std::size_t& pos) {
  if (pos >= json.size() || json[pos] != '"') {
    return "";
  }
  ++pos;
  std::string result;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      ++pos;
      switch (json[pos]) {
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
        default:
          result += json[pos];
          break;
      }
    } else {
      result += json[pos];
    }
    ++pos;
  }
  if (pos < json.size()) {
    ++pos;
  }
  return result;
}

std::string build_plain_chat_prompt(const std::string& prompt_text) {
  return std::string(kDefaultSystemPrompt) + "\nUser: " + prompt_text + "\nAssistant:";
}

std::string build_llama3_style_prompt(const std::string& prompt_text) {
  return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
         std::string(kDefaultSystemPrompt) +
         "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
         prompt_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

std::string build_llama2_style_prompt(const std::string& prompt_text) {
  return "[INST] <<SYS>>\n" + std::string(kDefaultSystemPrompt) +
         "\nReply naturally to the user's latest request only. Do not add explanation unless asked.\n<</SYS>>\n\n" +
         prompt_text + " [/INST]";
}

std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return s;
}

}  // namespace

SingleInstanceGuard::~SingleInstanceGuard() { release(); }

bool SingleInstanceGuard::acquire() {
#ifdef _WIN32
  const std::string mutex_name =
      env_or_default_string("LLAMA_INFER_INSTANCE_MUTEX", kDefaultSingleInstanceMutex);
  mutex_ = CreateMutexA(nullptr, FALSE, mutex_name.c_str());
  if (!mutex_) {
    return false;
  }
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    release();
    return false;
  }
  return true;
#else
  const std::string lock_path =
      env_or_default_string("LLAMA_INFER_LOCK_PATH", kDefaultSingleInstanceLockPath);
  fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
  if (fd_ < 0) {
    return false;
  }
  if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    release();
    return false;
  }
  return true;
#endif
}

void SingleInstanceGuard::release() {
#ifdef _WIN32
  HANDLE handle = static_cast<HANDLE>(mutex_);
  if (handle) {
    CloseHandle(handle);
    mutex_ = nullptr;
  }
#else
  if (fd_ >= 0) {
    flock(fd_, LOCK_UN);
    close(fd_);
    fd_ = -1;
  }
#endif
}

std::vector<int> parse_tokens(const std::string& csv) {
  std::vector<int> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stoi(item));
    }
  }
  return out;
}

std::string join_ints(const std::vector<int>& values, std::size_t limit) {
  std::ostringstream ss;
  const std::size_t n = (limit == 0 || limit > values.size()) ? values.size() : limit;
  for (std::size_t i = 0; i < n; ++i) {
    if (i > 0) {
      ss << ' ';
    }
    ss << values[i];
  }
  if (n < values.size()) {
    ss << " ...";
  }
  return ss.str();
}

std::string json_get_string(const std::string& json, const std::string& key) {
  const std::size_t v = json_find_key(json, key);
  if (v == std::string::npos) {
    return "";
  }
  std::size_t p = v;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) {
    ++p;
  }
  if (p >= json.size() || json[p] != '"') {
    return "";
  }
  return json_read_string(json, p);
}

int json_get_int(const std::string& json, const std::string& key, int def) {
  const std::size_t v = json_find_key(json, key);
  if (v == std::string::npos) {
    return def;
  }
  std::size_t p = v;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) {
    ++p;
  }
  if (p >= json.size()) {
    return def;
  }
  try {
    return std::stoi(json.substr(p));
  } catch (...) {
    return def;
  }
}

float json_get_float(const std::string& json, const std::string& key, float def) {
  const std::size_t v = json_find_key(json, key);
  if (v == std::string::npos) {
    return def;
  }
  std::size_t p = v;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) {
    ++p;
  }
  if (p >= json.size()) {
    return def;
  }
  try {
    return std::stof(json.substr(p));
  } catch (...) {
    return def;
  }
}

bool json_get_bool(const std::string& json, const std::string& key, bool def) {
  const std::size_t v = json_find_key(json, key);
  if (v == std::string::npos) {
    return def;
  }
  std::size_t p = v;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) {
    ++p;
  }
  if (p >= json.size()) {
    return def;
  }
  if (json.compare(p, 4, "true") == 0) {
    return true;
  }
  if (json.compare(p, 5, "false") == 0) {
    return false;
  }
  return def;
}

std::vector<std::string> json_get_string_array(const std::string& json,
                                               const std::string& key) {
  const std::size_t v = json_find_key(json, key);
  if (v == std::string::npos) {
    return {};
  }
  std::size_t p = v;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) {
    ++p;
  }
  if (p >= json.size() || json[p] != '[') {
    return {};
  }
  ++p;
  std::vector<std::string> result;
  while (p < json.size() && json[p] != ']') {
    while (p < json.size() &&
           (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == ',')) {
      ++p;
    }
    if (p >= json.size() || json[p] == ']') {
      break;
    }
    if (json[p] == '"') {
      result.push_back(json_read_string(json, p));
    } else {
      ++p;
    }
  }
  return result;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

std::string build_chat_prompt(const std::string& chat_template,
                              const std::string& prompt_text,
                              bool tinyllama_plain_fallback) {
  if (chat_template.empty()) {
    return prompt_text;
  }
  if (chat_template == "tinyllama") {
    return build_plain_chat_prompt(prompt_text);
  }
  if (chat_template == "tinyllama-chatml") {
    if (tinyllama_plain_fallback) {
      return build_plain_chat_prompt(prompt_text);
    }
    return "<|system|>\n" + std::string(kDefaultSystemPrompt) + "</s>\n<|user|>\n" +
           prompt_text + "</s>\n<|assistant|>\n";
  }
  if (chat_template == "llama2") {
    return build_llama2_style_prompt(prompt_text);
  }
  if (chat_template == "llama3") {
    return build_llama3_style_prompt(prompt_text);
  }
  if (chat_template == "mistral") {
    return "[INST] " + prompt_text + " [/INST]";
  }
  if (chat_template == "phi3") {
    return "<|system|>\n" + std::string(kDefaultSystemPrompt) + "<|end|>\n<|user|>\n" +
           prompt_text + "<|end|>\n<|assistant|>\n";
  }
  if (chat_template == "qwen2") {
    return "<|im_start|>system\n" + std::string(kDefaultSystemPrompt) +
           "<|im_end|>\n<|im_start|>user\n" + prompt_text +
           "<|im_end|>\n<|im_start|>assistant\n";
  }
  if (chat_template == "llama4") {
    return build_llama3_style_prompt(prompt_text);
  }
  throw std::runtime_error("unsupported --chat-template value: " + chat_template);
}

std::vector<std::string> default_stop_texts_for_template(
    const std::string& chat_template) {
  if (chat_template == "llama2") {
    return {"</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "\nQuestion:",
            "\nUser:", "\nSystem:", "\nAssistant:", "\nExplanation:", "</", "<|"};
  }
  if (chat_template == "llama3") {
    return {"<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"};
  }
  if (chat_template == "mistral") {
    return {"</s>", "[INST]"};
  }
  if (chat_template == "phi3") {
    return {"<|end|>", "<|user|>", "<|system|>", "<|assistant|>"};
  }
  if (chat_template == "qwen2") {
    return {"<|im_end|>", "<|im_start|>"};
  }
  if (chat_template == "llama4") {
    return {"<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|begin_of_text|>"};
  }
  if (chat_template == "tinyllama") {
    return {"</s>", "\nUser:"};
  }
  if (chat_template == "tinyllama-chatml") {
    return {"</s>", "<|user|>", "<|system|>", "<|assistant|>", "\nUser:"};
  }
  return {};
}

std::string sanitize_stream_text(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string normalize_whitespace(std::string text) {
  text = std::regex_replace(text, std::regex("\\s{2,}"), " ");
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.erase(text.begin());
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }
  return text;
}

std::string cleanup_repeated_words(std::string text) {
  static const std::regex repeated_word(R"(\b([A-Za-z][A-Za-z'-]*)\b(?:\s+\1\b)+)",
                                        std::regex_constants::icase);
  static const std::regex repeated_segment(R"(\b([A-Za-z]{3,})\1\b)",
                                           std::regex_constants::icase);
  static const std::regex repeated_letter(R"(([A-Za-z])\1{4,})");
  static const std::regex repeated_punct(R"(([!?.,])\1+)");

  for (int i = 0; i < 4; ++i) {
    const std::string next = std::regex_replace(text, repeated_word, "$1");
    if (next == text) {
      break;
    }
    text = next;
  }
  for (int i = 0; i < 4; ++i) {
    const std::string next = std::regex_replace(text, repeated_segment, "$1");
    if (next == text) {
      break;
    }
    text = next;
  }
  text = std::regex_replace(text, repeated_letter, "$1");
  text = std::regex_replace(text, repeated_punct, "$1");
  return normalize_whitespace(text);
}

std::string normalized_alpha_token(const std::string& token) {
  std::string out;
  for (unsigned char ch : token) {
    if (std::isalpha(ch)) {
      out.push_back(static_cast<char>(std::tolower(ch)));
    }
  }
  return out;
}

bool is_prefix_like_duplicate(const std::string& left, const std::string& right) {
  if (left.size() < 3 || right.size() < 3) {
    return false;
  }
  if (left == right) {
    return true;
  }
  return (right.starts_with(left) || left.starts_with(right));
}

std::string collapse_adjacent_near_duplicate_words(const std::string& text) {
  std::stringstream ss(text);
  std::vector<std::string> tokens;
  std::string token;
  while (ss >> token) {
    if (!tokens.empty()) {
      const std::string prev_norm = normalized_alpha_token(tokens.back());
      const std::string cur_norm = normalized_alpha_token(token);
      if (is_prefix_like_duplicate(prev_norm, cur_norm)) {
        if (cur_norm.size() >= prev_norm.size()) {
          tokens.back() = token;
        }
        continue;
      }
    }
    tokens.push_back(token);
  }

  std::ostringstream joined;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      joined << ' ';
    }
    joined << tokens[i];
  }
  return normalize_whitespace(joined.str());
}

std::string trim_incomplete_trailing_tail(std::string text) {
  const std::size_t last_period = text.find_last_of('.');
  const std::size_t last_exclaim = text.find_last_of('!');
  const std::size_t last_question = text.find_last_of('?');
  std::size_t last_terminal = last_period;
  if (last_exclaim != std::string::npos &&
      (last_terminal == std::string::npos || last_exclaim > last_terminal)) {
    last_terminal = last_exclaim;
  }
  if (last_question != std::string::npos &&
      (last_terminal == std::string::npos || last_question > last_terminal)) {
    last_terminal = last_question;
  }
  if (last_terminal == std::string::npos || last_terminal + 1 >= text.size()) {
    return normalize_whitespace(text);
  }
  const std::string trailing = normalize_whitespace(text.substr(last_terminal + 1));
  if (!trailing.empty() && trailing.find_first_of(".!?") == std::string::npos) {
    text.resize(last_terminal + 1);
  }
  return normalize_whitespace(text);
}

std::string cleanup_repeated_sentences(const std::string& text) {
  static const std::regex sentence_re(R"([^.!?]+[.!?]+|[^.!?]+$)");
  std::sregex_iterator it(text.begin(), text.end(), sentence_re);
  std::sregex_iterator end;
  std::vector<std::string> deduped;
  std::string previous_key;

  for (; it != end; ++it) {
    std::string sentence = normalize_whitespace(it->str());
    if (sentence.empty()) {
      continue;
    }
    const std::string key = to_lower_copy(sentence);
    if (key == previous_key) {
      continue;
    }
    deduped.push_back(sentence);
    previous_key = key;
  }

  if (deduped.empty()) {
    return normalize_whitespace(text);
  }

  std::ostringstream joined;
  for (std::size_t i = 0; i < deduped.size(); ++i) {
    if (i > 0) {
      joined << ' ';
    }
    joined << deduped[i];
  }
  return normalize_whitespace(joined.str());
}

std::string normalize_final_response_text(const std::string& text) {
  std::string cleaned = sanitize_stream_text(text);
  const std::vector<std::regex> cut_markers = {
      std::regex(R"((?:^|\n)\s*Explanation\s*:)", std::regex_constants::icase),
      std::regex(R"((?:^|\n)\s*Question\s*:)", std::regex_constants::icase),
      std::regex(R"((?:^|\n)\s*User\s*:)", std::regex_constants::icase),
      std::regex(R"((?:^|\n)\s*System\s*:)", std::regex_constants::icase),
      std::regex(R"(\[INST\])", std::regex_constants::icase),
      std::regex(R"(<<SYS>>)", std::regex_constants::icase),
      std::regex(R"(<</SYS>>)", std::regex_constants::icase),
  };
  for (const auto& marker : cut_markers) {
    std::smatch match;
    if (std::regex_search(cleaned, match, marker) && match.position() > 0) {
      cleaned = cleaned.substr(0, static_cast<std::size_t>(match.position()));
      break;
    }
  }

  cleaned = std::regex_replace(cleaned, std::regex(R"(^\s*Assistant\s*:?\s*)", std::regex_constants::icase), "");
  cleaned = std::regex_replace(cleaned, std::regex(R"(^\s*Answer\s*:?\s*)", std::regex_constants::icase), "");
  cleaned = std::regex_replace(cleaned, std::regex(R"(^\s*(?:\[/?INST\]|<<\/?SYS>>|</?s>)\s*)", std::regex_constants::icase), "");

  return cleanup_repeated_sentences(
      trim_incomplete_trailing_tail(
          collapse_adjacent_near_duplicate_words(
              cleanup_repeated_words(normalize_whitespace(cleaned)))));
}

std::size_t find_first_stop_pos(const std::string& text,
                                const std::vector<std::string>& stops) {
  std::size_t best = std::string::npos;
  for (const auto& stop : stops) {
    if (stop.empty()) {
      continue;
    }
    const std::size_t p = text.find(stop);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best = p;
    }
  }
  return best;
}

bool has_complete_sentence(const std::string& text) {
  if (text.size() < 16) {
    return false;
  }
  for (char ch : text) {
    if (ch == '.' || ch == '!' || ch == '?') {
      return true;
    }
  }
  return false;
}

bool is_safetensors_model_dir(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  const std::filesystem::path model_path(path);
  if (std::filesystem::is_regular_file(model_path, ec) && !ec) {
    return model_path.extension() == ".safetensors";
  }
  if (!std::filesystem::is_directory(model_path, ec) || ec) {
    return false;
  }
  for (const auto& entry : std::filesystem::directory_iterator(model_path, ec)) {
    if (ec) {
      return false;
    }
    if (entry.is_regular_file(ec) && !ec &&
        entry.path().extension() == ".safetensors") {
      return true;
    }
  }
  return false;
}

std::string guess_chat_template_from_model_path(const std::string& model_path) {
  const std::string name = to_lower_copy(model_path);
  if (name.find("llama-4") != std::string::npos ||
      name.find("llama4") != std::string::npos) {
    return "llama4";
  }
  if (name.find("llama-3") != std::string::npos ||
      name.find("llama3") != std::string::npos) {
    return "llama3";
  }
  if (name.find("llama-2") != std::string::npos ||
      name.find("llama2") != std::string::npos) {
    return "llama2";
  }
  if (name.find("qwen") != std::string::npos) {
    return "qwen2";
  }
  if (name.find("phi-3") != std::string::npos ||
      name.find("phi3") != std::string::npos) {
    return "phi3";
  }
  if (name.find("mistral") != std::string::npos) {
    return "mistral";
  }
  if (name.find("tinyllama") != std::string::npos) {
    return "tinyllama-chatml";
  }
  return "";
}

std::string auto_detect_tokenizer_path(const std::string& model_path) {
  auto first_existing_tokenizer = [](const std::filesystem::path& dir) -> std::string {
    std::error_code sec;
    const std::vector<std::filesystem::path> local = {
        dir / "tokenizer.json",
        dir / "tokenizer.model",
        dir / "hf" / "tokenizer.json",
        dir / "hf" / "tokenizer.model",
    };
    for (const auto& candidate : local) {
      if (!candidate.empty() && std::filesystem::exists(candidate, sec) &&
          std::filesystem::is_regular_file(candidate, sec) && !sec) {
        return candidate.string();
      }
    }
    return "";
  };
  auto split_name_tokens = [](const std::string& name) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char ch : name) {
      if (std::isalnum(ch)) {
        cur.push_back(static_cast<char>(std::tolower(ch)));
      } else if (!cur.empty()) {
        if (cur.size() >= 2) {
          out.push_back(cur);
        }
        cur.clear();
      }
    }
    if (!cur.empty() && cur.size() >= 2) {
      out.push_back(cur);
    }
    return out;
  };

  std::error_code ec;
  const std::filesystem::path model(model_path);
  const std::filesystem::path base =
      std::filesystem::is_directory(model, ec) ? model : model.parent_path();
  if (const std::string local = first_existing_tokenizer(base); !local.empty()) {
    return local;
  }

  const std::filesystem::path model_name =
      std::filesystem::is_directory(model, ec) ? model.filename() : model.stem();
  const auto model_tokens = split_name_tokens(model_name.string());
  if (std::filesystem::is_directory(base, ec) && !ec) {
    int best_score = -1;
    std::filesystem::path best_dir;
    for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_directory(ec) || ec) {
        continue;
      }
      const auto dir_tokens = split_name_tokens(entry.path().filename().string());
      int score = 0;
      for (const auto& tok : model_tokens) {
        for (const auto& dt : dir_tokens) {
          if (dt.find(tok) != std::string::npos || tok.find(dt) != std::string::npos) {
            ++score;
            break;
          }
        }
      }
      if (score > best_score) {
        best_score = score;
        best_dir = entry.path();
      }
    }
    if (best_score > 0) {
      if (const std::string best = first_existing_tokenizer(best_dir); !best.empty()) {
        return best;
      }
    }
  }

  return "";
}

}  // namespace app::main_helpers
