#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace app::main_helpers {

class SingleInstanceGuard {
 public:
  SingleInstanceGuard() = default;
  ~SingleInstanceGuard();

  SingleInstanceGuard(const SingleInstanceGuard&) = delete;
  SingleInstanceGuard& operator=(const SingleInstanceGuard&) = delete;

  bool acquire();

 private:
  void release();

#ifdef _WIN32
  void* mutex_ = nullptr;
#else
  int fd_ = -1;
#endif
};

std::vector<int> parse_tokens(const std::string& csv);
std::string join_ints(const std::vector<int>& values, std::size_t limit = 0);

std::string json_get_string(const std::string& json, const std::string& key);
int json_get_int(const std::string& json, const std::string& key, int def);
float json_get_float(const std::string& json, const std::string& key, float def);
bool json_get_bool(const std::string& json, const std::string& key, bool def);
std::vector<std::string> json_get_string_array(const std::string& json, const std::string& key);
std::string json_escape(const std::string& s);

std::string build_chat_prompt(const std::string& chat_template,
                              const std::string& prompt_text,
                              bool tinyllama_plain_fallback);
std::vector<std::string> default_stop_texts_for_template(const std::string& chat_template);

std::string sanitize_stream_text(std::string s);
std::size_t find_first_stop_pos(const std::string& text, const std::vector<std::string>& stops);
bool has_complete_sentence(const std::string& text);

bool is_safetensors_model_dir(const std::string& path);
std::string guess_chat_template_from_model_path(const std::string& model_path);
std::string auto_detect_tokenizer_path(const std::string& model_path);

}  // namespace app::main_helpers
