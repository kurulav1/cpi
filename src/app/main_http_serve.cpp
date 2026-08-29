// The in-binary OpenAI-compatible HTTP server (--serve).
//
// Why it exists: the Node bridge is a fine web UI host, but headless serving
// should not need an interpreter next to the engine. This routes HTTP straight
// into the same BatchWorker the stdin transport drives, so continuous batching,
// grammar-constrained decoding, preempt-and-resume and the sampling knobs are
// the shared ones rather than a second implementation.
//
// Endpoints: GET /health, GET /v1/models, POST /v1/completions,
// POST /v1/chat/completions (both with "stream": true for SSE).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "app/batch_worker.hpp"
#include "app/main_helpers.hpp"
#include "app/main_modes.hpp"
#include "engine/batch_scheduler.hpp"
#include "grammar/grammar.hpp"
#include "grammar/grammar_sampler.hpp"
#include "grammar/json_schema_to_grammar.hpp"
#include "model/tokenizer.hpp"
#include "model/wordpiece_tokenizer.hpp"
#include "net/http_server.hpp"

// The embedding backend, chosen the same way cpi_embed chooses it: CUDA when
// present, else Metal. Without either, /v1/embeddings reports unavailable
// rather than failing the build (a CPU-only cpi still serves chat).
#if CPI_HAS_CUDA
#include "engine/bert_embedder.hpp"
#define CPI_SERVE_HAS_EMBEDDER 1
using ServeEmbedder = engine::BertEmbedder;
#elif defined(CPI_ENABLE_METAL)
#include "engine/metal_bert_embedder.hpp"
#define CPI_SERVE_HAS_EMBEDDER 1
using ServeEmbedder = engine::MetalBertEmbedder;
#endif

namespace app::main_modes {

using app::main_helpers::build_chat_prompt;
using app::main_helpers::json_escape;
using app::main_helpers::json_get_bool;
using app::main_helpers::json_get_float;
using app::main_helpers::json_get_int;
using app::main_helpers::json_get_raw_value;
using app::main_helpers::json_get_string;
using app::main_helpers::json_get_raw_array;
using app::main_helpers::json_get_string_array;
using app::main_helpers::unwrap_json_schema;

namespace {

// Cuts `text` at the first stop string it contains, returning true when it cut.
//
// The batched worker turns only SINGLE-token stop strings into stop ids and, by
// its own comment, leaves multi-token ones "the transport's problem". The serial
// transport does that matching; this one never did, so a multi-token stop was
// accepted and then silently ignored. A stop like "\n\nUser:" is ordinary in
// OpenAI clients and did nothing here.
// Compiles OpenAI `tools` + `tool_choice` into a JSON schema the grammar sampler
// can enforce, so a requested tool call cannot come back malformed.
//
// The shape is a union over the allowed tools:
//   {"anyOf":[{"type":"object",
//              "properties":{"name":{"enum":["<tool>"]},"arguments":<parameters>},
//              "required":["name","arguments"]}, ...]}
//
// tool_choice decides which tools are in the union:
//   "none"                    no constraint, tools ignored
//   "auto" (default)          no constraint; the model may answer in prose, which
//                             is what the parameter means, so there is nothing to
//                             enforce and nothing is claimed
//   "required"                the union of every offered tool
//   {"type":"function",...}    that one tool alone
//
// Returns an empty string when nothing should be constrained. Sets *err (and
// returns empty) when the request is malformed, so the caller can answer 400
// rather than generate something unconstrained.
std::string tool_choice_schema(const std::string& body, std::string* err) {
  const std::vector<std::string> tools = json_get_raw_array(body, "tools");
  if (tools.empty()) {
    return "";
  }
  std::string choice_raw = json_get_raw_value(body, "tool_choice");
  std::string mode = "auto";
  std::string only;
  if (!choice_raw.empty()) {
    if (choice_raw.front() == '"') {
      mode = json_get_string(body, "tool_choice");
    } else {
      // {"type":"function","function":{"name":"..."}}
      mode = "function";
      const std::string fn = json_get_raw_value(choice_raw, "function");
      only = json_get_string(fn.empty() ? choice_raw : fn, "name");
      if (only.empty()) {
        if (err) *err = "tool_choice names no function";
        return "";
      }
    }
  }
  if (mode == "none" || mode == "auto") {
    return "";
  }
  if (mode != "required" && mode != "function") {
    if (err) *err = "unsupported tool_choice '" + mode + "'";
    return "";
  }

  std::string branches;
  int used = 0;
  for (const std::string& tool : tools) {
    const std::string fn = json_get_raw_value(tool, "function");
    const std::string src = fn.empty() ? tool : fn;
    const std::string name = json_get_string(src, "name");
    if (name.empty()) {
      if (err) *err = "a tool entry has no function name";
      return "";
    }
    if (mode == "function" && name != only) {
      continue;
    }
    std::string params = json_get_raw_value(src, "parameters");
    if (params.empty()) {
      params = R"({"type":"object","properties":{}})";
    }
    if (used++) {
      branches += ",";
    }
    branches += R"({"type":"object","properties":{"name":{"enum":[")" + name +
                R"("]},"arguments":)" + params +
                R"(},"required":["name","arguments"]})";
  }
  if (used == 0) {
    if (err) *err = "tool_choice names a function that is not in tools: '" + only + "'";
    return "";
  }
  return R"({"anyOf":[)" + branches + "]}";
}

// Rewrites a grammar-constrained {"name":...,"arguments":...} reply into an
// OpenAI tool_calls message body. `text` is the model's output.
std::string tool_calls_json(const std::string& id, const std::string& text) {
  const std::string name = json_get_string(text, "name");
  std::string args = json_get_raw_value(text, "arguments");
  if (args.empty()) {
    args = "{}";
  }
  // `arguments` is a STRING in the OpenAI shape, not an object, so the JSON the
  // model produced is escaped into one.
  return R"("content":null,"tool_calls":[{"id":"call_)" + id + R"(","type":"function",)" +
         R"("function":{"name":")" + json_escape(name) + R"(","arguments":")" + json_escape(args) +
         R"("}}])";
}

bool cut_at_stop(std::string& text, const std::vector<std::string>& stops) {
  std::size_t cut = std::string::npos;
  for (const auto& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t at = text.find(stop);
    if (at != std::string::npos && at < cut) cut = at;
  }
  if (cut == std::string::npos) return false;
  text.erase(cut);
  return true;
}

// A per-request mailbox. The worker thread pushes deltas in; the connection
// thread pops them out and writes SSE frames (or accumulates the full text).
struct Pending {
  std::mutex mu;
  std::condition_variable cv;
  std::string pending_text;  // delta text not yet written by the connection
  std::string full_text;     // everything so far
  std::string finish_reason;
  std::string error;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  bool done = false;
};

// OpenAI's usage object. Clients and dashboards read token counts from here to
// bill, budget context, and decide when to compact a conversation; responses
// used to carry none at all, so callers had to re-tokenize to find out.
std::string usage_json(int prompt_tokens, int completion_tokens) {
  return ",\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
         ",\"completion_tokens\":" + std::to_string(completion_tokens) +
         ",\"total_tokens\":" + std::to_string(prompt_tokens + completion_tokens) + "}";
}

// Base64 decoder for data: image URLs. Hand-rolled, per policy; skips
// whitespace, stops at padding, and returns false on any other stray byte
// rather than quietly producing a corrupt image.
bool base64_decode(const std::string& in, std::string* out) {
  auto value = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  out->clear();
  out->reserve(in.size() * 3 / 4);
  int acc = 0;
  int bits = 0;
  for (const char ch : in) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    if (c == '=') break;
    const int v = value(c);
    if (v < 0) return false;
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(static_cast<char>((acc >> bits) & 0xFF));
    }
  }
  return true;
}

// One image found in an OpenAI-style message: either an inline data: URL or a
// local path (the latter is a convenience for same-host clients).
struct ImageRef {
  std::string data;  // decoded bytes when inline
  std::string path;  // filesystem path when not
  bool inline_data = false;
};

// Pulls image_url entries out of a message's content array. OpenAI vision bodies
// carry content as [{"type":"text",...},{"type":"image_url","image_url":{"url":...}}];
// a plain string content has no images and returns nothing.
std::vector<ImageRef> parse_images(const std::string& message_object) {
  std::vector<ImageRef> out;
  std::size_t pos = 0;
  const std::string key = "\"url\"";
  while ((pos = message_object.find(key, pos)) != std::string::npos) {
    const std::size_t colon = message_object.find(':', pos + key.size());
    if (colon == std::string::npos) break;
    std::size_t q = message_object.find('"', colon);
    if (q == std::string::npos) break;
    std::size_t end = q + 1;
    std::string url;
    while (end < message_object.size()) {
      if (message_object[end] == '\\' && end + 1 < message_object.size()) {
        url.push_back(message_object[end + 1]);
        end += 2;
        continue;
      }
      if (message_object[end] == '"') break;
      url.push_back(message_object[end]);
      ++end;
    }
    pos = end + 1;

    ImageRef ref;
    const std::string data_prefix = "data:";
    if (url.rfind(data_prefix, 0) == 0) {
      const std::size_t comma = url.find(',');
      if (comma == std::string::npos) continue;
      if (url.find(";base64", 0) == std::string::npos) continue;
      if (!base64_decode(url.substr(comma + 1), &ref.data)) continue;
      ref.inline_data = true;
    } else {
      ref.path = url;
    }
    out.push_back(std::move(ref));
  }
  return out;
}

// Extracts the text of a message whose content is either a plain string or an
// OpenAI content array (in which case the "text" parts are concatenated).
std::string message_text(const std::string& message_object) {
  const std::string raw = json_get_raw_value(message_object, "content");
  if (raw.empty() || raw.front() != '[') return json_get_string(message_object, "content");
  std::string text;
  std::size_t pos = 0;
  const std::string key = "\"text\"";
  while ((pos = message_object.find(key, pos)) != std::string::npos) {
    const std::size_t colon = message_object.find(':', pos + key.size());
    if (colon == std::string::npos) break;
    const std::size_t q = message_object.find('"', colon);
    if (q == std::string::npos) break;
    std::size_t end = q + 1;
    std::string part;
    while (end < message_object.size()) {
      if (message_object[end] == '\\' && end + 1 < message_object.size()) {
        // Keep the escape intact; json_get_string-style unescaping happens later.
        part.push_back(message_object[end]);
        part.push_back(message_object[end + 1]);
        end += 2;
        continue;
      }
      if (message_object[end] == '"') break;
      part.push_back(message_object[end]);
      ++end;
    }
    pos = end + 1;
    if (!text.empty()) text += "\n";
    text += part;
  }
  return text;
}

struct ChatMessage {
  std::string role;
  std::string content;
  std::string raw;  // the message object, so images can be pulled from it
};

// Extracts messages from an OpenAI "messages" array. Hand-rolled because the
// shared JSON helpers are scalar-only, and a chat body is the one place the
// server must walk an array of objects. Order is preserved.
std::vector<ChatMessage> parse_messages(const std::string& body) {
  std::vector<ChatMessage> out;
  const std::string arr = json_get_raw_value(body, "messages");
  if (arr.empty() || arr.front() != '[') return out;

  // Walk objects at depth 1, respecting strings and escapes.
  std::size_t i = 1;
  int depth = 0;
  std::size_t obj_start = std::string::npos;
  bool in_string = false;
  bool escaped = false;
  for (; i < arr.size(); ++i) {
    const char c = arr[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{') {
      if (depth == 0) obj_start = i;
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0 && obj_start != std::string::npos) {
        const std::string obj = arr.substr(obj_start, i - obj_start + 1);
        ChatMessage m;
        m.role = json_get_string(obj, "role");
        m.content = message_text(obj);
        m.raw = obj;
        out.push_back(std::move(m));
        obj_start = std::string::npos;
      }
    } else if (c == ']' && depth == 0) {
      break;
    }
  }
  return out;
}

// Renders a chat conversation into a single prompt string. The engine's chat
// templates take one user turn, so a multi-turn history is flattened with role
// labels ahead of the templated final turn: enough for the common
// system + alternating turns case without inventing a second template engine.
std::string chat_prompt_from_messages(const std::vector<ChatMessage>& messages,
                                      const std::string& chat_template, bool with_image) {
  std::string system_text;
  std::string history;
  std::string last_user;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const std::string& role = messages[i].role;
    const std::string& content = messages[i].content;
    if (content.empty()) continue;
    if (role == "system") {
      if (!system_text.empty()) system_text += "\n";
      system_text += content;
    } else if (i + 1 == messages.size() && role == "user") {
      last_user = content;
    } else if (role == "user") {
      history += "User: " + content + "\n";
    } else if (role == "assistant") {
      history += "Assistant: " + content + "\n";
    }
  }
  if (last_user.empty() && !messages.empty()) last_user = messages.back().content;

  std::string turn;
  // The vision splice looks for this placeholder in the templated prompt; without
  // it image_prompt::expand refuses (rather than silently dropping the picture).
  if (with_image) turn += "<|image|>\n";
  if (!system_text.empty()) turn += system_text + "\n\n";
  if (!history.empty()) turn += history + "\n";
  turn += last_user;
  return build_chat_prompt(chat_template, turn, /*tinyllama_plain_fallback=*/true);
}

std::string iso_created() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now).count());
}

std::string error_json(const std::string& message, const std::string& type) {
  return "{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"" + type + "\"}}";
}

// Constant-time-ish comparison so a wrong key cannot be recovered byte by byte
// from response timing. Lengths differing is already public (and unavoidable).
bool token_matches(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff |= static_cast<unsigned char>(a[i] ^ b[i]);
  }
  return diff == 0;
}

// "Authorization: Bearer <token>", tolerating the header's usual whitespace.
std::string bearer_token(const net::HttpRequest& req) {
  std::string h = req.header("authorization");
  const std::string prefix = "Bearer ";
  if (h.size() > prefix.size() &&
      std::equal(prefix.begin(), prefix.end(), h.begin(), [](char x, char y) {
        return std::tolower(static_cast<unsigned char>(x)) ==
               std::tolower(static_cast<unsigned char>(y));
      })) {
    return h.substr(prefix.size());
  }
  return std::string();
}

// Auth, routing, the refuse-to-expose gate and startup logging, shared by both
// serving backends. Only generation differs between them (continuous batching
// when the engine has a scheduler, serialized when it does not), so everything
// security-relevant lives here once rather than in two copies that can drift.
using CompletionFn = std::function<void(const net::HttpRequest&, net::HttpResponder&, bool chat)>;
using EmbeddingFn = std::function<void(const net::HttpRequest&, net::HttpResponder&)>;

void serve_routes(const HttpServeOptions& opts, const std::string& model_name, const char* mode,
                  const CompletionFn& completion, const EmbeddingFn& embeddings,
                  const std::function<void()>& run_forever) {
  if (opts.api_key.empty() && opts.host != "127.0.0.1" && opts.host != "localhost") {
    throw std::runtime_error(
        "[serve] refusing to bind " + opts.host +
        " without --api-key: that would expose an unauthenticated API to the network. "
        "Pass --api-key <token> (or CPI_API_KEY), or keep the default --host 127.0.0.1.");
  }

  net::HttpServer server;
  std::string start_error;
  const bool ok = server.start(
      opts.host, opts.port,
      [&](const net::HttpRequest& req, net::HttpResponder& res) {
        // /health stays open so a load balancer needs no credential; everything
        // under /v1 requires the bearer token when one is configured.
        if (req.path == "/health" || req.path == "/api/health") {
          res.send(200, "application/json",
                   "{\"status\":\"ok\",\"model\":\"" + json_escape(model_name) + "\"}");
          return;
        }
        if (!opts.api_key.empty() && !token_matches(bearer_token(req), opts.api_key)) {
          res.send(
              401, "application/json",
              error_json("missing or invalid Authorization bearer token", "invalid_request_error"),
              "WWW-Authenticate: Bearer\r\n");
          return;
        }
        if (req.path == "/v1/embeddings") {
          if (req.method != "POST") {
            res.send(405, "application/json", error_json("use POST", "invalid_request_error"));
            return;
          }
          embeddings(req, res);
          return;
        }
        if (req.path == "/v1/models") {
          res.send(200, "application/json",
                   "{\"object\":\"list\",\"data\":[{\"id\":\"" + json_escape(model_name) +
                       "\",\"object\":\"model\",\"owned_by\":\"cpi\"}]}");
          return;
        }
        if (req.path == "/v1/chat/completions" || req.path == "/v1/completions") {
          if (req.method != "POST") {
            res.send(405, "application/json", error_json("use POST", "invalid_request_error"));
            return;
          }
          completion(req, res, req.path == "/v1/chat/completions");
          return;
        }
        res.send(404, "application/json",
                 error_json("unknown route: " + req.path, "invalid_request_error"));
      },
      &start_error);

  if (!ok) throw std::runtime_error("[serve] " + start_error);
  std::fprintf(stderr, "[serve] listening on http://%s:%d (model=%s, auth=%s, mode=%s)\n",
               opts.host.c_str(), server.port(), model_name.c_str(),
               opts.api_key.empty() ? "off" : "bearer", mode);
  std::fprintf(stderr,
               "[serve] POST /v1/chat/completions, POST /v1/completions, GET /v1/models, "
               "GET /v1/embeddings, GET /health\n");
  run_forever();
  server.stop();
}

}  // namespace

void run_http_server(engine::BatchScheduler& sched, model::Tokenizer& tokenizer,
                     const HttpServeOptions& opts) {
  BatchDefaults defaults;
  defaults.stop_texts = opts.stop_texts;
  defaults.add_bos = opts.add_bos;
  defaults.max_new = opts.max_new;
  defaults.temp = opts.temp;
  defaults.top_k = opts.top_k;
  defaults.top_p = opts.top_p;
  defaults.repeat_penalty = opts.repeat_penalty;
  defaults.no_repeat_ngram = opts.no_repeat_ngram;

  std::mutex reg_mu;
  std::unordered_map<std::string, std::shared_ptr<Pending>> registry;
  std::atomic<std::uint64_t> next_id{1};

  BatchWorker worker(sched, tokenizer, defaults, [&](const BatchEvent& e) {
    std::shared_ptr<Pending> p;
    {
      std::lock_guard<std::mutex> lk(reg_mu);
      const auto it = registry.find(e.id);
      if (it == registry.end()) return;
      p = it->second;
    }
    {
      std::lock_guard<std::mutex> lk(p->mu);
      switch (e.type) {
        case BatchEvent::Type::Start:
          break;
        case BatchEvent::Type::Delta:
          p->pending_text += e.text;
          p->full_text += e.text;
          break;
        case BatchEvent::Type::Done:
          p->full_text = e.text;
          p->finish_reason = e.finish_reason;
          p->prompt_tokens = e.prompt_tokens;
          p->completion_tokens = e.generated;
          p->done = true;
          break;
        case BatchEvent::Type::Error:
          p->error = e.error;
          p->done = true;
          break;
      }
    }
    p->cv.notify_all();
  });

  const std::string model_name = opts.model_name;

  // Shared body for both completion routes: submit, then either stream SSE
  // frames or block for the final text. `chat` selects the response shape.
  const auto handle_completion = [&](const net::HttpRequest& req, net::HttpResponder& res,
                                     bool chat) {
    const std::string& body = req.body;
    std::string prompt;
    std::vector<ImageRef> images;
    if (chat) {
      const auto messages = parse_messages(body);
      if (messages.empty()) {
        res.send(400, "application/json",
                 error_json("'messages' must be a non-empty array", "invalid_request_error"));
        return;
      }
      for (const auto& m : messages) {
        auto found = parse_images(m.raw);
        images.insert(images.end(), found.begin(), found.end());
      }
      prompt = chat_prompt_from_messages(messages, opts.chat_template, !images.empty());
    } else {
      prompt = json_get_string(body, "prompt");
      if (prompt.empty()) {
        res.send(400, "application/json",
                 error_json("'prompt' is required", "invalid_request_error"));
        return;
      }
    }

    BatchOverrides ov;
    // max_tokens is the OpenAI name; max_new is CPI's own, accepted as an alias.
    ov.max_new = json_get_int(body, "max_tokens", json_get_int(body, "max_new", -1));
    ov.temp = json_get_float(body, "temperature", -1.0f);
    ov.top_p = json_get_float(body, "top_p", -1.0f);
    ov.top_k = json_get_int(body, "top_k", -1);
    ov.repeat_penalty =
        json_get_float(body, "repetition_penalty", json_get_float(body, "repeat_penalty", -1.0f));
    ov.no_repeat_ngram = json_get_int(body, "no_repeat_ngram", -1);
    ov.stop_texts = json_get_string_array(body, "stop");
    // Suppress EOS until this many tokens are generated. The scheduler has
    // implemented it per row all along; this is the first path that fills it in.
    ov.min_new = std::max(0, json_get_int(body, "min_new", 0));
    // response_format: {"type":"json_schema","json_schema":{...}} and the plain
    // CPI form {"json_schema":{...}} both reach the same grammar path.
    ov.json_schema = json_get_raw_value(body, "json_schema");
    if (ov.json_schema.empty()) {
      const std::string rf = json_get_raw_value(body, "response_format");
      if (!rf.empty()) ov.json_schema = json_get_raw_value(rf, "json_schema");
    }
    ov.json_schema = unwrap_json_schema(ov.json_schema);

    // tools/tool_choice compile to the same grammar the schema path uses, so a
    // required tool call is enforced by the sampler rather than hoped for. A
    // malformed tools block is a 400: the alternative is generating something
    // unconstrained and letting the caller discover it by parsing.
    bool tool_mode = false;
    {
      std::string tool_err;
      const std::string tool_schema = tool_choice_schema(body, &tool_err);
      if (!tool_err.empty()) {
        res.send(400, "application/json", error_json(tool_err, "invalid_request_error"));
        return;
      }
      if (!tool_schema.empty()) {
        ov.json_schema = tool_schema;
        tool_mode = true;
      }
    }

    const bool stream = json_get_bool(body, "stream", false);
    const std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(next_id.fetch_add(1));

    // Vision requests take a different engine entry point than the batch
    // scheduler, so they run as an exclusive task on the worker thread. Only the
    // first image is used: the towers this serves splice one image span.
    if (!images.empty()) {
      if (!opts.multimodal) {
        res.send(400, "application/json",
                 error_json("this model has no vision tower; remove the image or load a "
                            "multimodal model",
                            "invalid_request_error"));
        return;
      }
      std::string image_path = images.front().path;
      std::string temp_path;
      if (images.front().inline_data) {
        // The image pipeline reads a file (the PNG decoder is path-based), so an
        // inline data: URL is staged to a temp file and removed afterwards.
        std::error_code ec;
        const auto dir = std::filesystem::temp_directory_path(ec);
        temp_path = (dir / ("cpi_serve_" + id + ".png")).string();
        std::ofstream f(temp_path, std::ios::binary);
        f.write(images.front().data.data(),
                static_cast<std::streamsize>(images.front().data.size()));
        f.close();
        image_path = temp_path;
      }
      if (image_path.empty()) {
        res.send(400, "application/json",
                 error_json("could not read the supplied image", "invalid_request_error"));
        return;
      }

      const int max_new = ov.max_new >= 0 ? ov.max_new : defaults.max_new;
      const float temp = ov.temp >= 0.0f ? ov.temp : defaults.temp;
      std::string text;
      std::string vision_error;
      const std::vector<int> base = tokenizer.encode(prompt, defaults.add_bos);
      worker.run_exclusive([&]() {
        try {
          const std::vector<int> outs = opts.multimodal(base, image_path, max_new, temp, nullptr);
          text = app::main_helpers::sanitize_stream_text(tokenizer.decode(outs));
        } catch (const std::exception& e) {
          vision_error = e.what();
        }
      });
      if (!temp_path.empty()) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
      }
      if (!vision_error.empty()) {
        res.send(500, "application/json", error_json(vision_error, "server_error"));
        return;
      }
      const std::string created_v = iso_created();
      if (stream) {
        // Vision generation is not streamed by the engine, so the whole answer
        // arrives as one content chunk followed by the terminator.
        res.begin_sse();
        res.sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                created_v + ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"" +
                json_escape(text) + "\"},\"finish_reason\":null}]}");
        res.sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                created_v + ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
        res.sse("[DONE]");
        return;
      }
      res.send(200, "application/json",
               "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + created_v +
                   ",\"model\":\"" + json_escape(model_name) +
                   "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                   "\"content\":\"" +
                   json_escape(text) + "\"},\"finish_reason\":\"stop\"}]}");
      return;
    }
    auto pending = std::make_shared<Pending>();
    {
      std::lock_guard<std::mutex> lk(reg_mu);
      registry[id] = pending;
    }
    const auto unregister = [&]() {
      std::lock_guard<std::mutex> lk(reg_mu);
      registry.erase(id);
    };

    std::string err;
    if (!worker.submit(id, prompt, ov, &err)) {
      unregister();
      res.send(400, "application/json", error_json(err, "invalid_request_error"));
      return;
    }

    const std::string created = iso_created();
    if (stream) {
      res.begin_sse();
      // The role-only first chunk clients expect before any content.
      if (chat) {
        res.sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                created + ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                "\"finish_reason\":null}]}");
      }
      std::string finish;
      // A streaming server cannot emit text that might turn out to be the start of
      // a stop string, so the last (longest stop - 1) bytes are withheld until the
      // next chunk decides them. Matching emitted + chunk is not enough: by the
      // time the whole stop is visible, its first bytes have already gone out. A
      // request stopping at "11 12 13" streamed "11 12 " before cutting.
      std::string held;
      std::size_t max_stop = 0;
      for (const auto& st : ov.stop_texts) max_stop = std::max(max_stop, st.size());
      bool stop_cut = false;
      while (true) {
        std::string chunk;
        bool finished = false;
        {
          std::unique_lock<std::mutex> lk(pending->mu);
          pending->cv.wait_for(lk, std::chrono::milliseconds(200),
                               [&]() { return !pending->pending_text.empty() || pending->done; });
          chunk.swap(pending->pending_text);
          finished = pending->done;
          finish = pending->finish_reason;
          if (!pending->error.empty() && chunk.empty()) {
            const std::string e = pending->error;
            lk.unlock();
            res.sse(error_json(e, "server_error"));
            res.sse("[DONE]");
            unregister();
            return;
          }
        }
        if (!ov.stop_texts.empty() && (!chunk.empty() || finished)) {
          held += chunk;
          std::string candidate = held;
          if (cut_at_stop(candidate, ov.stop_texts)) {
            chunk = candidate;  // everything before the stop, nothing after
            held.clear();
            stop_cut = true;
            finished = true;
          } else if (finished) {
            chunk.swap(held);  // no stop and no more tokens: release the tail
          } else {
            // Any stop ending in the new data starts at most (len - 1) bytes
            // before it, so retaining that much is enough to catch every split.
            const std::size_t keep = std::min(held.size(), max_stop > 0 ? max_stop - 1 : 0);
            chunk = held.substr(0, held.size() - keep);
            held.erase(0, held.size() - keep);
          }
        }
        if (!chunk.empty()) {
          const std::string payload =
              chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                      json_escape(chunk) + "\"},\"finish_reason\":null}]}")
                   : ("{\"id\":\"" + id +
                      "\",\"object\":\"text_completion\",\"created\":" + created + ",\"model\":\"" +
                      json_escape(model_name) + "\",\"choices\":[{\"index\":0,\"text\":\"" +
                      json_escape(chunk) + "\",\"finish_reason\":null}]}");
          // A failed write means the client hung up: cancel so its KV blocks go
          // back to the pool instead of decoding into a closed socket.
          if (!res.sse(payload)) {
            worker.cancel(id);
            unregister();
            return;
          }
        }
        if (finished) break;
      }
      const std::string reason =
          (stop_cut || finish.empty()) ? "stop" : (finish == "length" ? "length" : "stop");
      res.sse(chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + reason +
                      "\"}]}")
                   : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"" + reason +
                      "\"}]}"));
      res.sse("[DONE]");
      unregister();
      return;
    }

    // Non-streaming: wait for the worker to finish this request.
    std::string text;
    std::string finish;
    std::string error;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    {
      std::unique_lock<std::mutex> lk(pending->mu);
      pending->cv.wait(lk, [&]() { return pending->done; });
      text = pending->full_text;
      finish = pending->finish_reason;
      error = pending->error;
      prompt_tokens = pending->prompt_tokens;
      completion_tokens = pending->completion_tokens;
    }
    unregister();
    if (!error.empty()) {
      res.send(500, "application/json", error_json(error, "server_error"));
      return;
    }
    // Multi-token stop strings are matched here, on the decoded text, which is the
    // only place they can be seen. A cut also decides the reason: the request
    // ended because the caller asked it to, whatever the worker reported.
    const bool cut = cut_at_stop(text, ov.stop_texts);
    const std::string reason = (!cut && finish == "length") ? "length" : "stop";
    const std::string usage = usage_json(prompt_tokens, completion_tokens);
    // A tool call comes back as tool_calls with finish_reason tool_calls, which is
    // what an OpenAI client dispatches on; the raw JSON is never handed over as
    // message content.
    // A truncated generation is not a tool call. Reporting one anyway produced a
    // call with an empty name and finish_reason "tool_calls", which a caller would
    // dispatch on: the worst shape this can fail in, since it looks successful.
    // When the reply was cut off, or no name can be read out of it, the raw text
    // and the real finish_reason go back instead.
    const bool tool_ok =
        tool_mode && reason != "length" && !json_get_string(text, "name").empty();
    const std::string message_body =
        tool_ok ? tool_calls_json(id, text) : ("\"content\":\"" + json_escape(text) + "\"");
    const std::string chat_reason = tool_ok ? "tool_calls" : reason;
    const std::string payload =
        chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"," +
                message_body + "},\"finish_reason\":\"" + chat_reason + "\"}]" + usage + "}")
             : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"text\":\"" + json_escape(text) +
                "\",\"finish_reason\":\"" + reason + "\"}]" + usage + "}");
    res.send(200, "application/json", payload);
  };

  // Embedding model (optional). Loaded once here so /v1/embeddings does not pay
  // a per-request init, and guarded by its own mutex because the batching worker
  // owns the generation engine on another thread.
  std::mutex embed_mu;
  bool embed_ready = false;
  std::string embed_error;
#if defined(CPI_SERVE_HAS_EMBEDDER)
  ServeEmbedder embedder;
  model::WordPieceTokenizer embed_tokenizer;
  if (!opts.embed_model_dir.empty()) {
    try {
      embedder.initialize(opts.embed_model_dir);
      const auto& ecfg = embedder.config();
      embed_tokenizer.load(opts.embed_model_dir, ecfg.lowercase, ecfg.strip_accents);
      embed_ready = true;
      std::fprintf(stderr, "[serve] embeddings ready dim=%d model=%s\n", embedder.dim(),
                   opts.embed_model_dir.c_str());
    } catch (const std::exception& e) {
      embed_error = e.what();
      std::fprintf(stderr, "[serve] embedding model failed to load: %s\n", embed_error.c_str());
    }
  }
#else
  if (!opts.embed_model_dir.empty()) {
    embed_error = "this build has no GPU backend for embeddings";
  }
#endif

  const auto handle_embeddings = [&](const net::HttpRequest& req, net::HttpResponder& res) {
    if (!embed_ready) {
      const std::string why =
          embed_error.empty() ? std::string("no embedding model loaded (pass --embed-model <dir>)")
                              : embed_error;
      res.send(503, "application/json", error_json(why, "server_error"));
      return;
    }
#if defined(CPI_SERVE_HAS_EMBEDDER)
    std::vector<std::string> inputs = json_get_string_array(req.body, "input");
    if (inputs.empty()) {
      const std::string one = json_get_string(req.body, "input");
      if (!one.empty()) inputs.push_back(one);
    }
    if (inputs.empty()) {
      res.send(400, "application/json",
               error_json("'input' must be a string or array of strings", "invalid_request_error"));
      return;
    }
    const std::string input_type = json_get_string(req.body, "input_type");
    std::ostringstream data;
    int total_tokens = 0;
    int dim = 0;
    try {
      std::lock_guard<std::mutex> lk(embed_mu);
      const auto& ecfg = embedder.config();
      const std::string prefix = (input_type == "query") ? ecfg.query_prefix : ecfg.doc_prefix;
      dim = embedder.dim();
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<int> ids =
            embed_tokenizer.encode(prefix + inputs[i], embedder.max_tokens());
        const std::vector<float> v = embedder.embed(ids);
        total_tokens += static_cast<int>(ids.size());
        if (i) data << ",";
        data << "{\"object\":\"embedding\",\"index\":" << i << ",\"embedding\":[";
        for (std::size_t d = 0; d < v.size(); ++d) {
          if (d) data << ",";
          char buf[24];
          std::snprintf(buf, sizeof(buf), "%.7g", v[d]);
          data << buf;
        }
        data << "]}";
      }
    } catch (const std::exception& e) {
      res.send(500, "application/json", error_json(e.what(), "server_error"));
      return;
    }
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[" << data.str() << "],\"model\":\""
        << json_escape(opts.embed_model_dir) << "\",\"usage\":{\"prompt_tokens\":" << total_tokens
        << ",\"total_tokens\":" << total_tokens << "},\"dim\":" << dim << "}";
    res.send(200, "application/json", out.str());
#endif
  };

  serve_routes(opts, model_name, "batched", handle_completion, handle_embeddings, [&]() {
    // The worker owns the engine and runs on this thread until the process is
    // stopped; HTTP threads only enqueue work.
    worker.run();
  });
}

void run_http_server_serial(GenerateStreamFn generate, model::Tokenizer& tokenizer,
                            const HttpServeOptions& opts) {
  const std::string model_name = opts.model_name;
  // Built once per server, not once per request. This decodes the whole
  // vocabulary (~26 ms at Gemma's 262144 tokens) and depends only on the
  // tokenizer, so every schema request used to repeat identical work before
  // generating its first token.
  const std::shared_ptr<const grammar::TokenTables> token_tables =
      grammar::build_token_tables(tokenizer.token_pieces());
  // One request at a time: these engines have no paged pool to interleave with,
  // so concurrency is queueing rather than batching. Stated in the startup log
  // so nobody benchmarks this expecting the batched path's numbers.
  std::mutex engine_mu;

  const auto handle_completion = [&](const net::HttpRequest& req, net::HttpResponder& res,
                                     bool chat) {
    const std::string& body = req.body;
    std::string prompt;
    std::vector<ImageRef> images;
    if (chat) {
      const auto messages = parse_messages(body);
      if (messages.empty()) {
        res.send(400, "application/json",
                 error_json("'messages' must be a non-empty array", "invalid_request_error"));
        return;
      }
      for (const auto& m : messages) {
        auto found = parse_images(m.raw);
        images.insert(images.end(), found.begin(), found.end());
      }
      prompt = chat_prompt_from_messages(messages, opts.chat_template, !images.empty());
    } else {
      prompt = json_get_string(body, "prompt");
      if (prompt.empty()) {
        res.send(400, "application/json",
                 error_json("'prompt' is required", "invalid_request_error"));
        return;
      }
    }

    const int max_new =
        std::max(1, json_get_int(body, "max_tokens", json_get_int(body, "max_new", opts.max_new)));
    const float temp = std::max(0.0f, json_get_float(body, "temperature", opts.temp));
    const bool stream = json_get_bool(body, "stream", false);

    // Schema-constrained decoding, same request forms the batched path accepts.
    // This transport used to pass no constraints at all, so a client that asked
    // for JSON matching a schema silently got unconstrained prose back.
    std::string json_schema = json_get_raw_value(body, "json_schema");
    if (json_schema.empty()) {
      const std::string rf = json_get_raw_value(body, "response_format");
      if (!rf.empty()) json_schema = json_get_raw_value(rf, "json_schema");
    }
    json_schema = unwrap_json_schema(json_schema);
    // Per-request stops on top of the server's template defaults. Without these
    // the model emits its end-of-turn marker and keeps going, inventing the next
    // speaker's turn until it hits max_tokens.
    //
    // Split by how they can actually be detected. A marker like <|eot_id|> is a
    // single special token that DECODES TO NOTHING, so searching the decoded text
    // for its spelling never matches; it has to be caught by token id. Ordinary
    // multi-token stop strings only exist in the text, so they are matched there.
    std::vector<std::string> stop_texts;
    std::vector<int> stop_ids;
    for (const auto& s : opts.stop_texts) {
      if (s.empty()) continue;
      const auto t = tokenizer.encode(s, /*add_bos=*/false);
      if (t.size() == 1) {
        stop_ids.push_back(t[0]);
      } else {
        stop_texts.push_back(s);
      }
    }
    for (const auto& s : json_get_string_array(body, "stop")) {
      if (s.empty()) continue;
      const auto t = tokenizer.encode(s, /*add_bos=*/false);
      if (t.size() == 1) {
        stop_ids.push_back(t[0]);
      } else {
        stop_texts.push_back(s);
      }
    }
    const std::string id =
        (chat ? "chatcmpl-" : "cmpl-") +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() % 100000);
    const std::string created = iso_created();

    std::string image_path;
    std::string temp_path;
    if (!images.empty()) {
      if (!opts.multimodal) {
        res.send(400, "application/json",
                 error_json("this model has no vision tower; remove the image or load a "
                            "multimodal model",
                            "invalid_request_error"));
        return;
      }
      image_path = images.front().path;
      if (images.front().inline_data) {
        std::error_code ec;
        const auto dir = std::filesystem::temp_directory_path(ec);
        temp_path = (dir / ("cpi_serve_" + id + ".png")).string();
        std::ofstream f(temp_path, std::ios::binary);
        f.write(images.front().data.data(),
                static_cast<std::streamsize>(images.front().data.size()));
        f.close();
        image_path = temp_path;
      }
      if (image_path.empty()) {
        res.send(400, "application/json",
                 error_json("could not read the supplied image", "invalid_request_error"));
        return;
      }
    }

    if (stream) res.begin_sse();
    if (stream && chat) {
      res.sse("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + created +
              ",\"model\":\"" + json_escape(model_name) +
              "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
              "\"finish_reason\":null}]}");
    }

    // Incremental detokenization, matching the batched path: decode the whole
    // history each step and emit the difference, so multi-byte pieces are never
    // split mid-character.
    std::vector<int> ids;
    std::string prev_text;
    std::string full_text;
    bool client_gone = false;
    const auto emit_delta = [&](const std::string& delta) {
      if (delta.empty()) return true;
      full_text += delta;
      if (!stream) return true;
      const std::string payload =
          chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                  created + ",\"model\":\"" + json_escape(model_name) +
                  "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + json_escape(delta) +
                  "\"},\"finish_reason\":null}]}")
               : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created +
                  ",\"model\":\"" + json_escape(model_name) +
                  "\",\"choices\":[{\"index\":0,\"text\":\"" + json_escape(delta) +
                  "\",\"finish_reason\":null}]}");
      return res.sse(payload);
    };

    std::string error;
    bool hit_stop = false;
    int prompt_token_count = 0;
    {
      std::lock_guard<std::mutex> lk(engine_mu);
      try {
        const std::vector<int> base = tokenizer.encode(prompt, opts.add_bos);
        prompt_token_count = static_cast<int>(base.size());
        if (!image_path.empty()) {
          // Vision generation is not incremental in these engines; one answer.
          const std::vector<int> outs = opts.multimodal(base, image_path, max_new, temp, nullptr);
          emit_delta(app::main_helpers::sanitize_stream_text(tokenizer.decode(outs)));
        } else {
          // Build the grammar here rather than in the caller: the sampler must
          // outlive the generate call, and every serial engine wants the same one.
          std::unique_ptr<grammar::GrammarSampler> sampler;
          engine::GenerationConstraints constraints;
          // Both of these reach the stdin transport (main_modes) and the batch
          // scheduler implements min_new per row, but neither was ever parsed
          // here, so over HTTP the fields were unreachable: a request could ask
          // for them and the server would quietly generate as if it had not.
          constraints.min_new_tokens = std::max(0, json_get_int(body, "min_new", 0));
          constraints.seed = json_get_int(body, "seed", -1);
          constraints.stop_token_ids = &stop_ids;
          if (!json_schema.empty()) {
            try {
              grammar::Grammar g =
                  grammar::Grammar::parse(grammar::json_schema_to_grammar(json_schema));
              sampler = std::make_unique<grammar::GrammarSampler>(
                  std::move(g), tokenizer.token_pieces(), tokenizer.eos_id(), token_tables);
              constraints.grammar = sampler.get();
            } catch (const std::exception& e) {
              error = std::string("invalid json_schema: ") + e.what();
            }
          }
          if (error.empty()) {
            generate(
                base, max_new, temp,
                [&](int token) {
                  if (std::find(stop_ids.begin(), stop_ids.end(), token) != stop_ids.end()) {
                    hit_stop = true;
                    return false;  // never emit the marker itself
                  }
                  ids.push_back(token);
                  const std::string decoded =
                      app::main_helpers::sanitize_stream_text(tokenizer.decode(ids));
                  if (decoded.size() > prev_text.size()) {
                    std::string delta = decoded.substr(prev_text.size());
                    // Stop-text check against the whole text so far: a marker split
                    // across two decode steps still matches, and the part of the
                    // delta before it is still emitted.
                    const std::string candidate = prev_text + delta;
                    std::size_t cut = std::string::npos;
                    for (const auto& s : stop_texts) {
                      if (s.empty()) continue;
                      const std::size_t at = candidate.find(
                          s, prev_text.size() > s.size() ? prev_text.size() - s.size() : 0);
                      if (at != std::string::npos && at < cut) cut = at;
                    }
                    if (cut != std::string::npos) {
                      if (cut > prev_text.size()) {
                        emit_delta(candidate.substr(prev_text.size(), cut - prev_text.size()));
                      }
                      hit_stop = true;
                      return false;
                    }
                    prev_text = decoded;
                    if (!emit_delta(delta)) {
                      client_gone = true;
                      return false;  // stop generating for a client that hung up
                    }
                  }
                  return true;
                },
                // Always passed. A null grammar is fine, and constraints also
                // carry seed and min_new_tokens, so gating on the sampler meant
                // those two silently did nothing unless the same request also
                // asked for a json_schema. main_modes already does it this way.
                &constraints);
          }
        }
      } catch (const std::exception& e) {
        error = e.what();
      }
    }
    if (!temp_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(temp_path, ec);
    }
    if (client_gone) return;

    if (!error.empty()) {
      if (stream) {
        res.sse(error_json(error, "server_error"));
        res.sse("[DONE]");
      } else {
        res.send(500, "application/json", error_json(error, "server_error"));
      }
      return;
    }
    const std::string reason =
        (!hit_stop && static_cast<int>(ids.size()) >= max_new) ? "length" : "stop";
    if (stream) {
      res.sse(chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + reason +
                      "\"}]}")
                   : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"" + reason +
                      "\"}]}"));
      res.sse("[DONE]");
      return;
    }
    const std::string usage = usage_json(prompt_token_count, static_cast<int>(ids.size()));
    res.send(
        200, "application/json",
        chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                "\"content\":\"" +
                json_escape(full_text) + "\"},\"finish_reason\":\"" + reason + "\"}]" + usage + "}")
             : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"text\":\"" + json_escape(full_text) +
                "\",\"finish_reason\":\"" + reason + "\"}]" + usage + "}"));
  };

  // No embedder on this path (it would need its own model load); the route
  // reports why rather than 404ing.
  const auto handle_embeddings = [&](const net::HttpRequest&, net::HttpResponder& res) {
    res.send(503, "application/json",
             error_json("embeddings are served by the batched backend; this model runs on the "
                        "op-plan engine",
                        "server_error"));
  };

  std::mutex done_mu;
  std::condition_variable done_cv;
  serve_routes(opts, model_name, "serial", handle_completion, handle_embeddings, [&]() {
    // Nothing to pump: requests run on their own connection threads under
    // engine_mu. Park until the process is stopped.
    std::unique_lock<std::mutex> lk(done_mu);
    done_cv.wait(lk, []() { return false; });
  });
}

}  // namespace app::main_modes
