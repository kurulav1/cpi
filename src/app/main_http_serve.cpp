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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "app/batch_worker.hpp"
#include "app/main_helpers.hpp"
#include "app/main_modes.hpp"
#include "engine/batch_scheduler.hpp"
#include "model/tokenizer.hpp"
#include "model/wordpiece_tokenizer.hpp"
#include "net/http_server.hpp"

// The embedding backend, chosen the same way cpi_embed chooses it: CUDA when
// present, else Metal. Without either, /v1/embeddings reports unavailable
// rather than failing the build (a CPU-only cpi still serves chat).
#if CPI_HAS_CUDA
#  include "engine/bert_embedder.hpp"
#  define CPI_SERVE_HAS_EMBEDDER 1
using ServeEmbedder = engine::BertEmbedder;
#elif defined(CPI_ENABLE_METAL)
#  include "engine/metal_bert_embedder.hpp"
#  define CPI_SERVE_HAS_EMBEDDER 1
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
using app::main_helpers::json_get_string_array;

namespace {

// A per-request mailbox. The worker thread pushes deltas in; the connection
// thread pops them out and writes SSE frames (or accumulates the full text).
struct Pending {
  std::mutex mu;
  std::condition_variable cv;
  std::string pending_text;   // delta text not yet written by the connection
  std::string full_text;      // everything so far
  std::string finish_reason;
  std::string error;
  bool done = false;
};

// Extracts message contents from an OpenAI "messages" array. Hand-rolled because
// the shared JSON helpers are scalar-only, and a chat body is the one place the
// server must walk an array of objects. Returns role/content pairs in order.
std::vector<std::pair<std::string, std::string>> parse_messages(const std::string& body) {
  std::vector<std::pair<std::string, std::string>> out;
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
        out.emplace_back(json_get_string(obj, "role"), json_get_string(obj, "content"));
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
std::string chat_prompt_from_messages(
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& chat_template) {
  std::string system_text;
  std::string history;
  std::string last_user;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const std::string& role = messages[i].first;
    const std::string& content = messages[i].second;
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
  if (last_user.empty() && !messages.empty()) last_user = messages.back().second;

  std::string turn;
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
      std::equal(prefix.begin(), prefix.end(), h.begin(),
                 [](char x, char y) { return std::tolower(static_cast<unsigned char>(x)) ==
                                             std::tolower(static_cast<unsigned char>(y)); })) {
    return h.substr(prefix.size());
  }
  return std::string();
}

}  // namespace

void run_http_server(engine::BatchScheduler& sched, model::Tokenizer& tokenizer,
                     const HttpServeOptions& opts) {
  // Refuse the combination that silently publishes an unauthenticated model to
  // the network. Loopback without a key stays fine (that is the dev default),
  // and an explicit key makes any bind address the operator's choice.
  if (opts.api_key.empty() && opts.host != "127.0.0.1" && opts.host != "localhost") {
    throw std::runtime_error(
        "[serve] refusing to bind " + opts.host +
        " without --api-key: that would expose an unauthenticated API to the network. "
        "Pass --api-key <token> (or CPI_API_KEY), or keep the default --host 127.0.0.1.");
  }

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
    if (chat) {
      const auto messages = parse_messages(body);
      if (messages.empty()) {
        res.send(400, "application/json",
                 error_json("'messages' must be a non-empty array", "invalid_request_error"));
        return;
      }
      prompt = chat_prompt_from_messages(messages, opts.chat_template);
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
    ov.repeat_penalty = json_get_float(body, "repetition_penalty",
                                       json_get_float(body, "repeat_penalty", -1.0f));
    ov.no_repeat_ngram = json_get_int(body, "no_repeat_ngram", -1);
    ov.stop_texts = json_get_string_array(body, "stop");
    // response_format: {"type":"json_schema","json_schema":{...}} and the plain
    // CPI form {"json_schema":{...}} both reach the same grammar path.
    ov.json_schema = json_get_raw_value(body, "json_schema");
    if (ov.json_schema.empty()) {
      const std::string rf = json_get_raw_value(body, "response_format");
      if (!rf.empty()) ov.json_schema = json_get_raw_value(rf, "json_schema");
    }

    const bool stream = json_get_bool(body, "stream", false);
    const std::string id =
        (chat ? "chatcmpl-" : "cmpl-") + std::to_string(next_id.fetch_add(1));
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
      while (true) {
        std::string chunk;
        bool finished = false;
        {
          std::unique_lock<std::mutex> lk(pending->mu);
          pending->cv.wait_for(lk, std::chrono::milliseconds(200), [&]() {
            return !pending->pending_text.empty() || pending->done;
          });
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
        if (!chunk.empty()) {
          const std::string payload =
              chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                      json_escape(chunk) + "\"},\"finish_reason\":null}]}")
                   : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" +
                      created + ",\"model\":\"" + json_escape(model_name) +
                      "\",\"choices\":[{\"index\":0,\"text\":\"" + json_escape(chunk) +
                      "\",\"finish_reason\":null}]}");
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
      const std::string reason = finish.empty() ? "stop" : (finish == "length" ? "length" : "stop");
      res.sse(chat ? ("{\"id\":\"" + id +
                      "\",\"object\":\"chat.completion.chunk\",\"created\":" + created +
                      ",\"model\":\"" + json_escape(model_name) +
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
    {
      std::unique_lock<std::mutex> lk(pending->mu);
      pending->cv.wait(lk, [&]() { return pending->done; });
      text = pending->full_text;
      finish = pending->finish_reason;
      error = pending->error;
    }
    unregister();
    if (!error.empty()) {
      res.send(500, "application/json", error_json(error, "server_error"));
      return;
    }
    const std::string reason = finish == "length" ? "length" : "stop";
    const std::string payload =
        chat ? ("{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) +
                "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
                json_escape(text) + "\"},\"finish_reason\":\"" + reason + "\"}]}")
             : ("{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created +
                ",\"model\":\"" + json_escape(model_name) + "\",\"choices\":[{\"index\":0,\"text\":\"" +
                json_escape(text) + "\",\"finish_reason\":\"" + reason + "\"}]}");
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
          embed_error.empty()
              ? std::string("no embedding model loaded (pass --embed-model <dir>)")
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
          res.send(401, "application/json",
                   error_json("missing or invalid Authorization bearer token",
                              "invalid_request_error"),
                   "WWW-Authenticate: Bearer\r\n");
          return;
        }
        if (req.path == "/v1/embeddings") {
          if (req.method != "POST") {
            res.send(405, "application/json", error_json("use POST", "invalid_request_error"));
            return;
          }
          handle_embeddings(req, res);
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
          handle_completion(req, res, req.path == "/v1/chat/completions");
          return;
        }
        res.send(404, "application/json",
                 error_json("unknown route: " + req.path, "invalid_request_error"));
      },
      &start_error);

  if (!ok) {
    throw std::runtime_error("[serve] " + start_error);
  }
  std::fprintf(stderr, "[serve] listening on http://%s:%d (model=%s, auth=%s)\n", opts.host.c_str(),
               server.port(), model_name.c_str(), opts.api_key.empty() ? "off" : "bearer");
  std::fprintf(stderr,
               "[serve] POST /v1/chat/completions, POST /v1/completions, GET /v1/models, "
               "GET /health\n");

  // The worker owns the engine and runs on this thread until the process is
  // stopped; HTTP threads only enqueue work.
  worker.run();
  server.stop();
}

}  // namespace app::main_modes
