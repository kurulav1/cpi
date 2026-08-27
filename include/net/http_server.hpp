#pragma once

// A small hand-rolled HTTP/1.1 server (no third-party dependency, per CPI policy).
//
// Enough HTTP for an inference API and no more: request line + headers + a
// Content-Length body, fixed responses, and Server-Sent Events for token
// streaming. Thread-per-connection, which suits a handful of concurrent chat
// streams; the engine work behind them is serialized by the batching worker
// anyway, so the accept path is never the bottleneck.
//
// Winsock on Windows, BSD sockets elsewhere, behind one interface.
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace net {

struct HttpRequest {
  std::string method;
  std::string path;   // path only, query stripped
  std::string query;  // raw query string ("" when absent)
  std::string body;
  std::unordered_map<std::string, std::string> headers;  // lowercased names

  std::string header(const std::string& lowercase_name) const {
    const auto it = headers.find(lowercase_name);
    return it == headers.end() ? std::string() : it->second;
  }
};

// Per-connection writer. A handler either sends one complete response, or opens
// an SSE stream and pushes events until the client leaves.
class HttpResponder {
public:
  explicit HttpResponder(std::intptr_t socket);
  ~HttpResponder();

  HttpResponder(const HttpResponder&) = delete;
  HttpResponder& operator=(const HttpResponder&) = delete;

  // One-shot response. `extra_headers` must already be CRLF-terminated lines.
  void send(int status, const std::string& content_type, const std::string& body,
            const std::string& extra_headers = "");

  // Opens a text/event-stream response. The stream ends when the handler returns.
  void begin_sse();
  // Writes one "data: <payload>\n\n" frame. Returns false once the peer is gone,
  // which is how a disconnected client cancels its generation.
  bool sse(const std::string& payload);

  bool alive() const {
    return alive_;
  }
  bool sse_open() const {
    return sse_open_;
  }

private:
  bool write_all(const char* data, std::size_t len);

  std::intptr_t sock_;
  bool alive_ = true;
  bool sse_open_ = false;
  bool responded_ = false;
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponder&)>;

class HttpServer {
public:
  HttpServer();
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds and starts the accept loop on a background thread. `host` is an IPv4
  // literal ("127.0.0.1" to stay local, "0.0.0.0" to expose). Port 0 asks the OS
  // for a free port, readable afterwards through port(). Returns false with
  // *error set when the bind fails (the common one: port already in use).
  bool start(const std::string& host, int port, HttpHandler handler, std::string* error);

  // Stops accepting and waits for the accept thread. In-flight connections are
  // closed by the OS when the process exits; this is a serve-until-killed server.
  void stop();

  int port() const {
    return port_;
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int port_ = 0;
};

// Percent-decoding for query values and path segments.
std::string url_decode(const std::string& s);

}  // namespace net
