// Hand-rolled HTTP/1.1 server. See http_server.hpp for the scope rationale.
#include "net/http_server.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#  define CPI_CLOSESOCKET closesocket
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
using socket_t = int;
static constexpr socket_t kInvalidSocket = -1;
#  define CPI_CLOSESOCKET ::close
#endif

namespace net {
namespace {

// Winsock needs a process-wide startup; refcounted so several servers (or a
// server plus a future client) can coexist without one's teardown breaking the
// other. A no-op everywhere else.
struct SocketSubsystem {
  SocketSubsystem() {
#if defined(_WIN32)
    WSADATA wsa;
    ok = WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#endif
  }
  ~SocketSubsystem() {
#if defined(_WIN32)
    if (ok) WSACleanup();
#endif
  }
  bool ok = true;
};

SocketSubsystem& socket_subsystem() {
  static SocketSubsystem instance;
  return instance;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
  return s.substr(b, e - b);
}

const char* status_text(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Payload Too Large";
    case 500:
      return "Internal Server Error";
    case 503:
      return "Service Unavailable";
    default:
      return "OK";
  }
}

// A request body big enough to be a mistake or an attack. Long chat histories
// are still comfortably under this.
constexpr std::size_t kMaxBody = 64u << 20;
constexpr std::size_t kMaxHeaders = 64u << 10;

}  // namespace

std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
        std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
      out.push_back(static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16)));
      i += 2;
    } else if (s[i] == '+') {
      out.push_back(' ');
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

HttpResponder::HttpResponder(std::intptr_t socket) : sock_(socket) {}

HttpResponder::~HttpResponder() = default;

bool HttpResponder::write_all(const char* data, std::size_t len) {
  const auto s = static_cast<socket_t>(sock_);
  std::size_t sent = 0;
  while (sent < len) {
    const int n = ::send(s, data + sent, static_cast<int>(len - sent), 0);
    if (n <= 0) {
      alive_ = false;
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

void HttpResponder::send(int status, const std::string& content_type, const std::string& body,
                         const std::string& extra_headers) {
  if (responded_ || sse_open_) return;
  responded_ = true;
  std::ostringstream head;
  head << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Connection: close\r\n"
       << extra_headers << "\r\n";
  const std::string h = head.str();
  if (write_all(h.data(), h.size())) write_all(body.data(), body.size());
}

void HttpResponder::begin_sse() {
  if (responded_ || sse_open_) return;
  sse_open_ = true;
  static const char kHead[] =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream\r\n"
      "Cache-Control: no-cache\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      // No chunked framing: the stream is delimited by the close. Every SSE
      // client handles that, and it keeps the writer trivial.
      "Connection: close\r\n\r\n";
  write_all(kHead, sizeof(kHead) - 1);
}

bool HttpResponder::sse(const std::string& payload) {
  if (!sse_open_ || !alive_) return false;
  std::string frame;
  frame.reserve(payload.size() + 10);
  frame += "data: ";
  frame += payload;
  frame += "\n\n";
  return write_all(frame.data(), frame.size());
}

struct HttpServer::Impl {
  socket_t listener = kInvalidSocket;
  std::thread accept_thread;
  std::atomic<bool> running{false};
  HttpHandler handler;

  // Reads one request off `sock`. Returns false on a malformed or oversized
  // request (the caller answers 400 and closes).
  static bool read_request(socket_t sock, HttpRequest* req) {
    std::string buf;
    char chunk[8192];
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
      const int n = ::recv(sock, chunk, sizeof(chunk), 0);
      if (n <= 0) return false;
      buf.append(chunk, static_cast<std::size_t>(n));
      header_end = buf.find("\r\n\r\n");
      if (header_end == std::string::npos && buf.size() > kMaxHeaders) return false;
    }

    std::istringstream head(buf.substr(0, header_end));
    std::string line;
    if (!std::getline(head, line)) return false;
    {
      std::istringstream rl(line);
      std::string target;
      std::string version;
      if (!(rl >> req->method >> target >> version)) return false;
      const auto qpos = target.find('?');
      if (qpos == std::string::npos) {
        req->path = target;
      } else {
        req->path = target.substr(0, qpos);
        req->query = target.substr(qpos + 1);
      }
    }
    while (std::getline(head, line)) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      req->headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }

    std::size_t content_length = 0;
    const std::string cl = req->header("content-length");
    if (!cl.empty()) {
      try {
        content_length = static_cast<std::size_t>(std::stoull(cl));
      } catch (const std::exception&) {
        return false;
      }
      if (content_length > kMaxBody) return false;
    }
    req->body = buf.substr(header_end + 4);
    while (req->body.size() < content_length) {
      const int n = ::recv(sock, chunk, sizeof(chunk), 0);
      if (n <= 0) return false;
      req->body.append(chunk, static_cast<std::size_t>(n));
    }
    if (req->body.size() > content_length) req->body.resize(content_length);
    return true;
  }

  void serve(socket_t sock) {
    // Token streams are long and bursty; Nagle would sit on single-token frames.
    int one = 1;
    ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));

    HttpRequest req;
    if (read_request(sock, &req)) {
      HttpResponder res(static_cast<std::intptr_t>(sock));
      if (req.method == "OPTIONS") {
        res.send(200, "text/plain", "",
                 "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                 "Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
      } else {
        try {
          handler(req, res);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "[http] handler error: %s\n", e.what());
          res.send(500, "application/json",
                   std::string("{\"error\":{\"message\":\"internal error\"}}"));
        }
      }
    } else {
      HttpResponder res(static_cast<std::intptr_t>(sock));
      res.send(400, "application/json", "{\"error\":{\"message\":\"malformed request\"}}");
    }
    CPI_CLOSESOCKET(sock);
  }
};

HttpServer::HttpServer() : impl_(new Impl()) {}

HttpServer::~HttpServer() {
  stop();
}

bool HttpServer::start(const std::string& host, int port, HttpHandler handler, std::string* error) {
  const auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  if (!socket_subsystem().ok) return fail("socket subsystem unavailable");

  impl_->handler = std::move(handler);
  impl_->listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (impl_->listener == kInvalidSocket) return fail("socket() failed");

  int yes = 1;
  ::setsockopt(impl_->listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes),
               sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    CPI_CLOSESOCKET(impl_->listener);
    impl_->listener = kInvalidSocket;
    return fail("bad host address: " + host);
  }
  if (::bind(impl_->listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    CPI_CLOSESOCKET(impl_->listener);
    impl_->listener = kInvalidSocket;
    return fail("bind failed on " + host + ":" + std::to_string(port) + " (port in use?)");
  }
  if (::listen(impl_->listener, 64) != 0) {
    CPI_CLOSESOCKET(impl_->listener);
    impl_->listener = kInvalidSocket;
    return fail("listen failed");
  }

  // Port 0 asks the OS to pick; report what it picked.
  sockaddr_in bound{};
#if defined(_WIN32)
  int blen = sizeof(bound);
#else
  socklen_t blen = sizeof(bound);
#endif
  if (::getsockname(impl_->listener, reinterpret_cast<sockaddr*>(&bound), &blen) == 0) {
    port_ = ntohs(bound.sin_port);
  } else {
    port_ = port;
  }

  impl_->running.store(true);
  impl_->accept_thread = std::thread([this]() {
    while (impl_->running.load()) {
      const socket_t client = ::accept(impl_->listener, nullptr, nullptr);
      if (client == kInvalidSocket) {
        if (!impl_->running.load()) break;
        continue;
      }
      std::thread([this, client]() { impl_->serve(client); }).detach();
    }
  });
  return true;
}

void HttpServer::stop() {
  if (!impl_ || !impl_->running.exchange(false)) return;
  if (impl_->listener != kInvalidSocket) {
    CPI_CLOSESOCKET(impl_->listener);
    impl_->listener = kInvalidSocket;
  }
  if (impl_->accept_thread.joinable()) impl_->accept_thread.join();
}

}  // namespace net
