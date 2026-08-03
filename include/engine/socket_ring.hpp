#pragma once

// Ring all-reduce over real TCP sockets -- the transport-logic prototype for a future multi-node
// collective (the layer NcclCollective replaces on a real cluster; see collective.hpp). Unlike the
// single-process LocalCollective (one address space, buffers copied device-to-device), this exercises
// the DISTRIBUTED control flow that only appears when ranks are genuinely separate endpoints:
//   * ring connection setup (each rank listens, connects to its successor, accepts its predecessor)
//     with connect-retry -- the handshake-ordering / deadlock surface,
//   * the two-phase ring algorithm (scatter-reduce then all-gather, 2*(world-1) steps),
//   * byte-level send/recv with partial-IO loops (a short read/write is normal on a stream socket and
//     is the classic transport bug a single-process test can never surface).
//
// Ranks may be separate processes or threads; they share NO state and talk ONLY over 127.0.0.1
// sockets, so the isolation that matters for correctness holds either way. Portable over Winsock and
// BSD sockets. Header-only (inline); it is verification/prototype infrastructure, not on the hot path.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace engine {

#if defined(_WIN32)
using socket_t = SOCKET;
inline constexpr socket_t kInvalidSocket = INVALID_SOCKET;
inline void close_socket(socket_t s) { closesocket(s); }
inline void socket_startup() {
  static std::once_flag once;
  std::call_once(once, [] {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
  });
}
#else
using socket_t = int;
inline constexpr socket_t kInvalidSocket = -1;
inline void close_socket(socket_t s) { ::close(s); }
inline void socket_startup() {}
#endif

// Write exactly n bytes, looping over short writes (a stream socket may accept fewer than requested).
inline void send_all(socket_t s, const void* buf, std::size_t n) {
  const char* p = static_cast<const char*>(buf);
  std::size_t sent = 0;
  while (sent < n) {
    const int k = ::send(s, p + sent, static_cast<int>(n - sent), 0);
    if (k <= 0) throw std::runtime_error("send_all: socket send failed");
    sent += static_cast<std::size_t>(k);
  }
}

// Read exactly n bytes, looping over short reads (a single recv may return fewer than requested, or a
// message may span several TCP segments -- handling this is the whole point of the exercise).
inline void recv_all(socket_t s, void* buf, std::size_t n) {
  char* p = static_cast<char*>(buf);
  std::size_t got = 0;
  while (got < n) {
    const int k = ::recv(s, p + got, static_cast<int>(n - got), 0);
    if (k <= 0) throw std::runtime_error("recv_all: socket recv failed / peer closed");
    got += static_cast<std::size_t>(k);
  }
}

// A rank's two ring links: a socket to its successor (send) and from its predecessor (recv). Connecting
// the ring is where handshake ordering bites -- every rank listens first, then connect-retries its
// successor (whose listener may not be up yet), then accepts its predecessor.
class SocketRing {
public:
  // rank in [0, world); every rank uses base_port + rank as its listen port on 127.0.0.1.
  SocketRing(int rank, int world, uint16_t base_port) : rank_(rank), world_(world) {
    socket_startup();
    if (world <= 1) return;  // trivial ring: no links, all-reduce is a no-op

    const int next = (rank + 1) % world;

    // 1) Listen on our own port so our predecessor can connect to us.
    socket_t listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener == kInvalidSocket) throw std::runtime_error("SocketRing: socket() failed");
    int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one),
                 sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(base_port + rank));
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
      throw std::runtime_error("SocketRing: bind() failed (port in use?)");
    if (::listen(listener, 1) != 0) throw std::runtime_error("SocketRing: listen() failed");

    // 2) Connect to our successor, retrying until its listener is up (handshake ordering).
    send_sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (send_sock_ == kInvalidSocket) throw std::runtime_error("SocketRing: socket() failed");
    sockaddr_in naddr{};
    naddr.sin_family = AF_INET;
    naddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    naddr.sin_port = htons(static_cast<uint16_t>(base_port + next));
    for (int attempt = 0; attempt < 2000; ++attempt) {
      if (::connect(send_sock_, reinterpret_cast<sockaddr*>(&naddr), sizeof(naddr)) == 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if (attempt == 1999) throw std::runtime_error("SocketRing: connect() to successor timed out");
    }

    // 3) Accept our predecessor's connection.
    recv_sock_ = ::accept(listener, nullptr, nullptr);
    if (recv_sock_ == kInvalidSocket) throw std::runtime_error("SocketRing: accept() failed");
    close_socket(listener);

    // Disable Nagle so the small per-step chunk exchanges don't stall on coalescing.
    ::setsockopt(send_sock_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one),
                 sizeof(one));
  }

  ~SocketRing() {
    if (send_sock_ != kInvalidSocket) close_socket(send_sock_);
    if (recv_sock_ != kInvalidSocket) close_socket(recv_sock_);
  }

  SocketRing(const SocketRing&) = delete;
  SocketRing& operator=(const SocketRing&) = delete;

  // In-place ring all-reduce (sum) of `count` floats. On return every rank holds the elementwise sum
  // over all ranks' inputs. Classic two-phase ring: scatter-reduce leaves block (rank+1) fully reduced
  // on each rank, then all-gather rotates the reduced blocks around the ring.
  void all_reduce_sum(float* data, int count) {
    if (world_ <= 1) return;

    // Balanced block split (floor/ceil), same partition policy as the TP/EP/PP shards. Block b spans
    // the same [off,off+len) on every rank, so neighbours agree on each step's transfer size.
    std::vector<int> off(world_), len(world_);
    int rem = count, o = 0;
    for (int b = 0; b < world_; ++b) {
      len[b] = rem / (world_ - b);
      off[b] = o;
      o += len[b];
      rem -= len[b];
    }

    std::vector<float> tmp;
    for (int b = 0; b < world_; ++b) tmp.resize(std::max<int>(tmp.size(), len[b]));

    // Phase 1 -- scatter-reduce: world-1 steps. Send one block to the successor while receiving the
    // predecessor's block and accumulating it. Send-then-recv is deadlock-free here because each block
    // is far smaller than the socket send buffer, so send() returns before recv() is posted.
    for (int step = 0; step < world_ - 1; ++step) {
      const int send_b = (rank_ - step + 2 * world_) % world_;
      const int recv_b = (rank_ - step - 1 + 2 * world_) % world_;
      exchange_add(data + off[send_b], len[send_b], data + off[recv_b], len[recv_b], tmp.data());
    }

    // Phase 2 -- all-gather: world-1 steps. Rotate the fully-reduced blocks so every rank ends with all
    // of them. Received block overwrites (it is already the final sum), it is not accumulated.
    for (int step = 0; step < world_ - 1; ++step) {
      const int send_b = (rank_ - step + 1 + 2 * world_) % world_;
      const int recv_b = (rank_ - step + 2 * world_) % world_;
      exchange_copy(data + off[send_b], len[send_b], data + off[recv_b], len[recv_b], tmp.data());
    }
  }

  int rank() const { return rank_; }
  int world() const { return world_; }

private:
  // Send send_buf to successor, recv predecessor's block into tmp, ADD tmp into recv_buf.
  void exchange_add(const float* send_buf, int send_len, float* recv_buf, int recv_len, float* tmp) {
    send_all(send_sock_, send_buf, static_cast<std::size_t>(send_len) * sizeof(float));
    recv_all(recv_sock_, tmp, static_cast<std::size_t>(recv_len) * sizeof(float));
    for (int i = 0; i < recv_len; ++i) recv_buf[i] += tmp[i];
  }
  // Send send_buf to successor, recv predecessor's block into tmp, OVERWRITE recv_buf with it.
  void exchange_copy(const float* send_buf, int send_len, float* recv_buf, int recv_len, float* tmp) {
    send_all(send_sock_, send_buf, static_cast<std::size_t>(send_len) * sizeof(float));
    recv_all(recv_sock_, tmp, static_cast<std::size_t>(recv_len) * sizeof(float));
    std::memcpy(recv_buf, tmp, static_cast<std::size_t>(recv_len) * sizeof(float));
  }

  int rank_ = 0;
  int world_ = 1;
  socket_t send_sock_ = kInvalidSocket;
  socket_t recv_sock_ = kInvalidSocket;
};

}  // namespace engine
