#pragma once

// Transport-agnostic continuous-batching worker.
//
// The scheduler-driving loop (admit / step / cancel, preempt-and-resume,
// per-request grammar and incremental detokenization) is the same work whether
// requests arrive as JSON lines on stdin or as HTTP requests on a socket. It
// used to live inside the stdin/stdout worker, which is why the in-binary HTTP
// server could not exist without copying it. The loop lives here and takes an
// event sink; a transport supplies requests through submit()/cancel() and
// renders the events however it likes.
//
// Threading: submit() and cancel() are safe from any thread. run() owns the
// engine and must be called from exactly one thread (the engine is not
// thread-safe); the sink is invoked from that thread.
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/batch_scheduler.hpp"
#include "model/tokenizer.hpp"

namespace grammar {
class GrammarSampler;
}

namespace app {

// Server-side defaults; a request may override any of them.
struct BatchDefaults {
  std::vector<std::string> stop_texts;
  bool add_bos = true;
  int max_new = 128;
  float temp = 0.0f;
  int top_k = 0;
  float top_p = 0.0f;
  float repeat_penalty = 1.0f;
  int no_repeat_ngram = 0;
};

// Per-request overrides. Negative/empty means "use the server default", which
// keeps "absent" distinct from "explicitly 0" for every field that has a
// meaningful zero (temperature 0 is greedy, not "unset").
struct BatchOverrides {
  int max_new = -1;
  float temp = -1.0f;
  int min_new = 0;
  int top_k = -1;
  float top_p = -1.0f;
  float repeat_penalty = -1.0f;
  int no_repeat_ngram = -1;
  bool has_add_bos = false;
  bool add_bos = true;
  std::vector<std::string> stop_texts;  // empty => defaults
  std::string json_schema;              // raw schema; compiled to a grammar at admit
};

struct BatchEvent {
  enum class Type { Start, Delta, Done, Error };
  Type type = Type::Delta;
  std::string id;
  std::string text;           // Delta: the new text; Done: the full text
  std::string finish_reason;  // Done only
  std::string error;          // Error only
  int generated = 0;
  double elapsed_ms = 0.0;
  double tok_per_s = 0.0;
};

class BatchWorker {
 public:
  using Sink = std::function<void(const BatchEvent&)>;

  BatchWorker(engine::BatchScheduler& sched, model::Tokenizer& tokenizer, BatchDefaults defaults,
              Sink sink);
  ~BatchWorker();

  BatchWorker(const BatchWorker&) = delete;
  BatchWorker& operator=(const BatchWorker&) = delete;

  // Tokenizes `prompt` and queues it. Returns false (with *error set) when the
  // prompt is empty or encodes to nothing; the request is never queued then, so
  // the caller can fail the connection synchronously instead of through the sink.
  bool submit(const std::string& id, const std::string& prompt, const BatchOverrides& overrides,
              std::string* error);

  // Reclaims a request's KV blocks (client disconnected, or an explicit cancel).
  // Safe for ids that are queued, running, or already gone.
  void cancel(const std::string& id);

  // Runs `task` on the worker thread with the engine to itself, then returns.
  // Multimodal generation goes through a different engine entry point than the
  // scheduler (image embeddings, not just tokens), and the engine is not
  // thread-safe, so an HTTP thread cannot simply call it while decode steps are
  // in flight. The task runs between steps; in-flight text requests pause for
  // its duration rather than racing it. Blocks until the task has run.
  void run_exclusive(const std::function<void()>& task);

  // Asks run() to return once the queue drains and no request is active.
  void stop();

  // The scheduler loop. Returns after stop() (or, for the stdio transport, when
  // its reader signals shutdown) and the in-flight work drains.
  void run();

 private:
  struct Incoming {
    std::string id;
    std::vector<int> tokens;
    engine::StreamParams params;
    std::string json_schema;
  };

  void emit(BatchEvent::Type type, const std::string& id, const std::string& text,
            const std::string& finish_reason, const std::string& error, int generated,
            double elapsed_ms, double tok_per_s);

  engine::BatchScheduler& sched_;
  model::Tokenizer& tokenizer_;
  BatchDefaults defaults_;
  Sink sink_;

  struct ExclusiveTask {
    const std::function<void()>* fn = nullptr;
    bool done = false;
  };

  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable exclusive_cv_;
  std::deque<Incoming> queue_;
  std::deque<std::string> cancels_;
  std::deque<ExclusiveTask*> exclusive_;
  bool stopping_ = false;
};

}  // namespace app
