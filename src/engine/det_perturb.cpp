#include "engine/det_perturb.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace cpi::det {
namespace {

// Parsed once. A determinism run must not have the perturbation appear or vanish partway
// through, and re-reading the environment per token would let it.
struct Config {
  bool enabled = false;
  int step = -1;
};

const Config& config() {
  static const Config cfg = [] {
    Config c;
    const char* e = std::getenv("CPI_DET_PERTURB");
    if (e == nullptr || e[0] == '\0') {
      return c;
    }
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || v < 0 || v > 1000000) {
      std::fprintf(stderr,
                   "[det-perturb] ignoring CPI_DET_PERTURB=%s: expected a step index in [0, "
                   "1000000]\n",
                   e);
      return c;
    }
    c.enabled = true;
    c.step = static_cast<int>(v);
    // Loud, because a perturbed run that looks like a normal one is the failure this whole
    // switch exists to prevent. Anything reading stdout sees only the [verify] lines, so this
    // goes to stderr where it cannot be mistaken for a result.
    std::fprintf(stderr,
                 "[det-perturb] ACTIVE: the token at index %d will be replaced. This build is "
                 "deliberately non-deterministic; no output from this run is evidence of "
                 "anything except that a check can fail.\n",
                 c.step);
    return c;
  }();
  return cfg;
}

}  // namespace

bool perturb_enabled() {
  return config().enabled;
}

int perturb_step() {
  return config().step;
}

int perturb_token(int step, int token) {
  const Config& cfg = config();
  if (!cfg.enabled || step != cfg.step) {
    return token;
  }
  // Any different, always-valid id. Going down avoids needing the vocabulary size at each call
  // site, since id 0 is the only case that cannot decrement and it has 1 available instead.
  return token == 0 ? 1 : token - 1;
}

std::string perturb_description() {
  const Config& cfg = config();
  if (!cfg.enabled) {
    return std::string();
  }
  return "token index " + std::to_string(cfg.step) + " replaced";
}

}  // namespace cpi::det
