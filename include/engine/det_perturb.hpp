#pragma once

#include <string>

// Deliberately corrupting the token stream, so a determinism check can be shown to FAIL.
//
// Every row in docs/determinism-scope.md is a check reporting "identical". A check that has
// never reported anything else is not evidence: the divergence branch is the untested branch,
// because it only runs when something goes wrong. Four false positives in one week shared that
// shape, including a divergence-reporting path that crashed the one time it fired.
//
// So the failure case gets a switch. CPI_DET_PERTURB=<step> replaces the token generated at
// index <step> with a different valid token, deterministically, in every engine that emits
// tokens. Setting it on one side of any comparison must make that comparison report divergence,
// name <step> as the first differing index, and exit non-zero.
//
// The perturbation is at the token rather than in a kernel on purpose. A single ULP in the LM
// head is more faithful to a real numeric fault, but whether it flips a token inside 64 steps
// depends on hitting a near-tie, so a control built on it can quietly test nothing. This one
// changes the stream at an index known in advance, which is what lets the control assert the
// reported index rather than merely that something differed. It exercises the comparison
// plumbing, not the arithmetic.
namespace cpi::det {

// True when CPI_DET_PERTURB is set to a non-negative step. Read once, at first use.
[[nodiscard]] bool perturb_enabled();

// The step index that will be corrupted, or -1 when disabled.
[[nodiscard]] int perturb_step();

// Returns token unchanged unless perturbation is enabled and step matches the configured index,
// in which case it returns a different, always-valid token id. Engines call this at the single
// point where a generated token becomes part of the output, passing the zero-based index of that
// token within the generation.
[[nodiscard]] int perturb_token(int step, int token);

// One line naming the active perturbation, for a banner. Empty when disabled.
[[nodiscard]] std::string perturb_description();

}  // namespace cpi::det
