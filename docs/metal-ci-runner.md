# Standing up the Metal GPU runner

`.github/workflows/metal-gpu.yml` is the only thing that runs CPI's Metal kernels against a
reference. It is **inert until you give it a runner** — this is how.


## ⚠ Read this first: THIS REPOSITORY IS PUBLIC

A self-hosted runner executes whatever a workflow tells it to, on your machine, as the user
that owns it, inside your network. On a **public** repo that is a live remote-code-execution
path: anyone can fork it, open a pull request that edits `CMakeLists.txt` (or the workflow
itself), and have your Mac run it. GitHub's own guidance is blunt about this — self-hosted
runners are recommended for private repositories only.

`metal-gpu.yml` therefore refuses to run for pull requests that come from a fork:

```yaml
if: >-
  vars.HAS_METAL_RUNNER == 'true' &&
  (github.event_name != 'pull_request' ||
   github.event.pull_request.head.repo.full_name == github.repository)
```

Fork PRs never reach the runner; pushes to `main`, your own branches, and manual dispatch do.
**Do not relax this** to give fork PRs coverage — that is the whole exposure. Also worth doing,
belt and braces:

- Settings → Actions → General → **Fork pull request workflows**: require approval for **all**
  outside collaborators (the default only covers first-time ones).
- Prefer an **ephemeral Tart VM** over your daily Mac, so a compromised job dies with the VM
  and never sees your keys, your SSH agent or your home directory.
- Do not put anything on that machine you would mind losing. The rented box this backend was
  built on is a good shape for it: disposable, isolated, nothing personal on it.

If you would rather not accept any of this, the honest alternative is to keep running
`tools/metal_verify.sh --require-gpu` by hand (last section) and skip the runner. That is a
real position — it just means the checks only run when someone remembers.
## Why it matters more than it looks

The `metal` job in `ci.yml` is compile-only, and not by oversight: GitHub's macOS runners are
GPU-less VMs where `MTLCreateSystemDefaultDevice()` returns nil, so every Metal test there
reports SKIP. A green macOS CI means the shaders type-check and link. It has never meant a
kernel is correct.

That gap is not hypothetical. `cpi_gemm_f16` was dispatched with half the threads its tile
needed for weeks — every fp16 prompt of ≥16 tokens decoded from a corrupted prefill — while CI
stayed green, because no gate in the repo executed that kernel. It was caught by hand, on a
rented Mac, by accident. See the GEMM note in the README.

The other half of the risk is structural: the op-plan and its builder are **shared with CUDA**.
A CUDA-side refactor can break Metal with nothing on either side going red.

## What you need

One Apple Silicon machine that stays on, with a Metal device. "VM" is not the blocker —
GPU-less is:

- **Tart** (<https://tart.run>) — macOS VMs on Apple Silicon that expose a real paravirtual
  `MTLDevice`. Ephemeral per-job runners, so a bad job cannot poison the next one. Recommended.
- **A Mac mini with the runner agent** — simplest, and fine. State persists between jobs, so
  a wedged build directory is yours to clean.

A rented Apple Silicon Mac works for a session but is not a runner: the moment it lapses, Metal
is unguarded again.

## Setup

1. **Register the runner** with these labels — the workflow targets all four:

   ```
   self-hosted, macOS, ARM64, metal-gpu
   ```

2. **Install the toolchain.** Xcode is *not* required and should not be added: the offline
   `metal` compiler ships with Xcode, not the Command Line Tools, and requiring a `.metallib`
   would mean a ~15 GB download on the runner. The engine compiles shaders at runtime through
   `newLibraryWithSource`, using the Metal framework's own compiler service, which every Mac
   has. So:

   ```sh
   xcode-select --install    # Command Line Tools: clang + make
   brew install cmake python3
   ```

3. **Put checkpoints somewhere** and point `METAL_MODELS_DIR` at them (repo variable). Goldens
   whose `.ll2c` is missing are skipped **loudly** — the run still passes but says what it did
   not cover. A runner with only Qwen2.5-0.5B is worth having; it just covers less.

   The gate looks for: `qwen.ll2c` (or `Qwen2.5-0.5B-Instruct.ll2c`), `Qwen3-0.6B.ll2c`,
   `gemma-2b.ll2c`, `llama8b.ll2c` (or `Llama-3.1-8B-Instruct.ll2c`).

4. **Set the repo variables** (Settings → Secrets and variables → Actions → Variables):

   | Variable | Value | Effect |
   | --- | --- | --- |
   | `HAS_METAL_RUNNER` | `true` | Enables the job. Until set, it is skipped entirely. |
   | `METAL_MODELS_DIR` | e.g. `/Users/ci/models` | Where the gate looks for checkpoints. Defaults to `$HOME`. |

## Why the job is gated on a variable

Because a job targeting a self-hosted label with no matching runner **does not fail — it queues
forever and hangs every PR**. The variable makes the workflow cost nothing and block nothing
until the hardware exists. Do not replace it with a `runs-on` guess.

## Why `--require-gpu`

The job passes `--require-gpu` to `tools/metal_verify.sh`, which turns "no Metal device" from a
skip into a failure. On a runner whose entire purpose is to exercise the GPU, a missing device
is a misconfigured host — and a job that goes green having verified nothing is worse than one
that goes red. That is not a theoretical concern either: CMake once forced `-mavx2` on arm64,
GitHub's older clang silently ignored the unsupported flag, and CI passed while real hardware
(clang 21) errored.

## Verifying it works

Push to `main` touching any of the paths in the workflow's `paths:` filter, or run it via
**workflow_dispatch**. A healthy run reports something like:

```
== 7 checks ran, 0 failed, 2 skipped ==
```

To convince yourself the gate has teeth rather than just being green, break something on
purpose. In `src/engine/plan_metal_engine.cpp` set `kGemmFBM = 128` without touching the shader
and re-run: `qwen2.5-0.5b-longprompt-86x64.txt` must FAIL while the 12-token goldens still pass.
That is the exact blind spot that hid the original bug, reproduced on demand.

## Running it by hand

The same gate, no CI required — this is what to run on any Mac before trusting a change:

```sh
cmake -S . -B build -DCPI_ENABLE_CUDA=OFF -DCPI_ENABLE_METAL=ON
cmake --build build --config Release -j
./tools/metal_verify.sh --build build --models ~/models --require-gpu
```
