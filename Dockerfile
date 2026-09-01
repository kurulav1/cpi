# syntax=docker/dockerfile:1.7
#
# Two targets:
#   --target cuda  (default)  the engine and nothing else. No Node, no npm, no web/.
#   --target web              the same engine plus the Node web UI, as before.
#
# The point of the split is that the no-dependencies claim should be true of the
# artifact people download, not only of the source tree. An image that ships a
# Node runtime to serve a static page does not read as "no runtime dependencies"
# however the README phrases it.
#
# Build:
#   docker build --target cuda -t cpi:cuda .
#   docker build --target web  -t cpi:web  .

FROM node:22-bookworm-slim AS web-deps
WORKDIR /app/web
COPY web/package*.json ./
RUN npm install

FROM web-deps AS web-build
COPY web/ ./
RUN npm run build

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS engine-build
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Ubuntu 22.04's apt cmake is 3.22, but CMakeLists.txt requires >= 3.24.
# Install an official static cmake into /usr/local instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libsentencepiece-dev \
    ca-certificates \
    wget \
    && wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
       | tar --strip-components=1 -xz -C /usr/local \
    && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src

# The build host has no GPU, so the CMakeLists default of CMAKE_CUDA_ARCHITECTURES=native
# cannot probe a device and falls back to a low arch lacking __dp4a (needs sm_61+). Pin a
# real arch list instead.
#
# 120-real is Blackwell (RTX 50-series), and it is compiled rather than left to the
# 90-virtual PTX on purpose. Without it the driver JIT-compiles kernels lazily on first
# launch, measured at about 13 s folded into the first forward for a 0.5B on an RTX 5090,
# and a --rm container throws the JIT cache away every run, so every run pays it. That is
# the first thing a person trying the one-line docker command would experience, on the
# card most likely to be running it. Needs a 12.8+ base, which is why the images moved.
#
# 90-virtual stays as the forward-compatibility tail for GPUs newer than this list.
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;120-real;90-virtual"
# Bounded rather than $(nproc). Each .cu is now compiled for seven targets, and nvcc
# holds a lot of memory per translation unit, so an unbounded -j on a many-core host
# is a way to get the OOM killer instead of a binary. Raise it with
# --build-arg BUILD_JOBS=N where there is memory for it.
ARG BUILD_JOBS=4
# The build context carries no .git, so CMake's git probe resolves to "unknown" and
# a published binary cannot say which commit produced it. For a project whose output
# is hashes people paste somewhere, "which build produced this" is the other half of
# the claim. The CI job passes the tagged commit; a local build that does not bother
# still reports "unknown", which is honest rather than wrong.
ARG CPI_GIT_SHA=unknown
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCPI_ENABLE_CUDA=ON -DCPI_REQUIRE_CUDA=ON \
      -DCPI_GIT_SHA="${CPI_GIT_SHA}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build -j"${BUILD_JOBS}" --target cpi

# ---------------------------------------------------------------------------
# cpi:cuda -- the engine alone.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS cuda
ARG DEBIAN_FRONTEND=noninteractive

# libgomp1 is the OpenMP runtime the CPU paths link against; without it the binary
# will not start at all. A release once shipped a Linux artifact missing it and died
# on stock Ubuntu, so it is named here rather than assumed to come with the base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libsentencepiece0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=engine-build /app/build/cpi /usr/local/bin/cpi

# Deliberately no CPI_HOST here. The web image sets it because the Node server reads
# it; the engine does not read it at all, so setting it in this image would announce
# a behaviour that does not happen. The bind address is --host on the command line,
# and binding anything but loopback requires --api-key (the server refuses to expose
# an unauthenticated API to a network). See the docker run line in the README.
EXPOSE 8080
VOLUME ["/models"]

ENTRYPOINT ["cpi"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# cpi:web -- the engine plus the Node web UI.
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS node-runtime

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS web
ARG DEBIAN_FRONTEND=noninteractive
ENV NODE_ENV=production
WORKDIR /app/web

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libsentencepiece0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node-runtime /usr/local/ /usr/local/
COPY web/package*.json ./
RUN npm install --omit=dev

COPY web/server ./server
COPY web/.env.example ./.env.example
COPY --from=web-build /app/web/dist ./dist
COPY --from=engine-build /app/build/cpi /app/bin/cpi

# CPI_HOST=0.0.0.0: inside a container the server must bind all interfaces for
# the port mapping to reach it; the host's -p flag decides actual exposure. The
# admin surface still refuses non-localhost clients unless CPI_ADMIN_TOKEN is set.
#
# CPU-in-container notes (verified by running, Docker Desktop / WSL2):
# - Without a GPU the engine falls back to the CPU path automatically; no
#   separate CPU image. Docker Desktop's WSL2 backend hands the GPU to
#   containers even WITHOUT --gpus; force CPU with -e CUDA_VISIBLE_DEVICES=-1.
# - Set OMP_NUM_THREADS to the host's PHYSICAL core count for CPU inference:
#   the default (all logical CPUs) oversubscribes SMT under the WSL2 scheduler
#   and measured 2.4x slower on both prefill and decode.
ENV PORT=3001 \
    CPI_HOST=0.0.0.0 \
    CPI_BIN=/app/bin/cpi \
    LLAMA_MODEL_DIRS=/models \
    LLAMA_MODEL_PATH=/models/model.ll2c \
    LLAMA_TOKENIZER_PATH=/models/tokenizer.json \
    LLAMA_CHAT_TEMPLATE=tinyllama

EXPOSE 3001

CMD ["npm", "run", "start"]
