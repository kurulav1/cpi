# syntax=docker/dockerfile:1.7

FROM node:22-bullseye-slim AS web-deps
WORKDIR /app/web
COPY web/package*.json ./
RUN npm install

FROM web-deps AS web-build
COPY web/ ./
RUN npm run build

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS engine-build
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
# can't probe a device and falls back to a low arch lacking __dp4a (needs sm_61+). Pin a
# real arch list (Turing..Hopper) plus 90-virtual PTX for forward-compat on newer GPUs.
# Newer-than-Hopper GPUs (e.g. sm_120 Blackwell) run via the 90-virtual PTX, which means
# the driver JIT-compiles the whole engine on first launch -- several minutes, and --rm
# containers lose the JIT cache every run. To serve such GPUs routinely, bump the base
# images to a CUDA that can target them (12.8+ for Blackwell) and add the -real arch here.
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;90-virtual"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build -j"$(nproc)" --target cpi

FROM node:22-bullseye-slim AS node-runtime

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime
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
ENV PORT=3001 \
    CPI_HOST=0.0.0.0 \
    CPI_BIN=/app/bin/cpi \
    LLAMA_MODEL_DIRS=/models \
    LLAMA_MODEL_PATH=/models/model.ll2c \
    LLAMA_TOKENIZER_PATH=/models/tokenizer.json \
    LLAMA_CHAT_TEMPLATE=tinyllama

EXPOSE 3001

CMD ["npm", "run", "start"]
