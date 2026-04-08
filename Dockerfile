# syntax=docker/dockerfile:1.7

FROM node:22-bullseye-slim AS web-deps
WORKDIR /app/web
COPY web/package*.json ./
RUN npm install

FROM web-deps AS web-build
COPY web/ ./
RUN npm run build

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS engine-build
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" --target llama_infer

FROM node:22-bullseye-slim AS node-runtime

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV NODE_ENV=production
WORKDIR /app/web

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsentencepiece0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node-runtime /usr/local/ /usr/local/
COPY web/package*.json ./
RUN npm install --omit=dev

COPY web/server ./server
COPY web/.env.example ./.env.example
COPY --from=web-build /app/web/dist ./dist
COPY --from=engine-build /app/build/llama_infer /app/bin/llama_infer

ENV PORT=3001 \
    LLAMA_INFER_BIN=/app/bin/llama_infer \
    LLAMA_MODEL_DIRS=/models \
    LLAMA_MODEL_PATH=/models/model.ll2c \
    LLAMA_TOKENIZER_PATH=/models/tokenizer.json \
    LLAMA_CHAT_TEMPLATE=tinyllama

EXPOSE 3001

CMD ["npm", "run", "start"]
