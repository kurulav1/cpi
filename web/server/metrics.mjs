// Minimal, dependency-free Prometheus metrics for the CPI server.
// Exposed at GET /metrics in the standard text exposition format so a
// Prometheus-compatible scraper (VictoriaMetrics / vmagent) can collect them and
// KEDA can autoscale on them. Kept tiny on purpose -- no prom-client dependency.

const reqTotal = new Map(); // "route|status" -> count
const durSum = new Map(); // route -> total seconds
const durCount = new Map(); // route -> request count
let inflight = 0;
let generationsTotal = 0;
let generatedTokensTotal = 0;
let decodeMsSum = 0;
let readyProbe = () => 0;

// Called by the server to report model-loaded state at scrape time (0/1).
export function setReadyProbe(fn) {
  readyProbe = typeof fn === "function" ? fn : () => 0;
}

// Request lifecycle (driven by middleware). `inflight` is the live concurrency,
// the primary KEDA autoscaling signal for an engine without continuous batching.
export function reqStart() {
  inflight += 1;
}
export function reqEnd(route, status, seconds) {
  inflight = Math.max(0, inflight - 1);
  const key = `${route}|${status}`;
  reqTotal.set(key, (reqTotal.get(key) || 0) + 1);
  durSum.set(route, (durSum.get(route) || 0) + seconds);
  durCount.set(route, (durCount.get(route) || 0) + 1);
}

// Called once per completed generation with the token count and decode time.
export function recordGeneration(tokens, decodeMs) {
  generationsTotal += 1;
  if (Number.isFinite(tokens)) generatedTokensTotal += tokens;
  if (Number.isFinite(decodeMs)) decodeMsSum += decodeMs;
}

function esc(v) {
  return String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

export function render() {
  const lines = [];
  const ready = readyProbe() ? 1 : 0;

  lines.push("# HELP cpi_model_ready Whether the model is loaded and ready (1) or not (0).");
  lines.push("# TYPE cpi_model_ready gauge");
  lines.push(`cpi_model_ready ${ready}`);

  lines.push("# HELP cpi_inflight_requests In-flight API requests (live concurrency).");
  lines.push("# TYPE cpi_inflight_requests gauge");
  lines.push(`cpi_inflight_requests ${inflight}`);

  lines.push("# HELP cpi_http_requests_total Total API requests by route and status.");
  lines.push("# TYPE cpi_http_requests_total counter");
  for (const [key, val] of reqTotal) {
    const [route, status] = key.split("|");
    lines.push(`cpi_http_requests_total{route="${esc(route)}",status="${esc(status)}"} ${val}`);
  }

  lines.push("# HELP cpi_http_request_duration_seconds Request duration sum/count by route.");
  lines.push("# TYPE cpi_http_request_duration_seconds summary");
  for (const [route, sum] of durSum) {
    lines.push(`cpi_http_request_duration_seconds_sum{route="${esc(route)}"} ${sum}`);
  }
  for (const [route, count] of durCount) {
    lines.push(`cpi_http_request_duration_seconds_count{route="${esc(route)}"} ${count}`);
  }

  lines.push("# HELP cpi_generations_total Completed generation requests.");
  lines.push("# TYPE cpi_generations_total counter");
  lines.push(`cpi_generations_total ${generationsTotal}`);

  lines.push("# HELP cpi_generated_tokens_total Tokens generated across all requests.");
  lines.push("# TYPE cpi_generated_tokens_total counter");
  lines.push(`cpi_generated_tokens_total ${generatedTokensTotal}`);

  lines.push("# HELP cpi_generation_decode_ms_sum Summed decode time (ms); with tokens gives tok/s.");
  lines.push("# TYPE cpi_generation_decode_ms_sum counter");
  lines.push(`cpi_generation_decode_ms_sum ${decodeMsSum}`);

  return lines.join("\n") + "\n";
}
