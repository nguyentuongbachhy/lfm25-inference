#!/usr/bin/env bash
# ==============================================================================
# Automated Server Test & Benchmark Runner for lfm25-inference
# Runs:
#   1. All OpenAI and Ollama API endpoints verification
#   2. Multi-turn Radix tree prefix caching benchmark
#   3. High-concurrency continuous batching benchmark (C = 1, 2, 4, 8)
# ==============================================================================

set -e

PORT="${1:-8088}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="/tmp/lfm25_server_${PORT}.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SERVER_URL="${BASE_URL}"
export no_proxy="*"
export NO_PROXY="*"

echo "======================================================================"
echo " LFM 2.5 SERVING VERIFICATION & BENCHMARK SUITE"
echo " Target: ${BASE_URL}"
echo "======================================================================"

# Check if server is already running
SERVER_STARTED_BY_SCRIPT=0
if curl --noproxy "*" -s "${BASE_URL}/health" | grep -q '"status":"ok"'; then
  echo "Detected existing server running on ${BASE_URL}. Using it directly."
else
  echo "Building release binary if needed..."
  cargo build --release --manifest-path "${REPO_DIR}/Cargo.toml"

  echo "Starting lfm25-inference server on ${HOST}:${PORT}..."
  "${REPO_DIR}/target/release/lfm25-inference" \
    --serve "${HOST}:${PORT}" \
    --hardware-profile "${REPO_DIR}/docs/serving/fp8-splitk-hardware-ps16.cost-model.json" \
    > "${LOG_FILE}" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED_BY_SCRIPT=1

  cleanup() {
    if [ "${SERVER_STARTED_BY_SCRIPT}" -eq 1 ]; then
      echo ""
      echo "Stopping server (PID: ${SERVER_PID})..."
      kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  echo "Waiting for server to become healthy..."
  HEALTHY=0
  for i in {1..40}; do
    if curl --noproxy "*" -s "${BASE_URL}/health" | grep -q '"status":"ok"'; then
      echo "Server is UP and ready!"
      HEALTHY=1
      break
    fi
    sleep 1
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: Server exited unexpectedly! Log output:"
      cat "${LOG_FILE}"
      exit 1
    fi
  done

  if [ "${HEALTHY}" -ne 1 ]; then
    echo "ERROR: Server timed out waiting for health check."
    exit 1
  fi
fi

echo ""
python3 "${SCRIPT_DIR}/serving/test_all_endpoints.py" "${BASE_URL}"

echo ""
python3 "${SCRIPT_DIR}/serving/bench_prefix_caching.py" "${BASE_URL}"

echo ""
python3 "${SCRIPT_DIR}/serving/bench_concurrency.py" "${BASE_URL}"

echo ""
echo "======================================================================"
echo " ALL TESTS AND BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "======================================================================"

