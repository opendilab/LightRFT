#!/bin/bash
#
# Start SGLang teacher server for On-Policy Distillation.
# Prints TEACHER_URL on success for use with start_training.sh.
#
# Usage:
#   bash examples/on_policy_distillation/start_teacher.sh
#   # Then in another terminal:
#   TEACHER_URL=http://127.0.0.1:13141/generate bash examples/on_policy_distillation/start_training.sh
#

set -euo pipefail

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_GPU="${TEACHER_GPU:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.7}"

LOG_DIR="rft_logs/teacher"
mkdir -p "$LOG_DIR"
TEACHER_LOG="${LOG_DIR}/teacher_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing process on the port
if lsof -Pi :"$TEACHER_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port $TEACHER_PORT in use, killing existing process..."
    lsof -ti:"$TEACHER_PORT" | xargs kill -9 2>/dev/null || true
    sleep 3
fi

echo "Starting teacher server on GPU $TEACHER_GPU..."
echo "  Model: $TEACHER_MODEL_PATH"
echo "  Log:   $TEACHER_LOG"

CUDA_VISIBLE_DEVICES=$TEACHER_GPU python3 -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$TEACHER_PORT" \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-radix-cache \
    --max-running-requests 64 \
    >> "$TEACHER_LOG" 2>&1 &

TEACHER_PID=$!

# Wait for health check
max_wait=600
waited=0
while ! curl -sf "http://$TEACHER_IP:$TEACHER_PORT/health" >/dev/null 2>&1; do
    if [ $waited -ge $max_wait ]; then
        echo "ERROR: Teacher server failed to start in ${max_wait}s"
        tail -30 "$TEACHER_LOG"
        kill "$TEACHER_PID" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
        echo "ERROR: Teacher server process died"
        tail -30 "$TEACHER_LOG"
        exit 1
    fi
    printf "."
    sleep 5
    waited=$((waited + 5))
done

TEACHER_URL="http://$TEACHER_IP:$TEACHER_PORT/generate"
echo ""
echo "========================================="
echo "Teacher server ready!"
echo "  PID:  $TEACHER_PID"
echo "  URL:  $TEACHER_URL"
echo "========================================="
echo ""
echo "Export for training:"
echo "  export TEACHER_URL=$TEACHER_URL"

# Keep running in foreground
wait "$TEACHER_PID"
