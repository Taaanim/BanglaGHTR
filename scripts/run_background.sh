#!/usr/bin/env bash
# =============================================================================
# BANGHTR-X v2: Safe Background Training Launcher
# Designed for Shared GPU Servers (Strict Non-Interference)
# =============================================================================
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"

# 1. GPU Isolation - strictly use GPU 0 (the RTX 4090)
export CUDA_VISIBLE_DEVICES=0

# 2. Virtual Environment - strictly use project-isolated .venv
if [ -d "$WORKSPACE_DIR/.venv" ]; then
    PYTHON_EXEC="$WORKSPACE_DIR/.venv/bin/python"
else
    echo "❌ Error: Project virtual environment (.venv) not found!"
    exit 1
fi

# 3. Create log directory
mkdir -p "$WORKSPACE_DIR/logs"

# 4. Check if training is already running
PID_FILE="$WORKSPACE_DIR/logs/train.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is ALREADY running with PID: $OLD_PID"
        echo "👉 You can monitor it in Monitor_Training.ipynb or run: tail -f logs/train_progress.log"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# 5. Launch detached in background with nohup
echo "🚀 Launching BANGHTR-X v2 Training Pipeline in background..."
nohup "$PYTHON_EXEC" -u scripts/train_pipeline.py > "$WORKSPACE_DIR/logs/train_stdout.log" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "============================================================================="
echo "✅ Training successfully launched in background with PID: $TRAIN_PID"
echo "============================================================================="
echo "🔒 Shared Server Safety:"
echo "   - Isolated to GPU 0 (RTX 4090)"
echo "   - Modest CPU workers (4 workers, 20 cores left untouched)"
echo "   - Using isolated .venv (zero impact on system or other users)"
echo "   - Safe against SSH disconnection / offline laptop shutdown"
echo ""
echo "📊 Monitoring Options:"
echo "   1. Open 'Monitor_Training.ipynb' in Jupyter to watch live curves & tables"
echo "   2. In terminal: tail -f logs/train_progress.log"
echo "   3. Check status file: cat logs/training_status.json"
echo "============================================================================="
