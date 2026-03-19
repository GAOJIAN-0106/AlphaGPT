#!/bin/bash
# ==============================================================
# AlphaGPT Daily Trading Pipeline
#
# 每日收盘后自动运行：信号生成 → 风控过滤 → 下单
#
# Usage:
#   ./scripts/daily_run.sh              # 默认 dry-run 模式
#   ./scripts/daily_run.sh sim          # 模拟盘模式
#   ./scripts/daily_run.sh live         # 实盘模式（需二次确认）
#
# Crontab example (每天16:00自动运行):
#   0 16 * * 1-5 /home/gj/AlphaGPT/scripts/daily_run.sh sim >> /home/gj/AlphaGPT/logs/daily.log 2>&1
# ==============================================================

set -e

# Config
CAPITAL=300000
TOPN=5
MODE="${1:-dry}"  # dry / sim / live
PROJECT_DIR="/home/gj/AlphaGPT"
PYTHON="/home/gj/miniconda3/bin/python"
SIGNAL_DIR="${PROJECT_DIR}/signals"
LOG_DIR="${PROJECT_DIR}/logs"

# Setup
cd "$PROJECT_DIR"
mkdir -p "$SIGNAL_DIR" "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOG_FILE="${LOG_DIR}/daily_${DATE}.log"

log() {
    echo "[${DATE} ${TIME}] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "AlphaGPT Daily Pipeline - ${DATE}"
log "Mode: ${MODE}, Capital: ${CAPITAL}, Top-N: ${TOPN}"
log "=========================================="

# Step 1: Generate signals
log "Step 1: Generating signals..."
$PYTHON scripts/generate_signals.py --output "$SIGNAL_DIR/" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    log "ERROR: Signal generation failed!"
    exit 1
fi
log "Step 1: Done"

# Step 2: Risk filter (Top-N small capital mode)
log "Step 2: Applying risk filter (Top-${TOPN})..."
$PYTHON scripts/risk_filter.py \
    --input "${SIGNAL_DIR}/latest.json" \
    --capital "$CAPITAL" \
    --mode topn \
    --topn "$TOPN" \
    --output "${SIGNAL_DIR}/topn_signals.json" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    log "ERROR: Risk filter failed!"
    exit 1
fi
log "Step 2: Done"

# Step 3: Execute orders
log "Step 3: Order execution (mode=${MODE})..."
$PYTHON scripts/order_router.py \
    --input "${SIGNAL_DIR}/topn_signals.json" \
    --mode "$MODE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    log "WARNING: Order execution had issues, check log"
fi
log "Step 3: Done"

# Summary
log "=========================================="
log "Pipeline complete!"
log "  Signals: ${SIGNAL_DIR}/signals_${DATE}.json"
log "  Filtered: ${SIGNAL_DIR}/topn_signals.json"
log "  Log: ${LOG_FILE}"
log "=========================================="
