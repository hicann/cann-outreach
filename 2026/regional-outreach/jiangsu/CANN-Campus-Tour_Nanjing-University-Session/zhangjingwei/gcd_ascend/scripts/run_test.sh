#!/bin/bash
#==============================================================================
# run_test.sh - Build and run unit tests for gcd_ascend operator
#
# Usage:
#   bash scripts/run_test.sh [Debug|Release]
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
BUILD_TYPE="${1:-Release}"

# 1. Build the operator
echo "=== Step 1: Build operator ==="
bash "${PROJECT_DIR}/build.sh" "${BUILD_TYPE}"

# 2. Locate operator output
OUTPUT_DIR="${PROJECT_DIR}/build/output"
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Build output not found at ${OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "=== Step 2: Run unit tests ==="

# 3. Run Python tests
export PYTHONPATH="${OUTPUT_DIR}:${PYTHONPATH:-}"
python3 "${SCRIPT_DIR}/test_gcd.py" --op-dir "${OUTPUT_DIR}"

echo ""
echo "=== Done ==="
