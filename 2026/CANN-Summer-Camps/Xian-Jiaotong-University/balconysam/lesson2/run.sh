#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}/custom_op"
bash build.sh

cd "${SCRIPT_DIR}/test"
rm -rf build
mkdir -p build
cd build

cmake ..
cmake --build . -j"$(nproc)"

if [ -f "./execute_add_op" ]; then
    ./execute_add_op
else
    echo "test executable not found."
    exit 1
fi