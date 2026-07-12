#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    for candidate in "$HOME/Ascend/cann-8.5.0" \
        "/usr/local/Ascend/ascend-toolkit/latest" \
        "/usr/local/Ascend/ascend-toolkit"; do
        if [[ -f "$candidate/set_env.sh" ]]; then
            ASCEND_TOOLKIT_HOME="$candidate"
            break
        fi
    done
fi

: "${ASCEND_TOOLKIT_HOME:?Unable to locate the Ascend toolkit}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"
source "$ASCEND_TOOLKIT_HOME/set_env.sh"

echo "[1/4] Building custom operator"
cd "$SCRIPT_DIR/custom_op"
bash build.sh

echo "[2/4] Installing custom operator"
RUN_FILE="$(find build_out -maxdepth 1 -name 'custom_opp*.run' -print -quit)"
[[ -n "$RUN_FILE" ]] || { echo "Operator installer was not generated." >&2; exit 1; }
"$RUN_FILE" --install-path="$HOME"
source "$HOME/vendors/customize/bin/set_env.bash"

echo "[3/4] Building correctness test"
cd "$SCRIPT_DIR"
g++ -std=c++11 \
    -I"$ASCEND_TOOLKIT_HOME/include" \
    -I"$HOME/vendors/customize/op_api/include" \
    -L"$ASCEND_TOOLKIT_HOME/lib64" \
    -L"$HOME/vendors/customize/op_api/lib" \
    test/main.cpp -lcust_opapi -lnnopbase -lacl_rt -o execute_op

echo "[4/4] Running correctness test and profiler"
./execute_op
rm -rf prof
msprof op --output=prof ./execute_op

python3 - <<'PY'
import csv
import glob
import sys

files = glob.glob("prof/**/OpBasicInfo.csv", recursive=True)
if not files:
    raise SystemExit("OpBasicInfo.csv was not generated")

with open(files[0], newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("OpBasicInfo.csv contains no operator records")

print("Profiler CSV:", files[0])
for row in rows:
    print(row)

duration_keys = [key for key in rows[0] if "duration" in key.lower() or "time" in key.lower()]
print("Timing columns:", duration_keys)
duration = float(rows[0]["Task Duration(us)"])
print(f"Task Duration: {duration:.6f} us")
if duration >= 80.0:
    raise SystemExit(f"Performance target missed: {duration:.6f} us >= 80 us")
print("Performance target passed: duration < 80 us")
PY
