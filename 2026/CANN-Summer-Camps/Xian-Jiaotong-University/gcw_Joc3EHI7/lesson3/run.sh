#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN Environment"
echo "=========================================="

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
    elif [ -d "/usr/local/Ascend/cann-8.5.0" ]; then
        export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/cann-8.5.0"
    else
        echo "[ERROR] Please set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH."
        exit 1
    fi
fi

if [ ! -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    echo "[ERROR] Cannot find $ASCEND_TOOLKIT_HOME/set_env.sh"
    exit 1
fi

source "$ASCEND_TOOLKIT_HOME/set_env.sh"
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
echo "[OK] CANN environment loaded: $ASCEND_HOME_PATH"

echo "=========================================="
echo " 2. Updating CANN Path"
echo "=========================================="
sed -i '/"ASCEND_CANN_PACKAGE_PATH": {/,/}/ s|"value": ".*"|"value": "'"$ASCEND_HOME_PATH"'"|' custom_op/CMakePresets.json

echo "=========================================="
echo " 3. Building Custom Operator"
echo "=========================================="
cd custom_op
bash build.sh
installer=$(ls build_out/custom_opp*.run | head -n 1)
if [ -z "$installer" ]; then
    echo "[ERROR] Custom operator installer was not generated."
    exit 1
fi
chmod +x "$installer"
"$installer" --install-path="${HOME}"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 4. Building Test Case"
echo "=========================================="
g++ -std=c++11 \
    -I"$ASCEND_HOME_PATH/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"$ASCEND_HOME_PATH/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op
echo "[OK] Test case built successfully."

echo "=========================================="
echo " 5. Running Performance Test"
echo "=========================================="
rm -rf prof
mkdir -p prof
source "${HOME}/vendors/customize/bin/set_env.bash"
msprof op --output=prof ./execute_op

echo "=========================================="
echo " 6. Reading Performance Result"
echo "=========================================="
python3 - <<'PY'
import csv
import glob
import sys

csv_files = glob.glob("prof/*/OpBasicInfo.csv")
if not csv_files:
    sys.exit("[ERROR] OpBasicInfo.csv was not generated.")

csv_file = csv_files[0]
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    headers = reader.fieldnames or []

print(f"Performance CSV: {csv_file}")
if not rows:
    sys.exit("[ERROR] OpBasicInfo.csv is empty.")

duration_cols = [name for name in headers if "duration" in name.lower() and "us" in name.lower()]
durations = []

for index, row in enumerate(rows, 1):
    print(f"row {index}:")
    for name in headers:
        print(f"  {name}: {row.get(name, '')}")
    for name in duration_cols:
        try:
            durations.append(float(row.get(name, "")))
        except ValueError:
            pass

if durations:
    max_duration = max(durations)
    print(f"Max duration(us): {max_duration:.3f}")
    if max_duration < 80:
        print("Performance target: PASS (<80us)")
    else:
        print("Performance target: WARNING (>=80us)")
else:
    print("Performance target: duration column not found; please check the CSV above.")
PY
