#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    if [ -d "/home/developer/Ascend/cann-9.0.0" ]; then
        export ASCEND_TOOLKIT_HOME="/home/developer/Ascend/cann-9.0.0"
    elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
        export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
    else
        echo "ASCEND_TOOLKIT_HOME is not set and default CANN path was not found."
        exit 1
    fi
fi

source "$ASCEND_TOOLKIT_HOME/set_env.sh"
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
echo "CANN environment: $ASCEND_TOOLKIT_HOME"

sed -i '/"ASCEND_CANN_PACKAGE_PATH": {/,/}/ s|\s*"value": ".*"| "value": "'"$ASCEND_TOOLKIT_HOME"'"|' custom_op/CMakePresets.json
sed -i '/"ASCEND_COMPUTE_UNIT": {/,/}/ s|\s*"value": ".*"| "value": "ascend910b;ascend910_93"|' custom_op/CMakePresets.json

cd custom_op
bash build.sh
./build_out/custom_opp*.run --install-path=${HOME}
cd ..

g++ -I"$ASCEND_TOOLKIT_HOME/include" \
    -I${HOME}/vendors/customize/op_api/include \
    -L"$ASCEND_TOOLKIT_HOME/lib64" \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

rm -rf prof
mkdir -p prof
source ${HOME}/vendors/customize/bin/set_env.bash
msprof op --output=prof ./execute_op

python3 - <<'PY'
import csv
import glob

csv_file = glob.glob('prof/*/OpBasicInfo.csv')[0]
with open(csv_file, newline='') as f:
    rows = list(csv.DictReader(f))

print(f'Profiling file: {csv_file}')
for row in rows:
    if 'add_custom_template' in ''.join(row.values()).lower():
        print(row)
PY
