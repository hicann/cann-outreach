#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    echo "Error: ASCEND_TOOLKIT_HOME is not set."
    exit 1
fi
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
echo "CANN environment initialized."

cd custom_op
bash build.sh
./build_out/custom_opp*.run --install-path="${HOME}"
cd ..

g++ -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

rm -rf prof
mkdir -p prof
source "${HOME}/vendors/customize/bin/set_env.bash"
msprof op --output=prof ./execute_op

python3 -c "
import csv
import glob

csv_file = glob.glob('prof/*/OpBasicInfo.csv')[0]
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print('OpBasicInfo:', csv_file)
for row in rows:
    print(row)
"
