# ============================================================================
# 运行脚本 — Sign 算子
# 编译 → 生成数据 → 运行 → 精度验证
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="sign"
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in --skip-build) SKIP_BUILD=1 ;; esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 失败"
npu-smi info | grep -q "Ascend" || die "未检测到 NPU 设备"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译 ==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 失败"
    make -j4  || die "make 失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py 失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel 运行失败"
[ -f output/output.bin ] || die "output.bin 不存在"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "验证失败"

echo "=== 完成 ==="
exit 0
