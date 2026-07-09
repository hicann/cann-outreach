#!/bin/bash
if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

BUILD_DIR="build_out"
mkdir -p "${BUILD_DIR}"
rm -rf "${BUILD_DIR:?}"/*

target=package
if [ "$1"x != ""x ]; then
    target=$1
fi

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SOURCE_PACKAGE=True \
    -DENABLE_BINARY_PACKAGE=True \
    -DASCEND_COMPUTE_UNIT=ascend910b \
    -DENABLE_TEST=True \
    -Dvendor_name=customize \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_PYTHON_EXECUTABLE=python3 \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/${BUILD_DIR}" \
    -DENABLE_CROSS_COMPILE=False \
    -DCMAKE_CROSS_PLATFORM_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
    -DASCEND_PACK_SHARED_LIBRARY=False
cmake --build "${BUILD_DIR}" --target binary -j"$(nproc)"
cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"
