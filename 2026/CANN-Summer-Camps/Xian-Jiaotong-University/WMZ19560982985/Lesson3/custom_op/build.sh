#!/bin/bash
set -e

if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi

echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"


BUILD_DIR="build_out"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"


ENABLE_CROSS="-DENABLE_CROSS_COMPILE=False"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=False"


target=package
if [ "$1"x != ""x ]; then
    target=$1
fi


cmake -S . \
      -B "$BUILD_DIR" \
      --preset=default \
      ${ENABLE_CROSS} \
      ${ENABLE_BINARY} \
      ${ENABLE_LIBRARY}


cmake --build "$BUILD_DIR" --target binary -j$(nproc)

cmake --build "$BUILD_DIR" --target "$target" -j$(nproc)