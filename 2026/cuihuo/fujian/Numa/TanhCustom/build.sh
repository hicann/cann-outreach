#!/bin/bash

# Source CANN environment
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# Set required environment variables
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASC_DIR=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/cmake

# Get script directory
script_path=$(realpath $(dirname $0))
cd $script_path

BUILD_DIR="build_out"
HOST_NATIVE_DIR="host_native_tiling"
mkdir -p build_out
rm -rf build_out/*

ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake -S . -B "$BUILD_DIR" --preset=default
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
cmake --build "$BUILD_DIR" --target $target -j$(nproc)
