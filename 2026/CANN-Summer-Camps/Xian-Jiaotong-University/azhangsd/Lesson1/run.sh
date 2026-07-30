#!/bin/bash
source /opt/conda/Ascend/cann-9.0.0/set_env.sh
mkdir -p build
cd build/
cmake ..
make -j8
./mul_test
