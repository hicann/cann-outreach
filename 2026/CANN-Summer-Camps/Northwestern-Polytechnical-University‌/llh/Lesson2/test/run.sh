#!/bin/bash
# ============================================================================
# SubCustomTemplate 测试运行脚本 (外层 standalone)
# 用法: bash run.sh
# ============================================================================
set -e

source $ASCEND_TOOLKIT_HOME/set_env.sh
source ${HOME}/vendors/customize/bin/set_env.bash

g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

./execute_op
