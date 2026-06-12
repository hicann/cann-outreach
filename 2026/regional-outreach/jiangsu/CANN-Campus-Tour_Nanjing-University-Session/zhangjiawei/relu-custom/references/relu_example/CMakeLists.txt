cmake_minimum_required(VERSION 3.14)
project(relu)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME})
endif()
add_definitions(-DASCENDC_DT_INT32)

include(cmake/package_script/func.cmake)
include(cmake/package_script/operator_cmake.cmake)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
# op_api: 使用独立的编译方式，见 examples/run.sh
# tests/ut tests/st: 暂未包含在构建中，可参考 examples/test_aclnn_relu.cpp 编写
