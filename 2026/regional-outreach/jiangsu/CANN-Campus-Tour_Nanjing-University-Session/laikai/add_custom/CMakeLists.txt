cmake_minimum_required(VERSION 3.16)
project(add_custom LANGUAGES CXX)

set(SOC_VERSION "Ascend910B1" CACHE STRING "Ascend SoC target")
if (DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
elseif (DEFINED ENV{ASCEND_HOME_PATH})
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
else()
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${DEFAULT_ASCEND_CANN_PACKAGE_PATH}" CACHE PATH
    "Ascend CANN installation directory")

if (EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
elseif (EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
else()
    message(FATAL_ERROR "Cannot find ascendc_kernel_cmake under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

ascendc_library(add_custom_kernel STATIC op_kernel/add_custom.cpp)

add_library(add_custom_host OBJECT op_host/add_custom.cpp)

target_include_directories(add_custom_host PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel"
    "${ASCEND_CANN_PACKAGE_PATH}/include")
