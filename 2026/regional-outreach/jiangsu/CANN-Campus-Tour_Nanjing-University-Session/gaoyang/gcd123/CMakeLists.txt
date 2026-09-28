# Whole-project CMakeLists for the Gcd123 custom operator.
# Layout follows the official CANN FrameworkLaunch custom-operator sample:
#   cmake .. && make -j binary package
cmake_minimum_required(VERSION 3.16.0)
project(gcd123_op)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ASCEND_PATH / ASCEND_TOOLKIT_HOME point to the local CANN installation.
if(NOT DEFINED ASCEND_PATH)
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_PATH $ENV{ASCEND_TOOLKIT_HOME})
    else()
        set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

# Soc type of the target AI processor, e.g. ascend910b4.
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b4" CACHE STRING "Target SoC version")
endif()

list(APPEND CMAKE_MODULE_PATH ${ASCEND_PATH}/compiler/cmake
                             ${ASCEND_PATH}/lib64/cmake
                             ${ASCEND_PATH}/cmake)

find_package(okn-env REQUIRED)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
